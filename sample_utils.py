import torch
from torch.nn import functional as F
from tqdm import tqdm
from datasets import Dataset as HFDataset
import os
from lens import download_model, LENS
import numpy as np
import evaluate
import fasttext
from nltk.util import ngrams
from nltk import word_tokenize
from typing import List

def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def add_gumbel_noise_multiple_samples(logits, temperature, num_samples):
    """
    Generates multiple Gumbel-distributed samples from given logits.

    Args:
        logits (torch.Tensor): The input logits.
        temperature (float): The temperature parameter.
        num_samples (int): The number of samples to generate.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, *logits.shape) containing the sampled values.
    """
    logits = logits.to(torch.float64)  # Ensure float64 for precision
    noise = torch.rand((num_samples, *logits.shape), dtype=torch.float64, device=logits.device)
    gumbel_noise = (- torch.log(noise)) ** temperature
    expanded_logits = logits.expand((num_samples, *logits.shape))
    expanded_logits = expanded_logits.exp() / gumbel_noise

    return expanded_logits


@torch.no_grad()
def diff_sample(model, tokenizer, prompt=None, batch_size=1, alg='origin', steps=512, temperature=1., cfg_scale=2.,
                context_length=2048, eps=1e-5, dim=32000, device='cuda'):
    batch_size = batch_size if prompt is None else prompt.shape[0]
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    for i in tqdm(range(steps), total=steps):
        mask_index = (x == dim)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[:, :prompt.shape[1]] = dim
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits, un_logits = logits[mask_index], un_logits[mask_index]
            else:
                logits = model(x)[mask_index]

        if cfg_scale > 0.:
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + dim
            transfer_index_t_s = torch.rand(*x0.shape, device='cuda') < p_transfer
            logits_with_noise = add_gumbel_noise(logits[transfer_index_t_s], temperature=temperature)
            x0[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)
            x[mask_index] = x0.clone()
        elif alg == 'greedy':
            logits_with_noise = add_gumbel_noise_multiple_samples(logits, temperature=temperature, num_samples=1)[0]
            x0 = torch.argmax(logits_with_noise, dim=-1)
            logits = logits.to(torch.float64)
            p = F.softmax(logits, dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            num_mask_token = mask_index.sum()
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
            if number_transfer_tokens > 0:
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + dim
                x0_[transfer_index] = x0[transfer_index].clone()
                x[mask_index] = x0_
        else:
            raise NotImplementedError(alg)

    return x


def _get_logits(model, x, prompt, mask_index, cfg_scale, dim):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[:, :prompt.shape[1]] = dim
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_)
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits, un_logits = logits[mask_index], un_logits[mask_index]
        else:
            logits = model(x)[mask_index]
    
    if cfg_scale > 0.:
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

    return logits


def _compute_posterior(x, logits, mask_index, temperature, s, t, i, steps, dim):
    """
    Compute the posterior diffusion sample based on the greedy approach
    from the SMDM paper: https://arxiv.org/pdf/2410.18514v1
    """
    device = x.device
    
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0_pred = torch.argmax(logits_with_noise, dim=-1)
    logits = logits.to(torch.float64)
    p = F.softmax(logits, dim=-1)
    confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0_pred, -1)).squeeze(dim=-1)
    num_mask_token = mask_index.sum()
    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
    
    xs = x.clone()
    if number_transfer_tokens > 0:
        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
        x0_ = torch.zeros_like(x0_pred, device=device, dtype=torch.long) + dim
        x0_[transfer_index] = x0_pred[transfer_index].clone()
        xs[mask_index] = x0_
    
    return xs

@torch.no_grad()
def diff_sample_svdd(model, tokenizer, input_embeds, sent_embedder, embedding_type='semantic', prompt=None, batch_size=1, steps=512, temperature=1., cfg_scale=2.,
                context_length=2048, M=5, eps=1e-5, dim=32000, split_str="Translation:", device='cuda'):
    """
    Sampling with soft-value based diffusion decoding.
    Args:
        model: trained masked diffusion denoiser
        tokenizer: tokenizer for the model
        input_embeds: input embeddings for the prompt from the verifier
        sent_embedder: sentence embedding model (verifier)
        prompt: input ids for conditional generation
        batch_size: batch size
        steps: number of sampling steps
        temperature: temperature for sampling
        cfg_scale: classifier free guidance scale for computing logits
        context_length: context length
        M: number of x0 samples to be considered
        eps: epsilon for sampling
        dim: mask token id
        device: torch device
    """
    batch_size = batch_size if prompt is None else prompt.shape[0]
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    for i in tqdm(range(steps), total=steps):
        mask_index = (x == dim)
        logits = _get_logits(model, x, prompt, mask_index, cfg_scale, dim)
        t = timesteps[i]
        s = timesteps[i + 1]

        if i < steps - 1:
            xs_list = []
            rewards_list = []
            for _ in range(M):
                xs = _compute_posterior(x, logits, mask_index, temperature, s, t, i, steps, dim)
                xs_logits = _get_logits(model, xs, prompt, mask_index, cfg_scale, dim)
                logits_with_noise = add_gumbel_noise(xs_logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                x0_full = x.clone()
                x0_full[mask_index] = x0
                reward = _decode_and_reward(
                    sent_embedder=sent_embedder, 
                    tokenizer=tokenizer, 
                    x_candidates=x0_full, 
                    input_embeds=input_embeds, 
                    embedding_type=embedding_type,
                    emptiness='<s>', 
                    split_str=split_str
                )
                rewards_list.append(reward)
                xs_list.append(xs)
            
            # select the xs with the best reward
            xs_list = torch.stack(xs_list)
            rewards_list = torch.stack(rewards_list)
            best_rewards_idx = torch.argmax(rewards_list, dim=0)
            x = xs_list[best_rewards_idx, torch.arange(batch_size)]
        else:
            x = _compute_posterior(x, logits, mask_index, temperature, s, t, i, steps, dim)

    return x


def _decode_and_reward(
        sent_embedder, 
        tokenizer, 
        x_candidates, 
        input_embeds, 
        embedding_type='semantic', 
        emptiness='<s>', 
        split_str='Translation:'
    ):
    """
    Decodes the candidate x0 predictions, passes them through a verifier and obtains
    a reward for each candidate. 
    """
    candidate_sents = tokenizer.batch_decode(x_candidates, skip_special_tokens=True)
    candidate_sents = [sent.partition(split_str)[2] for sent in candidate_sents]
    # Handling potentially empty sentences
    candidate_sents = [sent.strip() if len(sent) > 0 else emptiness for sent in candidate_sents]
    if embedding_type == 'semantic':
        candidate_embs = sent_embedder.encode(candidate_sents, convert_to_tensor=True).to(input_embeds.device)
    else:
        candidate_embs = get_fasttext_embeddings(sent_embedder, candidate_sents)
        candidate_embs = candidate_embs.to(input_embeds.device)

    candidate_rewards = torch.cosine_similarity(candidate_embs, input_embeds, dim=-1)
    return candidate_rewards


def get_fasttext_embeddings(ft_model, sentences):
    """
    sentences: List of strings (batch of sentences)
    Returns: torch.Tensor of shape [batch_size, embedding_dim]
    """
    batch_embeddings = []
    embedding_dim = ft_model.get_dimension()

    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())  # Tokenize and lowercase
        word_embeddings = []

        for token in tokens:
            word_vec = ft_model.get_word_vector(token)  # returns a numpy array
            word_embeddings.append(word_vec)

        if word_embeddings:
            sent_embedding = np.mean(word_embeddings, axis=0)
        else:
            sent_embedding = np.zeros(embedding_dim)

        batch_embeddings.append(sent_embedding)
    batch_embeddings = np.array(batch_embeddings)

    return torch.tensor(batch_embeddings, dtype=torch.bfloat16)  # Shape: [batch_size, embedding_dim]

@torch.no_grad()
def ar_sample_kvcache(gpt, tokenizer, prompt, temperature=1., context_length=2048, device='cuda'):
    gpt.eval()
    gpt.reset_cache()

    prev_pos = 0
    for cur_pos in range(prompt.shape[1], context_length):
        input_pos = torch.arange(cur_pos, dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = gpt(prompt[:, prev_pos:cur_pos], input_pos=input_pos)[:, -1]

        next_token = top_p_sample(logits, p=0.95)

        prompt = torch.cat([prompt, next_token], dim=-1)
        prev_pos = cur_pos
        if next_token[0] == torch.tensor([tokenizer.eos_token_id], device=device):
            break
    return prompt


def top_p_sample(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above p
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the mask to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Apply mask to logits (shape [1, vocab_size])
    logits[0, sorted_indices[sorted_indices_to_remove]] = -float("Inf")

    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1)


def process_data(path, name='bible', **kwargs):
    if name == 'bible':
        src_file_name = "Test.sourc"
        tgt_file_name = "Test.tgt"
        with open (os.path.join(path, src_file_name), 'r') as fsrc, open (os.path.join(path, tgt_file_name), 'r') as ftgt:
            src_lines = [line.strip() for line in fsrc.readlines()]
            tgt_lines = [line.strip() for line in ftgt.readlines()]
        
    elif name == 'wikilarge':
        src_file_name = "test.8turkers.tok.norm"
        tgt_file_name = "test.8turkers.tok.turk."

        with open(os.path.join(path, src_file_name), 'r') as fsrc:
            src_lines = [line.strip() for line in fsrc.readlines()]
        
        tgt_lines = []
        for i in range(8):
            with open(os.path.join(path, f"{tgt_file_name}{i}"), 'r') as fsrc:
                lines = fsrc.readlines()
                for j, line in enumerate(lines):
                    line = line.strip()
                    if i == 0:
                        tgt_lines.append([line])
                    else:
                        tgt_lines[j].append(line)
    else:
        raise ValueError(f"Invalid dataset name: {name}")

    dataset = HFDataset.from_dict(
        {
            "src": src_lines, 
            "tgt": tgt_lines,
        }
    )

    def _preprocess(example):
        sep_string = " #Simplified:" if name == 'wikilarge' else " Translation:"
        example['src'] = [i + sep_string for i in example['src']]
        return example

    dataset = dataset.map(_preprocess, batched=True, num_proc=8)
    print(dataset[0])
    return dataset


def calculate_pinc(
        sources: List[str], 
        candidates: List[str], 
        max_n: int = 4
    ):
    """
    Calculate PINC scores for batches of source-candidate pairs
    
    Args:
        sources (List[str]): List of original texts
        candidates (List[str]): List of generated texts
        max_n (int): Maximum n-gram size (default: 4)
        
    Returns:
        PINC score (float)
    """
    assert len(sources) == len(candidates), "Input lists must be same length"
    
    # Pre-tokenize all texts
    src_batch = [[t.lower() for t in word_tokenize(s)] for s in sources]
    cand_batch = [[t.lower() for t in word_tokenize(c)] for c in candidates]
    
    scores = [0.0] * len(sources)
    
    for n in range(1, max_n + 1):
        # Generate n-grams for entire batch
        src_ngrams = [set(ngrams(tokens, n)) for tokens in src_batch]
        cand_ngrams = [set(ngrams(tokens, n)) for tokens in cand_batch]
        
        # Calculate scores for this n-gram level
        for i in range(len(sources)):
            unique = cand_ngrams[i] - src_ngrams[i]
            denom = len(cand_ngrams[i])
            scores[i] += len(unique) / denom if denom else 0.0
    
    # Average across all n-gram sizes
    scores = [score / max_n for score in scores]
    score = sum(scores) / len(scores)

    return score


def avg_scores(predictions_list, references, sources, data_name):
    """
    Computes the BLEU, ROUGE and METEOR scores using Minimum Bayes Risk (MBR) decoding.

    Parameters:
    - predictions_list (list of list of str): List where each element contains multiple candidate predictions for an input.
    - references (list of str): List of reference texts corresponding to each input.

    Returns:
    - float: The average BLEU score of MBR-selected candidates against references.
    """
    print("Computing scores ...")
    bleu = evaluate.load("bleu")
    lens_path = download_model("davidheineman/lens")
    lens = LENS(lens_path, rescale=True)
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")
    sari = evaluate.load("sari")

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    meteor_scores = []
    pinc_scores = []
    bert_scores = []
    sle_scores = []
    sle_delta_scores = []
    sari_scores = []
    lens_scores = []
    num_candiates = len(predictions_list[0])

    if isinstance(references[0], str):
        references=[[ref] for ref in references]
    
    for i in range(num_candiates):
        candidates = [preds[i] for preds in predictions_list]
        bleu_result = bleu.compute(predictions=candidates, references=references)
        bleu_scores.append(bleu_result['bleu'])
        if data_name == 'bible':
            rouge_result = rouge.compute(predictions=candidates, references=references)
            rouge1_scores.append(rouge_result['rouge1'])
            rouge2_scores.append(rouge_result['rouge2'])
            rougeL_scores.append(rouge_result['rougeL'])

            meteor_result = meteor.compute(predictions=candidates, references=references)
            meteor_scores.append(meteor_result['meteor'])
            
            pinc_score = calculate_pinc(sources=sources, candidates=candidates)
            pinc_scores.append(pinc_score)

            bert_score_result = bert_score.compute(predictions=candidates, references=references, lang='en')
            avg_f1 = sum(bert_score_result['f1']) / len(bert_score_result['f1'])
            bert_scores.append(avg_f1)

        elif data_name == 'wikilarge':  
            # sle_score = calculate_sle(sources=sources, candidates=candidates)
            # sle_scores.append(sle_score['sle'])
            # sle_delta_scores.append(sle_score['sle_delta'])

            sari_score = sari.compute(sources=sources, predictions=candidates, references=references)
            sari_scores.append(sari_score['sari'])

            lens_score = lens.score(sources, candidates, references, batch_size=64, devices=[0])
            lens_score = sum(lens_score) / len(lens_score)
            lens_scores.append(lens_score)
        else:
            raise ValueError("Invalid data name")

    if data_name == 'bible':
        return {
            "BLEU": {'mean': np.mean(np.array(bleu_scores)) * 100, 'std': np.std(np.array(bleu_scores)) * 100},
            "ROUGE1": {'mean': np.mean(np.array(rouge1_scores)) * 100, 'std': np.std(np.array(rouge1_scores)) * 100},
            "ROUGE2": {'mean': np.mean(np.array(rouge2_scores)) * 100, 'std': np.std(np.array(rouge2_scores)) * 100},
            "ROUGEL": {'mean': np.mean(np.array(rougeL_scores)) * 100, 'std': np.std(np.array(rougeL_scores)) * 100},
            "METEOR": {'mean': np.mean(np.array(meteor_scores)) * 100, 'std': np.std(np.array(meteor_scores)) * 100},
            "PINC": {'mean': np.mean(np.array(pinc_scores)) * 100, 'std': np.std(np.array(pinc_scores)) * 100},
            "BERT_SCORE": {'mean': np.mean(np.array(bert_scores)) * 100, 'std': np.std(np.array(bert_scores)) * 100},
        }
    elif data_name == 'wikilarge':  
        return {
            "BLEU": {'mean': np.mean(np.array(bleu_scores)) * 100, 'std': np.std(np.array(bleu_scores)) * 100},
            "SARI": {'mean': np.mean(np.array(sari_scores)), 'std': np.std(np.array(sari_scores))},
            # "SLE": {'mean': np.mean(np.array(sle_scores)), 'std': np.std(np.array(sle_scores))},
            # "SLE_DELTA": {'mean': np.mean(np.array(sle_delta_scores)), 'std': np.std(np.array(sle_delta_scores))},
            "LENS": {'mean': np.mean(np.array(lens_scores)), 'std': np.std(np.array(lens_scores))},
        }
    else:
        return {
            "BLEU": {'mean': np.mean(np.array(bleu_scores)) * 100, 'std': np.std(np.array(bleu_scores)) * 100},
        }
        
