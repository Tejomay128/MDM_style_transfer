import torch
import argparse
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import random
import numpy as np
import json
import fasttext

from sample_utils import diff_sample, diff_sample_svdd, avg_scores, process_data, get_fasttext_embeddings
from lit_gpt.diffmodel import TransEncoder, Config

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="", help="name of the model used to fetch the architecture config")
    parser.add_argument("--model_ckpt", type=str, default="", help="path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="", help="path to the dataset directory")
    parser.add_argument("--data_name", type=str, default="bible", help="name of the data")
    parser.add_argument("--output_dir", type=str, default="", help="path to the output directory")
    parser.add_argument("--tokenizer_path", type=str, default="", help="path to the tokenizer, either huggingface config name or path to the tokenizer")

    parser.add_argument("--epsilon", type=float, default=1e-3, help="min value of difference between t=1 and t=t_max")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="classifier free guidance scale")
    parser.add_argument("--seq_len", type=int, default=128, help="sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--nfe", type=int, default=64, help="Number of function evaluations")
    parser.add_argument("--M", type=int, default=1, help="If greater than 1, uses svdd decoding")
    parser.add_argument("--embedding_type", type=str, default="semantic", help="Used during SVDD. One of [semantic, static]")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    args = parser.parse_args()
    return args


def _seed_everything(seed: int = 42):
    """
    Seed all random number generators for reproducibility in PyTorch experiments.
    
    Args:
        seed (int): The seed value to set. Default is 42.
    """
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  


def main():
    args = get_args()
    out_file_path = os.path.join(args.output_dir, 'sents.jsonl')
    
    dataset = process_data(path=args.data_dir, name=args.data_name)
    
    if not os.path.exists(out_file_path):
        os.makedirs(args.output_dir, exist_ok=True)
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        assert args.M > 0, "Number of x0 samples to be considered should be atleast 1"
        
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                padding_side="right", use_fast=True)
        if not tokenizer.pad_token_id:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = 32000 if args.tokenizer_path == 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' else tokenizer.mask_token_id
        
        config = Config.from_name(args.model_name)
        if not args.tokenizer_path == 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T':
            config.vocab_size = tokenizer.vocab_size
            config.padded_vocab_size = tokenizer.vocab_size

        model = TransEncoder(config).to(device)
        ckpt_state_dict = torch.load(args.model_ckpt, map_location=device)
        model_ckpt = ckpt_state_dict['ema']
        model.load_state_dict(model_ckpt)

        if args.M > 1:
            if args.embedding_type == 'semantic':
                sentence_embedder = SentenceTransformer("all-mpnet-base-v2")
                sentence_embedder.eval()
            else:
                FASTTEXT_MODEL_PATH = "cc.en.300.bin"
                sentence_embedder = fasttext.load_model(FASTTEXT_MODEL_PATH)

        generated_sents = []
        length = len(dataset)
        print("CFG scale: ", args.cfg_scale)
        iterations = length // args.batch_size if length % args.batch_size == 0 else length // args.batch_size + 1
        for n_sample in range(args.n_samples):
            _seed_everything(args.seed + n_sample)  # separate random seeds for each sampling iteration
            print(f"Sampling with random seed: {args.seed + n_sample}")
            generated_text_per_sample = []
            for i in tqdm(range(iterations), desc="Sampling: "):
                end_index = (i + 1) * args.batch_size if ((i + 1) * args.batch_size) < length else length
                data = dataset[i * args.batch_size: end_index]
                input_ids = tokenizer(
                    data['src'], 
                    padding="longest", 
                    truncation=True, 
                    return_tensors="pt", 
                    add_special_tokens=False
                )['input_ids'].to(device)
                
                if args.M > 1:
                    input_data = [data.replace(" Translation:", "") for data in data['src']]
                    input_data = [data.replace(" #Simplified:", "") for data in input_data]
                    input_data = [data.replace("<ASV> ", "") for data in input_data]
                    input_data = [data.replace("<BBE> ", "") for data in input_data]
                    if args.embedding_type == 'semantic':
                        input_embeds = sentence_embedder.encode(input_data, convert_to_tensor=True, show_progress_bar=False)
                    else:
                        input_embeds = get_fasttext_embeddings(ft_model=sentence_embedder, sentences=input_data)
                    input_embeds = input_embeds.to(device)
                    out_ids = diff_sample_svdd(
                        model=model,
                        tokenizer=tokenizer,
                        input_embeds=input_embeds,
                        sent_embedder=sentence_embedder,
                        embedding_type=args.embedding_type,
                        prompt=input_ids,
                        batch_size=args.batch_size,
                        steps=args.nfe,
                        temperature=1.0,
                        cfg_scale=args.cfg_scale,
                        context_length=args.seq_len,
                        eps=args.epsilon,
                        M=args.M,
                        dim=32000 if args.tokenizer_path == 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' else tokenizer.mask_token_id,
                        split_str="Translation:" if args.data_name == 'bible' else "#Simplified:",
                        device=device
                    )
                else:
                    out_ids = diff_sample(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=input_ids,
                        batch_size=args.batch_size,
                        alg='greedy',
                        steps=args.nfe,
                        temperature=1.0,
                        cfg_scale=args.cfg_scale,
                        context_length=args.seq_len,
                        eps=args.epsilon,
                        dim=32000 if args.tokenizer_path == 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' else tokenizer.mask_token_id,
                        device=device,
                    )
                generated_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                split_str = "Translation: " if args.data_name == 'bible' else "#Simplified: "
                for sent in generated_text:
                    sent = sent.partition(split_str)[2]
                    generated_text_per_sample.append(sent)
               
            for idx, sent in enumerate(generated_text_per_sample):
                if n_sample == 0:
                    generated_sents.append([sent])
                else:
                    generated_sents[idx].append(sent)
        
        os.makedirs(args.output_dir, exist_ok=True)

        with open(out_file_path, 'w') as f:
            for idx, sents in enumerate(generated_sents):
                json.dump({'idx': idx, 'sents': sents}, f)
                f.write('\n')
    
    generated_sents = []
    with open(out_file_path, 'r') as f:
        for data in f:
            data_dict = json.loads(data)
            generated_sents.append(data_dict['sents'])
    
    # out_file_path = out_file_path.replace("_wikilarge_more", "_wikilarge")
    # out_file_path = out_file_path.replace("_more", "")
    # with open(out_file_path, 'r') as f:
    #     for idx, batch in enumerate(f):
    #         data_dict = json.loads(batch)
    #         generated_sents[idx].extend(data_dict['sents'])
    
    print("Number of samples:", len(generated_sents[0]))
    reference_sents = dataset['tgt']
    source_sents = dataset['src']

    scores = avg_scores(generated_sents, reference_sents, source_sents, data_name=args.data_name)
    print(f"Results for {out_file_path}....")
    # print(scores)

    for k,v in scores.items():
        # print(f"{k}: {v['mean']:.3f}({v['std']:.3f})")
        print(f"{k}: $ {v['mean']:.3f}_{{\pm {v['std']:.3f}}} $")


if __name__ == '__main__':
    main()