import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import random
import numpy as np
from tqdm import tqdm
import os
import json
from evaluate import load

from sample_utils import process_data, avg_scores

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="", help="name of the model used to fetch the architecture config")
    parser.add_argument("--model_ckpt", type=str, default="", help="path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="", help="path to the dataset directory")
    parser.add_argument("--data_name", type=str, default="bible", help="name of the data")
    parser.add_argument("--output_dir", type=str, default="", help="path to the output directory")
    parser.add_argument("--tokenizer_path", type=str, default="", help="path to the tokenizer, either huggingface config name or path to the tokenizer")

    parser.add_argument("--seq_len", type=int, default=128, help="sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--device", type=str, default=0, help="CUDA device ordinal value, -1 if cpu")
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


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True)

    # Ensure the padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for models using relative positional encoding
    tokenizer.padding_side = "left"

    model.eval()  # Set model to evaluation mode
    return model, tokenizer

def generate_text_batch(model, tokenizer, split_str, input_texts, device, max_seq_len=256, temperature=0.7, top_p=0.9):
    inputs = tokenizer(input_texts, return_tensors="pt", padding='longest', truncation=True, max_length=max_seq_len // 2)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_seq_len,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_texts = [sent.partition(split_str)[2] for sent in generated_texts]
    return generated_texts


def main():
    args = get_args()
    model, tokenizer = load_model_and_tokenizer(args.model_ckpt)
    dataset = process_data(path=args.data_dir, name=args.data_name)
    out_file_path = os.path.join(args.output_dir, 'sents.jsonl')

    if not os.path.exists(out_file_path):
        os.makedirs(args.output_dir, exist_ok=True)
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        generated_sents = []
        length = len(dataset)
        iterations = length // args.batch_size if length % args.batch_size == 0 else length // args.batch_size + 1
        for num_sample in range(args.n_samples):
            _seed_everything(args.seed + num_sample)
            print(f"Sampling with random seed: {args.seed + num_sample}")

            generated_texts_per_sample = []
            for i in tqdm(range(iterations), desc="Sampling: "):
                end_index = (i + 1) * args.batch_size if ((i + 1) * args.batch_size) < length else length
                batch = dataset[i * args.batch_size: end_index]
                split_str = "Translation:" if args.data_name == 'bible' else "#Simplified:"
                generated_text = generate_text_batch(
                    model, 
                    tokenizer, 
                    split_str,
                    input_texts=batch['src'], 
                    device=device,
                    max_seq_len=args.seq_len,
                    temperature=1.0,
                    top_p=0.95,
                )
                generated_texts_per_sample.extend(generated_text)
            
            for idx, sent in enumerate(generated_texts_per_sample):
                if num_sample == 0:
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
        for batch in f:
            data_dict = json.loads(batch)
            generated_sents.append(data_dict['sents'])
    
    out_file_path = out_file_path.replace("_wikilarge_more", "_wikilarge")
    with open(out_file_path, 'r') as f:
        for idx, batch in enumerate(f):
            data_dict = json.loads(batch)
            generated_sents[idx].extend(data_dict['sents'])
    
    print("Number of samples:", len(generated_sents[0]))
    reference_sents = dataset['tgt']
    source_sents = dataset['src']

    scores = avg_scores(generated_sents, reference_sents, source_sents, args.data_name)
    print(f"Results for {out_file_path}....")

    for k,v in scores.items():
        # print(f"{k}: {v['mean']:.3f}({v['std']:.3f})")
        print(f"{k}: $ {v['mean']:.3f}_{{\pm {v['std']:.3f}}} $")
    

if __name__ == "__main__":
    main()