import torch
import argparse
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import random
import numpy as np
import json

from sample_utils import ar_sample_kvcache, process_data, mbr_scores
from lit_gpt.model_cache import GPTCache, Config

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="", help="name of the model used to fetch the architecture config")
    parser.add_argument("--model_ckpt", type=str, default="", help="path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="", help="path to the dataset directory")
    parser.add_argument("--data_name", type=str, default="bible", help="name of the data")
    parser.add_argument("--output_dir", type=str, default="", help="path to the output directory")
    parser.add_argument("--tokenizer_path", type=str, default="", help="path to the tokenizer, either huggingface config name or path to the tokenizer")

    parser.add_argument("--seq_len", type=int, default=128, help="sequence length")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--device", type=str, default=0, help="CUDA device ordinal value, -1 if cpu")

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
    _seed_everything(42)
    out_file_path = os.path.join(args.output_dir, 'sents.jsonl')
    
    if args.data_name == 'bible':
        dataset = process_data(path=args.data_dir, name=args.data_name)
    elif args.data_name == 'translation':
        dataset = process_data(path=args.data_dir, name=args.data_name, src_lang='en', tgt_lang='ro')
    else:
        raise NotImplementedError
    
    if not os.path.exists(out_file_path):
        os.makedirs(args.output_dir, exist_ok=True)
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                                                padding_side="right", use_fast=True)
        
        config = Config.from_name(args.model_name)

        model = GPTCache(config).to(device)
        ckpt_state_dict = torch.load(args.model_ckpt, map_location=device)
        model_ckpt = ckpt_state_dict['model']
        model.load_state_dict(model_ckpt)
        model.eval()

        generated_sents = {}
        length = len(dataset)
        it = 0
        for i in tqdm(range(length), desc="Sampling: "):
            data = dataset[i]
            input_ids = tokenizer(data['src'], return_tensors="pt")['input_ids'].to(device)
            for _ in range(args.n_samples):
                out_ids = ar_sample_kvcache(
                    gpt=model,
                    tokenizer=tokenizer,
                    prompt=input_ids,
                    temperature=1.0,
                    context_length=args.seq_len,
                    device=device
                )
                sent = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
                sent = sent.partition("Translation: ")[2]
                # print(sent.partition("Translation : "))
                generated_sents[i] = generated_sents.get(i, []) + [sent]            
        
        os.makedirs(args.output_dir, exist_ok=True)

        # with open(os.path.join(args.output_dir, 'sents.txt'), 'w') as f:
        #     for sent in generated_sents:
        #         f.write(sent + "\n")
        with open(os.path.join(out_file_path), 'w') as f:
            for idx, sents in generated_sents.items():
                json.dump({'idx': idx, 'sents': sents}, f)
                f.write('\n')
    
    generated_sents = []
    with open(out_file_path, 'r') as f:
        for data in f:
            data_dict = json.loads(data)
            generated_sents.append(data_dict['sents'])
    # generated_sents = [generated_sents.partition("Translation: ")[2] for generated_sents in generated_sents]
    reference_sents = dataset['tgt']
    source_sents = dataset['src']
    print(len(generated_sents), len(reference_sents))
    scores = mbr_scores(generated_sents, reference_sents, source_sents)
    print(f"Results: ")
    print(scores)


if __name__ == '__main__':
    main()