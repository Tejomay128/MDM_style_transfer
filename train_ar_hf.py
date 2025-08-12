import torch
from datasets import Dataset
from torch.optim.lr_scheduler import LambdaLR
import copy
import argparse
import os
import yaml
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="", help="name of the model used to fetch the architecture config")
    parser.add_argument("--tokenizer_path", type=str, default="", help="path to the tokenizer, either huggingface config name or path to the tokenizer")
    parser.add_argument("--compile", action='store_true', default=False, help="use torch.compile")
    parser.add_argument("--data_dir", type=str, default="", help="path to the dataset directory")
    parser.add_argument("--data_name", type=str, default="", help="name of the data")

    parser.add_argument("--seq_len", type=int, default=128, help="sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.03, help="weight decay regularization parameter")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="number of warmup steps")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="number of gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--log_every", type=int, default=10, help="log every")
    parser.add_argument("--log_dir", type=str, default=".", help="log directory")
    parser.add_argument("--eval_every", type=int, default=10, help="eval every")
    parser.add_argument("--save_every", type=int, default=10, help="save every")
    parser.add_argument("--save_after", type=int, default=0, help="min training steps after which saving should start")
    parser.add_argument("--save_dir", type=str, default=".", help="save directory")

    args = parser.parse_args()
    return args
    

def create_dataset(name, mode, path, tokenizer, seq_len=256, num_proc=8):
    if name == 'bible':
        if mode == 'train':
            src_file_name = ["Train1.sourc", "Train2.sourc", "Train3.sourc", "Train4.sourc"]
            tgt_file_name = ["Train1.tgt", "Train2.tgt", "Train3.tgt", "Train4.tgt"]
        elif mode == 'val':
            src_file_name = ["Dev.sourc"]
            tgt_file_name = ["Dev.tgt"]
        else:
            src_file_name = ["Test.sourc"]
            tgt_file_name = ["Test.tgt"]

        src_lines = [line.strip().lower() for src_file in src_file_name for line in open(os.path.join(path, src_file)).readlines()]
        tgt_lines = [line.strip().lower() for tgt_file in tgt_file_name for line in open(os.path.join(path, tgt_file)).readlines()]

        if mode == 'val':
            src_lines = src_lines
            tgt_lines = tgt_lines

    elif name == 'wikilarge':
        if mode == 'train':
            src_file_name = "wiki.full.aner.ori.train.src"
            tgt_file_name = "wiki.full.aner.ori.train.dst"
        elif mode == 'val':
            src_file_name = "wiki.full.aner.ori.valid.src"
            tgt_file_name = "wiki.full.aner.ori.valid.dst"
        elif mode == 'test':
            src_file_name = "wiki.full.aner.ori.test.src"
            tgt_file_name = "wiki.full.aner.ori.test.dst"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        with open (os.path.join(path, src_file_name), 'r') as fsrc, open (os.path.join(path, tgt_file_name), 'r') as ftgt:
            src_lines = [line.strip() for line in fsrc.readlines()]
            tgt_lines = [line.strip() for line in ftgt.readlines()]
        
        # if mode == 'train' or mode == 'val':
        #     # Remove -LRB- <text> -RRB-
        #     pattern = r"-LRB-.*?RRB-"
        #     for i in tqdm(range(len(src_lines))):
        #         src_lines[i] = re.sub(pattern, '', src_lines[i])
        #         tgt_lines[i] = re.sub(pattern, '', tgt_lines[i])
        #         # extra whitespace removal
        #         src_lines[i] = re.sub(r'\s+', ' ', src_lines[i]).strip()
        #         tgt_lines[i] = re.sub(r'\s+', ' ', tgt_lines[i]).strip()

    else:
        raise ValueError(f"Invalid dataset: {name}")
    
    dataset = Dataset.from_dict(
            {
                "src": src_lines, 
                "tgt": tgt_lines,
            }
        )

    def _tokenize(example):
        sep_string = " #Simplified:" if name == 'wikilarge' else " Translation:"
        example['src'] = [i + sep_string for i in example['src']]
        example['tgt'] = [i + tokenizer.eos_token for i in example['tgt']]

        input_tokens = tokenizer(example['src'], max_length=seq_len // 2, truncation=True, padding=False, add_special_tokens=False)
        output_tokens = tokenizer(example['tgt'], max_length=seq_len // 2, truncation=True, padding=False, add_special_tokens=False)

        tokens = {
            'input_ids':    [i + o for i, o in zip(input_tokens['input_ids'], output_tokens['input_ids'])],
            'src_length':   [len(i) for i in input_tokens['input_ids']],
            'length':       [len(i) + len(o) for i,o in zip(input_tokens['input_ids'], output_tokens['input_ids'])],
        }   
        return tokens
    
    tokenized_dataset = dataset.map(_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)

    def _pad(example):
        example['attention_mask'] = [[1] * len(i) + [0] * (seq_len - len(i)) for i in example['input_ids']]
        example['input_ids'] =      [i + [tokenizer.pad_token_id] * (seq_len - len(i)) for i in example['input_ids']]
        return example
    
    tokenized_dataset = tokenized_dataset.map(_pad, batched=True, num_proc=num_proc, load_from_cache_file=True)

    def _make_labels(example):
        example['labels'] = copy.deepcopy(example['input_ids'])
        # example['labels'] = [-100 if i == tokenizer.pad_token_id else i for i in example['labels']]
        # setting to -100 makes the loss function ignore loss computation of those positions
        example['labels'] = [
            -100 if (idx < src_length and idx >= length) else label 
            for idx, (src_length, length, label) in enumerate(zip(example['src_length'], example['length'], example['labels']))
            ]
        return example

    tokenized_dataset = tokenized_dataset.map(_make_labels, batched=True, num_proc=num_proc, load_from_cache_file=True)

    print("SRC text from the dataset: ", tokenized_dataset[0]['src'])
    print("TGT text from the dataset: ", tokenized_dataset[0]['tgt'])
    print("Tokenized sample from the dataset: ", tokenized_dataset[0]['input_ids'])
    tokenized_dataset = tokenized_dataset.with_format('torch')
    return tokenized_dataset


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if optimizer is None:
            optimizer = self.optimizer

        # Retrieve warmup steps from TrainingArguments
        warmup_steps = self.args.get_warmup_steps(num_training_steps)

        # Custom learning rate schedule: Linear warmup and inverse square root decay
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))  # Linear warmup

            return (warmup_steps ** 0.5) / (current_step ** 0.5)  # Inverse square root decay

        return LambdaLR(optimizer, lr_lambda)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        self.lr_scheduler = self.create_scheduler(num_training_steps, self.optimizer)


def main():
    args = get_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_dataset(args.data_name, 'train', args.data_dir, tokenizer, seq_len=args.seq_len, num_proc=8)
    val_dataset = create_dataset(args.data_name, 'val', args.data_dir, tokenizer, seq_len=args.seq_len, num_proc=8)

    os.makedirs(args.save_dir, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_every,
        save_strategy="steps",
        save_steps=args.save_every,
        save_total_limit=10,
        logging_dir=args.log_dir,
        logging_steps=args.log_every,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        max_grad_norm=args.clip_grad_norm,
        adam_beta1=0.9,
        adam_beta2=0.95,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        torch_compile=args.compile,
        bf16=True,
    )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()