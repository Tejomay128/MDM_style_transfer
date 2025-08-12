import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import yaml

from train_utils import MaskedDiffusionTrainer
from data_utils import get_bible_dataset_full, get_wiki_large

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="", help="name of the model used to fetch the architecture config")
    parser.add_argument("--use_pretrained", action='store_true', default=False, help="whether to use pre-trained model weights")
    parser.add_argument("--pretrain_path", type=str, default="", help="path to the model's pre-trained weights")
    parser.add_argument("--training_mode", type=str, default='diff', help="training mode, one of ['diff', 'ar']")
    parser.add_argument("--tokenizer_path", type=str, default="", help="path to the tokenizer, either huggingface config name or path to the tokenizer")
    parser.add_argument("--compile", action='store_true', default=False, help="use torch.compile")
    parser.add_argument("--data_dir", type=str, default="", help="path to the dataset directory")
    parser.add_argument("--processed_data_dir", type=str, default="", help="path to the processed dataset")
    parser.add_argument("--data_name", type=str, default="", help="name of the data")

    parser.add_argument("--epsilon", type=float, default=1e-3, help="min value of difference between t=1 and t=t_max")
    parser.add_argument("--cfg", type=float, default=0.2, help="condition drop ratio for classifier free guidance")
    parser.add_argument("--seq_len", type=int, default=128, help="sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.03, help="weight decay regularization parameter")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="number of warmup steps")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--log_every", type=int, default=10, help="log every")
    parser.add_argument("--log_dir", type=str, default=".", help="log directory")
    parser.add_argument("--eval_every", type=int, default=10, help="eval every")
    parser.add_argument("--save_every", type=int, default=10, help="save every")
    parser.add_argument("--save_after", type=int, default=0, help="min training steps after which saving should start")
    parser.add_argument("--save_dir", type=str, default=".", help="save directory")
    parser.add_argument("--device", type=str, default=0, help="CUDA device ordinal value, -1 if cpu")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    if args.use_pretrained:
        assert args.tokenizer_path == 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right", use_fast=True)
    if not tokenizer.pad_token_id:
        print("Pad token not found, setting it to eos token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Vocab size:", tokenizer.vocab_size)

    if args.data_name == 'bible':
        train_data = get_bible_dataset_full(
            path=args.data_dir,
            mode='train',
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_proc=8
        )

        val_data = get_bible_dataset_full(
            path=args.data_dir,
            mode='val',
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_proc=8
        )
    elif args.data_name == 'wikilarge':
        train_data = get_wiki_large(
            path=args.data_dir,
            mode='train',
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_proc=8
        )

        val_data = get_wiki_large(
            path=args.data_dir,
            mode='val',
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_proc=8
        )
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.save_dir, exist_ok=True)

    args_dict = vars(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)


    trainer = MaskedDiffusionTrainer(
        model_name=args.model_name,
        use_pretrained=args.use_pretrained,
        pretrain_path=args.pretrain_path,
        training_mode=args.training_mode,
        tokenizer_path=args.tokenizer_path,
        compile_model=args.compile,
        tokenizer=tokenizer,
        epsilon=args.epsilon,
        cfg=args.cfg,
        lr=args.lr,
        min_lr=args.min_lr,
        max_length=args.seq_len,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        clip_grad_norm=args.clip_grad_norm,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        log_every=args.log_every,
        log_dir=args.log_dir,
        save_every=args.save_every,
        save_after=args.save_after,
        save_dir=args.save_dir,
        device=device
    )
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()