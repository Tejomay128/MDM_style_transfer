from datasets import Dataset as HFDataset
import re
from tqdm import tqdm
import os


def get_bible_dataset(path, mode, tokenizer, seq_len=256, num_proc=8):
    if mode == 'train':
        src_file_name = "Train.sourc"
        tgt_file_name = "Train.tgt"
    elif mode == 'val':
        src_file_name = "Dev.sourc"
        tgt_file_name = "Dev.tgt"
    else:
        src_file_name = "Test.sourc"
        tgt_file_name = "Test.tgt"

    src_lines = [line.strip() for line in open(os.path.join(path, src_file_name)).readlines()]
    tgt_lines = [line.strip() for line in open(os.path.join(path, tgt_file_name)).readlines()]

    # print("SRC:", src_lines[0])
    # print("TGT:", tgt_lines[0])
    dataset = HFDataset.from_dict(
        {
            "src": src_lines, 
            "tgt": tgt_lines,
        }
    )

    def _tokenize(example):
        example['src'] = [tokenizer.eos_token + i + " Translation:" for i in example['src']]
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
        example['input_ids'] =      [i + [tokenizer.pad_token_id] * (seq_len - len(i)) for i in example['input_ids']]
        example['attention_mask'] = [[1] * len(i) + [0] * (seq_len - len(i)) for i in example['input_ids']]
        return example
    
    tokenized_dataset = tokenized_dataset.map(_pad, batched=True, num_proc=num_proc, load_from_cache_file=True)

    print("SRC text from the dataset: ", tokenized_dataset[0]['src'])
    print("TGT text from the dataset: ", tokenized_dataset[0]['tgt'])
    print("Tokenized sample from the dataset: ", tokenized_dataset[0]['input_ids'])
    print("length of the source of the tokenized text: ", tokenized_dataset[0]['src_length'])
    tokenized_dataset = tokenized_dataset.with_format('torch')
    return tokenized_dataset


def get_bible_dataset_full(path, mode, tokenizer, seq_len=256, num_proc=8):
    if mode == 'train':
        src_file_name = ["Train1.sourc", "Train2.sourc", "Train3.sourc", "Train4.sourc"]
        tgt_file_name = ["Train1.tgt", "Train2.tgt", "Train3.tgt", "Train4.tgt"]
    elif mode == 'val':
        src_file_name = ["Dev.sourc"]
        tgt_file_name = ["Dev.tgt"]
    elif mode == 'test':
        src_file_name = ["Test.sourc"]
        tgt_file_name = ["Test.tgt"]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    src_lines = [line.strip() for src_file in src_file_name for line in open(os.path.join(path, src_file)).readlines()]
    tgt_lines = [line.strip() for tgt_file in tgt_file_name for line in open(os.path.join(path, tgt_file)).readlines()]

    # print("SRC:", src_lines[0])
    # print("TGT:", tgt_lines[0])
    if mode == 'val':
        src_lines = src_lines[:10000]
        tgt_lines = tgt_lines[:10000]
    
    dataset = HFDataset.from_dict(
        {
            "src": src_lines, 
            "tgt": tgt_lines,
        }
    )

    def _tokenize(example):
        example['src'] = [i + " Translation:" for i in example['src']]
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
        example['input_ids'] =      [i + [tokenizer.pad_token_id] * (seq_len - len(i)) for i in example['input_ids']]
        example['attention_mask'] = [[1] * len(i) + [0] * (seq_len - len(i)) for i in example['input_ids']]
        return example
    
    tokenized_dataset = tokenized_dataset.map(_pad, batched=True, num_proc=num_proc, load_from_cache_file=True)

    print("SRC text from the dataset: ", tokenized_dataset[0]['src'])
    print("TGT text from the dataset: ", tokenized_dataset[0]['tgt'])
    print("Tokenized sample from the dataset: ", tokenized_dataset[0]['input_ids'])
    print("length of the source of the tokenized text: ", tokenized_dataset[0]['src_length'])
    tokenized_dataset = tokenized_dataset.with_format('torch')
    return tokenized_dataset


def get_wiki_large(path, mode, tokenizer, seq_len=256, num_proc=8):
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
        src_lines = [line.strip().lower() for line in fsrc.readlines()]
        tgt_lines = [line.strip().lower() for line in ftgt.readlines()]

    dataset = HFDataset.from_dict(
        {
            "src": src_lines, 
            "tgt": tgt_lines,
        }
    )

    def _tokenize(example):
        example['src'] = [i + " #Simplified:" for i in example['src']]
        example['tgt'] = [i + tokenizer.eos_token for i in example['tgt']]

        input_tokens = tokenizer(example['src'], padding=False, add_special_tokens=False)['input_ids']
        output_tokens = tokenizer(example['tgt'], padding=False, add_special_tokens=False)['input_ids']

        tokens = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }   
        return tokens

    tokenized_dataset = dataset.map(_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_tokens']) <= seq_len // 2)

    def _pad(example):
        example['input_ids'] =      [i + o[:seq_len // 2] for i, o in zip(example['input_tokens'], example['output_tokens'])]
        example['src_length'] =     [len(i) for i in example['input_tokens']]
        example['length'] =         [len(i) for i in example['input_ids']]
        example['attention_mask'] = [[1] * len(i) + [0] * (seq_len - len(i)) for i in example['input_ids']]
        example['input_ids'] =      [i + [tokenizer.pad_token_id] * (seq_len - len(i)) for i in example['input_ids']]
        return example
    
    tokenized_dataset = tokenized_dataset.map(_pad, batched=True, num_proc=num_proc, load_from_cache_file=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['input_tokens', 'output_tokens'])

    print("SRC text from the dataset: ", tokenized_dataset[0]['src'])
    print("TGT text from the dataset: ", tokenized_dataset[0]['tgt'])
    print("Tokenized sample from the dataset: ", tokenized_dataset[0]['input_ids'])
    print("Attention mask from the dataset: ", tokenized_dataset[0]['attention_mask'])

    tokenized_dataset = tokenized_dataset.with_format('torch')
    return tokenized_dataset
 
