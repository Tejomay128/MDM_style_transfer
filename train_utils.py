"""
Training code adapted from the repository of SMDM: https://github.com/ML-GSAI/SMDM/tree/main
"""

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from functools import partial
from safetensors.torch import load_file
from flash_attn.losses.cross_entropy import CrossEntropyLoss

from lit_gpt.diffmodel import TransEncoder
from lit_gpt.diffmodel import Config as TransEncoderConfig
from lit_gpt.model import GPT
from lit_gpt.model import Config as GPTConfig
from ema import ExponentialMovingAverage


class MaskedDiffusionTrainer:
    def __init__(
            self,
            model_name,
            use_pretrained,
            pretrain_path,
            training_mode,
            tokenizer_path,
            tokenizer,
            epsilon,
            cfg,
            lr,
            min_lr,
            max_length,
            weight_decay,
            warmup_steps,
            clip_grad_norm,
            batch_size,
            num_epochs,
            eval_every,
            log_every,
            log_dir,
            save_every,
            save_after,
            save_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
        """
        Args:
            model_name: name of the model used to fetch the architecture config
            use_pretrained: whether to use pre-trained model weights during initialization or not
            pretrain_path: path to the model's pre-trained weights
            training_mode: ar (for autoregressive LM training) or diff (for masked diffusion LM training)
            tokenizer: tokenizer used for tokenization
            epsilon: min amount of noise to be added in the forward process
            cfg: condition drop ratio for classifier free guidance
            lr: learning rate
            min_lr: minimum value upto which learning rate can decay
            max_length: maximum length of the sequence
            weight_decay: weight decay regularization parameter
            warmup_steps: number of warmup steps
            clip_grad_norm: gradient norm clipping max value
            batch_size: batch size
            num_epochs: number of epochs
            eval_every: eval every
            log_every: log every
            log_dir: log directory
            save_every: save every
            save_after: min training steps after which saving should start
            save_dir: save directory
        """
        self.model_name = model_name
        self.pretrain_path = pretrain_path
        self.use_pretrained = use_pretrained
        self.training_mode = training_mode
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.cfg = cfg
        self.lr = lr
        self.min_lr = min_lr
        self.max_length = max_length
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.clip_grad_norm = clip_grad_norm
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.eval_every = eval_every
        self.log_every = log_every
        self.save_every = save_every
        self.save_after = save_after
        self.save_dir = save_dir
        self.device = device

        self.neg_infinity = -1e8
        print(log_dir, save_dir)

        self.logger = SummaryWriter(log_dir=log_dir)
        # os.makedirs(self.save_dir, exist_ok=True)
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_id = 32000

        # Initialize
        if self.training_mode == 'diff':
            config = TransEncoderConfig.from_name(model_name)
        elif self.training_mode == 'ar':
            config = GPTConfig.from_name(model_name)
        else:
            raise NotImplementedError
        
        if not tokenizer_path == 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T':
            config.vocab_size = tokenizer.vocab_size
            config.padded_vocab_size = tokenizer.vocab_size
            if tokenizer.mask_token_id:
                self.mask_id = tokenizer.mask_token_id
            else:
                self.mask_id = self.vocab_size

        if self.training_mode == 'diff':
            self.model = TransEncoder(config).to(device)
            self.loss_fn = CrossEntropyLoss(reduction='none')
            # Keep track of the exponential moving average of the weights during training
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)
            
        elif self.training_mode == 'ar':
            self.model = GPT(config).to(device)
            self.loss_fn = CrossEntropyLoss()
            
        else:
            raise NotImplementedError
        
        self.model.apply(partial(self.model._init_weights ,n_layer=config.n_layer))

        if self.use_pretrained:
            print("Using pretrained model...")
            ckpt_dic = load_file(pretrain_path)
            self.model.load_state_dict(ckpt_dic)
        else:
            print("Training from scratch...")

        # number of parameters
        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        print(f'Number of parameters: {(num_params / 1e6): .2f}M')

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr, 
            weight_decay=self.weight_decay, 
            betas=(0.9, 0.95)
        )


    def _adjust_lr(self, step):
        """
        Implements an inverse square root scheduler with warmup for the learning rate.
        """
        if step <= self.warmup_steps:
            lr = self.lr * step / self.warmup_steps
        else:
            # lr = max(1e-5, self.lr * (0.999 ** (step - self.warmup_steps)))
            lr = self.lr * (self.warmup_steps ** 0.5) / (step ** 0.5)  # inverse square root schedule
            lr = max(self.min_lr, lr)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        return lr
    

    def forward_process(self, batch):
        """
        Forward diffusion process for masked diffusion training.
        Returns:
            noisy_batch: Partially masked sequence
            p_mask: proportion of masking
        """
        b, l = batch.shape
        t = torch.rand((b,), device=batch.device)

        p_mask = (1 - self.epsilon) * t + self.epsilon
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        noisy_batch = torch.where(mask_indices, self.mask_id, batch)
        return noisy_batch, p_mask
    

    def _ar_step(self, data, mode='train'):
        """
        One training step of autoregressive training.
        """
        assert mode in ['train', 'eval']

        input_ids = data['input_ids'].to(self.device)
        prompt_length = data['src_length'].to(self.device) # [0, len-1]

        self.optim.zero_grad()
        length = data['length'].to(self.device) # [prompt + answer] length
        max_length = length.max().item()
        input_ids = input_ids[:, :max_length]
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input_ids)
            temp_tensor = torch.arange(logits.size(1), device=input_ids.device).expand(logits.size(0), logits.size(1))
            logits_index = (temp_tensor >= (prompt_length - 1).unsqueeze(1)) & (temp_tensor < (length - 1).unsqueeze(1))
            logits = logits[logits_index]
            input_ids_index = (temp_tensor >= prompt_length.unsqueeze(1)) & (temp_tensor < length.unsqueeze(1))
            targets = input_ids[input_ids_index]
            loss = self.loss_fn(logits, targets)

        if mode == 'train':
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optim.step()
        
        return loss.item()


    def _diff_step(self, data, mode='train'):
        """
        One training step of masked diffusion training. First computes a noisy batch and
        passes it through the neural network. The cross entropy loss is then computed on the 
        masked positions in the noisy batch.
        """
        assert mode in ['train', 'eval']

        input_ids = data['input_ids'].to(self.device)
        prompt_length = data['src_length'].to(self.device) # [0, len-1]

        self.optim.zero_grad()

        noisy_input, p_mask = self.forward_process(batch=input_ids)
        temp_tensor = torch.arange(noisy_input.size(1), device=noisy_input.device).expand(noisy_input.size(0), noisy_input.size(1))
        prompt_index = (temp_tensor < prompt_length.unsqueeze(1))
        noisy_input[prompt_index] = input_ids[prompt_index].clone()
        mask_indices = (noisy_input == self.mask_id)
        
        # Randomly drops the conditional sequence by completely masking it out (Randomness determined by self.cfg)
        # See classifier-free guidance paper: https://arxiv.org/abs/2207.12598 and
        # scaling MDMs paper: https://openreview.net/forum?id=WNvvwK0tut
        if  mode == 'train' and torch.rand(1) < self.cfg:
            noisy_input[prompt_index] = self.mask_id

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # training with bfloat16 precision
            logits = self.model(noisy_input)
            loss = self.loss_fn(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * self.max_length - prompt_length.sum())
        
        if mode == 'train':
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optim.step()

        return loss.item()

    
    def train(self, train_loader, val_loader=None):
        steps = 0
        for i in range(self.num_epochs):
            for data in tqdm(train_loader, desc='Epoch {}'.format(i+1)):
                steps += 1
                curr_lr = self._adjust_lr(steps)

                # Run a training step for autoregressive LM training
                if self.training_mode == 'ar':
                    loss = self._ar_step(data)

                # Run a training step for masked diffusion training
                elif self.training_mode == 'diff':
                    loss = self._diff_step(data)   
                else:
                    raise NotImplementedError

                if self.training_mode == 'diff':
                    self.ema.update(self.model.parameters())    

                # log loss
                if steps % self.log_every == 0:
                    self.logger.add_scalar('train/loss', loss, steps)
                    self.logger.add_scalar('train/lr', curr_lr, steps)
                
                # save model
                if steps % self.save_every == 0 and steps >= self.save_after:
                    self._save(steps)
            
                # evaluate
                if val_loader is not None and steps % self.eval_every == 0:
                    print("Evaluating...")
                    with torch.no_grad():
                        if self.training_mode == 'diff':
                            self.ema.store(self.model.parameters())
                            self.ema.copy_to(self.model.parameters())
                        
                        self.model.eval()
                        eval_loss = 0.
                        for val_data in val_loader:
                            if self.training_mode == 'diff':
                                eval_loss_per_step = self._diff_step(val_data, mode='eval')
                            elif self.training_mode == 'ar':
                                eval_loss_per_step = self._ar_step(val_data, mode='eval')
                            else:
                                raise NotImplementedError
                            eval_loss += eval_loss_per_step

                        eval_loss /= len(val_loader)
                        self.logger.add_scalar('val/loss', eval_loss, steps)
                        self.model.train()

                        if self.training_mode == 'diff':
                            self.ema.restore(self.model.parameters())


    def _save(self, step):
        """
        Saves a checkpoint of the model weights and ema weights.
        """
        print("Saving model at step {}...".format(step))
        model_state_dict = self.model.state_dict()
        ema_state_dict = {}
        if self.training_mode == 'diff':
            ema_params = self.ema.shadow_params
            buffer_names = set(name for name, _ in self.model.named_buffers())
            # add buffer names same as the non-ema model
            ema_state_dict = {b: v for b, v in self.model.state_dict().items() if b in buffer_names}
            for i, k in enumerate([key for key in model_state_dict.keys() if key not in buffer_names]):
                ema_state_dict[k] = ema_params[i]
        state = {
            'step': step,
            'optimizer': self.optim.state_dict(),
            'model': model_state_dict,
            'ema': ema_state_dict,
        }
        torch.save(state, os.path.join(self.save_dir, 'model_step_{}'.format(step)))
