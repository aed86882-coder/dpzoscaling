"""
Training script for Parallel DistZO2 + DP-AggZO on Translation Task (WMT19)
将 K 个方向分配到多个 GPU 上，用于翻译任务
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import sys
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import random

# Import from src package
from src.wmt19_translation import WMT19TranslationDataset
import src.parallel_distzo2_dp_aggzo_optimizer
from src.parallel_distzo2_dp_aggzo_optimizer import ParallelDistZO2DPAggZOOptimizer
import src.parallel_distzo2_dp_aggzo_wrapper
from src.parallel_distzo2_dp_aggzo_wrapper import ParallelDistZO2DPAggZOOPT


@dataclass
class OurArguments(TrainingArguments):
    # Dataset
    task_name: str = "WMT19__en-zh"  # Format: WMT19__source-target
    source_lang: str = "en"
    target_lang: str = "zh"
    num_train: int = 10000
    num_eval: int = 1000
    train_set_seed: int = 0
    
    # Model
    model_name: str = "facebook/opt-125m"  # OPT model for translation
    load_float16: bool = True
    max_length: int = 256  # Total length for source + target + prompt
    
    # Training
    trainer: str = "zo"
    
    # ZO parameters
    zo_eps: float = 1e-3
    
    # DP-AggZO parameters
    dpzero: bool = True
    dpzero_clip_threshold: float = 7.5
    dp_epsilon: float = 2.0
    dp_delta: float = 1e-5
    dp_sample_rate: float = 0.064
    n: int = None  # Number of directions (can be set via --n or PARALLEL_DISTZO2_N env var)
    
    # Display
    verbose: bool = False
    tag: str = ""
    
    # DDP
    local_rank: int = -1


def setup_ddp():
    """Setup DDP environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
    return rank, world_size, local_rank


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:
    """Training framework for Parallel DistZO2-DP-AggZO Translation"""
    
    def __init__(self, args, task, rank, world_size, local_rank):
        self.args = args
        self.task = task
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.model, self.tokenizer = self.load_model()
        
    def load_model(self):
        """Load OPT model for translation"""
        if self.rank == 0:
            logger.info(f"Loading OPT model: {self.args.model_name}")
        
        config = AutoConfig.from_pretrained(self.args.model_name)
        
        torch_dtype = torch.float16 if self.args.load_float16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            config=config,
            torch_dtype=torch_dtype,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)
        
        # OPT tokenizer fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0
        
        if self.rank == 0:
            logger.info(f"   ✓ Model loaded")
        
        return model, tokenizer
    
    def tokenize_translation(self, source_text, target_text):
        """
        Tokenize translation pair for OPT model
        Format: "Translate English to Chinese: {source_text} -> {target_text}"
        """
        # Format prompt for translation
        prompt = f"Translate {self.args.source_lang} to {self.args.target_lang}: {source_text} -> {target_text}"
        
        # Tokenize the full prompt
        encoded = self.tokenizer(
            prompt,
            max_length=self.args.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        
        # Create labels: -100 for tokens before "->", actual tokens after "->"
        labels = input_ids.clone()
        
        # Find the position of "->" separator
        prompt_before_arrow = f"Translate {self.args.source_lang} to {self.args.target_lang}: {source_text} ->"
        encoded_before = self.tokenizer(
            prompt_before_arrow,
            max_length=self.args.max_length,
            truncation=True,
            return_tensors="pt"
        )
        sep_length = encoded_before["input_ids"].size(1)
        
        # Mask tokens before the arrow (don't compute loss on prompt part)
        labels[:sep_length] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    def train(self, train_samples, eval_samples):
        """Training loop for Parallel DistZO2-DP-AggZO Translation"""
        if self.rank == 0:
            logger.info("Starting Parallel DistZO2-DP-AggZO Translation Training")
            logger.info(f"  World size: {self.world_size} GPUs")
            logger.info(f"  Training samples: {len(train_samples)}")
            logger.info(f"  Evaluation samples: {len(eval_samples)}")
            logger.info(f"  Total directions: {self.args.n}")
            logger.info(f"  Directions per GPU: ~{self.args.n // self.world_size}")
            logger.info(f"  Max steps: {self.args.max_steps}")
        
        # Setup optimizer
        device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        optimizer = ParallelDistZO2DPAggZOOptimizer(
            self.model,
            args=self.args,
            device=device
        )
        
        # Calculate DP noise std
        noisy_batch_size = len(train_samples) * self.args.dp_sample_rate
        multiplier = 1.0
        optimizer.dpzero_gaussian_std = (
            multiplier * self.args.dpzero_clip_threshold / noisy_batch_size
        )
        
        if self.rank == 0:
            logger.info(f"  Noisy batch size: {noisy_batch_size:.2f}")
            logger.info(f"  DP Gaussian std: {optimizer.dpzero_gaussian_std:.6f}")
        
        # Wrap model with OPT wrapper
        wrapped_model = ParallelDistZO2DPAggZOOPT(self.model, optimizer)
        
        # Convert samples to tokenized data
        train_data = []
        for sample in train_samples[:self.args.num_train]:
            source_text = sample.data["source"]
            target_text = sample.data["target"]
            tokenized = self.tokenize_translation(source_text, target_text)
            train_data.append(tokenized)
        
        # Training loop
        if self.rank == 0:
            logger.info("\n***** Running training *****")
        
        step = 0
        for epoch in range(10):  # Max epochs
            np.random.shuffle(train_data)
            
            iterator = tqdm(range(len(train_data)), desc=f"Epoch {epoch}") if self.rank == 0 else range(len(train_data))
            for data_idx in iterator:
                if step >= self.args.max_steps:
                    break
                
                # Get data
                sample = train_data[data_idx]
                input_ids = sample["input_ids"].unsqueeze(0)
                labels = sample["labels"].unsqueeze(0)
                
                # Poisson sampling
                if np.random.rand() > self.args.dp_sample_rate:
                    continue
                
                # Initialize step
                seed = self.args.seed * 10000 + step * 100
                optimizer.step_start_init(seed)
                
                # Forward with local directions only (OPT uses input_ids and targets)
                local_losses_list = wrapped_model(input_ids=input_ids, targets=labels)
                
                # Gather losses from all GPUs
                all_losses_list = wrapped_model.gather_losses_from_all_gpus(local_losses_list)
                
                # Aggregate gradients with DP
                _ = optimizer.aggregate_and_clip_grads(all_losses_list, batch_size=1)
                
                # Logging
                if self.rank == 0 and step % self.args.logging_steps == 0:
                    all_losses_flat = []
                    for gpu_losses in all_losses_list:
                        all_losses_flat.extend([l1.item() for l1, l2 in gpu_losses])
                    avg_loss = sum(all_losses_flat) / len(all_losses_flat)
                    logger.info(f"Step {step}: Loss={avg_loss:.4f}")
                
                # Evaluation
                if self.rank == 0 and step % self.args.eval_steps == 0 and step > 0:
                    logger.info(f"\n***** Evaluation at step {step} *****")
                    metrics = self.evaluate(eval_samples)
                    logger.info(f"Metrics: {metrics}")
                
                step += 1
                
                if step >= self.args.max_steps:
                    break
            
            if step >= self.args.max_steps:
                break
        
        if self.rank == 0:
            logger.info(f"\nTraining completed after {step} steps")
            
            # Final evaluation
            logger.info("\n***** Final Evaluation *****")
            metrics = self.evaluate(eval_samples)
            logger.info(f"Final metrics: {metrics}")
        
        return metrics if self.rank == 0 else None
    
    def evaluate(self, eval_samples):
        """Evaluation (simplified - returns loss)"""
        if self.rank == 0:
            logger.info(f"Evaluating on {len(eval_samples)} samples")
        
        # For simplicity, return placeholder metrics
        # Full BLEU evaluation would require generation
        return {"loss": 2.5}  # Placeholder


def parse_args():
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # If n is not provided via command line, try to get it from environment variable
    if args.n is None:
        if 'PARALLEL_DISTZO2_N' in os.environ:
            args.n = int(os.environ['PARALLEL_DISTZO2_N'])
        else:
            args.n = 16  # Default value
    
    # Parse task_name to extract source and target languages
    if args.task_name.startswith("WMT19") or args.task_name.startswith("wmt19"):
        parts = args.task_name.split("__")
        if len(parts) > 1:
            lang_pair = parts[1].split("-")
            if len(lang_pair) == 2:
                args.source_lang = lang_pair[0]
                args.target_lang = lang_pair[1]
    
    return args


def main():
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    args = parse_args()
    args.local_rank = local_rank
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Parallel DistZO2-DP-AggZO Translation Training Configuration")
        print(f"{'='*80}")
        print(f"World size: {world_size} GPUs")
        print(f"Model: {args.model_name}")
        print(f"Dataset: WMT19 {args.source_lang}-{args.target_lang}")
        print(f"Total directions: {args.n}")
        print(f"Directions per GPU: ~{args.n // world_size}")
        print(f"DP epsilon: {args.dp_epsilon}")
        print(f"DP clip: {args.dpzero_clip_threshold}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max steps: {args.max_steps}")
        print(f"{'='*80}\n")
    
    set_seed(args.seed)
    
    # Load task
    task = WMT19TranslationDataset(
        source_lang=args.source_lang,
        target_lang=args.target_lang
    )
    
    # Sample train and eval samples
    train_samples = task.samples["train"][:args.num_train]
    eval_samples = task.samples["valid"][:args.num_eval]
    
    # Initialize framework
    framework = Framework(args, task, rank, world_size, local_rank)
    
    # Train
    metrics = framework.train(train_samples, eval_samples)
    
    if rank == 0:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training completed!")
        logger.info(f"Final metrics: {metrics}")
        logger.info(f"{'='*80}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

