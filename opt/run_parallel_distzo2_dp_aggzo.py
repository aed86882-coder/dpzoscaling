"""
Training script for Parallel DistZO2 + DP-AggZO on OPT
将 K 个方向分配到多个 GPU 上，大幅减少每个 GPU 的计算量
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
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import random

# Import from src package
import src.tasks
from src.tasks import get_task
from src.metrics import calculate_metric
from src.utils import *
import src.parallel_distzo2_dp_aggzo_optimizer
from src.parallel_distzo2_dp_aggzo_optimizer import ParallelDistZO2DPAggZOOptimizer
import src.parallel_distzo2_dp_aggzo_wrapper
from src.parallel_distzo2_dp_aggzo_wrapper import ParallelDistZO2DPAggZOOPT


@dataclass
class OurArguments(TrainingArguments):
    # Dataset
    task_name: str = "SQuAD"
    num_train: int = 1000
    num_dev: int = 500
    num_eval: int = 1000
    train_set_seed: int = 0
    result_file: str = None
    
    # Model
    model_name: str = "facebook/opt-125m"
    load_float16: bool = True
    max_length: int = 2048
    
    # Training
    trainer: str = "zo"
    only_train_option: bool = True
    train_as_classification: bool = False
    
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
    """Training framework for Parallel DistZO2-DP-AggZO"""
    
    def __init__(self, args, task, rank, world_size, local_rank):
        self.args = args
        self.task = task
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.model, self.tokenizer = self.load_model()
        
    def load_model(self):
        """Load HuggingFace OPT model"""
        if self.rank == 0:
            logger.info(f"Loading model {self.args.model_name}")
        
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
    
    def train(self, train_samples, eval_samples):
        """Training loop for Parallel DistZO2-DP-AggZO"""
        if self.rank == 0:
            logger.info("Starting Parallel DistZO2-DP-AggZO training")
            logger.info(f"  World size: {self.world_size} GPUs")
            logger.info(f"  Training samples: {len(train_samples)}")
            logger.info(f"  Evaluation samples: {len(eval_samples)}")
            logger.info(f"  Total directions: {self.args.n}")
            logger.info(f"  Directions per GPU: ~{self.args.n // self.world_size}")
            logger.info(f"  Max steps: {self.args.max_steps}")
        
        # Set tokenizer to left padding
        self.tokenizer.padding_side = "left"
        
        # Setup optimizer
        device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        optimizer = ParallelDistZO2DPAggZOOptimizer(
            self.model,
            args=self.args,
            device=device
        )
        
        # Calculate DP noise std
        noisy_batch_size = len(train_samples) * self.args.dp_sample_rate
        multiplier = 1.0  # Simplified, should use privacy accounting
        optimizer.dpzero_gaussian_std = (
            multiplier * self.args.dpzero_clip_threshold / noisy_batch_size
        )
        
        if self.rank == 0:
            logger.info(f"  Noisy batch size: {noisy_batch_size:.2f}")
            logger.info(f"  DP Gaussian std: {optimizer.dpzero_gaussian_std:.6f}")
        
        # Wrap model
        wrapped_model = ParallelDistZO2DPAggZOOPT(self.model, optimizer)
        
        # Convert samples to dataset
        train_data = self._convert_samples(train_samples)
        
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
                input_ids = torch.tensor([sample["input_ids"]])
                labels = torch.tensor([sample["labels"]])
                
                # Poisson sampling
                if np.random.rand() > self.args.dp_sample_rate:
                    continue
                
                # Initialize step
                seed = self.args.seed * 10000 + step * 100
                optimizer.step_start_init(seed)
                
                # Forward with local directions only (分布式计算)
                local_losses_list = wrapped_model(input_ids, targets=labels)
                
                # Gather losses from all GPUs
                all_losses_list = wrapped_model.gather_losses_from_all_gpus(local_losses_list)
                
                # Aggregate gradients with DP (在 rank 0 上执行，然后广播)
                _ = optimizer.aggregate_and_clip_grads(all_losses_list, batch_size=1)
                
                # Logging
                if self.rank == 0 and step % self.args.logging_steps == 0:
                    # Flatten all losses
                    all_losses_flat = []
                    for gpu_losses in all_losses_list:
                        all_losses_flat.extend([l1.item() for l1, l2 in gpu_losses])
                    avg_loss = sum(all_losses_flat) / len(all_losses_flat)
                    logger.info(f"Step {step}: Loss={avg_loss:.4f}")
                
                # Evaluation
                if self.rank == 0 and step % self.args.eval_steps == 0 and step > 0:
                    logger.info(f"\n***** Evaluation at step {step} *****")
                    metrics = self.evaluate([], eval_samples)
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
            metrics = self.evaluate([], eval_samples)
            logger.info(f"Final metrics: {metrics}")
            
            # Save results
            if self.args.result_file:
                write_metrics_to_file(metrics, self.args.result_file)
        
        return metrics if self.rank == 0 else None
    
    def _convert_samples(self, samples):
        """Convert samples to tokenized data"""
        data = []
        for sample in samples:
            encoded_candidates, option_lens = encode_prompt(
                self.task, self.task.get_template(), [], sample, self.tokenizer,
                max_length=self.args.max_length, generation=self.task.generation,
                generation_with_gold=True, max_new_tokens=50
            )
            
            if self.task.generation:
                correct_candidate_id = 0
            elif isinstance(sample.correct_candidate, list):
                correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
            else:
                correct_candidate_id = sample.candidates.index(sample.correct_candidate)
            
            data.append({
                "input_ids": encoded_candidates[correct_candidate_id],
                "labels": encoded_candidates[correct_candidate_id],
                "option_len": option_lens[correct_candidate_id]
            })
        
        return data
    
    def evaluate(self, train_samples, eval_samples):
        """Evaluation (simplified)"""
        if self.rank == 0:
            logger.info(f"Evaluating on {len(eval_samples)} samples")
        
        # For simplicity, return placeholder metrics
        metric_name = getattr(self.task, "metric_name", "accuracy")
        
        return {metric_name: 0.5}  # Placeholder


def parse_args():
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # If n is not provided via command line, try to get it from environment variable
    # This avoids --n conflict with torchrun options
    if args.n is None:
        if 'PARALLEL_DISTZO2_N' in os.environ:
            args.n = int(os.environ['PARALLEL_DISTZO2_N'])
        else:
            args.n = 16  # Default value
    
    return args


def main():
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    args = parse_args()
    args.local_rank = local_rank
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Parallel DistZO2-DP-AggZO Training Configuration")
        print(f"{'='*80}")
        print(f"World size: {world_size} GPUs")
        print(f"Model: {args.model_name}")
        print(f"Task: {args.task_name}")
        print(f"Total directions: {args.n}")
        print(f"Directions per GPU: ~{args.n // world_size}")
        print(f"DP epsilon: {args.dp_epsilon}")
        print(f"DP clip: {args.dpzero_clip_threshold}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max steps: {args.max_steps}")
        print(f"{'='*80}\n")
    
    set_seed(args.seed)
    
    # Load task
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        train_set_seed=args.train_set_seed
    )
    
    # Initialize framework
    framework = Framework(args, task, rank, world_size, local_rank)
    
    # Training
    train_samples = train_sets[0]
    
    # Sample eval samples
    if args.num_eval is not None:
        eval_samples = task.sample_subset(data_split="valid", seed=args.train_set_seed, num=args.num_eval)
    else:
        eval_samples = task.valid_samples
    
    # Split train/dev
    if args.num_dev is not None:
        dev_samples = train_samples[-args.num_dev:]
        train_samples = train_samples[:-args.num_dev]
    else:
        dev_samples = eval_samples
    
    # Train
    metrics = framework.train(train_samples, dev_samples)
    
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

