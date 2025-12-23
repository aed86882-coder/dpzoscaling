"""
Parallel DistZO2 + DP-AggZO Wrapper for Seq2Seq Models (mBART, T5, etc.)
用于翻译任务的 Encoder-Decoder 模型
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np


class ParallelDistZO2DPAggZOSeq2Seq(nn.Module):
    """
    Parallel版本的 Seq2Seq 模型包装器（用于翻译任务）
    
    支持 Encoder-Decoder 架构的模型，如 mBART, T5 等
    每个 GPU 只计算 K/num_gpus 个方向
    """
    
    def __init__(self, model, optimizer, tokenizer, source_lang="en", target_lang="zh"):
        super().__init__()
        self.model = model
        self.opt = optimizer
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Seq2Seq 模型结构 (以 mBART 为例)
        # Encoder
        if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
            self.encoder = model.model.encoder
            self.decoder = model.model.decoder
            self.lm_head = model.lm_head
        else:
            # 如果没有 encoder/decoder 结构，假设是单层模型
            self.encoder = None
            self.decoder = model.model if hasattr(model, 'model') else model
            self.lm_head = model.lm_head if hasattr(model, 'lm_head') else None
        
        # 初始化：Encoder 和 Decoder 的主要层都保持在 GPU 上
        # 但由于 Seq2Seq 模型通常较大，我们可以考虑 offload 某些层
        # 这里为了简化，先保持所有层在 GPU 上，后续可以根据需要优化
        
        print(f"✓ Parallel DistZO2-DP-AggZO Seq2Seq initialized")
        print(f"  - Rank {self.opt.rank}/{self.opt.world_size}")
        print(f"  - Total directions: {self.opt.n_directions}")
        print(f"  - Local directions: {self.opt.local_n_directions}")
        print(f"  - Direction range: [{self.opt.direction_start}, {self.opt.direction_end})")
        print(f"  - Source lang: {source_lang}, Target lang: {target_lang}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for translation task
        
        Args:
            input_ids: Source language token IDs
            attention_mask: Attention mask for source
            labels: Target language token IDs (for training)
        
        Returns:
            local_losses_list: List of (loss1, loss2) for this GPU's directions
        """
        device = self.opt.device
        
        # Prepare inputs
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # Update parameters if we have gradients from previous step
        if self.opt.projected_grads is not None and self.opt.last_step_seed is not None:
            self._update_model()
        
        # For Seq2Seq models, we need to handle encoder and decoder separately
        # 简化版本：使用模型的 forward 方法，然后在不同方向上计算 loss
        # 由于 Seq2Seq 模型结构复杂，这里使用简化策略：
        # 对模型整体进行扰动，而不是逐层处理（后续可以优化为逐层）
        
        # Generate seeds for all N directions (consistent across all GPUs)
        np.random.seed(self.opt.current_step_seed)
        all_random_seeds = np.random.randint(1000000000, size=self.opt.n_directions)
        
        # Extract seeds for this GPU's directions
        local_random_seeds = all_random_seeds[self.opt.direction_start:self.opt.direction_end]
        
        local_losses_list = []
        
        # Process each local direction
        for local_idx in range(self.opt.local_n_directions):
            seed = int(local_random_seeds[local_idx])
            
            # Perturb model parameters (+eps)
            self._perturb_model(seed, scaling=1.0)
            
            # Forward pass 1
            with torch.no_grad():
                outputs1 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss1 = outputs1.loss if hasattr(outputs1, 'loss') else outputs1[0]
            
            # Perturb model parameters (-2*eps, total: -eps)
            self._perturb_model(seed, scaling=-2.0)
            
            # Forward pass 2
            with torch.no_grad():
                outputs2 = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss2 = outputs2.loss if hasattr(outputs2, 'loss') else outputs2[0]
            
            # Restore parameters (+eps, total: 0)
            self._perturb_model(seed, scaling=1.0)
            
            local_losses_list.append((loss1, loss2))
        
        return local_losses_list
    
    def _perturb_model(self, seed, scaling):
        """Perturb all model parameters"""
        torch.manual_seed(seed)
        for param in self.model.parameters():
            if param.requires_grad:
                z = torch.randn_like(param)
                param.data.add_(z * self.opt.zo_eps * scaling)
    
    def _update_model(self):
        """Update model parameters using aggregated gradients"""
        if self.opt.projected_grads is None:
            return
        
        # Generate same random seeds as in forward pass
        np.random.seed(self.opt.last_step_seed)
        all_random_seeds = np.random.randint(1000000000, size=self.opt.n_directions)
        
        # Update with each direction's gradient
        for direction_idx in range(self.opt.n_directions):
            seed = int(all_random_seeds[direction_idx])
            noisy_grad = self.opt.projected_grads[direction_idx]
            
            torch.manual_seed(seed)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    z = torch.randn_like(param)
                    
                    # Apply weight decay selectively
                    no_decay = any(nd in name for nd in ["bias", "layer_norm", "layernorm", "ln", "norm"])
                    if no_decay:
                        param.data.sub_(self.opt.lr * noisy_grad * z)
                    else:
                        param.data.sub_(self.opt.lr * (noisy_grad * z + self.opt.weight_decay * param.data))
    
    def gather_losses_from_all_gpus(self, local_losses_list):
        """
        Gather losses from all GPUs using all_gather
        
        Args:
            local_losses_list: List of (loss1, loss2) from this GPU
        
        Returns:
            all_losses_list: List of losses from all GPUs
                            Shape: [world_size][local_n_directions][2]
        """
        if not dist.is_initialized() or self.opt.world_size == 1:
            return [local_losses_list]
        
        # Convert local losses to tensors
        local_loss_tensor = torch.tensor(
            [[l1.item(), l2.item()] for l1, l2 in local_losses_list],
            device=self.opt.device
        )  # Shape: [local_n_directions, 2]
        
        # Gather sizes from all GPUs
        local_size = torch.tensor([local_loss_tensor.size(0)], device=self.opt.device)
        size_list = [torch.zeros_like(local_size) for _ in range(self.opt.world_size)]
        dist.all_gather(size_list, local_size)
        
        # Prepare for all_gather with padding
        max_size = max([s.item() for s in size_list])
        
        # Pad local tensor if needed
        if local_loss_tensor.size(0) < max_size:
            padding = torch.zeros(
                max_size - local_loss_tensor.size(0), 2,
                device=self.opt.device
            )
            local_loss_tensor = torch.cat([local_loss_tensor, padding], dim=0)
        
        # All gather
        gathered_tensors = [
            torch.zeros(max_size, 2, device=self.opt.device)
            for _ in range(self.opt.world_size)
        ]
        dist.all_gather(gathered_tensors, local_loss_tensor)
        
        # Convert back to list of (loss1, loss2) tuples
        all_losses_list = []
        for rank_idx, (tensor, size) in enumerate(zip(gathered_tensors, size_list)):
            rank_losses = []
            actual_size = int(size.item())
            for i in range(actual_size):
                loss1 = tensor[i, 0]
                loss2 = tensor[i, 1]
                rank_losses.append((loss1, loss2))
            all_losses_list.append(rank_losses)
        
        return all_losses_list

