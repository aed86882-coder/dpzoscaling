"""
Parallel DistZO2 + DP-AggZO Optimizer
结合 ParallelZOOptimizer 的思想，将 K 个方向分配到多个 GPU 上
每个 GPU 只计算 K/num_gpus 个方向，大幅减少计算量
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from collections import deque
import numpy as np


class ParallelDistZO2DPAggZOOptimizer:
    """
    Parallel 版本的 DistZO2-DP-AggZO Optimizer
    
    核心思想:
    1. 将 K 个方向均匀分配到所有 GPU 上
    2. 每个 GPU 只计算 K/num_gpus 个方向的前向传播
    3. 使用 all_gather 收集所有方向的 loss
    4. 在 rank 0 上应用 DP（裁剪+噪声）并广播
    5. 所有 GPU 使用相同的聚合梯度更新参数
    
    优势:
    - 每个 GPU 的计算量减少到 K/num_gpus
    - 理论加速比接近 num_gpus 倍（忽略通信开销）
    - 保持完整的 DP 保护
    """
    
    def __init__(self, model: nn.Module, args, device='cuda:0', offload_device='cpu'):
        self.model = model
        self.args = args
        self.device = device
        self.offload_device = offload_device
        
        # DDP Info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # DistZO2 streams
        self.upload_stream = torch.cuda.Stream(device=device)
        self.compute_stream = torch.cuda.Stream(device=device)
        self.offload_stream = torch.cuda.Stream(device=device)
        
        # DP-AggZO parameters
        self.n_directions = getattr(args, 'n', getattr(args, 'num_directions', 16))
        self.dp_epsilon = args.dp_epsilon
        self.dp_clip_threshold = args.dpzero_clip_threshold
        self.zo_eps = args.zo_eps
        self.lr = args.learning_rate
        self.weight_decay = args.weight_decay
        
        # Parallel distribution of directions
        # 每个 GPU 负责的方向数量
        self.local_n_directions = self.n_directions // self.world_size
        if self.rank < self.n_directions % self.world_size:
            self.local_n_directions += 1
        
        # 每个 GPU 负责的方向索引范围
        self.direction_start = sum([
            self.n_directions // self.world_size + (1 if i < self.n_directions % self.world_size else 0)
            for i in range(self.rank)
        ])
        self.direction_end = self.direction_start + self.local_n_directions
        
        print(f"[Rank {self.rank}] Responsible for directions {self.direction_start}-{self.direction_end-1} "
              f"(total {self.local_n_directions}/{self.n_directions})")
        
        # Gradient tracking
        self.projected_grads = None  # Store N gradients
        self.current_step_seed = None
        self.last_step_seed = None
        self.seed_queue = deque(maxlen=2)
        
        # DP noise standard deviation
        self.dpzero_gaussian_std = 0.0

    def step_start_init(self, seed):
        """Initialize step with seed"""
        self.current_step_seed = seed
        self.seed_queue.append(seed)
        if len(self.seed_queue) == 2:
            self.last_step_seed = self.seed_queue[0]
        if dist.is_initialized():
            torch.cuda.synchronize(self.device)

    def task_upload(self, module):
        """Upload module to GPU"""
        with torch.cuda.stream(self.upload_stream):
            module.to(self.device)
        return module

    def task_offload(self, module):
        """Offload module to CPU"""
        self.offload_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.offload_stream):
            module.to(self.offload_device)
        return module

    def _perturb_parameters(self, module, scaling, seed):
        """Perturb module parameters with Gaussian noise"""
        torch.manual_seed(seed)
        for param in module.parameters():
            if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.size(), 
                               device=param.device, dtype=param.dtype)
                param.data.add_(z * self.zo_eps * scaling)

    def _zo_update_multi_direction(self, module, module_id):
        """Update module parameters using aggregated gradients from N directions"""
        if self.projected_grads is None:
            return
        
        # Generate same random seeds as in the forward pass
        np.random.seed(self.last_step_seed + module_id)
        random_seeds = np.random.randint(1000000000, size=self.n_directions)
        
        # Update with each direction's gradient
        for direction_idx in range(self.n_directions):
            seed = int(random_seeds[direction_idx])
            noisy_grad = self.projected_grads[direction_idx]
            
            torch.manual_seed(seed)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    z = torch.normal(mean=0, std=1, size=param.size(), 
                                   device=param.device, dtype=param.dtype)
                    
                    # Apply weight decay selectively
                    no_decay = any(nd in name for nd in ["bias", "layer_norm", "layernorm", "ln", "norm"])
                    if no_decay:
                        param.data.sub_(self.lr * noisy_grad * z)
                    else:
                        param.data.sub_(self.lr * (noisy_grad * z + self.weight_decay * param.data))

    def aggregate_and_clip_grads(self, losses_list_all_gpus, batch_size):
        """
        Aggregate gradients from all GPUs and apply DP
        
        Args:
            losses_list_all_gpus: List of (loss1, loss2) tuples from all GPUs
                                  Shape: [world_size, local_n_directions, 2]
            batch_size: Batch size for normalization
        
        Returns:
            Noisy aggregated gradients (N-dimensional tensor)
        """
        # Compute gradient for each direction
        all_grads = []
        for gpu_losses in losses_list_all_gpus:
            for loss1, loss2 in gpu_losses:
                grad = (loss1 - loss2) / (2 * self.zo_eps)
                all_grads.append(grad)
        
        # Stack: [N]
        stacked_grads = torch.stack(all_grads) / self.n_directions
        
        # 只在 rank 0 上执行 DP 操作
        if self.rank == 0:
            # Clip
            norms = torch.norm(stacked_grads)
            scaling_factor = min(1.0, self.dp_clip_threshold / (norms.item() + 1e-8))
            clipped_grads = stacked_grads * scaling_factor
            
            # Average over batch
            mean_grad = clipped_grads / batch_size
            
            # Add Gaussian noise
            noise = torch.randn(self.n_directions, device=self.device) * self.dpzero_gaussian_std
            noisy_grad = mean_grad + noise
        else:
            noisy_grad = torch.zeros(self.n_directions, device=self.device)
        
        # Broadcast from rank 0 to ensure all GPUs have same gradient
        if dist.is_initialized():
            dist.broadcast(noisy_grad, src=0)
        
        self.projected_grads = noisy_grad
        return noisy_grad
    
    def update_module(self, module, module_seed):
        """Update module parameters with aggregated gradients"""
        if self.projected_grads is None:
            return
        
        for direction_idx in range(self.n_directions):
            seed = module_seed + direction_idx * 10000
            noisy_grad = self.projected_grads[direction_idx]
            
            torch.manual_seed(seed)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    z = torch.randn_like(param)
                    # Apply weight decay selectively
                    no_decay = any(nd in name for nd in ["bias", "layer_norm", "layernorm", "ln", "norm"])
                    if no_decay:
                        param.data.sub_(self.lr * noisy_grad * z)
                    else:
                        param.data.sub_(self.lr * (noisy_grad * z + self.weight_decay * param.data))

