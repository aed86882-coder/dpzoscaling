"""
Parallel DistZO2 + DP-AggZO Wrapper for OPT Models
每个 GPU 只计算分配给它的方向子集
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np


class ParallelDistZO2DPAggZOOPT(nn.Module):
    """
    Parallel版本的 OPT 模型包装器
    
    每个 GPU 只计算 K/num_gpus 个方向，然后通过 all_gather 收集所有结果
    """
    
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.opt = optimizer
        
        # OPT model structure
        self.embed_tokens = model.model.decoder.embed_tokens
        self.embed_positions = model.model.decoder.embed_positions
        self.final_layer_norm = model.model.decoder.final_layer_norm
        self.lm_head = model.lm_head
        
        # Init Upload (keep embeddings and head on GPU)
        self.embed_tokens.to(self.opt.device)
        self.embed_positions.to(self.opt.device)
        self.final_layer_norm.to(self.opt.device)
        self.lm_head.to(self.opt.device)
        
        # Layers Offload to CPU (DistZO2 style)
        self.layers = model.model.decoder.layers
        for layer in self.layers:
            layer.to(self.opt.offload_device)
        
        print(f"✓ Parallel DistZO2-DP-AggZO OPT initialized")
        print(f"  - Rank {self.opt.rank}/{self.opt.world_size}")
        print(f"  - Total directions: {self.opt.n_directions}")
        print(f"  - Local directions: {self.opt.local_n_directions}")
        print(f"  - Direction range: [{self.opt.direction_start}, {self.opt.direction_end})")
        print(f"  - Layers to offload: {len(self.layers)}")
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass with local directions only
        每个 GPU 只计算分配给它的方向
        
        Returns:
            local_losses_list: List of (loss1, loss2) for this GPU's directions
        """
        device = self.opt.device
        batch_size, seq_len = input_ids.size()
        input_ids = input_ids.to(device)
        targets = targets.to(device) if targets is not None else None
        
        # Prepare OPT inputs
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Determine dtype from model
        model_dtype = next(self.embed_tokens.parameters()).dtype
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=model_dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        
        # Embeddings (same for all directions to save memory)
        with torch.no_grad():
            tok_emb = self.embed_tokens(input_ids)
            pos_emb = self.embed_positions(attention_mask, 0)
            base_hidden_states = tok_emb + pos_emb
        
        # Initialize hidden states for local directions only
        local_hidden_states_list = [base_hidden_states.clone() for _ in range(self.opt.local_n_directions)]
        local_hidden_states_list2 = [base_hidden_states.clone() for _ in range(self.opt.local_n_directions)]
        
        # Generate seeds for all N directions (consistent across all GPUs)
        np.random.seed(self.opt.current_step_seed)
        all_random_seeds = np.random.randint(1000000000, size=self.opt.n_directions)
        
        # Extract seeds for this GPU's directions
        local_random_seeds = all_random_seeds[self.opt.direction_start:self.opt.direction_end]
        
        # Process layers layer-wise
        num_layers = len(self.layers)
        for layer_idx in range(num_layers):
            # Upload current layer
            self.opt.task_upload(self.layers[layer_idx])
            
            # Prefetch next layer
            if layer_idx + 1 < num_layers:
                self.opt.task_upload(self.layers[layer_idx + 1])
            
            # Update parameters if we have gradients from previous step
            if self.opt.projected_grads is not None and self.opt.last_step_seed is not None:
                module_seed = self.opt.last_step_seed + layer_idx
                self.opt.update_module(self.layers[layer_idx], module_seed)
            
            # Process local directions only
            new_hidden_states_list = []
            new_hidden_states_list2 = []
            
            for local_idx in range(self.opt.local_n_directions):
                x1 = local_hidden_states_list[local_idx]
                x2 = local_hidden_states_list2[local_idx]
                
                seed = int(local_random_seeds[local_idx])
                
                # Perturb +eps
                torch.manual_seed(seed)
                for param in self.layers[layer_idx].parameters():
                    if param.requires_grad:
                        z = torch.randn_like(param)
                        param.data.add_(z * self.opt.zo_eps)
                
                with torch.no_grad():
                    out1 = self.layers[layer_idx](x1, attention_mask=causal_mask)
                    out1 = out1[0] if isinstance(out1, tuple) else out1
                
                # Perturb -2*eps (total: -eps)
                torch.manual_seed(seed)
                for param in self.layers[layer_idx].parameters():
                    if param.requires_grad:
                        z = torch.randn_like(param)
                        param.data.add_(z * self.opt.zo_eps * (-2))
                
                with torch.no_grad():
                    out2 = self.layers[layer_idx](x2, attention_mask=causal_mask)
                    out2 = out2[0] if isinstance(out2, tuple) else out2
                
                # Restore to original (total: 0)
                torch.manual_seed(seed)
                for param in self.layers[layer_idx].parameters():
                    if param.requires_grad:
                        z = torch.randn_like(param)
                        param.data.add_(z * self.opt.zo_eps)
                
                new_hidden_states_list.append(out1.detach())
                new_hidden_states_list2.append(out2.detach())
            
            local_hidden_states_list = new_hidden_states_list
            local_hidden_states_list2 = new_hidden_states_list2
            
            # Offload previous layer
            if layer_idx > 0:
                self.opt.task_offload(self.layers[layer_idx - 1])
        
        # Offload last layer
        self.opt.task_offload(self.layers[num_layers - 1])
        
        # Final layer norm and LM head (for local directions)
        local_losses_list = []
        for local_idx in range(self.opt.local_n_directions):
            x1 = local_hidden_states_list[local_idx]
            x2 = local_hidden_states_list2[local_idx]
            
            with torch.no_grad():
                x1 = self.final_layer_norm(x1)
                x2 = self.final_layer_norm(x2)
                
                logits1 = self.lm_head(x1)
                logits2 = self.lm_head(x2)
            
            # Compute losses
            loss_fct = nn.CrossEntropyLoss()
            vocab_size = logits1.size(-1)
            loss1 = loss_fct(logits1.view(-1, vocab_size), targets.view(-1))
            loss2 = loss_fct(logits2.view(-1, vocab_size), targets.view(-1))
            
            local_losses_list.append((loss1, loss2))
        
        return local_losses_list
    
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
        
        # Gather sizes from all GPUs (since they might be different)
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

