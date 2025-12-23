"""
Simple test for Parallel DistZO2 + DP-AggZO
验证多 GPU 上的方向分配和聚合是否正确
"""

import torch
import sys
import os

# Import from src package
from transformers import AutoModelForCausalLM
import src.parallel_distzo2_dp_aggzo_optimizer
from src.parallel_distzo2_dp_aggzo_optimizer import ParallelDistZO2DPAggZOOptimizer
import src.parallel_distzo2_dp_aggzo_wrapper
from src.parallel_distzo2_dp_aggzo_wrapper import ParallelDistZO2DPAggZOOPT
import torch.distributed as dist


class SimpleArgs:
    """Simple argument container"""
    def __init__(self, n_directions=16):
        self.n = n_directions
        self.learning_rate = 1e-5
        self.zo_eps = 1e-3
        self.dp_epsilon = 2.0
        self.dp_delta = 1e-5
        self.dpzero_clip_threshold = 7.5
        self.dp_sample_rate = 0.064
        self.weight_decay = 0.0


def test_single_gpu():
    """Test on single GPU (no DDP)"""
    print("="*80)
    print("Test 1: Single GPU Mode")
    print("="*80)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("\n⚠️  Warning: No CUDA available, running on CPU")
    
    # Load model
    model_name = "facebook/opt-125m"
    print(f"\n1. Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    print(f"   ✓ Model loaded")
    
    # Setup args
    args = SimpleArgs(n_directions=8)
    print(f"\n2. Configuration:")
    print(f"   - N directions: {args.n}")
    print(f"   - Single GPU mode (no distribution)")
    
    # Setup optimizer
    print(f"\n3. Setting up optimizer...")
    optimizer = ParallelDistZO2DPAggZOOptimizer(
        model,
        args=args,
        device=device
    )
    
    # Calculate DP noise std
    optimizer.dpzero_gaussian_std = args.dpzero_clip_threshold / 64.0
    print(f"   ✓ Optimizer ready")
    
    # Wrap model
    print(f"\n4. Wrapping model...")
    wrapped_model = ParallelDistZO2DPAggZOOPT(model, optimizer)
    
    # Prepare test data
    vocab_size = model.config.vocab_size
    batch_size = 1
    seq_len = 20
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = input_ids.clone()
    
    print(f"\n5. Running test iteration...")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    
    # Test one step
    seed = 42 * 1000
    optimizer.step_start_init(seed)
    
    # Forward
    local_losses_list = wrapped_model(input_ids, targets=targets)
    
    # Gather (should return same list in single GPU mode)
    all_losses_list = wrapped_model.gather_losses_from_all_gpus(local_losses_list)
    
    # Aggregate
    aggregated_grad = optimizer.aggregate_and_clip_grads(all_losses_list, batch_size=1)
    
    # Verify
    print(f"\n6. Verification:")
    print(f"   - Local losses computed: {len(local_losses_list)}")
    print(f"   - Expected: {args.n}")
    print(f"   - Status: {'✅ PASS' if len(local_losses_list) == args.n else '❌ FAIL'}")
    print(f"   - Aggregated gradient shape: {aggregated_grad.shape}")
    print(f"   - Expected shape: torch.Size([{args.n}])")
    print(f"   - Status: {'✅ PASS' if aggregated_grad.shape == torch.Size([args.n]) else '❌ FAIL'}")
    
    print("\n✅ Single GPU test completed!\n")


def test_multi_gpu():
    """Test on multiple GPUs with DDP"""
    print("="*80)
    print("Test 2: Multi-GPU Mode (DDP)")
    print("="*80)
    
    # Check if running with torchrun
    if 'RANK' not in os.environ:
        print("\n⚠️  This test requires torchrun.")
        print("   Run with: torchrun --nproc_per_node=2 test_parallel_distzo2_dp_aggzo.py --multi-gpu")
        return
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Initialize DDP
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = f'cuda:{local_rank}'
    
    if rank == 0:
        print(f"\n✓ DDP initialized with {world_size} GPUs")
    
    # Load model
    model_name = "facebook/opt-125m"
    if rank == 0:
        print(f"\n1. Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # Setup args
    args = SimpleArgs(n_directions=16)
    if rank == 0:
        print(f"\n2. Configuration:")
        print(f"   - Total directions: {args.n}")
        print(f"   - World size: {world_size}")
        print(f"   - Directions per GPU: ~{args.n // world_size}")
    
    # Setup optimizer
    optimizer = ParallelDistZO2DPAggZOOptimizer(
        model,
        args=args,
        device=device
    )
    optimizer.dpzero_gaussian_std = args.dpzero_clip_threshold / 64.0
    
    # Wrap model
    wrapped_model = ParallelDistZO2DPAggZOOPT(model, optimizer)
    
    # Prepare test data
    vocab_size = model.config.vocab_size
    batch_size = 1
    seq_len = 20
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = input_ids.clone()
    
    if rank == 0:
        print(f"\n3. Running distributed test iteration...")
    
    # Test one step
    seed = 42 * 1000
    optimizer.step_start_init(seed)
    
    # Forward (each GPU computes its assigned directions)
    local_losses_list = wrapped_model(input_ids, targets=targets)
    
    # Gather from all GPUs
    all_losses_list = wrapped_model.gather_losses_from_all_gpus(local_losses_list)
    
    # Aggregate (DP operations on rank 0, then broadcast)
    aggregated_grad = optimizer.aggregate_and_clip_grads(all_losses_list, batch_size=1)
    
    # Verify
    if rank == 0:
        print(f"\n4. Verification:")
        print(f"   - GPU {rank}: Local directions computed: {len(local_losses_list)}")
        print(f"   - Expected: {optimizer.local_n_directions}")
        print(f"   - Total directions from all GPUs: {sum([len(losses) for losses in all_losses_list])}")
        print(f"   - Expected: {args.n}")
        print(f"   - Status: {'✅ PASS' if sum([len(losses) for losses in all_losses_list]) == args.n else '❌ FAIL'}")
        print(f"   - Aggregated gradient shape: {aggregated_grad.shape}")
        print(f"   - Expected shape: torch.Size([{args.n}])")
        print(f"   - Status: {'✅ PASS' if aggregated_grad.shape == torch.Size([args.n]) else '❌ FAIL'}")
    
    # Verify gradient consistency across GPUs
    grad_tensor = aggregated_grad.clone()
    grad_list = [torch.zeros_like(grad_tensor) for _ in range(world_size)]
    dist.all_gather(grad_list, grad_tensor)
    
    if rank == 0:
        consistent = all(torch.allclose(grad_list[0], grad_list[i]) for i in range(world_size))
        print(f"   - Gradient consistency across GPUs: {'✅ PASS' if consistent else '❌ FAIL'}")
        print("\n✅ Multi-GPU test completed!")
    
    dist.destroy_process_group()


def main():
    import sys
    
    if '--multi-gpu' in sys.argv:
        test_multi_gpu()
    else:
        test_single_gpu()
        print("\n" + "="*80)
        print("To test multi-GPU mode, run:")
        print("torchrun --nproc_per_node=2 test_parallel_distzo2_dp_aggzo.py --multi-gpu")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()

