"""
Parameter sweep script for WMT19 Translation Scaling Law experiments
Automates running experiments with different hyperparameters
"""

import subprocess
import os
import sys
import math
from itertools import product


def run_experiment(config):
    """
    Run a single experiment with given configuration
    
    Args:
        config: dict with experiment parameters
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: {config['name']}")
    print(f"{'='*80}")
    for key, value in config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Build environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = config.get('cuda_visible_devices', '0,1')
    env['MODEL'] = config.get('model', 'facebook/opt-125m')
    env['SOURCE_LANG'] = config.get('source_lang', 'en')
    env['TARGET_LANG'] = config.get('target_lang', 'zh')
    env['NUM_TRAIN'] = str(config.get('num_train', 10000))
    env['NUM_EVAL'] = str(config.get('num_eval', 1000))
    env['STEPS'] = str(config.get('steps', 1000))
    env['EVAL_STEPS'] = str(config.get('eval_steps', 100))
    env['N'] = str(config['n'])  # Number of ZO directions
    env['DP_EPS'] = str(config.get('dp_epsilon', 2.0))
    env['DP_CLIP'] = str(config.get('dp_clip', 7.5))
    env['DP_SAMPLE_RATE'] = str(config.get('dp_sample_rate', 0.064))
    env['MAX_LENGTH'] = str(config.get('max_length', 256))
    
    # Learning rate scaling: lr = base_lr * sqrt(N)
    base_lr = config.get('base_lr', 1e-5)
    lr = base_lr * math.sqrt(config['n'])
    env['LR'] = str(lr)
    
    # ZO epsilon
    env['EPS'] = str(config.get('zo_eps', 1e-3))
    
    # Seed
    env['SEED'] = str(config.get('seed', 0))
    
    # Max samples (for scaling law)
    if 'max_samples' in config:
        env['MAX_SAMPLES'] = str(config['max_samples'])
    
    # Build command
    script_path = os.path.join(os.path.dirname(__file__), 'examples', 'parallel_distzo2_dp_aggzo_translation.sh')
    
    try:
        result = subprocess.run(
            ['bash', script_path],
            env=env,
            cwd=os.path.dirname(__file__),
            check=True,
            capture_output=False  # Show output in real-time
        )
        print(f"\n✓ Experiment '{config['name']}' completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment '{config['name']}' failed with exit code {e.returncode}\n")
        return False


def main():
    """Main parameter sweep"""
    
    # Define parameter grids for scaling law experiments
    n_list = [4, 16]  # ZO direction numbers (core scaling parameter)
    eps_list = [2.0, 8.0]  # Privacy budget
    data_list = [10000, 50000]  # Data sizes
    
    # Additional fixed parameters
    base_config = {
        'model': 'facebook/opt-125m',
        'source_lang': 'en',
        'target_lang': 'zh',
        'num_train': 10000,  # Will be overridden by max_samples
        'num_eval': 1000,
        'steps': 1000,
        'eval_steps': 100,
        'dp_clip': 7.5,
        'dp_sample_rate': 0.064,
        'max_length': 256,
        'zo_eps': 1e-3,
        'base_lr': 1e-5,
        'seed': 0,
        'cuda_visible_devices': '0,1'
    }
    
    # Generate all combinations
    experiments = []
    exp_id = 0
    
    for n, dp_eps, max_samples in product(n_list, eps_list, data_list):
        exp_id += 1
        config = base_config.copy()
        config['n'] = n
        config['dp_epsilon'] = dp_eps
        config['max_samples'] = max_samples
        config['name'] = f"exp_{exp_id:02d}_N{n}_eps{dp_eps}_data{max_samples}"
        experiments.append(config)
    
    print(f"\n{'='*80}")
    print(f"Scaling Law Parameter Sweep")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Parameters:")
    print(f"  N (directions): {n_list}")
    print(f"  DP Epsilon: {eps_list}")
    print(f"  Max Samples: {data_list}")
    print(f"{'='*80}\n")
    
    # Ask for confirmation
    response = input(f"Run {len(experiments)} experiments? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    results = []
    for i, config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")
        success = run_experiment(config)
        results.append((config['name'], success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("Experiment Summary")
    print(f"{'='*80}")
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print(f"\nTotal: {len(results)} experiments")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

