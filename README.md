## Environment Setup

Create and activate the conda environment using `environments.yml`:

```bash
conda env create -n dpzero -f environments.yml
conda activate dpzero

```

## Run Example

```bash
cd opt

CUDA_VISIBLE_DEVICES=0,1 \
MODEL=facebook/opt-125m \
SOURCE_LANG=en \
TARGET_LANG=zh \
LR=1e-5 \
BATCH_SIZE=32 \
MAX_LENGTH=1024 \
DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0416 \
STEP=1000 \
SEED=42 \
NUM_DIRECTION=64 \
RANDOM_DIRECTION_SEED=100 \
DPZERO_THRESHOLD=25 \
DP_DELTA=1e-5 \
MAX_SAMPLES=10000 \
bash examples/parallel_distzo2_dp_aggzo_translation.sh

```



## Parameter Description

- **LR**: Learning Rate (default: 1e-5)
- **BATCH_SIZE**: Batch size for training (default: 4)
- **MAX_LENGTH**: Maximum sequence length (default: 256)
- **DPZERO_PRIVACY_EPS** / **DP_EPS**: Differential Privacy epsilon (default: 2.0)
- **DP_DELTA**: Differential Privacy delta (default: 1e-5)
- **DPZERO_THRESHOLD** / **DP_CLIP**: Gradient clipping threshold (default: 7.5)
- **DP_SAMPLE_RATE**: Poisson sampling rate (default: 0.064)
- **STEP** / **STEPS**: Number of training steps (default: 1000)
- **SEED**: Random seed for reproducibility (default: 0)
- **NUM_DIRECTION** / **N**: Number of ZO directions (default: 16)
- **RANDOM_DIRECTION_SEED**: Random seed for direction sampling (default: 42)
- **MAX_SAMPLES**: Maximum number of training samples (default: all available, use for scaling law experiments)
