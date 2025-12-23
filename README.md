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
MAX_LENGTH=1024 \
DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.05 \
STEP=1000 \
SEED=42 \
NUM_DIRECTION=64 \
RANDOM_DIRECTION_SEED=100 \
DPZERO_THRESHOLD=25 \
DP_DELTA=1e-5 \
MAX_SAMPLES=10000 \
bash examples/parallel_distzo2_dp_aggzo_translation.sh

```
