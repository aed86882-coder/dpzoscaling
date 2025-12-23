#!/bin/bash
# Parallel DistZO2 + DP-AggZO for WMT19 Translation Task
# 将 K 个方向分配到多个 GPU 上，用于翻译任务
#
# Example usage:
# CUDA_VISIBLE_DEVICES=0,1 MODEL=facebook/mbart-large-cc25 SOURCE_LANG=en TARGET_LANG=zh STEPS=100 N=16 bash examples/parallel_distzo2_dp_aggzo_translation.sh

MODEL=${MODEL:-facebook/opt-125m}
SOURCE_LANG=${SOURCE_LANG:-en}
TARGET_LANG=${TARGET_LANG:-zh}

LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
NUM_TRAIN=${NUM_TRAIN:-10000}
NUM_EVAL=${NUM_EVAL:-1000}
STEPS=${STEPS:-1000}
EVAL_STEPS=${EVAL_STEPS:-100}
DP_EPS=${DP_EPS:-2.0}
DP_CLIP=${DP_CLIP:-7.5}
DP_SAMPLE_RATE=${DP_SAMPLE_RATE:-0.064}
N=${N:-16}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-256}

TAG=parallel-distzo2-dp-aggzo-translation-$STEPS-$N-$LR-$EPS-$SEED-$DP_EPS-$DP_CLIP

# 检测 GPU 数量
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=1
else
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS[@]}
fi

echo "============================================"
echo "Parallel DistZO2 + DP-AggZO Translation Training"
echo "============================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Model: $MODEL"
echo "Dataset: WMT19 $SOURCE_LANG-$TARGET_LANG"
echo "Learning Rate: $LR"
echo "ZO Eps: $EPS"
echo "Training Steps: $STEPS"
echo "Eval Steps: $EVAL_STEPS"
echo "Seed: $SEED"
echo ""
echo "DP Parameters:"
echo "  - DP Epsilon: $DP_EPS"
echo "  - DP Clip: $DP_CLIP"
echo "  - DP Sample Rate: $DP_SAMPLE_RATE"
echo "  - Total Directions: $N"
echo "  - Directions per GPU: ~$((N / NUM_GPUS))"
echo ""
echo "Memory Optimization:"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Max Length: $MAX_LENGTH"
echo ""
echo "Output: result/translation-$TAG"
echo "============================================"
echo ""

# Export N as environment variable to avoid --n conflict with torchrun
export PARALLEL_DISTZO2_N=$N

# 根据 GPU 数量选择运行方式
if [ $NUM_GPUS -eq 1 ]; then
    echo "Running on single GPU (no DDP)"
    python run_parallel_distzo2_dp_aggzo_translation.py \
        --model_name $MODEL \
        --task_name "WMT19__${SOURCE_LANG}-${TARGET_LANG}" \
        --source_lang $SOURCE_LANG \
        --target_lang $TARGET_LANG \
        --output_dir result/translation-$TAG \
        --tag $TAG \
        --train_set_seed $SEED \
        --num_train $NUM_TRAIN \
        --num_eval $NUM_EVAL \
        --logging_steps 10 \
        --max_steps $STEPS \
        --eval_steps $EVAL_STEPS \
        --trainer zo \
        --load_float16 \
        --learning_rate $LR \
        --zo_eps $EPS \
        --lr_scheduler_type "constant" \
        --dpzero \
        --dpzero_clip_threshold $DP_CLIP \
        --dp_epsilon $DP_EPS \
        --dp_delta 1e-5 \
        --dp_sample_rate $DP_SAMPLE_RATE \
        --max_length $MAX_LENGTH \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        "$@"
else
    echo "Running on $NUM_GPUS GPUs with DDP (torchrun)"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
        run_parallel_distzo2_dp_aggzo_translation.py \
        --model_name $MODEL \
        --task_name "WMT19__${SOURCE_LANG}-${TARGET_LANG}" \
        --source_lang $SOURCE_LANG \
        --target_lang $TARGET_LANG \
        --output_dir result/translation-$TAG \
        --tag $TAG \
        --train_set_seed $SEED \
        --num_train $NUM_TRAIN \
        --num_eval $NUM_EVAL \
        --logging_steps 10 \
        --max_steps $STEPS \
        --eval_steps $EVAL_STEPS \
        --trainer zo \
        --load_float16 \
        --learning_rate $LR \
        --zo_eps $EPS \
        --lr_scheduler_type "constant" \
        --dpzero \
        --dpzero_clip_threshold $DP_CLIP \
        --dp_epsilon $DP_EPS \
        --dp_delta 1e-5 \
        --dp_sample_rate $DP_SAMPLE_RATE \
        --max_length $MAX_LENGTH \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        "$@"
fi

unset PARALLEL_DISTZO2_N

echo ""
echo "============================================"
echo "Translation Training completed!"
echo "Tag: $TAG"
echo "============================================"

