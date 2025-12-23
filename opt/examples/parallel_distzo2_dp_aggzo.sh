#!/bin/bash
# Parallel DistZO2 + DP-AggZO training script for OPT
# 将 K 个方向分配到多个 GPU 上，大幅提升效率
#
# Example usage:
# CUDA_VISIBLE_DEVICES=0,1 MODEL=facebook/opt-125m TASK=SQuAD MODE=ft LR=1e-5 EPS=1e-3 STEPS=100 N=16 DP_EPS=2.0 DP_CLIP=7.5 bash examples/parallel_distzo2_dp_aggzo.sh

MODEL=${MODEL:-facebook/opt-125m}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-1000}
EVAL_STEPS=${EVAL_STEPS:-250}
DP_EPS=${DP_EPS:-2.0}
DP_CLIP=${DP_CLIP:-7.5}
DP_SAMPLE_RATE=${DP_SAMPLE_RATE:-0.064}
N=${N:-16}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_LENGTH=${MAX_LENGTH:-1024}

MODE=${MODE:-ft}
TAG=parallel-distzo2-dp-aggzo-$MODE-$STEPS-$N-$LR-$EPS-$SEED-$DP_EPS-$DP_CLIP

TASK_ARGS=""
case $TASK in
    CB)
        DEV=100
        ;;
    Copa)
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD)
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP)
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

# 检测 GPU 数量
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=1
else
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS[@]}
fi

echo "============================================"
echo "Parallel DistZO2 + DP-AggZO Training Configuration"
echo "============================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Mode: $MODE"
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
echo "Output: result/$TASK-${MODEL_NAME}-$TAG"
echo "============================================"
echo ""

# Export N as environment variable to avoid --n conflict with torchrun
export PARALLEL_DISTZO2_N=$N

# 根据 GPU 数量选择运行方式
if [ $NUM_GPUS -eq 1 ]; then
    echo "Running on single GPU (no DDP)"
    python run_parallel_distzo2_dp_aggzo.py \
        --model_name $MODEL \
        --task_name $TASK \
        --output_dir result/$TASK-${MODEL_NAME}-$TAG \
        --tag $TAG \
        --train_set_seed $SEED \
        --num_train $TRAIN \
        --num_dev $DEV \
        --num_eval $EVAL \
        --logging_steps 10 \
        --max_steps $STEPS \
        --trainer zo \
        --load_float16 \
        --learning_rate $LR \
        --zo_eps $EPS \
        --lr_scheduler_type "constant" \
        --evaluation_strategy steps \
        --save_strategy steps \
        --save_total_limit 1 \
        --eval_steps $EVAL_STEPS \
        --save_steps $EVAL_STEPS \
        --train_as_classification \
        --dpzero \
        --dpzero_clip_threshold $DP_CLIP \
        --dp_epsilon $DP_EPS \
        --dp_delta 1e-5 \
        --dp_sample_rate $DP_SAMPLE_RATE \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        $TASK_ARGS \
        "$@"
else
    echo "Running on $NUM_GPUS GPUs with DDP (torchrun)"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
        run_parallel_distzo2_dp_aggzo.py \
        --model_name $MODEL \
        --task_name $TASK \
        --output_dir result/$TASK-${MODEL_NAME}-$TAG \
        --tag $TAG \
        --train_set_seed $SEED \
        --num_train $TRAIN \
        --num_dev $DEV \
        --num_eval $EVAL \
        --logging_steps 10 \
        --max_steps $STEPS \
        --trainer zo \
        --load_float16 \
        --learning_rate $LR \
        --zo_eps $EPS \
        --lr_scheduler_type "constant" \
        --evaluation_strategy steps \
        --save_strategy steps \
        --save_total_limit 1 \
        --eval_steps $EVAL_STEPS \
        --save_steps $EVAL_STEPS \
        --train_as_classification \
        --dpzero \
        --dpzero_clip_threshold $DP_CLIP \
        --dp_epsilon $DP_EPS \
        --dp_delta 1e-5 \
        --dp_sample_rate $DP_SAMPLE_RATE \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        $TASK_ARGS \
        "$@"
fi

unset PARALLEL_DISTZO2_N

echo ""
echo "============================================"
echo "Training completed!"
echo "Tag: $TAG"
echo "============================================"

