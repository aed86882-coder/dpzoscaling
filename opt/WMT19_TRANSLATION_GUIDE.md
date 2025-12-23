# WMT19 翻译任务使用指南

已成功添加 WMT19 翻译任务支持到 Parallel DistZO2 + DP-AggZO 框架中。

---

## 📋 新增文件

1. **`src/wmt19_translation.py`** - WMT19 数据集类
2. **`src/parallel_distzo2_dp_aggzo_wrapper_seq2seq.py`** - Seq2Seq 模型包装器
3. **`run_parallel_distzo2_dp_aggzo_translation.py`** - 翻译任务训练脚本
4. **`examples/parallel_distzo2_dp_aggzo_translation.sh`** - 便捷启动脚本

---

## 🚀 快速开始

### 1. 快速测试（单 GPU，小模型）

```bash
cd /root/autodl-tmp/dpscal/opt

CUDA_VISIBLE_DEVICES=0 \
MODEL=facebook/opt-125m \
SOURCE_LANG=en \
TARGET_LANG=zh \
NUM_TRAIN=1000 \
NUM_EVAL=100 \
STEPS=50 \
EVAL_STEPS=10 \
N=8 \
MAX_LENGTH=256 \
DP_SAMPLE_RATE=1.0 \
bash examples/parallel_distzo2_dp_aggzo_translation.sh
```

### 2. 标准训练（多 GPU）

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL=facebook/opt-125m \
SOURCE_LANG=en \
TARGET_LANG=zh \
NUM_TRAIN=10000 \
NUM_EVAL=1000 \
STEPS=1000 \
EVAL_STEPS=100 \
N=16 \
BATCH_SIZE=4 \
MAX_LENGTH=256 \
DP_SAMPLE_RATE=0.064 \
bash examples/parallel_distzo2_dp_aggzo_translation.sh
```

### 3. 大规模训练（大模型）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL=facebook/opt-1.3b \
SOURCE_LANG=en \
TARGET_LANG=zh \
NUM_TRAIN=50000 \
NUM_EVAL=5000 \
STEPS=5000 \
EVAL_STEPS=250 \
N=32 \
BATCH_SIZE=4 \
MAX_LENGTH=256 \
DP_SAMPLE_RATE=0.032 \
bash examples/parallel_distzo2_dp_aggzo_translation.sh
```

---

## ⚙️ 参数说明

### 模型选择

| 模型 | 大小 | 特点 | 推荐场景 |
|------|------|------|----------|
| `facebook/opt-125m` | ~250MB | 小模型，快速测试 | 快速测试 |
| `facebook/opt-1.3b` | ~2.5GB | 中等模型 | 标准训练 |
| `facebook/opt-2.7b` | ~5.4GB | 较大模型 | 高质量训练 |
| `facebook/opt-6.7b` | ~13GB | 大型模型 | 最佳质量（需要大 GPU） |

### 数据集参数

- `SOURCE_LANG`: 源语言 (默认: `en`)
- `TARGET_LANG`: 目标语言 (默认: `zh`)
- `NUM_TRAIN`: 训练样本数 (默认: `10000`)
- `NUM_EVAL`: 评估样本数 (默认: `1000`)

### 序列长度参数

- `MAX_LENGTH`: 总序列长度，包括 prompt + 源语言 + 目标语言 (默认: `256`)

### DP-AggZO 参数

- `N`: 方向数量 (默认: `16`)
- `DP_EPS`: DP epsilon (默认: `2.0`)
- `DP_CLIP`: 梯度裁剪阈值 (默认: `7.5`)
- `DP_SAMPLE_RATE`: Poisson 采样率 (默认: `0.064`)

---

## 🔧 数据集加载

WMT19 数据集会从 HuggingFace 自动下载。如果下载失败，代码会创建一个虚拟数据集用于测试。

### 环境变量（可选）

如果 HuggingFace 访问有问题，可以设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/
```

---

## 📊 与问答任务的主要区别

| 特性 | SQuAD (问答) | WMT19 (翻译) |
|------|--------------|--------------|
| **模型类型** | Decoder-only (OPT) | Decoder-only (OPT) |
| **输入格式** | 问题 + 上下文 | "Translate en to zh: {source} -> {target}" |
| **输出** | 答案文本 | 目标语言文本 |
| **Wrapper** | `ParallelDistZO2DPAggZOOPT` | `ParallelDistZO2DPAggZOOPT` |
| **Loss** | Cross-entropy | Cross-entropy (only on target part) |
| **评估指标** | F1 Score | BLEU Score (需要额外实现) |

---

## 🎯 性能优化建议

### GPU 数量与方向数匹配

建议 K 能被 num_gpus 整除：

| K | 推荐 GPU 数量 | 每 GPU 方向数 |
|---|--------------|--------------|
| 8 | 1, 2, 4 | 8, 4, 2 |
| 16 | 1, 2, 4 | 16, 8, 4 |
| 32 | 1, 2, 4, 8 | 32, 16, 8, 4 |

### 显存优化

如果遇到 OOM：

1. 减少 `BATCH_SIZE` (8 → 4 → 2 → 1)
2. 减少序列长度 (`MAX_SOURCE_LEN`, `MAX_TARGET_LEN`)
3. 减少方向数 `N` (32 → 16 → 8)
4. 使用更小的模型

### 训练速度

- 翻译任务比分类任务慢（更长序列）
- 多 GPU 可以显著加速（3-4x）
- 建议至少使用 2 个 GPU

---

## ⚠️ 注意事项

1. **模型选择**: 可以使用不同大小的 OPT 模型，根据需要选择
2. **序列长度**: 翻译任务格式为 "Translate X to Y: {source} -> {target}"，需要足够的长度容纳完整 prompt 和目标文本
3. **Prompt 格式**: 使用 "->" 作为分隔符，只对目标语言部分计算 loss
4. **Tokenization**: OPT 使用标准的 BPE tokenizer，无需额外依赖

---

## 📝 示例输出

训练过程中会看到：

```
============================================
Parallel DistZO2-DP-AggZO Translation Training Configuration
============================================
World size: 2 GPUs
Model: facebook/mbart-large-cc25
Dataset: WMT19 en-zh
Total directions: 16
Directions per GPU: ~8
DP epsilon: 2.0
DP clip: 7.5
Learning rate: 1e-05
Max steps: 1000
============================================

[Rank 0] Responsible for directions 0-7 (total 8/16)
[Rank 1] Responsible for directions 8-15 (total 8/16)

Step 0: Loss=8.2341
Step 10: Loss=7.9123
...
```

---

## 🔗 相关资源

- **WMT19 数据集**: https://huggingface.co/datasets/wmt19
- **mBART 模型**: https://huggingface.co/facebook/mbart-large-cc25
- **OPUS-MT 模型**: https://huggingface.co/Helsinki-NLP/opus-mt-en-zh

---

## ✅ 检查清单

运行前确认：

- [ ] 已安装 `datasets` 和 `sentencepiece`
- [ ] 有足够的 GPU 显存（建议至少 16GB per GPU）
- [ ] 设置了正确的环境变量（如需要）
- [ ] 选择了合适的模型（测试用小模型，训练用大模型）

