#!/bin/bash

# ==============================================================================
# SALD 训练启动脚本 (支持单卡/多卡)
# 用法:
#   bash run_train.sh --gpu 0          (单卡运行，指定 ID 为 0)
#   bash run_train.sh --multi 2        (多卡运行，使用 2 张显卡)
# ==============================================================================

# 默认参数
GPU_ID="0"
NUM_PROCESSES=1
MULTI_GPU=false

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU_ID="$2"
      NUM_PROCESSES=1
      MULTI_GPU=false
      shift 2
      ;;
    --multi)
      NUM_PROCESSES="$2"
      MULTI_GPU=true
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 环境变量优化 (针对 4090 显存碎片)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
export HF_ENDPOINT="https://hf-mirror.com"

echo "🚀 准备点火训练..."

if [ "$MULTI_GPU" = true ]; then
  echo "📡 模式: 多卡并行 (数量: $NUM_PROCESSES)"
  accelerate launch \
    --multi_gpu \
    --num_processes $NUM_PROCESSES \
    --mixed_precision fp16 \
    train_stage1.py
else
  echo "🎯 模式: 单卡训练 (GPU ID: $GPU_ID)"
  CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    train_stage1.py
fi