#!/bin/bash

# 1. 配置路径
CONFIG_FILE="configs/stage2_config.yaml"
LOG_DIR="logs/stage2"
RESUME_PATH="checkpoints/stage1_sald/2026-02-25_15-10-27/sald_stage1_latest.pth"

# 2. 自动创建日志目录
mkdir -p "$LOG_DIR"

# 3. 生成带时间戳的日志文件名
LOG_FILE="${LOG_DIR}/train_stage2_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "🚀 Starting Stage 2 Training..."
echo "📝 Log file: $LOG_FILE"

# 4. 执行加速器训练，并重定向所有输出到日志文件
# 同时也使用 tee 命令让你在屏幕上也能看到实时进度（可选）
export PYTORCH_ALLOC_CONF="max_split_size_mb:32"
export XFORMERS_DISABLED=1

accelerate launch --num_processes 1 --gpu_ids 0 \
    train_stage2.py \
    --config "$CONFIG_FILE" \
    --resume_from_stage1 "$RESUME_PATH" > "$LOG_FILE" 2>&1

echo "✅ Training session finished. Check log at: $LOG_FILE"