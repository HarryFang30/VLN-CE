#!/bin/bash
# ==========================================
# VLN-CE 数据采集后台运行脚本
# ==========================================
# 
# 使用方法：
#   ./run_collect.sh train 1000     # 采集 1000 个 train clips
#   ./run_collect.sh val 200        # 采集 200 个 val clips
#   ./run_collect.sh train 5000 1   # 采集 5000 个 train clips，使用 GPU 1
#
# 日志保存在: /root/autodl-tmp/dataset_with_actions/collect_${SPLIT}.log
# ==========================================

set -e

# 参数
SPLIT=${1:-train}
NUM_CLIPS=${2:-1000}
GPU=${3:-0}
OUTPUT_DIR="/root/autodl-tmp/dataset_with_actions"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 日志文件
LOG_FILE="${OUTPUT_DIR}/collect_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "🚀 VLN-CE 数据采集"
echo "=========================================="
echo "  Split: ${SPLIT}"
echo "  Clips: ${NUM_CLIPS}"
echo "  GPU: ${GPU}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${LOG_FILE}"
echo "=========================================="

# 切换到 VLN-CE 目录
cd /root/VLN-CE

# 激活环境（如果需要）
# source activate base

# 后台运行
echo "📝 开始采集，日志写入: ${LOG_FILE}"
echo "📝 使用 'tail -f ${LOG_FILE}' 查看实时日志"
echo "📝 使用 'pkill -f collect_with_actions' 停止采集"

nohup python collect_with_actions.py \
    --split ${SPLIT} \
    --num-clips ${NUM_CLIPS} \
    --output ${OUTPUT_DIR} \
    --gpu ${GPU} \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo "=========================================="
echo "✅ 采集进程已启动 (PID: ${PID})"
echo "=========================================="
echo ""
echo "常用命令："
echo "  查看日志:    tail -f ${LOG_FILE}"
echo "  查看进度:    grep '✅ Clip' ${LOG_FILE} | tail -5"
echo "  查看统计:    grep 'Successful\\|Failed' ${LOG_FILE}"
echo "  停止采集:    kill ${PID}"
echo ""


