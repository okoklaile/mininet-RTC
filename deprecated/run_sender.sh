#!/bin/bash
# AlphaRTC Sender 启动脚本

if [ $# -eq 0 ]; then
    echo "用法: $0 <algorithm_name>"
    echo "例如: $0 GCC"
    exit 1
fi

ALGORITHM=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查算法目录是否存在
if [ ! -d "${SCRIPT_DIR}/ccalgs/${ALGORITHM}" ]; then
    echo "错误: 算法目录 'ccalgs/${ALGORITHM}' 不存在"
    exit 1
fi

# 设置环境变量
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${SCRIPT_DIR}/pylib:${PYTHONPATH}"

# 切换到算法目录（即使sender不使用，也需要在此目录以便导入BandwidthEstimator）
cd "${SCRIPT_DIR}/ccalgs/${ALGORITHM}"

# 使用Python生成临时配置文件（包含算法名的日志文件）
python3 -c "
import json
with open('${SCRIPT_DIR}/config/sender_pyinfer.json', 'r') as f:
    config = json.load(f)

# 修改日志文件路径，加入算法名
if config['logging']['enabled']:
    config['logging']['log_output_path'] = '/home/wyq/桌面/mininet-RTC/output/${ALGORITHM}_sender.log'

with open('sender_pyinfer.json', 'w') as f:
    json.dump(config, f, indent=4)
"

echo "========================================="
echo "启动 AlphaRTC Sender"
echo "算法: ${ALGORITHM}"
echo "工作目录: $(pwd)"
echo "日志文件: ${ALGORITHM}_sender.log"
echo "========================================="

# 运行sender
"${SCRIPT_DIR}/bin/peerconnection_serverless" sender_pyinfer.json