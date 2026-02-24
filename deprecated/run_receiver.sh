#!/bin/bash
# AlphaRTC Receiver 启动脚本

if [ $# -eq 0 ]; then
    echo "用法: $0 <algorithm_name>"
    echo "例如: $0 GCC"
    exit 1
fi

ALGORITHM=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置环境变量
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${SCRIPT_DIR}/pylib:${PYTHONPATH}"

# 切换到算法目录
cd "${SCRIPT_DIR}/ccalgs/${ALGORITHM}"

# 使用Python生成临时配置文件（包含算法名的输出文件）
python3 -c "
import json
with open('${SCRIPT_DIR}/config/receiver_pyinfer.json', 'r') as f:
    config = json.load(f)

# 修改输出文件路径，加入算法名
config['save_to_file']['audio']['file_path'] = '/home/wyq/桌面/mininet-RTC/output/${ALGORITHM}_outaudio.wav'
config['save_to_file']['video']['file_path'] = '/home/wyq/桌面/mininet-RTC/output/${ALGORITHM}_outvideo.yuv'
config['logging']['log_output_path'] = '/home/wyq/桌面/mininet-RTC/output/${ALGORITHM}_webrtc.log'

with open('receiver_pyinfer.json', 'w') as f:
    json.dump(config, f, indent=4)
"

echo "========================================="
echo "启动 AlphaRTC Receiver"
echo "算法: ${ALGORITHM}"
echo "工作目录: $(pwd)"
echo "输出文件前缀: ${ALGORITHM}_"
echo "========================================="

# 运行receiver
"${SCRIPT_DIR}/bin/peerconnection_serverless" receiver_pyinfer.json