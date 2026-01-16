#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahimahi 多拥塞控制算法并行测试脚本

功能:
- 使用 Mahimahi 网络仿真工具
- 为每个算法创建独立的测试环境
- 所有算法同时测试
- 支持 Mahimahi trace 格式（uplink/downlink）
- 自动生成独立的输出文件

Mahimahi trace 格式:
- 每行一个时间戳（毫秒）
- 表示在该时间可以发送一个 MTU 包（1500字节）

使用方法:
    python3 mahimahi_multi_cc_test.py <trace_file> [--delay DELAY_MS] [--loss LOSS_PERCENT]
    例如: python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down --delay 20 --loss 0
"""

import os
import sys
import json
import time
import argparse
import subprocess
import signal
import tempfile
from pathlib import Path

# ============================================
# 配置区域 - 在这里选择要测试的算法
# ============================================
# 所有可用的算法（用于清理旧文件）
ALL_ALGORITHMS = [
    'GCC', 
    'BBR', 'dummy', 'PCC', 'Copa', 'Copa+',
    'Cubic', 'FARC', 'Gemini', 'HRCC', 'RL-DelayGCC', 'Schaferct',
]

# 可测试的算法列表 - 注释掉不想测试的算法
ALGORITHMS = [
    'GCC', 
    'BBR', 
    #'dummy',
    'FARC', 
    #'Gemini', 
    'HRCC', 
    'Schaferct',
    #'Copa',
    'Copa+',
    #'Cubic',
    #'PCC'
]

# 默认网络配置
PORT_BASE = 8000  # 基础端口，每个算法递增
DEFAULT_DELAY = 20  # 默认延迟（毫秒）
DEFAULT_LOSS = 0    # 默认丢包率（百分比）

# 测试时长（秒）- 会根据trace长度自动调整
TEST_DURATION = 60

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_PATH = os.path.join(SCRIPT_DIR, 'bin/peerconnection_serverless')
CONFIG_DIR = os.path.join(SCRIPT_DIR, 'config')
CCALGS_DIR = os.path.join(SCRIPT_DIR, 'ccalgs')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
MAHIMAHI_TRACES_DIR = os.path.join(SCRIPT_DIR, 'mahimahi_traces')

# ============================================
# Mahimahi Trace 处理
# ============================================

class MahimahiTrace:
    """Mahimahi trace 解析器"""
    def __init__(self, trace_file):
        self.trace_file = os.path.abspath(trace_file)
        self.duration_ms = 0
        self.parse_trace()
    
    def parse_trace(self):
        """解析 Mahimahi trace 文件，获取时长"""
        if not os.path.exists(self.trace_file):
            raise FileNotFoundError(f"Trace 文件不存在: {self.trace_file}")
        
        # 读取最后一行获取总时长
        with open(self.trace_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # 过滤空行
                non_empty_lines = [line.strip() for line in lines if line.strip()]
                if non_empty_lines:
                    try:
                        self.duration_ms = int(non_empty_lines[-1])
                    except ValueError:
                        raise ValueError(f"无法解析 trace 文件，最后一行不是有效数字: {non_empty_lines[-1]}")
        
        print(f"✓ 加载 Mahimahi trace: {os.path.basename(self.trace_file)}")
        print(f"  - 总时长: {self.duration_ms/1000:.1f}秒")
        print(f"  - 行数: {len(lines)}")
    
    def get_total_duration_sec(self):
        """获取trace总时长（秒），加5秒缓冲"""
        return int(self.duration_ms / 1000) + 5


def create_constant_trace(bandwidth_mbps, duration_sec, output_file):
    """
    创建恒定带宽的 Mahimahi trace 文件
    
    Args:
        bandwidth_mbps: 带宽（Mbps）
        duration_sec: 时长（秒）
        output_file: 输出文件路径
    """
    # Mahimahi trace: 每个时间戳表示可以发送一个 MTU (1500字节) 的包
    # 带宽(Mbps) -> 每毫秒可以发送的包数
    # bandwidth_mbps * 1e6 / 8 / 1500 = 每秒包数
    # 每毫秒包数 = 每秒包数 / 1000
    
    packets_per_ms = (bandwidth_mbps * 1e6 / 8 / 1500) / 1000
    interval_ms = 1.0 / packets_per_ms if packets_per_ms > 0 else 1000
    
    timestamps = []
    current_time = 0
    duration_ms = duration_sec * 1000
    
    while current_time <= duration_ms:
        timestamps.append(int(current_time))
        current_time += interval_ms
    
    with open(output_file, 'w') as f:
        for ts in timestamps:
            f.write(f"{ts}\n")
    
    print(f"  创建恒定带宽 trace: {bandwidth_mbps}Mbps, {duration_sec}秒 -> {output_file}")
    return output_file


# ============================================
# 配置文件生成
# ============================================

def create_config_for_algorithm(algo, port, test_duration, is_receiver=True):
    """为特定算法生成配置文件"""
    
    if is_receiver:
        template_path = os.path.join(CONFIG_DIR, 'receiver_pyinfer.json')
        with open(template_path, 'r') as f:
            config = json.load(f)
        
        config['serverless_connection']['receiver']['listening_port'] = port
        config['serverless_connection']['receiver']['listening_ip'] = '0.0.0.0'  # 监听所有接口（Mahimahi需要）
        config['serverless_connection']['autoclose'] = test_duration
        
        config['save_to_file']['audio']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_outaudio.wav')
        config['save_to_file']['video']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_outvideo.yuv')
        config['logging']['log_output_path'] = os.path.join(OUTPUT_DIR, f'{algo}_receiver.log')
        
        config_path = os.path.join(CCALGS_DIR, algo, 'receiver_mahimahi.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        template_path = os.path.join(CONFIG_DIR, 'sender_pyinfer.json')
        with open(template_path, 'r') as f:
            config = json.load(f)
        
        # 在 Mahimahi shell 中，100.64.0.1 是访问宿主机的网关地址
        config['serverless_connection']['sender']['dest_ip'] = '100.64.0.1'
        config['serverless_connection']['sender']['dest_port'] = port
        config['serverless_connection']['autoclose'] = test_duration
        
        config['save_to_file']['audio']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_inaudio.wav')
        config['save_to_file']['video']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_invideo.yuv')
        
        if config['logging']['enabled']:
            config['logging']['log_output_path'] = os.path.join(OUTPUT_DIR, f'{algo}_sender.log')
        
        config_path = os.path.join(CCALGS_DIR, algo, 'sender_mahimahi.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    return config_path


def setup_environment():
    """设置环境变量"""
    os.environ['LD_LIBRARY_PATH'] = os.path.join(SCRIPT_DIR, 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['PYTHONPATH'] = os.path.join(SCRIPT_DIR, 'pylib') + ':' + os.environ.get('PYTHONPATH', '')


# ============================================
# Mahimahi 测试执行
# ============================================

def run_mahimahi_test(algo, port, uplink_trace, downlink_trace, delay_ms, loss_percent, test_duration, work_dir):
    """
    在 Mahimahi 环境中运行单个算法的测试
    
    架构：
    - Receiver 在宿主机上运行（监听 0.0.0.0）
    - Sender 在 Mahimahi shell 中运行（连接到宿主机IP）
    
    返回: (receiver_process, sender_process, receiver_log, sender_log)
    """
    # 设置环境变量
    lib_path = os.path.join(SCRIPT_DIR, 'lib')
    py_path = os.path.join(SCRIPT_DIR, 'pylib')
    env_setup = f'export LD_LIBRARY_PATH={lib_path}:$LD_LIBRARY_PATH && export PYTHONPATH={py_path}:$PYTHONPATH'
    
    # 创建日志文件路径
    receiver_log = f'/tmp/{algo}_mahi_receiver_shell.log'
    sender_log = f'/tmp/{algo}_mahi_sender_shell.log'
    
    # Receiver 命令（在宿主机上直接运行，不通过 Mahimahi）
    receiver_cmd = f'{env_setup} && cd {work_dir} && {BIN_PATH} receiver_mahimahi.json > {receiver_log} 2>&1'
    receiver_proc = subprocess.Popen(receiver_cmd, shell=True, preexec_fn=os.setsid)
    
    # 等待 receiver 启动并监听端口（给足够时间让端口完全绑定）
    time.sleep(3)
    
    # Sender 命令（在 Mahimahi shell 中运行）
    # Mahimahi 会拦截 sender 的出站流量，应用带宽限制
    sender_cmd = f'{env_setup} && cd {work_dir} && {BIN_PATH} sender_mahimahi.json'
    
    # 构建 Mahimahi 包装命令
    mahimahi_cmd_base = f"mm-delay {delay_ms}"
    
    if loss_percent > 0:
        mahimahi_cmd_base += f" mm-loss uplink {loss_percent}"
    
    mahimahi_cmd_base += f" mm-link {uplink_trace} {downlink_trace}"
    
    # 启动 sender（在 Mahimahi 环境中）
    sender_full_cmd = f"{mahimahi_cmd_base} -- sh -c '{sender_cmd}' > {sender_log} 2>&1"
    sender_proc = subprocess.Popen(sender_full_cmd, shell=True, preexec_fn=os.setsid)
    
    return receiver_proc, sender_proc, receiver_log, sender_log


def cleanup_old_files():
    """清理旧的输出文件"""
    print("\n清理旧的输出文件...")
    for algo in ALL_ALGORITHMS:
        for ext in ['_outaudio.wav', '_outvideo.y4m', '_outvideo.yuv', '_inaudio.wav', 
                    '_invideo.yuv', '_receiver.log', '_sender.log']:
            old_file = os.path.join(OUTPUT_DIR, f'{algo}{ext}')
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                    print(f"  删除: {os.path.basename(old_file)}")
                except Exception as e:
                    print(f"  警告: 无法删除 {old_file}: {e}")


def run_multi_mahimahi_test(uplink_trace_file, downlink_trace_file=None, delay_ms=DEFAULT_DELAY, loss_percent=DEFAULT_LOSS):
    """运行多算法 Mahimahi 测试"""
    
    print("=" * 70)
    print("Mahimahi 多拥塞控制算法并行测试")
    print("=" * 70)
    print(f"测试算法: {', '.join(ALGORITHMS)}")
    print(f"Uplink trace: {os.path.basename(uplink_trace_file)}")
    
    # 解析 trace
    trace = MahimahiTrace(uplink_trace_file)
    test_duration = trace.get_total_duration_sec()
    
    # 如果没有指定 downlink trace，创建一个高带宽的恒定 trace
    if downlink_trace_file is None:
        downlink_trace_file = tempfile.mktemp(suffix='.mahi', prefix='downlink_')
        create_constant_trace(100, test_duration, downlink_trace_file)  # 100 Mbps downlink
    else:
        downlink_trace_file = os.path.abspath(downlink_trace_file)
    
    print(f"Downlink trace: {os.path.basename(downlink_trace_file)}")
    print(f"延迟: {delay_ms}ms")
    print(f"丢包率: {loss_percent}%")
    print(f"测试时长: {test_duration}秒")
    print("=" * 70)
    
    cleanup_old_files()
    setup_environment()
    
    print("\n生成配置文件并启动测试...")
    print("=" * 70)
    
    # 启动所有算法的测试
    processes = []
    
    for i, algo in enumerate(ALGORITHMS):
        port = PORT_BASE + i
        work_dir = os.path.join(CCALGS_DIR, algo)
        
        # 生成配置文件
        create_config_for_algorithm(algo, port, test_duration, is_receiver=True)
        create_config_for_algorithm(algo, port, test_duration, is_receiver=False)
        
        print(f"\n[{algo}] 启动测试 (端口: {port})")
        
        try:
            receiver_proc, sender_proc, recv_log, send_log = run_mahimahi_test(
                algo, port, uplink_trace_file, downlink_trace_file,
                delay_ms, loss_percent, test_duration, work_dir
            )
            
            processes.append({
                'algo': algo,
                'receiver': receiver_proc,
                'sender': sender_proc,
                'recv_log': recv_log,
                'send_log': send_log
            })
            
            print(f"  ✓ Receiver PID: {receiver_proc.pid}")
            print(f"  ✓ Sender PID: {sender_proc.pid}")
            
            # 给每个算法充足的启动时间，避免端口冲突
            time.sleep(2)
            
        except Exception as e:
            print(f"  ✗ 启动失败: {e}")
    
    print("\n" + "=" * 70)
    print(f"所有算法已启动，测试运行中... (预计 {test_duration} 秒)")
    print("=" * 70)
    print("\n提示:")
    print("  - 测试将自动运行完整个 trace 时长")
    print("  - 按 Ctrl+C 可以提前结束")
    print("  - 结果将保存在 output/ 目录")
    print()
    
    # 等待测试完成
    try:
        # 显示进度
        start_time = time.time()
        while time.time() - start_time < test_duration:
            elapsed = time.time() - start_time
            remaining = test_duration - elapsed
            print(f"\r  进度: {elapsed:.0f}/{test_duration}秒 (剩余 {remaining:.0f}秒)    ", end='', flush=True)
            time.sleep(1)
        print()  # 换行
        
    except KeyboardInterrupt:
        print("\n\n收到中断信号，停止测试...")
    
    # 停止所有进程
    print("\n停止测试...")
    for proc_info in processes:
        algo = proc_info['algo']
        print(f"  停止 [{algo}]...")
        
        for proc_type in ['receiver', 'sender']:
            proc = proc_info[proc_type]
            try:
                # 发送 SIGTERM 到进程组
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3)
            except:
                try:
                    # 如果 SIGTERM 失败，使用 SIGKILL
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    pass
    
    # 清理所有残留进程
    os.system('pkill -9 peerconnection_serverless 2>/dev/null')
    os.system('pkill -9 mm-link 2>/dev/null')
    os.system('pkill -9 mm-delay 2>/dev/null')
    os.system('pkill -9 mm-loss 2>/dev/null')
    
    # 清理临时文件
    if downlink_trace_file and 'downlink_' in downlink_trace_file and os.path.exists(downlink_trace_file):
        os.remove(downlink_trace_file)
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print(f"\n结果文件位于: {OUTPUT_DIR}/")
    for algo in ALGORITHMS:
        print(f"  [{algo}]")
        print(f"    - {algo}_receiver.log (接收端日志)")
        print(f"    - {algo}_outvideo.yuv (接收端视频)")
        print(f"    - {algo}_outaudio.wav (接收端音频)")
        print(f"    - {algo}_invideo.yuv (发送端视频)")
        print(f"    - {algo}_inaudio.wav (发送端音频)")
    print()


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Mahimahi 多拥塞控制算法并行测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 Mahimahi trace 文件测试
  python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down
  
  # 指定延迟和丢包率
  python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down --delay 50 --loss 1
  
  # 同时指定 uplink 和 downlink trace
  python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down --downlink mahimahi_traces/ATT-LTE-driving-2016.up
"""
    )
    
    parser.add_argument('uplink_trace', help='Uplink trace 文件路径（Mahimahi 格式）')
    parser.add_argument('--downlink', dest='downlink_trace', default=None,
                        help='Downlink trace 文件路径（可选，默认使用100Mbps恒定带宽）')
    parser.add_argument('--delay', type=int, default=DEFAULT_DELAY,
                        help=f'单向延迟（毫秒），默认: {DEFAULT_DELAY}ms')
    parser.add_argument('--loss', type=float, default=DEFAULT_LOSS,
                        help=f'丢包率（百分比），默认: {DEFAULT_LOSS}%%')
    
    args = parser.parse_args()
    
    # 检查 Mahimahi 是否安装
    try:
        subprocess.run(['which', 'mm-link'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("错误: Mahimahi 未安装或不在 PATH 中")
        print("请先安装 Mahimahi: https://github.com/ravinet/mahimahi")
        sys.exit(1)
    
    # 处理 trace 文件路径
    uplink_trace = args.uplink_trace
    if not os.path.isabs(uplink_trace):
        uplink_trace = os.path.join(SCRIPT_DIR, uplink_trace)
    
    if not os.path.exists(uplink_trace):
        print(f"错误: Uplink trace 文件不存在: {uplink_trace}")
        print(f"\n可用的 Mahimahi trace 文件:")
        if os.path.exists(MAHIMAHI_TRACES_DIR):
            for f in sorted(os.listdir(MAHIMAHI_TRACES_DIR)):
                if not f.startswith('.'):
                    print(f"  - mahimahi_traces/{f}")
        sys.exit(1)
    
    downlink_trace = args.downlink_trace
    if downlink_trace:
        if not os.path.isabs(downlink_trace):
            downlink_trace = os.path.join(SCRIPT_DIR, downlink_trace)
        if not os.path.exists(downlink_trace):
            print(f"错误: Downlink trace 文件不存在: {downlink_trace}")
            sys.exit(1)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 运行测试
    run_multi_mahimahi_test(uplink_trace, downlink_trace, args.delay, args.loss)


if __name__ == '__main__':
    main()
