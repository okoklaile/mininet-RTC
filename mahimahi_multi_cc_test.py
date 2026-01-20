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
    python3 mahimahi_multi_cc_test.py <trace_file> [--delay DELAY_MS] [--loss LOSS_PERCENT] [--queue QUEUE_SIZE]
    例如: python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down --delay 20 --loss 0 --queue 100
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
DEFAULT_QUEUE_SIZE = 1000  # 默认缓冲区大小（包数）

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
    
    # #region agent log
    import json as json_log
    with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:147","message":"create_config_for_algorithm ENTRY","data":{"algo":algo,"port":port,"test_duration":test_duration,"is_receiver":is_receiver},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"B,D"})+'\n')
    # #endregion
    
    if is_receiver:
        template_path = os.path.join(CONFIG_DIR, 'receiver_pyinfer.json')
        with open(template_path, 'r') as f:
            config = json.load(f)
        
        config['serverless_connection']['receiver']['listening_port'] = port
        config['serverless_connection']['receiver']['listening_ip'] = '0.0.0.0'  # 监听所有接口
        config['serverless_connection']['autoclose'] = test_duration
        
        config['save_to_file']['audio']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_outaudio.wav')
        config['save_to_file']['video']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_outvideo.yuv')
        config['logging']['log_output_path'] = os.path.join(OUTPUT_DIR, f'{algo}_receiver.log')
        
        config_path = os.path.join(CCALGS_DIR, algo, 'receiver_mahimahi.json')
        
        # #region agent log
        with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:163","message":"BEFORE writing receiver config","data":{"config_path":config_path,"exists":os.path.exists(os.path.dirname(config_path)),"port":port},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"B,D"})+'\n')
        # #endregion
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # #region agent log
        with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:165","message":"AFTER writing receiver config","data":{"config_path":config_path,"file_exists":os.path.exists(config_path),"file_size":os.path.getsize(config_path) if os.path.exists(config_path) else 0},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"B"})+'\n')
        # #endregion
    else:
        template_path = os.path.join(CONFIG_DIR, 'sender_pyinfer.json')
        with open(template_path, 'r') as f:
            config = json.load(f)
        
        # Sender 在 Mahimahi shell 内，连接到外部的 Receiver
        # 使用 shell 的 $MAHIMAHI_BASE 环境变量（mahimahi自动设置）
        config['serverless_connection']['sender']['dest_ip'] = '$$MAHIMAHI_BASE'  # 占位符，稍后替换
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

def run_mahimahi_test(algo, port, uplink_trace, downlink_trace, delay_ms, loss_percent, queue_size, test_duration, work_dir):
    """
    在 Mahimahi 环境中运行单个算法的测试
    
    正确架构：
    - Receiver 在 Mahimahi shell **外部**运行，监听 $MAHIMAHI_BASE
    - Sender 在 Mahimahi shell **内部**运行，连接到 Receiver
    - 这样 sender→receiver 的流量才会经过 mahimahi 的带宽限制
    
    参数:
    - queue_size: 缓冲区大小（包数），用于限制队列长度，防止 bufferbloat
    
    返回: (receiver_process, mahimahi_process, receiver_log, sender_log)
    """
    # #region agent log
    import json as json_log
    with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:199","message":"run_mahimahi_test ENTRY","data":{"algo":algo,"port":port,"work_dir":work_dir,"work_dir_exists":os.path.exists(work_dir)},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"A,D"})+'\n')
    # #endregion
    
    # 设置环境变量
    lib_path = os.path.join(SCRIPT_DIR, 'lib')
    py_path = os.path.join(SCRIPT_DIR, 'pylib')
    env_setup = f'export LD_LIBRARY_PATH={lib_path}:$LD_LIBRARY_PATH && export PYTHONPATH={py_path}:$PYTHONPATH'
    
    # 创建日志文件路径
    receiver_log = f'/tmp/{algo}_mahi_receiver.log'
    sender_log = f'/tmp/{algo}_mahi_sender.log'
    
    receiver_config = os.path.join(work_dir, 'receiver_mahimahi.json')
    sender_config = os.path.join(work_dir, 'sender_mahimahi.json')
    
    # 1. 先在**外部**启动 Receiver（不在mahimahi shell内）
    receiver_cmd = f'{env_setup} && cd {work_dir} && {BIN_PATH} {receiver_config} > {receiver_log} 2>&1'
    receiver_proc = subprocess.Popen(receiver_cmd, shell=True, preexec_fn=os.setsid)
    
    # 等待receiver启动并监听端口
    time.sleep(3)
    
    # 2. 在 Mahimahi shell **内部**启动 Sender
    # 在启动前，用sed替换配置文件中的$$MAHIMAHI_BASE为实际的环境变量值
    sender_cmd = f'{env_setup} && cd {work_dir} && sed -i "s/\\$\\$MAHIMAHI_BASE/$MAHIMAHI_BASE/g" {sender_config} && {BIN_PATH} {sender_config} > {sender_log} 2>&1'
    
    # 构建 Mahimahi 包装命令
    mahimahi_cmd_base = f"mm-delay {delay_ms}"
    
    if loss_percent > 0:
        mahimahi_cmd_base += f" mm-loss uplink {loss_percent}"
    
    # 添加队列大小限制（防止 bufferbloat）
    mahimahi_cmd_base += f" mm-link {uplink_trace} {downlink_trace}"
    mahimahi_cmd_base += f" --uplink-queue=droptail --uplink-queue-args=packets={queue_size}"
    mahimahi_cmd_base += f" --downlink-queue=droptail --downlink-queue-args=packets={queue_size}"
    
    # 启动 Mahimahi shell（只包含 sender）
    mahimahi_full_cmd = f"{mahimahi_cmd_base} -- sh -c '{sender_cmd}'"
    mahimahi_proc = subprocess.Popen(mahimahi_full_cmd, shell=True, preexec_fn=os.setsid)
    
    # 等待连接建立
    time.sleep(2)
    
    return receiver_proc, mahimahi_proc, receiver_log, sender_log


def cleanup_old_processes():
    """清理旧的进程，释放端口"""
    print("\n清理旧的进程...")
    
    # #region agent log
    import json as json_log
    with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:244","message":"cleanup_old_processes START","data":{},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"post-fix","hypothesisId":"FIX"})+'\n')
    # #endregion
    
    # 杀死所有 peerconnection_serverless 相关进程（包括Python包装器和C++二进制文件）
    for proc_name in ['peerconnection_serverless.origin_v4', 'peerconnection_serverless.origin', 
                      'peerconnection_serverless', 'mm-link', 'mm-delay', 'mm-loss']:
        try:
            result = subprocess.run(['pkill', '-9', proc_name], 
                                   capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"  ✓ 已终止旧的 {proc_name} 进程")
        except:
            pass
    
    # 额外使用 -f 选项匹配完整命令行，确保清理所有相关进程
    try:
        subprocess.run(['pkill', '-9', '-f', 'peerconnection_serverless'], 
                      capture_output=True, timeout=5)
    except:
        pass
    
    # 等待端口释放
    time.sleep(2)
    
    # 验证是否还有残留的 .origin 进程
    try:
        result = subprocess.run(['pgrep', '-f', 'peerconnection_serverless.origin'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            remaining_pids = result.stdout.strip().split('\n')
            print(f"  ⚠ 警告: 仍有 {len(remaining_pids)} 个 .origin 进程未清理，强制清理...")
            # #region agent log
            with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:265","message":"Remaining origin processes after cleanup","data":{"remaining_pids":remaining_pids},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"post-fix","hypothesisId":"FIX"})+'\n')
            # #endregion
            for pid in remaining_pids:
                try:
                    os.system(f'kill -9 {pid} 2>/dev/null')
                except:
                    pass
            time.sleep(1)
    except:
        pass
    
    # 最后再次使用killall确保彻底清理
    try:
        subprocess.run(['killall', '-9', 'peerconnection_serverless'], 
                      capture_output=True, timeout=5)
        subprocess.run(['killall', '-9', 'peerconnection_serverless.origin'], 
                      capture_output=True, timeout=5)
        subprocess.run(['killall', '-9', 'peerconnection_serverless.origin_v4'], 
                      capture_output=True, timeout=5)
        time.sleep(2)  # 等待端口完全释放
    except:
        pass
    
    # 验证关键端口是否已释放
    print("\n检查端口状态...")
    max_retries = 3
    for retry in range(max_retries):
        try:
            result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True, timeout=5)
            occupied_ports = []
            for port in range(PORT_BASE, PORT_BASE + len(ALGORITHMS)):
                if f":{port} " in result.stdout or f":{port}\n" in result.stdout:
                    occupied_ports.append(port)
            
            # #region agent log
            with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:268","message":"Port status after cleanup","data":{"occupied_ports":occupied_ports,"retry":retry},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"post-fix","hypothesisId":"FIX"})+'\n')
            # #endregion
            
            if occupied_ports:
                if retry < max_retries - 1:
                    print(f"  ⚠ 警告: 以下端口仍被占用: {occupied_ports}，重试清理 ({retry+1}/{max_retries})...")
                    # 查找占用端口的进程并强制杀死
                    for port in occupied_ports:
                        try:
                            lsof_result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                                        capture_output=True, text=True, timeout=5)
                            if lsof_result.stdout.strip():
                                pids = lsof_result.stdout.strip().split('\n')
                                for pid in pids:
                                    print(f"    强制终止占用端口{port}的进程 PID={pid}")
                                    os.system(f'kill -9 {pid} 2>/dev/null')
                        except:
                            pass
                    time.sleep(3)
                else:
                    print(f"  ✗ 错误: 以下端口仍被占用: {occupied_ports}")
                    print(f"  请手动运行: sudo lsof -ti:{occupied_ports[0]} | xargs kill -9")
                    sys.exit(1)
            else:
                print(f"  ✓ 端口 {PORT_BASE}-{PORT_BASE + len(ALGORITHMS)-1} 均可用")
                break
        except Exception as e:
            print(f"  跳过端口检查（工具不可用）: {e}")
            # #region agent log
            with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:280","message":"Port check failed","data":{"error":str(e)},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"post-fix","hypothesisId":"FIX"})+'\n')
            # #endregion
            break


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


def run_multi_mahimahi_test(uplink_trace_file, downlink_trace_file=None, delay_ms=DEFAULT_DELAY, loss_percent=DEFAULT_LOSS, queue_size=DEFAULT_QUEUE_SIZE, custom_duration=None):
    """运行多算法 Mahimahi 测试"""
    
    print("=" * 70)
    print("Mahimahi 多拥塞控制算法并行测试")
    print("=" * 70)
    print(f"测试算法: {', '.join(ALGORITHMS)}")
    print(f"Uplink trace: {os.path.basename(uplink_trace_file)}")
    
    # 解析 trace
    trace = MahimahiTrace(uplink_trace_file)
    
    # 使用用户指定的时长或trace的时长
    if custom_duration is not None:
        test_duration = custom_duration
        trace_duration = trace.get_total_duration_sec()
        if test_duration > trace_duration:
            print(f"  注意: 指定时长({test_duration}秒) > trace时长({trace_duration}秒), Mahimahi将循环使用trace")
    else:
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
    print(f"缓冲区大小: {queue_size} 包")
    if custom_duration is not None:
        print(f"测试时长: {test_duration}秒 (手动指定)")
    else:
        print(f"测试时长: {test_duration}秒 (根据trace自动计算)")
    print("=" * 70)
    
    cleanup_old_processes()  # 先清理旧进程，释放端口
    cleanup_old_files()
    setup_environment()
    
    print("\n生成配置文件并启动测试...")
    print("=" * 70)
    
    # 启动所有算法的测试
    processes = []
    
    for i, algo in enumerate(ALGORITHMS):
        port = PORT_BASE + i
        work_dir = os.path.join(CCALGS_DIR, algo)
        
        # #region agent log
        import json as json_log
        with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:294","message":"Starting algorithm iteration","data":{"algo":algo,"index":i,"port":port,"work_dir":work_dir},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"A"})+'\n')
        # #endregion
        
        # 生成配置文件
        create_config_for_algorithm(algo, port, test_duration, is_receiver=True)
        create_config_for_algorithm(algo, port, test_duration, is_receiver=False)
        
        print(f"\n[{algo}] 启动测试 (端口: {port})")
        
        try:
            receiver_proc, mahimahi_proc, recv_log, send_log = run_mahimahi_test(
                algo, port, uplink_trace_file, downlink_trace_file,
                delay_ms, loss_percent, queue_size, test_duration, work_dir
            )
            
            # #region agent log
            with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:310","message":"AFTER run_mahimahi_test returned","data":{"algo":algo,"receiver_pid":receiver_proc.pid,"mahimahi_pid":mahimahi_proc.pid,"mahimahi_running":mahimahi_proc.poll() is None},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"C,E"})+'\n')
            # #endregion
            
            processes.append({
                'algo': algo,
                'receiver': receiver_proc,
                'mahimahi': mahimahi_proc,
                'recv_log': recv_log,
                'send_log': send_log
            })
            
            print(f"  ✓ Receiver PID: {receiver_proc.pid}, Mahimahi shell PID: {mahimahi_proc.pid}")
            
            # 给每个算法充足的启动时间，避免端口冲突
            time.sleep(2)
            
        except Exception as e:
            # #region agent log
            import traceback
            with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:325","message":"Exception during algorithm startup","data":{"algo":algo,"error":str(e),"traceback":traceback.format_exc()},"timestamp":int(time.time()*1000),"sessionId":"debug-session","hypothesisId":"E"})+'\n')
            # #endregion
            print(f"  ✗ 启动失败: {e}")
    
    print("\n" + "=" * 70)
    print(f"所有算法已启动，测试运行中... (预计 {test_duration} 秒)")
    print("=" * 70)
    
    # #region agent log
    # 等待10秒后检查哪些receiver真正在接收数据
    print("\n等待10秒，检查连接状态...")
    time.sleep(10)
    import json as json_log
    for proc_info in processes:
        algo = proc_info['algo']
        log_file = os.path.join(OUTPUT_DIR, f'{algo}_receiver.log')
        log_size = 0
        log_lines = 0
        if os.path.exists(log_file):
            log_size = os.path.getsize(log_file)
            with open(log_file, 'r') as f:
                log_lines = len(f.readlines())
        
        receiver_running = proc_info['receiver'].poll() is None
        mahimahi_running = proc_info['mahimahi'].poll() is None
        
        with open('/home/wyq/桌面/mininet-RTC/.cursor/debug.log', 'a') as f_log: 
            f_log.write(json_log.dumps({"location":"mahimahi_multi_cc_test.py:372","message":"10 seconds status check","data":{"algo":algo,"log_size":log_size,"log_lines":log_lines,"receiver_running":receiver_running,"mahimahi_running":mahimahi_running},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"post-fix","hypothesisId":"NETWORK"})+'\n')
        
        status = "✓" if log_lines > 10 else "✗"
        print(f"  [{algo}] {status} Log: {log_lines} lines, Receiver: {receiver_running}, Mahimahi: {mahimahi_running}")
    # #endregion
    
    print("\n提示:")
    print("  - 测试将自动运行完整个 trace 时长")
    print("  - 按 Ctrl+C 可以提前结束")
    print("  - 结果将保存在 output/ 目录")
    print()
    
    # 等待测试完成
    try:
        # 显示进度（已经过了10秒的检查时间）
        start_time = time.time()
        remaining_duration = test_duration - 10  # 减去已经等待的10秒
        while time.time() - start_time < remaining_duration:
            elapsed = time.time() - start_time + 10  # 加上之前的10秒
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
        
        # 停止 mahimahi shell (包含sender)
        mahimahi_proc = proc_info['mahimahi']
        try:
            os.killpg(os.getpgid(mahimahi_proc.pid), signal.SIGTERM)
            mahimahi_proc.wait(timeout=3)
        except:
            try:
                os.killpg(os.getpgid(mahimahi_proc.pid), signal.SIGKILL)
            except:
                pass
        
        # 停止 receiver
        receiver_proc = proc_info['receiver']
        try:
            os.killpg(os.getpgid(receiver_proc.pid), signal.SIGTERM)
            receiver_proc.wait(timeout=3)
        except:
            try:
                os.killpg(os.getpgid(receiver_proc.pid), signal.SIGKILL)
            except:
                pass
    
    # 清理所有残留进程（使用 -f 匹配完整命令行）
    print("  清理残留进程...")
    os.system('pkill -9 peerconnection_serverless.origin_v4 2>/dev/null')
    os.system('pkill -9 peerconnection_serverless.origin 2>/dev/null')
    os.system('pkill -9 -f peerconnection_serverless 2>/dev/null')
    os.system('pkill -9 -f mm-link 2>/dev/null')
    os.system('pkill -9 -f mm-delay 2>/dev/null')
    os.system('pkill -9 -f mm-loss 2>/dev/null')
    time.sleep(1)  # 等待进程完全终止
    
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
  # 使用 Mahimahi trace 文件测试（自动使用trace的完整时长）
  python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down
  
  # 指定运行时间（60秒）
  python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down --duration 60
  
  # 指定延迟和丢包率
  python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down --delay 50 --loss 1
  
  # 同时指定 uplink 和 downlink trace
  python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down --downlink mahimahi_traces/ATT-LTE-driving-2016.up
  
  # 指定运行时间和网络参数
  python3 mahimahi_multi_cc_test.py mahimahi_traces/7Train1.down --duration 120 --delay 30 --loss 0.5
  
  # 指定缓冲区大小（包数）
  python3 mahimahi_multi_cc_test.py mahimahi_traces/ATT-LTE-driving-2016.down --queue 200
"""
    )
    
    parser.add_argument('uplink_trace', help='Uplink trace 文件路径（Mahimahi 格式）')
    parser.add_argument('--downlink', dest='downlink_trace', default=None,
                        help='Downlink trace 文件路径（可选，默认使用100Mbps恒定带宽）')
    parser.add_argument('--duration', type=int, default=None,
                        help='测试时长（秒），不指定则使用trace的完整时长')
    parser.add_argument('--delay', type=int, default=DEFAULT_DELAY,
                        help=f'单向延迟（毫秒），默认: {DEFAULT_DELAY}ms')
    parser.add_argument('--loss', type=float, default=DEFAULT_LOSS,
                        help=f'丢包率（百分比），默认: {DEFAULT_LOSS}%%')
    parser.add_argument('--queue', type=int, default=DEFAULT_QUEUE_SIZE,
                        help=f'缓冲区大小（包数），默认: {DEFAULT_QUEUE_SIZE} 包')
    
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
    
    # 验证duration参数
    if args.duration is not None and args.duration <= 0:
        print(f"错误: 测试时长必须大于0秒")
        sys.exit(1)
    
    # 运行测试
    run_multi_mahimahi_test(uplink_trace, downlink_trace, args.delay, args.loss, args.queue, args.duration)


if __name__ == '__main__':
    main()
