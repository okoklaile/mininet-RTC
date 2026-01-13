#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mininet 多拥塞控制算法并行测试脚本

功能:
- 为每个算法创建独立的一对一拓扑 (sender <-> receiver)
- 所有算法同时测试
- 可通过注释算法列表来选择测试的算法
- 支持网络trace场景，实时改变网络状况
- 自动生成独立的输出文件

使用方法:
    sudo python3 multi_cc_test.py [trace_file]
    例如: sudo python3 multi_cc_test.py trace/4G_3mbps.json
"""

from mininet.net import Mininet
from mininet.node import Host
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import time
import os
import json
import sys
import threading

# ============================================
# 配置区域 - 在这里选择要测试的算法
# ============================================
# 所有可用的算法（用于清理旧文件）
ALL_ALGORITHMS = [
    'GCC', 'BBR', 'dummy', 'PCC', 'Copa', 'Copa+',
    'Cubic', 'FARC', 'Gemini', 'HRCC', 'RL-DelayGCC', 'Schaferct',
]

# 可测试的算法列表 - 注释掉不想测试的算法
ALGORITHMS = [
    'GCC', 
    'BBR', 
    'dummy',
    'FARC', 
    #'Gemini', 
    'HRCC', 
    'Schaferct',
]

# 默认网络配置（当没有使用trace时）
PORT = 8000
DEFAULT_BANDWIDTH = '10Mbps'
DEFAULT_DELAY = '20ms'
DEFAULT_LOSS = 0
QUEUE_SIZE = 1000

# 测试时长（秒）- 如果使用trace，会根据trace总时长自动调整
TEST_DURATION = 60

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_PATH = os.path.join(SCRIPT_DIR, 'bin/peerconnection_serverless')
CONFIG_DIR = os.path.join(SCRIPT_DIR, 'config')
CCALGS_DIR = os.path.join(SCRIPT_DIR, 'ccalgs')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
TRACE_DIR = os.path.join(SCRIPT_DIR, 'trace')


# ============================================
# Trace解析和应用
# ============================================

class NetworkTrace:
    """网络trace解析器"""
    def __init__(self, trace_file):
        self.trace_file = trace_file
        self.trace_pattern = []
        self.total_duration = 0
        self.parse_trace()
    
    def parse_trace(self):
        """解析trace文件"""
        with open(self.trace_file, 'r') as f:
            data = json.load(f)
        
        # 提取uplink的trace_pattern
        if 'uplink' in data and 'trace_pattern' in data['uplink']:
            self.trace_pattern = data['uplink']['trace_pattern']
        else:
            raise ValueError("Trace文件格式错误：缺少 uplink.trace_pattern")
        
        # 计算总时长（毫秒）
        self.total_duration = sum(p['duration'] for p in self.trace_pattern)
        
        info(f"✓ 加载trace: {os.path.basename(self.trace_file)}\n")
        info(f"  - 总时长: {self.total_duration/1000:.1f}秒\n")
        info(f"  - 阶段数: {len(self.trace_pattern)}\n\n")
    
    def get_total_duration_sec(self):
        """获取trace总时长（秒）"""
        return int(self.total_duration / 1000) + 5


def apply_trace_to_links(net, hosts, trace, stop_event):
    """
    应用trace到所有链路
    在后台线程中运行，实时改变网络参数
    """
    info("=" * 70 + "\n")
    info("开始应用网络trace...\n")
    info("=" * 70 + "\n\n")
    
    start_time = time.time()
    
    for stage_idx, stage in enumerate(trace.trace_pattern):
        if stop_event.is_set():
            break
        
        duration_ms = stage['duration']
        capacity_kbps = stage['capacity']
        loss_rate = stage.get('loss', 0)
        rtt_ms = stage.get('rtt', 40)  # 默认40ms RTT
        jitter_ms = stage.get('jitter', 0)
        
        # 计算单向延迟（RTT的一半）
        delay_ms = rtt_ms / 2
        
        elapsed = time.time() - start_time
        info(f"[{elapsed:.1f}s] 阶段 {stage_idx+1}/{len(trace.trace_pattern)}: "
             f"带宽={capacity_kbps}kbps, 延迟={rtt_ms}ms, 丢包={loss_rate}%, "
             f"抖动={jitter_ms}ms, 持续={duration_ms}ms\n")
        
        # 应用到所有算法的链路
        for algo in hosts.keys():
            receiver = hosts[algo]['receiver']
            sender = hosts[algo]['sender']
            
            # 获取链路接口
            receiver_intf = receiver.intf(f'{receiver.name}-eth0')
            sender_intf = sender.intf(f'{sender.name}-eth0')
            
                        # 使用tc命令修改网络参数
            try:
                # 对于sender端的接口，使用netem修改参数
                # 注意：rate参数必须大于0，单位用kbit
                cmd = (
                    f'tc qdisc change dev {sender_intf.name} root '
                    f'netem rate {capacity_kbps}kbit '
                    f'delay {delay_ms}ms'
                )
                
                # 只在有抖动时添加抖动参数
                if jitter_ms > 0:
                    cmd += f' {jitter_ms}ms'
                
                # 只在有丢包时添加丢包参数
                if loss_rate > 0:
                    cmd += f' loss {loss_rate}%'
                
                # 执行命令
                result = sender.cmd(cmd)
                if result and 'Error' in result:
                    info(f"  ⚠️ {algo}: {result}\n")
                    
            except Exception as e:
                info(f"  ⚠️ 无法修改 {algo} 链路参数: {e}\n")
        
        # 等待这个阶段的持续时间
        sleep_time = duration_ms / 1000.0
        time.sleep(sleep_time)
    
    info("\n" + "=" * 70 + "\n")
    info("网络trace应用完成\n")
    info("=" * 70 + "\n\n")


# ============================================
# 原有函数
# ============================================

def create_config_for_algorithm(algo, receiver_ip, test_duration, is_receiver=True):
    """为特定算法生成配置文件"""
    
    if is_receiver:
        template_path = os.path.join(CONFIG_DIR, 'receiver_pyinfer.json')
        with open(template_path, 'r') as f:
            config = json.load(f)
        
        config['serverless_connection']['receiver']['listening_port'] = PORT
        config['serverless_connection']['receiver']['listening_ip'] = '0.0.0.0'
        config['serverless_connection']['autoclose'] = test_duration
        
        config['save_to_file']['audio']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_outaudio.wav')
        config['save_to_file']['video']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_outvideo.yuv')
        config['logging']['log_output_path'] = os.path.join(OUTPUT_DIR, f'{algo}_receiver.log')
        
        config_path = os.path.join(CCALGS_DIR, algo, 'receiver_pyinfer.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        template_path = os.path.join(CONFIG_DIR, 'sender_pyinfer.json')
        with open(template_path, 'r') as f:
            config = json.load(f)
        
        config['serverless_connection']['sender']['dest_ip'] = receiver_ip
        config['serverless_connection']['sender']['dest_port'] = PORT
        config['serverless_connection']['autoclose'] = test_duration
        
        if config['logging']['enabled']:
            config['logging']['log_output_path'] = os.path.join(OUTPUT_DIR, f'{algo}_sender.log')
        
        config_path = os.path.join(CCALGS_DIR, algo, 'sender_pyinfer.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    return config_path


def setup_environment():
    """设置环境变量"""
    os.environ['LD_LIBRARY_PATH'] = os.path.join(SCRIPT_DIR, 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['PYTHONPATH'] = os.path.join(SCRIPT_DIR, 'pylib') + ':' + os.environ.get('PYTHONPATH', '')


def run_multi_cc_test(trace_file=None):
    """运行多算法测试"""
    
    # 解析trace文件（如果提供）
    trace = None
    test_duration = TEST_DURATION
    
    if trace_file:
        try:
            trace = NetworkTrace(trace_file)
            test_duration = trace.get_total_duration_sec()
        except Exception as e:
            info(f"错误: 无法加载trace文件: {e}\n")
            return
    
    info("=" * 70 + "\n")
    info("Mininet 多拥塞控制算法并行测试\n")
    info("=" * 70 + "\n")
    info(f"测试算法: {', '.join(ALGORITHMS)}\n")
    info(f"测试时长: {test_duration}秒\n")
    
    if trace:
        info(f"网络场景: {os.path.basename(trace_file)} (动态trace)\n")
    else:
        info(f"网络配置: 带宽={DEFAULT_BANDWIDTH}, 延迟={DEFAULT_DELAY}, 丢包={DEFAULT_LOSS}% (静态)\n")
    
    info("=" * 70 + "\n\n")
    
    # 清理旧文件
    info("清理旧的输出文件...\n")
    for algo in ALL_ALGORITHMS:
        for ext in ['_outaudio.wav', '_outvideo.y4m','_outvideo.yuv', '_receiver.log', '_sender.log']:
            old_file = os.path.join(OUTPUT_DIR, f'{algo}{ext}')
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                    info(f"  删除: {os.path.basename(old_file)}\n")
                except Exception as e:
                    info(f"  警告: 无法删除 {old_file}: {e}\n")
    info("\n")
    
    setup_environment()
    
    # 创建Mininet网络
    net = Mininet(link=TCLink)
    hosts = {}
    
        # 为每个算法创建一对主机
    for i, algo in enumerate(ALGORITHMS):
        receiver_name = f'r{i}'
        sender_name = f's{i}'
        receiver_ip = f'10.0.{i}.1'
        sender_ip = f'10.0.{i}.2'
        
        info(f"创建拓扑: {algo} - {receiver_name}({receiver_ip}) <-> {sender_name}({sender_ip})\n")
        
        receiver = net.addHost(receiver_name, ip=receiver_ip)
        sender = net.addHost(sender_name, ip=sender_ip)
        
        # 创建链路 - 使用默认参数，trace会动态修改
        if trace:
            # 如果使用trace，先创建一个基础链路，后续由trace线程修改
            bw = 10  # 默认10Mbps，会被trace覆盖
            delay = '20ms'
            loss = 0
        else:
            # 静态配置
            bw = float(DEFAULT_BANDWIDTH.rstrip('Mbps'))
            delay = DEFAULT_DELAY
            loss = DEFAULT_LOSS
        
        net.addLink(receiver, sender, bw=bw, delay=delay, loss=loss, max_queue_size=QUEUE_SIZE)
        
        hosts[algo] = {
            'receiver': receiver,
            'sender': sender,
            'receiver_ip': receiver_ip,
            'sender_ip': sender_ip
        }
    
    info("\n启动网络...\n")
    net.start()
    time.sleep(2)
    
    info("\n生成配置文件并启动测试...\n")
    info("=" * 70 + "\n")
    
    # 启动所有receiver和sender
    processes = []
    for algo in ALGORITHMS:
        h = hosts[algo]
        receiver = h['receiver']
        sender = h['sender']
        receiver_ip = h['receiver_ip']
        
        create_config_for_algorithm(algo, receiver_ip, test_duration, is_receiver=True)
        create_config_for_algorithm(algo, receiver_ip, test_duration, is_receiver=False)
        
        work_dir = os.path.join(CCALGS_DIR, algo)
        
        info(f"[{algo}] 启动 Receiver 在 {receiver.name} ({receiver_ip}:{PORT})\n")
        receiver_cmd = f'cd {work_dir} && {BIN_PATH} receiver_pyinfer.json > /tmp/{algo}_receiver.out 2>&1'
        receiver_proc = receiver.popen(receiver_cmd, shell=True)
        processes.append(('receiver', algo, receiver_proc))
        
        time.sleep(1)
        
        info(f"[{algo}] 启动 Sender 在 {sender.name} -> {receiver_ip}:{PORT}\n")
        sender_cmd = f'cd {work_dir} && {BIN_PATH} sender_pyinfer.json > /tmp/{algo}_sender.out 2>&1'
        sender_proc = sender.popen(sender_cmd, shell=True)
        processes.append(('sender', algo, sender_proc))
    
        info("\n" + "=" * 70 + "\n")
    info(f"所有算法已启动，测试运行中... (预计 {test_duration} 秒)\n")
    info("=" * 70 + "\n\n")
    
    # 如果有trace，启动后台线程应用网络变化
    stop_event = threading.Event()
    trace_thread = None
    
    if trace:
        info("启动网络trace应用线程...\n\n")
        trace_thread = threading.Thread(
            target=apply_trace_to_links,
            args=(net, hosts, trace, stop_event)
        )
        trace_thread.daemon = True
        trace_thread.start()
    
    info("提示: \n")
    if trace:
        info("  - 测试将自动运行完整个trace时长\n")
    else:
        info("  - 测试将运行 {test_duration} 秒\n")
    info("  - 按 Ctrl+C 可以提前结束\n")
    info("  - 结果将保存在 output/ 目录\n\n")
    
    # 等待测试完成（不使用CLI，避免并发冲突）
    try:
        if trace:
            # 等待trace线程完成
            trace_thread.join()
        else:
            # 等待固定时长
            time.sleep(test_duration)
    except KeyboardInterrupt:
        info("\n收到中断信号，停止测试...\n")
        if trace_thread:
            stop_event.set()
    
    # 停止trace线程（如果还在运行）
    if trace_thread and trace_thread.is_alive():
        stop_event.set()
        trace_thread.join(timeout=2)
    
    # 清理
    info("\n停止测试...\n")
    for proc_type, algo, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    
    net.stop()
    
    info("\n" + "=" * 70 + "\n")
    info("测试完成！\n")
    info("=" * 70 + "\n")
    info(f"\n结果文件位于: {OUTPUT_DIR}/\n")
    for algo in ALGORITHMS:
        info(f"  - {algo}_receiver.log\n")
    info("\n")


if __name__ == '__main__':
    if os.geteuid() != 0:
        print("错误: 此脚本需要root权限运行")
        print("请使用: sudo python3 multi_cc_test.py [trace_file]")
        exit(1)
    
    setLogLevel('info')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查是否提供了trace文件
    trace_file = None
    if len(sys.argv) > 1:
        trace_file = sys.argv[1]
        if not os.path.isabs(trace_file):
            trace_file = os.path.join(SCRIPT_DIR, trace_file)
        if not os.path.exists(trace_file):
            print(f"错误: Trace文件不存在: {trace_file}")
            print(f"\n可用的trace文件:")
            for f in sorted(os.listdir(TRACE_DIR)):
                if f.endswith('.json'):
                    print(f"  - trace/{f}")
            exit(1)
    
    # 运行测试
    run_multi_cc_test(trace_file)