#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mininet 多拥塞控制算法并行测试脚本

功能:
- 为每个算法创建独立的一对一拓扑 (sender <-> receiver)
- 所有算法同时测试
- 可通过注释算法列表来选择测试的算法
- 支持网络trace场景，实时改变网络状况
- 支持随机trace系列循环测试
- 自动生成独立的输出文件

使用方法:
    1. 单个trace测试:
       sudo python3 test.py [trace_file]
       例如: sudo python3 test.py trace/4G_3mbps.json
    
    2. 批量trace系列测试 (遍历文件夹，每个trace独立运行，保存所有log):
       sudo python3 test.py --batch <trace文件夹>
       例如: sudo python3 test.py --batch trace/4G_series_200step
    
    3. 随机trace系列测试 (从4G_series_200step随机抽取，循环运行):
       sudo python3 test.py --random-series <总时长(秒)>
       例如: sudo python3 test.py --random-series 3600   # 跑1小时
             sudo python3 test.py --random-series 7200   # 跑2小时
    
    4. 静态网络配置测试:
       sudo python3 test.py
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
import random

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
    #'BBR', 
    #'dummy',
    #'FARC', 
    #'Gemini', 
    #'HRCC', 
    #'Schaferct',
    #'Copa',
    #'Copa+',
    #'Cubic',
    #'PCC'
]

# 默认网络配置（当没有使用trace时）
PORT = 8000
DEFAULT_BANDWIDTH = '10Mbps'
DEFAULT_DELAY = '30ms'
DEFAULT_LOSS = 1
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
        
        # 配置 sender 端保存文件路径
        config['save_to_file']['audio']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_inaudio.wav')
        config['save_to_file']['video']['file_path'] = os.path.join(OUTPUT_DIR, f'{algo}_invideo.yuv')
        
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
        for ext in ['_outaudio.wav', '_outvideo.y4m', '_outvideo.yuv', '_inaudio.wav', '_invideo.yuv', '_receiver.log', '_sender.log']:
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
    os.system('pkill -9 peerconnection_serverless 2>/dev/null')
    net.stop()
    
    info("\n" + "=" * 70 + "\n")
    info("测试完成！\n")
    info("=" * 70 + "\n")
    info(f"\n结果文件位于: {OUTPUT_DIR}/\n")
    for algo in ALGORITHMS:
        info(f"  [{algo}]\n")
        info(f"    - {algo}_receiver.log (接收端日志)\n")
        info(f"    - {algo}_outvideo.yuv (接收端视频)\n")
        info(f"    - {algo}_outaudio.wav (接收端音频)\n")
        info(f"    - {algo}_invideo.yuv (发送端视频)\n")
        info(f"    - {algo}_inaudio.wav (发送端音频)\n")
    info("\n")


def save_logs_with_trace_name(trace_file):
    """
    将当前测试的log文件重命名，加上trace名称和时间戳，避免被覆盖
    
    Args:
        trace_file: trace文件路径，用于提取名称
    """
    import datetime
    
    # 提取trace文件名（不含扩展名）
    trace_basename = os.path.splitext(os.path.basename(trace_file))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    info(f"\n保存log文件 (trace: {trace_basename})...\n")
    
    for algo in ALGORITHMS:
        # 重命名receiver.log
        src_log = os.path.join(OUTPUT_DIR, f'{algo}_receiver.log')
        if os.path.exists(src_log):
            dst_log = os.path.join(OUTPUT_DIR, f'{algo}_receiver_{trace_basename}_{timestamp}.log')
            try:
                os.rename(src_log, dst_log)
                info(f"  保存: {os.path.basename(dst_log)}\n")
            except Exception as e:
                info(f"  警告: 无法重命名 {src_log}: {e}\n")
        
        # 重命名sender.log
        src_log = os.path.join(OUTPUT_DIR, f'{algo}_sender.log')
        if os.path.exists(src_log):
            dst_log = os.path.join(OUTPUT_DIR, f'{algo}_sender_{trace_basename}_{timestamp}.log')
            try:
                os.rename(src_log, dst_log)
                info(f"  保存: {os.path.basename(dst_log)}\n")
            except Exception as e:
                info(f"  警告: 无法重命名 {src_log}: {e}\n")
    
    info("\n")


def run_trace_series_batch(series_dir):
    """
    批量运行trace系列测试 - 每个trace独立运行，保存所有log
    
    遍历指定文件夹下的所有trace文件，对每个trace：
    1. 启动网络和RTC应用
    2. 运行完整个trace
    3. 停止应用和网络
    4. 保存log（带trace名称和时间戳）
    5. 继续下一个trace
    
    Args:
        series_dir: trace系列文件夹路径
    """
    # 获取所有trace文件
    all_traces = sorted([
        os.path.join(series_dir, f) 
        for f in os.listdir(series_dir) 
        if f.endswith('.json')
    ])
    
    if not all_traces:
        info("错误: 找不到trace文件\n")
        return
    
    info("=" * 70 + "\n")
    info("批量Trace系列测试模式\n")
    info("=" * 70 + "\n")
    info(f"测试算法: {', '.join(ALGORITHMS)}\n")
    info(f"Trace文件夹: {series_dir}\n")
    info(f"找到trace数: {len(all_traces)}\n")
    info("=" * 70 + "\n")
    info("⚠️  注意: 每个trace独立运行，所有log都会保存\n")
    info("=" * 70 + "\n\n")
    
    setup_environment()
    
    # 遍历每个trace
    for idx, trace_file in enumerate(all_traces, 1):
        trace_basename = os.path.basename(trace_file)
        
        info("\n" + "=" * 70 + "\n")
        info(f"📊 [{idx}/{len(all_traces)}] 开始测试: {trace_basename}\n")
        info("=" * 70 + "\n\n")
        
        # 解析trace
        try:
            trace = NetworkTrace(trace_file)
            test_duration = trace.get_total_duration_sec()
        except Exception as e:
            info(f"❌ 错误: 无法加载trace文件: {e}\n")
            continue
        
        info(f"测试时长: {test_duration}秒\n\n")
        
        # 清理旧的临时文件（但不删除已保存的log）
        info("清理临时输出文件...\n")
        for algo in ALL_ALGORITHMS:
            for ext in ['_outaudio.wav', '_outvideo.y4m', '_outvideo.yuv', '_inaudio.wav', '_invideo.yuv', '_receiver.log', '_sender.log']:
                old_file = os.path.join(OUTPUT_DIR, f'{algo}{ext}')
                if os.path.exists(old_file):
                    try:
                        os.remove(old_file)
                    except Exception as e:
                        pass
        info("\n")
        
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
            
            # 创建链路 - 使用trace模式
            bw = 10  # 默认10Mbps，会被trace覆盖
            delay = '20ms'
            loss = 0
            
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
        info(f"所有算法已启动，开始应用trace...\n")
        info("=" * 70 + "\n\n")
        
        # 启动trace应用线程
        stop_event = threading.Event()
        trace_thread = threading.Thread(
            target=apply_trace_to_links,
            args=(net, hosts, trace, stop_event)
        )
        trace_thread.daemon = True
        trace_thread.start()
        
        # 等待trace完成
        try:
            trace_thread.join()
        except KeyboardInterrupt:
            info("\n收到中断信号，停止当前测试...\n")
            stop_event.set()
            break
        
        # 停止trace线程
        if trace_thread.is_alive():
            stop_event.set()
            trace_thread.join(timeout=2)
        
        # 清理进程
        info("\n停止进程...\n")
        for proc_type, algo, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        os.system('pkill -9 peerconnection_serverless 2>/dev/null')
        
        # 停止网络
        net.stop()
        
        # 保存log文件（带trace名称）
        save_logs_with_trace_name(trace_file)
        
        info("=" * 70 + "\n")
        info(f"✅ [{idx}/{len(all_traces)}] {trace_basename} 测试完成\n")
        info("=" * 70 + "\n\n")
        
        # 短暂休息，让系统稳定
        if idx < len(all_traces):
            info("等待3秒后继续下一个trace...\n\n")
            time.sleep(3)
    
    info("\n" + "=" * 70 + "\n")
    info("🎉 所有trace测试完成！\n")
    info("=" * 70 + "\n")
    info(f"\n所有结果文件位于: {OUTPUT_DIR}/\n")
    info(f"每个trace的log都已保存，文件名格式: <算法>_<类型>_<trace名称>_<时间戳>.log\n\n")


def run_random_trace_series(series_dir, total_duration_sec):
    """
    在指定总时长内，循环运行随机trace
    
    Args:
        series_dir: trace系列文件夹路径
        total_duration_sec: 总运行时长（秒）
    """
    # 获取所有可用的trace文件
    all_traces = [
        os.path.join(series_dir, f) 
        for f in os.listdir(series_dir) 
        if f.endswith('.json')
    ]
    
    if not all_traces:
        info("错误: 找不到trace文件\n")
        return
    
    info("=" * 70 + "\n")
    info("随机Trace系列测试模式\n")
    info("=" * 70 + "\n")
    info(f"测试算法: {', '.join(ALGORITHMS)}\n")
    info(f"Trace文件夹: {os.path.basename(series_dir)}\n")
    info(f"可用trace数: {len(all_traces)}\n")
    info(f"⏱️  总运行时长: {total_duration_sec}秒 ({total_duration_sec/60:.1f}分钟 / {total_duration_sec/3600:.2f}小时)\n")
    info("=" * 70 + "\n\n")
    
    # 清理旧文件
    info("清理旧的输出文件...\n")
    for algo in ALL_ALGORITHMS:
        for ext in ['_outaudio.wav', '_outvideo.y4m', '_outvideo.yuv', '_inaudio.wav', '_invideo.yuv', '_receiver.log', '_sender.log']:
            old_file = os.path.join(OUTPUT_DIR, f'{algo}{ext}')
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                    info(f"  删除: {os.path.basename(old_file)}\n")
                except Exception as e:
                    info(f"  警告: 无法删除 {old_file}: {e}\n")
    info("\n")
    
    setup_environment()
    
    # 创建Mininet网络（只创建一次）
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
        
        # 创建基础链路，后续由trace动态修改
        bw = 10  # 默认10Mbps
        delay = '20ms'
        loss = 0
        
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
    
    # 启动所有receiver和sender（只启动一次，使用总时长+buffer）
    processes = []
    test_duration_with_buffer = total_duration_sec + 60  # 加60秒buffer
    
    for algo in ALGORITHMS:
        h = hosts[algo]
        receiver = h['receiver']
        sender = h['sender']
        receiver_ip = h['receiver_ip']
        
        create_config_for_algorithm(algo, receiver_ip, test_duration_with_buffer, is_receiver=True)
        create_config_for_algorithm(algo, receiver_ip, test_duration_with_buffer, is_receiver=False)
        
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
    info(f"所有算法已启动，开始循环运行随机trace...\n")
    info("=" * 70 + "\n\n")
    
    # 开始循环运行trace
    start_time = time.time()
    trace_count = 0
    stop_event = threading.Event()
    
    try:
        while True:
            elapsed = time.time() - start_time
            remaining = total_duration_sec - elapsed
            
            if remaining <= 0:
                info("\n⏰ 达到总运行时长，停止测试\n")
                break
            
            # 随机选择一个trace
            trace_file = random.choice(all_traces)
            trace_count += 1
            
            info("\n" + "=" * 70 + "\n")
            info(f"🎲 第 {trace_count} 个trace (已运行: {elapsed/60:.1f}分钟, 剩余: {remaining/60:.1f}分钟)\n")
            info(f"📄 {os.path.basename(trace_file)}\n")
            
            # 加载并运行这个trace
            try:
                trace = NetworkTrace(trace_file)
                trace_duration = trace.get_total_duration_sec()
                
                info(f"⏱️  Trace时长: {trace_duration}秒 ({trace_duration/60:.1f}分钟)\n")
                info("=" * 70 + "\n\n")
                
                # 在新线程中运行trace
                stop_event.clear()
                trace_thread = threading.Thread(
                    target=apply_trace_to_links,
                    args=(net, hosts, trace, stop_event)
                )
                trace_thread.start()
                trace_thread.join()  # 等待trace完整跑完
                
                info(f"\n✓ Trace {trace_count} 完成\n")
                
            except Exception as e:
                info(f"⚠️ 运行trace时出错: {e}\n")
                continue
                
    except KeyboardInterrupt:
        info("\n收到中断信号，停止测试...\n")
        stop_event.set()
    
    # 清理
    info("\n停止测试...\n")
    for proc_type, algo, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    os.system('pkill -9 peerconnection_serverless 2>/dev/null')
    net.stop()
    
    total_elapsed = time.time() - start_time
    
    info("\n" + "=" * 70 + "\n")
    info("测试完成！\n")
    info("=" * 70 + "\n")
    info(f"✅ 总共运行了 {trace_count} 个trace\n")
    info(f"⏱️  实际运行时间: {total_elapsed/60:.1f}分钟 ({total_elapsed/3600:.2f}小时)\n")
    info(f"\n结果文件位于: {OUTPUT_DIR}/\n")
    for algo in ALGORITHMS:
        info(f"  [{algo}]\n")
        info(f"    - {algo}_receiver.log (接收端日志)\n")
        info(f"    - {algo}_outvideo.yuv (接收端视频)\n")
        info(f"    - {algo}_outaudio.wav (接收端音频)\n")
        info(f"    - {algo}_invideo.yuv (发送端视频)\n")
        info(f"    - {algo}_inaudio.wav (发送端音频)\n")
    info("\n")


if __name__ == '__main__':
    if os.geteuid() != 0:
        print("错误: 此脚本需要root权限运行")
        print("请使用:")
        print("  sudo python3 test.py [trace_file]                    # 单个trace测试")
        print("  sudo python3 test.py --batch <trace文件夹>           # 批量测试")
        print("  sudo python3 test.py --random-series <总时长(秒)>    # 随机系列测试")
        exit(1)
    
    setLogLevel('info')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # 批量trace系列模式
        if len(sys.argv) < 3:
            print("用法: sudo python3 test.py --batch <trace文件夹>")
            print("示例:")
            print("  sudo python3 test.py --batch trace/4G_series_200step")
            print("说明:")
            print("  - 遍历文件夹下的所有trace文件")
            print("  - 每个trace独立运行（启动→运行→关闭）")
            print("  - 所有log都会保存，不会覆盖")
            exit(1)
        
        series_dir = sys.argv[2]
        if not os.path.isabs(series_dir):
            series_dir = os.path.join(SCRIPT_DIR, series_dir)
        
        if not os.path.exists(series_dir):
            print(f"错误: 找不到trace文件夹: {series_dir}")
            exit(1)
        
        if not os.path.isdir(series_dir):
            print(f"错误: {series_dir} 不是一个文件夹")
            exit(1)
        
        # 运行批量trace系列测试
        run_trace_series_batch(series_dir)
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--random-series':
        # 随机trace系列模式
        if len(sys.argv) < 3:
            print("用法: sudo python3 test.py --random-series <总时长(秒)>")
            print("示例:")
            print("  sudo python3 test.py --random-series 3600   # 跑1小时")
            print("  sudo python3 test.py --random-series 7200   # 跑2小时")
            print("  sudo python3 test.py --random-series 300    # 跑5分钟")
            exit(1)
        
        try:
            total_duration = int(sys.argv[2])
        except ValueError:
            print("错误: 总时长必须是整数（秒）")
            exit(1)
        
        if total_duration <= 0:
            print("错误: 总时长必须大于0")
            exit(1)
        
        series_dir = os.path.join(TRACE_DIR, '4G_series_200step')
        
        if not os.path.exists(series_dir):
            print(f"错误: 找不到trace系列文件夹: {series_dir}")
            exit(1)
        
        # 运行随机trace系列测试
        run_random_trace_series(series_dir, total_duration)
    
    elif len(sys.argv) > 1:
        # 原有的单个trace模式
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
        
        # 运行单个trace测试
        run_multi_cc_test(trace_file)
    
    else:
        # 无trace，静态配置模式
        run_multi_cc_test(None)