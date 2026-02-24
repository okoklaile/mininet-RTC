#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 receiver.log 文件中的视频流统计数据（video_receive_stream2.cc:800）

功能:
- 提取 VideoReceiveStream stats 数据
- 从视频层的 cum_loss 和 packets_received 计算丢包率
- 分析视频质量指标（比特率、帧率、延迟、卡顿、丢包等）
- 生成多维度对比图表（单张矢量图输出）
- 生成视频质量评估报告

丢包率计算:
    丢包率 = cum_loss / (cum_loss + packets_received) × 100%

使用方法:
    python3 eval/eval_video_stats.py [--smooth] [--window WINDOW]
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 配置
# ============================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
RESULT_DIR = os.path.join(ROOT_DIR, 'eval_results')


# ============================================
# 数据解析
# ============================================

class VideoStatsParser:
    """解析 VideoReceiveStream 统计日志"""
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.video_stats = []
        self.bwe_stats = []
        self.parse_video_stats()
    
    def parse_video_stats(self):
        """从日志中提取 VideoReceiveStream stats 数据和 BWE 数据"""
        if not os.path.exists(self.log_path):
            raise ValueError(f"日志文件不存在: {self.log_path}")
        
        # 正则表达式匹配 VideoReceiveStream stats 行
        pattern = re.compile(
            r'VideoReceiveStream stats: (\d+), \{(.+?)\}(?:, interframe_delay_max_ms: (\d+)\})?'
        )
        
        # 正则表达式匹配 BWE 行
        bwe_pattern = re.compile(r'Send back BWE estimation: ([\d.e+]+) at time: (\d+)')
        
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 1. 解析 VideoReceiveStream stats
                if 'VideoReceiveStream stats:' in line:
                    try:
                        match = re.search(r'VideoReceiveStream stats: (\d+), \{(.+)\}', line)
                        if match:
                            timestamp = int(match.group(1))
                            stats_str = match.group(2)
                            stats = {'timestamp': timestamp}
                            kv_pattern = re.compile(r'(\w+):\s*(-?\d+(?:\.\d+)?)')
                            for kv_match in kv_pattern.finditer(stats_str):
                                key = kv_match.group(1)
                                value_str = kv_match.group(2)
                                try:
                                    if '.' in value_str:
                                        stats[key] = float(value_str)
                                    else:
                                        stats[key] = int(value_str)
                                except ValueError:
                                    stats[key] = value_str
                            self.video_stats.append(stats)
                    except Exception as e:
                        print(f"  ⚠️ 解析视频统计行时出错: {e}")
                
                # 2. 解析 Send back BWE estimation
                elif 'Send back BWE estimation:' in line:
                    try:
                        match = bwe_pattern.search(line)
                        if match:
                            bwe_val = float(match.group(1))
                            timestamp = int(match.group(2))
                            self.bwe_stats.append({
                                'timestamp': timestamp,
                                'bwe': bwe_val
                            })
                    except Exception as e:
                        pass
        
        print(f"  找到 {len(self.video_stats)} 条视频统计记录, {len(self.bwe_stats)} 条 BWE 记录")


class NetworkStatsParser:
    """
    解析网络层统计日志，用于提取丢包率
    
    注意：此类已废弃，不再使用。
    丢包率现在直接从视频层的 cum_loss 和 packets_received 计算。
    保留此类仅供参考或未来可能的网络层分析需求。
    """
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.packet_loss_rate = 0.0
        self.parse_network_stats()
    
    def parse_network_stats(self):
        """从日志中提取网络层数据并计算丢包率"""
        if not os.path.exists(self.log_path):
            return
        
        net_data = []
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "remote_estimator_proxy.cc" not in line:
                    continue
                try:
                    raw_json = line[line.index('{'):]
                    json_network = json.loads(raw_json)
                    net_data.append(json_network)
                except:
                    pass
        
        if not net_data:
            return
        
        # 计算丢包数（通过序列号差值）
        loss_count = 0
        last_seqNo = {}
        
        for item in net_data:
            packet_info = item["packetInfo"]
            ssrc = packet_info["ssrc"]
            sequence_number = packet_info["seqNum"]
            
            if ssrc in last_seqNo:
                loss_count += max(0, sequence_number - last_seqNo[ssrc] - 1)
            last_seqNo[ssrc] = sequence_number
        
        # 计算丢包率
        total_packets = loss_count + len(net_data)
        self.packet_loss_rate = (loss_count / total_packets * 100) if total_packets > 0 else 0.0


# ============================================
# 指标计算
# ============================================

class VideoMetrics:
    """计算视频质量指标"""
    
    @staticmethod
    def remove_outliers(data, multiplier=3):
        """
        使用 IQR (Interquartile Range) 方法剔除极端异常值
        multiplier=3.0 表示剔除极度异常值（通常 1.5 是温和异常，3.0 是极端异常）
        """
        if len(data) < 4:
            return data
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        return [x for x in data if lower_bound <= x <= upper_bound]

    @staticmethod
    def calculate_metrics(video_stats, bwe_stats=None, warmup=0):
        """
        计算各种视频质量指标
        
        参数:
        - video_stats: 视频统计数据列表
        - bwe_stats: BWE 统计数据列表
        - warmup: 预热时间（秒），剔除前 X 秒的数据
        
        返回:
        - 时间序列数据字典
        - 聚合统计指标
        """
        if not video_stats:
            return {}, {}
        
        # 确定基准时间
        all_timestamps = [s['timestamp'] for s in video_stats]
        if bwe_stats:
            all_timestamps.extend([s['timestamp'] for s in bwe_stats])
        base_time = min(all_timestamps)
        
        # 剔除预热期数据
        if warmup > 0:
            video_stats = [s for s in video_stats if (s['timestamp'] - base_time) / 1000.0 >= warmup]
            if bwe_stats:
                bwe_stats = [s for s in bwe_stats if (s['timestamp'] - base_time) / 1000.0 >= warmup]
        
        if not video_stats:
            print(f"  ⚠️ 剔除 {warmup}s 预热期后没有剩余视频数据")
            return {}, {}

        # 重新计算剔除后的时间序列数据
        timestamps = [s['timestamp'] for s in video_stats]
        # 保持 rel_times 相对于原始基准时间，或者相对于第一个有效点？
        # 通常相对于第一个有效点更好看图
        new_base_time = timestamps[0]
        rel_times = [(t - new_base_time) / 1000.0 for t in timestamps] 
        
        time_series = {
            'time': rel_times,
            'total_bps': [s.get('total_bps', 0) / 1e6 for s in video_stats],  # Mbps
            'render_fps': [s.get('render_fps', 0) for s in video_stats],
            'network_fps': [s.get('network_fps', 0) for s in video_stats],
            'decode_fps': [s.get('decode_fps', 0) for s in video_stats],
            'cur_delay_ms': [s.get('cur_delay_ms', 0) for s in video_stats],
            'jb_delay_ms': [s.get('jb_delay_ms', 0) for s in video_stats],
            'e2e_delay_ms': [s.get('e2e_delay_ms', 0) for s in video_stats],  # 端到端延迟
            'network_delay_ms': [s.get('network_delay_ms', 0) for s in video_stats],  # 网络延迟
            'frames_dropped': [s.get('frames_dropped', 0) for s in video_stats],
            'freeze_cnt': [s.get('freeze_cnt', 0) for s in video_stats],
            'freeze_dur_ms': [s.get('freeze_dur_ms', 0) for s in video_stats],
            'cum_loss': [s.get('cum_loss', 0) for s in video_stats],
            'packets_received': [s.get('packets_received', 0) for s in video_stats],
            'nack': [s.get('nack', 0) for s in video_stats],
            'decode_ms': [s.get('decode_ms', 0) for s in video_stats],
            'width': [s.get('width', 0) for s in video_stats],  # 视频宽度
            'height': [s.get('height', 0) for s in video_stats],  # 视频高度
        }
        
        # 处理 BWE 数据
        if bwe_stats:
            bwe_times = [(s['timestamp'] - new_base_time) / 1000.0 for s in bwe_stats]
            bwe_values = [s['bwe'] / 1e6 for s in bwe_stats]  # Mbps
            time_series['bwe_time'] = bwe_times
            time_series['bwe_mbps'] = bwe_values
        else:
            time_series['bwe_time'] = []
            time_series['bwe_mbps'] = []
        
        # 计算丢包率时间序列（百分比）
        time_series['packet_loss_rate'] = [
            (cum_loss / (cum_loss + packets_received) * 100) 
            if (cum_loss + packets_received) > 0 else 0.0
            for cum_loss, packets_received in zip(time_series['cum_loss'], time_series['packets_received'])
        ]
        
        # 聚合统计
        total_packet_loss = time_series['cum_loss'][-1] if time_series['cum_loss'] else 0
        total_packets_received = time_series['packets_received'][-1] if time_series['packets_received'] else 0
        
        # 使用视频层的 cum_loss 和 packets_received 计算丢包率
        total_packets_sent = total_packet_loss + total_packets_received
        video_packet_loss_rate = (total_packet_loss / total_packets_sent * 100) if total_packets_sent > 0 else 0.0
        
        # 计算总播放时长（秒）
        total_duration_s = rel_times[-1] if rel_times else 0
        total_freeze_duration_ms = time_series['freeze_dur_ms'][-1] if time_series['freeze_dur_ms'] else 0
        freeze_rate = (total_freeze_duration_ms / 1000.0 / total_duration_s * 100) if total_duration_s > 0 else 0
        
        # 计算端到端延迟和网络延迟（过滤掉 0 和负值）
        valid_e2e_delays = [d for d in time_series['e2e_delay_ms'] if d > 0]
        valid_network_delays = [d for d in time_series['network_delay_ms'] if d > 0]
        valid_cur_delays = [d for d in time_series['cur_delay_ms'] if d > 0]
        valid_jb_delays = [d for d in time_series['jb_delay_ms'] if d > 0]
        
        # 剔除极端异常点 (使用 3.0 IQR 阈值)
        clean_e2e = VideoMetrics.remove_outliers(valid_e2e_delays, multiplier=3.0)
        clean_network = VideoMetrics.remove_outliers(valid_network_delays, multiplier=3.0)
        clean_cur = VideoMetrics.remove_outliers(valid_cur_delays, multiplier=3.0)
        clean_jb = VideoMetrics.remove_outliers(valid_jb_delays, multiplier=3.0)
        
        # 计算分辨率相关指标
        widths = time_series['width']
        heights = time_series['height']
        avg_width = np.mean([w for w in widths if w > 0])
        avg_height = np.mean([h for h in heights if h > 0])
        
        # 计算分辨率变化次数（分辨率切换）
        resolution_changes = 0
        for i in range(1, len(widths)):
            if widths[i] != widths[i-1] or heights[i] != heights[i-1]:
                resolution_changes += 1
        
        # 计算平均像素数（用于QoE评分）
        avg_pixels = avg_width * avg_height if avg_width > 0 and avg_height > 0 else 0
        
        # --- 计算自定义 QoE 指标 (根据用户提供的公式) ---
        avg_bitrate_mbps = np.mean(time_series['total_bps'])
        # 假设最大带宽为 5.0 Mbps (multi_cc_test.py 中的默认值)
        U = min(1.0, avg_bitrate_mbps / 5.0) 
        qoe_recv_rate = 100 * U
        
        if clean_cur:
            d_max = np.max(clean_cur)
            d_min = np.min(clean_cur)
            d_95th = np.percentile(clean_cur, 95)
            if d_max > d_min:
                qoe_delay = 100 * (d_max - d_95th) / (d_max - d_min)
            else:
                qoe_delay = 100.0
        else:
            qoe_delay = 0.0
            
        L = video_packet_loss_rate / 100.0
        qoe_loss = 100 * (1 - L)
        
        # QoE = 0.2 * QoE_recv_rate + 0.2 * QoE_delay + 0.3 * QoE_loss
        qoe_total = 0.2 * qoe_recv_rate + 0.2 * qoe_delay + 0.3 * qoe_loss
        # ----------------------------------------------

        aggregated = {
            'avg_bitrate': avg_bitrate_mbps,
            'avg_render_fps': np.mean(time_series['render_fps']),
            'avg_network_fps': np.mean(time_series['network_fps']),
            'avg_delay': np.mean(clean_cur) if clean_cur else 0,
            'p95_delay': np.percentile(clean_cur, 95) if clean_cur else 0,
            'avg_jb_delay': np.mean(clean_jb) if clean_jb else 0,
            'p95_jb_delay': np.percentile(clean_jb, 95) if clean_jb else 0,
            'avg_e2e_delay': np.mean(clean_e2e) if clean_e2e else -1,
            'p95_e2e_delay': np.percentile(clean_e2e, 95) if clean_e2e else -1,
            'avg_network_delay': np.mean(clean_network) if clean_network else -1,
            'p95_network_delay': np.percentile(clean_network, 95) if clean_network else -1,
            'total_freeze_count': time_series['freeze_cnt'][-1] if time_series['freeze_cnt'] else 0,
            'total_freeze_duration': total_freeze_duration_ms,
            'freeze_rate': freeze_rate,
            'total_frames_dropped': time_series['frames_dropped'][-1] if time_series['frames_dropped'] else 0,
            'total_packet_loss': total_packet_loss,
            'total_packets_received': total_packets_received,
            'packet_loss_rate': video_packet_loss_rate,  # 使用视频层计算的丢包率
            'total_nack': time_series['nack'][-1] if time_series['nack'] else 0,
            'avg_decode_time': np.mean(time_series['decode_ms']),
            'resolution': f"{video_stats[0].get('width', 0)}x{video_stats[0].get('height', 0)}",
            'avg_width': avg_width,
            'avg_height': avg_height,
            'avg_pixels': avg_pixels,
            'resolution_changes': resolution_changes,
            'total_duration': total_duration_s,
            # 自定义 QoE 指标
            'qoe_recv_rate': qoe_recv_rate,
            'qoe_delay': qoe_delay,
            'qoe_loss': qoe_loss,
            'qoe_total': qoe_total,
        }
        
        return time_series, aggregated


# ============================================
# 平滑处理
# ============================================

def smooth_data(data, window_size=5):
    """对数据进行平滑处理"""
    if len(data) < window_size:
        return data
    
    if window_size % 2 == 0:
        window_size += 1
    
    poly_order = min(3, window_size - 1)
    try:
        return savgol_filter(data, window_size, poly_order)
    except:
        # 回退到移动平均
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        prefix = data[:window_size//2]
        suffix = data[-(window_size//2):]
        return np.concatenate([prefix, smoothed, suffix])


# ============================================
# 绘图配置
# ============================================

# 算法颜色映射 (保持与 Tradeoff 图一致)
COLOR_MAP = {
    'Neural-GCC': '#32CD32',       # LimeGreen
    'Neural-GCC-NoQoE': '#00BFFF', # DeepSkyBlue
    'Neural-GCC-NoKL': '#FF4500',  # OrangeRed
    'Neural-GCC-NoBC': '#FFD700',  # Gold
}

def get_algo_color(algo_name, default=None):
    """根据算法名称获取颜色"""
    # 精确匹配
    if algo_name in COLOR_MAP:
        return COLOR_MAP[algo_name]
    
    # 模糊匹配
    if 'NoQoE' in algo_name: return COLOR_MAP['Neural-GCC-NoQoE']
    if 'NoKL' in algo_name: return COLOR_MAP['Neural-GCC-NoKL']
    if 'NoBC' in algo_name: return COLOR_MAP['Neural-GCC-NoBC']
    if 'Neural-GCC' in algo_name: return COLOR_MAP['Neural-GCC']
    
    return default


# ============================================
# 绘图函数 (单张矢量图)
# ============================================

def save_single_plot(time_series_dict, metric_key, output_path, title_cn, ylabel, smooth=False, smooth_window=5):
    """绘制并保存单个指标的图表"""
    plt.figure(figsize=(8, 5))
    
    # 确保图例顺序：Neural-GCC 优先
    sorted_items = sorted(time_series_dict.items(), key=lambda x: x[0])
    # 尝试让 Neural-GCC 排在前面或特定顺序
    desired_order = ['Neural-GCC', 'Neural-GCC-NoQoE', 'Neural-GCC-NoKL', 'Neural-GCC-NoBC']
    
    # 创建排序索引
    def sort_key(item):
        name = item[0]
        for i, key in enumerate(desired_order):
            if key in name:
                return i
        return 999
        
    sorted_items = sorted(time_series_dict.items(), key=sort_key)
    
    for algo_name, (time_series, _) in sorted_items:
        # 特殊处理 BWE
        if metric_key == 'bwe_mbps':
            time = time_series.get('bwe_time', [])
            data = time_series.get('bwe_mbps', [])
        else:
            time = time_series['time']
            data = time_series.get(metric_key, [])
        
        if len(data) == 0:
            continue
            
        if smooth and len(data) > smooth_window:
            try:
                data = smooth_data(data, smooth_window)
            except:
                pass
        
        color = get_algo_color(algo_name)
        plt.plot(time, data, label=algo_name, linewidth=2, alpha=0.8, color=color)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    # plt.title(title_cn, fontsize=14, fontweight='bold') # 去掉标题
    plt.legend(loc='best', fontsize=19)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  - 保存: {os.path.basename(output_path)}")

def plot_multi_metrics(data_dict, output_dir, smooth=False, smooth_window=5):
    """
    绘制多指标对比图（单张输出）
    """
    print(f"正在生成基础指标图表...")
    
    metrics_config = [
        ('total_bps', 'Video Bitrate', 'Bitrate (Mbps)'),
        ('bwe_mbps', 'Send Back BWE', 'BWE (Mbps)'),
        ('render_fps', 'Render Frame Rate', 'Render FPS'),
        ('e2e_delay_ms', 'End-to-End Delay', 'E2E Delay (ms)'),
        ('network_delay_ms', 'Network Delay', 'Network Delay (ms)'),
        ('jb_delay_ms', 'Jitter Buffer Delay', 'JB Delay (ms)'),
        ('freeze_cnt', 'Freeze Count', 'Freeze Count'),
        ('packet_loss_rate', 'Packet Loss Rate', 'Packet Loss Rate (%)'),
        ('nack', 'NACK Count', 'NACK Count'),
    ]
    
    for metric_key, title_cn, ylabel in metrics_config:
        output_path = os.path.join(output_dir, f"ts_{metric_key}.pdf")
        save_single_plot(data_dict, metric_key, output_path, title_cn, ylabel, smooth, smooth_window)


def plot_fps_comparison(data_dict, output_dir, smooth=False, smooth_window=5):
    """绘制帧率对比图（单张输出）"""
    print(f"正在生成帧率对比图...")
    
    for algo_name, (time_series, _) in data_dict.items():
        plt.figure(figsize=(8, 5))
        time = time_series['time']
        
        network_fps = time_series['network_fps']
        render_fps = time_series['render_fps']
        decode_fps = time_series['decode_fps']
        
        if smooth:
            network_fps = smooth_data(network_fps, smooth_window)
            render_fps = smooth_data(render_fps, smooth_window)
            decode_fps = smooth_data(decode_fps, smooth_window)
        
        plt.plot(time, network_fps, label='Network FPS', linewidth=2, alpha=0.8)
        plt.plot(time, render_fps, label='Render FPS', linewidth=2, alpha=0.8)
        plt.plot(time, decode_fps, label='Decode FPS', linewidth=2, alpha=0.8, linestyle='--')
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frame Rate (FPS)', fontsize=12)
        # plt.title(f'{algo_name} - Frame Rate Comparison', fontsize=14, fontweight='bold') # 去掉标题
        plt.legend(loc='best', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"fps_compare_{algo_name}.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  - 保存: {os.path.basename(output_path)}")


def plot_resolution_timeline(data_dict, output_dir, smooth=False, smooth_window=5):
    """绘制分辨率随时间变化的图表（单张输出）"""
    print(f"正在生成分辨率时间线图...")
    
    for algo_name, (time_series, aggregated) in data_dict.items():
        plt.figure(figsize=(8, 5))
        time = time_series['time']
        
        widths = time_series['width']
        heights = time_series['height']
        
        pixels = [w * h / 1e6 for w, h in zip(widths, heights)]
        
        if smooth and len(pixels) > smooth_window:
            pixels = smooth_data(pixels, smooth_window)
        
        plt.plot(time, pixels, label='Resolution (MPixels)', linewidth=2, alpha=0.8, color='purple')
        
        resolution_changes = aggregated['resolution_changes']
        if resolution_changes > 0:
            for i in range(1, len(widths)):
                if widths[i] != widths[i-1] or heights[i] != heights[i-1]:
                    plt.axvline(x=time[i], color='red', linestyle='--', alpha=0.5, linewidth=1)
                    plt.text(time[i], max(pixels) * 0.9, f'{int(widths[i])}x{int(heights[i])}', 
                           rotation=90, va='top', fontsize=8, color='red')
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Resolution (MPixels)', fontsize=12)
        # plt.title(f'{algo_name} - Video Resolution', fontsize=14, fontweight='bold') # 去掉标题
        plt.legend(loc='best', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, f"resolution_{algo_name}.pdf")
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  - 保存: {os.path.basename(output_path)}")


def save_single_bar_plot(algos, data, metric_name, output_path, ylabel, default_color):
    """绘制单张柱状图"""
    plt.figure(figsize=(8, 5))
    
    # 为每个柱子生成颜色
    bar_colors = []
    for algo in algos:
        c = get_algo_color(algo)
        if c:
            bar_colors.append(c)
        else:
            bar_colors.append(default_color)
            
    bars = plt.bar(algos, data, color=bar_colors, alpha=0.8, edgecolor='black')
    plt.ylabel(ylabel, fontsize=12)
    # plt.title(metric_name, fontsize=14, fontweight='bold') # 去掉标题
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  - 保存: {os.path.basename(output_path)}")

def plot_qoe_metrics(data_dict, output_dir):
    """绘制QoE（用户体验质量）指标柱状图（单张输出）"""
    print(f"正在生成 QoE 统计图...")
    algos = list(data_dict.keys())
    
    metrics = [
        ('freeze_count', [data_dict[a][1]['total_freeze_count'] for a in algos], 'Freeze Count', 'coral'),
        ('freeze_duration', [data_dict[a][1]['total_freeze_duration'] / 1000.0 for a in algos], 'Freeze Duration (s)', 'lightcoral'),
        ('freeze_rate', [data_dict[a][1]['freeze_rate'] for a in algos], 'Freeze Rate (%)', 'salmon'),
        ('packet_loss', [data_dict[a][1]['packet_loss_rate'] for a in algos], 'Packet Loss Rate (%)', 'lightyellow'),
        ('avg_fps', [data_dict[a][1]['avg_render_fps'] for a in algos], 'Average FPS', 'plum'),
        ('avg_bitrate', [data_dict[a][1]['avg_bitrate'] for a in algos], 'Bitrate (Mbps)', 'lightcyan'),
        ('p95_e2e_delay', [data_dict[a][1]['p95_e2e_delay'] if data_dict[a][1]['p95_e2e_delay'] > 0 else 0 for a in algos], 'E2E Delay P95 (ms)', 'lavender'),
        ('p95_net_delay', [data_dict[a][1]['p95_network_delay'] if data_dict[a][1]['p95_network_delay'] > 0 else 0 for a in algos], 'Network Delay P95 (ms)', 'peachpuff'),
        ('avg_pixels', [data_dict[a][1]['avg_pixels'] / 1e6 for a in algos], 'Resolution (MPixels)', 'lightblue'),
    ]
    
    for name, data, ylabel, color in metrics:
        output_path = os.path.join(output_dir, f"bar_{name}.pdf")
        save_single_bar_plot(algos, data, name, output_path, ylabel, color)


def plot_custom_qoe(data_dict, output_dir):
    """绘制自定义 QoE 指标对比图（单张输出）"""
    print(f"正在生成自定义 QoE 图...")
    algos = list(data_dict.keys())
    
    qoe_recv = [data_dict[a][1]['qoe_recv_rate'] for a in algos]
    qoe_delay = [data_dict[a][1]['qoe_delay'] for a in algos]
    qoe_loss = [data_dict[a][1]['qoe_loss'] for a in algos]
    qoe_total = [data_dict[a][1]['qoe_total'] for a in algos]
    
    metrics = [
        ('qoe_recv', qoe_recv, 'Score', 'skyblue'),
        ('qoe_delay', qoe_delay, 'Score', 'lightgreen'),
        ('qoe_loss', qoe_loss, 'Score', 'salmon'),
        ('qoe_total', qoe_total, 'Score', 'gold')
    ]
    
    for name, data, ylabel, color in metrics:
        output_path = os.path.join(output_dir, f"qoe_score_{name}.pdf")
        save_single_bar_plot(algos, data, name, output_path, ylabel, color)


# ============================================
# 统计报告
# ============================================

def generate_report(data_dict, output_path, warmup=0):
    """生成视频质量统计报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("视频流质量统计报告 (VideoReceiveStream Stats Analysis)\n")
        if warmup > 0:
            f.write(f"注意: 已剔除前 {warmup}s 的预热期数据\n")
        f.write("=" * 100 + "\n\n")
        
        # 按平均比特率排序
        sorted_algos = sorted(data_dict.items(), 
                            key=lambda x: x[1][1]['avg_bitrate'], 
                            reverse=True)
        
        for i, (algo_name, (time_series, aggregated)) in enumerate(sorted_algos, 1):
            f.write(f"{i}. {algo_name}\n")
            f.write("-" * 90 + "\n")
            
            # 基本信息
            f.write(f"  视频分辨率:              {aggregated['resolution']} (平均: {aggregated['avg_width']:.0f}x{aggregated['avg_height']:.0f})\n")
            f.write(f"  分辨率变化次数:          {aggregated['resolution_changes']} 次\n")
            f.write(f"  平均像素数:              {aggregated['avg_pixels']/1e6:.2f} MPixels\n")
            f.write(f"  统计样本数:              {len(time_series['time'])} 个数据点\n")
            f.write(f"  总播放时长:              {aggregated['total_duration']:.2f} s\n")
            f.write("\n")
            
            # 比特率与帧率
            f.write("  【比特率与帧率】\n")
            f.write(f"    平均视频比特率:        {aggregated['avg_bitrate']:.3f} Mbps\n")
            f.write(f"    平均渲染帧率:          {aggregated['avg_render_fps']:.1f} FPS\n")
            f.write(f"    平均网络帧率:          {aggregated['avg_network_fps']:.1f} FPS\n")
            f.write("\n")
            
            # 延迟指标
            f.write("  【延迟指标 (Average & P95)】\n")
            f.write(f"    处理延迟:              Avg: {aggregated['avg_delay']:.2f} ms / P95: {aggregated['p95_delay']:.2f} ms\n")
            f.write(f"    抖动缓冲延迟:          Avg: {aggregated['avg_jb_delay']:.2f} ms / P95: {aggregated['p95_jb_delay']:.2f} ms\n")
            f.write(f"    平均解码耗时:          {aggregated['avg_decode_time']:.2f} ms\n")
            
            # 显示端到端延迟和网络延迟（如果有效）
            if aggregated['p95_e2e_delay'] > 0:
                f.write(f"    端到端延迟:            Avg: {aggregated['avg_e2e_delay']:.2f} ms / P95: {aggregated['p95_e2e_delay']:.2f} ms\n")
            else:
                f.write(f"    端到端延迟:            N/A (无有效数据)\n")
            
            if aggregated['p95_network_delay'] > 0:
                f.write(f"    网络延迟:              Avg: {aggregated['avg_network_delay']:.2f} ms / P95: {aggregated['p95_network_delay']:.2f} ms\n")
            else:
                f.write(f"    网络延迟:              N/A (无有效数据)\n")
            f.write("\n")
            
            # 用户体验质量
            f.write("  【用户体验质量 (QoE)】\n")
            f.write(f"    视频卡顿次数:          {aggregated['total_freeze_count']}\n")
            f.write(f"    卡顿总时长:            {aggregated['total_freeze_duration'] / 1000.0:.2f} s\n")
            f.write(f"    卡顿率:                {aggregated['freeze_rate']:.2f} %\n")
            f.write(f"    丢弃的帧数:            {aggregated['total_frames_dropped']}\n")
            f.write("\n")
            
            # 丢包与重传
            f.write("  【丢包与重传】\n")
            f.write(f"    累计丢包数:            {aggregated['total_packet_loss']}\n")
            f.write(f"    累计接收包数:          {aggregated['total_packets_received']}\n")
            f.write(f"    丢包率:                {aggregated['packet_loss_rate']:.2f} % (cum_loss / (cum_loss + packets_received))\n")
            f.write(f"    NACK请求次数:          {aggregated['total_nack']}\n")
            f.write("\n")
            
            # 自定义 QoE (根据用户公式)
            f.write("  【自定义 QoE 指标 (根据用户提供的公式)】\n")
            f.write(f"    QoE_recv_rate (100*U): {aggregated['qoe_recv_rate']:.2f}\n")
            f.write(f"    QoE_delay (100*norm):  {aggregated['qoe_delay']:.2f}\n")
            f.write(f"    QoE_loss (100*(1-L)):  {aggregated['qoe_loss']:.2f}\n")
            f.write(f"    综合 QoE (0.2+0.2+0.3): {aggregated['qoe_total']:.2f}\n")
            f.write("\n")
            
            # QoE评分（简单评分系统）
            qoe_score = calculate_qoe_score(aggregated)
            f.write(f"  【综合QoE评分】:         {qoe_score:.1f} / 100\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
    
    print(f"✓ 统计报告已保存: {output_path}")


def calculate_qoe_score(aggregated):
    """计算简单的QoE评分 (0-100)"""
    score = 0.0
    
    # 1. 帧率评分 (满分25)
    fps = aggregated['avg_render_fps']
    if fps >= 30: score += 25
    elif fps >= 24: score += 22
    elif fps >= 20: score += 18
    elif fps >= 15: score += 13
    else: score += max(0, fps / 30 * 25)
    
    # 2. 延迟评分 (满分20) - 基于 P95 延迟
    delay = aggregated['p95_delay']
    if delay <= 50: score += 20
    elif delay <= 100: score += 16
    elif delay <= 150: score += 12
    elif delay <= 200: score += 8
    elif delay <= 300: score += 6
    else: score += max(0, 20 - (delay - 300) / 30)
    
    # 3. 卡顿评分 (满分25)
    freeze_count = aggregated['total_freeze_count']
    freeze_duration = aggregated['total_freeze_duration']
    freeze_rate = aggregated['freeze_rate']
    
    if freeze_count == 0: score += 25
    elif freeze_count <= 2: score += 20
    elif freeze_count <= 5: score += 15
    elif freeze_count <= 10: score += 10
    else: score += max(0, 25 - freeze_count * 2)
    
    if freeze_duration > 5000: score -= 8
    if freeze_rate > 10: score -= 6
    elif freeze_rate > 5: score -= 3
    
    # 4. 丢包评分 (满分20) - 基于丢包率
    packet_loss_rate = aggregated['packet_loss_rate']
    if packet_loss_rate == 0: score += 20
    elif packet_loss_rate < 0.5: score += 18
    elif packet_loss_rate < 1.0: score += 15
    elif packet_loss_rate < 2.0: score += 12
    elif packet_loss_rate < 5.0: score += 8
    else: score += max(0, 20 - packet_loss_rate)
    
    # 5. 分辨率评分 (满分10)
    avg_pixels = aggregated['avg_pixels']
    resolution_changes = aggregated['resolution_changes']
    
    if avg_pixels >= 0.9: score += 8
    elif avg_pixels >= 0.6: score += 7
    elif avg_pixels >= 0.3: score += 6
    elif avg_pixels >= 0.2: score += 4
    else: score += max(0, avg_pixels / 0.3 * 6)
    
    if resolution_changes == 0: score += 2
    elif resolution_changes <= 2: score += 1
    
    return min(100, max(0, score))


# ============================================
# 主程序
# ============================================

def main():
    parser = argparse.ArgumentParser(description='分析 VideoReceiveStream 统计数据')
    parser.add_argument('--input', type=str, help='指定要分析的日志文件夹路径')
    parser.add_argument('--smooth', action='store_true', help='启用平滑处理')
    parser.add_argument('--window', type=int, default=5, help='平滑窗口大小')
    parser.add_argument('--warmup', type=float, default=0, help='预热时间（秒），剔除开头的数据')
    args = parser.parse_args()

    # 创建结果目录
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    # 创建图片存放目录
    FIGURES_DIR = os.path.join(RESULT_DIR, 'figures')
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
        
    # 查找日志文件
    log_files = []
    
    if args.input:
        if os.path.exists(args.input):
            if os.path.isdir(args.input):
                log_files = glob.glob(os.path.join(args.input, '*_receiver.log'))
            elif os.path.isfile(args.input):
                log_files = [args.input]
        else:
            print(f"错误: 找不到路径 {args.input}")
            return
    else:
        log_files = glob.glob(os.path.join(OUTPUT_DIR, '*_receiver.log'))
    
    if not log_files:
        search_path = args.input if args.input else OUTPUT_DIR
        print(f"错误: 在 {search_path} 中没有找到日志文件 (*_receiver.log)")
        return
    
    print(f"找到 {len(log_files)} 个日志文件\n")
    
    # 解析所有日志
    data_dict = {}
    
    for log_file in sorted(log_files):
        algo_name = os.path.basename(log_file).replace('_receiver.log', '')
        print(f"处理: {algo_name}...")
        
        try:
            parser = VideoStatsParser(log_file)
            if not parser.video_stats:
                print(f"  警告: 未找到视频统计数据\n")
                continue
            
            time_series, aggregated = VideoMetrics.calculate_metrics(parser.video_stats, parser.bwe_stats, args.warmup)
            if not time_series:
                continue
                
            data_dict[algo_name] = (time_series, aggregated)
            
            print(f"  平均比特率: {aggregated['avg_bitrate']:.3f} Mbps")
            print(f"  平均帧率: {aggregated['avg_render_fps']:.1f} FPS")
            print(f"  丢包率: {aggregated['packet_loss_rate']:.2f} %\n")
            
        except Exception as e:
            print(f"  错误: {e}\n")
            continue
    
    if not data_dict:
        print("错误: 没有成功提取任何数据")
        return
    
    print(f"成功处理 {len(data_dict)} 个算法的数据\n")
    print("=" * 100)
    print("生成图表和报告...")
    print(f"图片将保存在: {FIGURES_DIR}")
    print("=" * 100 + "\n")
    
    # 生成图表
    # 1. 基础时间序列图（单张）
    plot_multi_metrics(data_dict, FIGURES_DIR, args.smooth, args.window)
    
    # 2. 帧率对比图（单张）
    plot_fps_comparison(data_dict, FIGURES_DIR, args.smooth, args.window)
    
    # 3. 分辨率时间线图（单张）
    plot_resolution_timeline(data_dict, FIGURES_DIR, args.smooth, args.window)
    
    # 4. QoE指标图（单张）
    plot_qoe_metrics(data_dict, FIGURES_DIR)
    
    # 5. 自定义 QoE 对比图（单张）
    plot_custom_qoe(data_dict, FIGURES_DIR)
    
    # 6. 生成报告
    report_path = os.path.join(RESULT_DIR, "video_quality_report.txt")
    generate_report(data_dict, report_path, args.warmup)
    
    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)
    print(f"\n图片保存在: {FIGURES_DIR}/")
    print("  - ts_*.pdf             # 时间序列图 (Bitrate, Delay, Loss, etc.)")
    print("  - fps_compare_*.pdf    # 帧率对比图")
    print("  - resolution_*.pdf     # 分辨率变化图")
    print("  - bar_*.pdf            # 柱状统计图")
    print("  - qoe_score_*.pdf      # QoE 评分图")
    print()


if __name__ == '__main__':
    main()
