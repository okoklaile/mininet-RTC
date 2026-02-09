#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 receiver.log 文件中的视频流统计数据（video_receive_stream2.cc:800）

功能:
- 提取 VideoReceiveStream stats 数据
- 从视频层的 cum_loss 和 packets_received 计算丢包率
- 分析视频质量指标（比特率、帧率、延迟、卡顿、丢包等）
- 生成多维度对比图表
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
    def remove_outliers(data, multiplier=10):
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
# 绘图函数
# ============================================

def plot_multi_metrics(data_dict, output_path, smooth=False, smooth_window=5):
    """
    绘制多指标对比图（3x3子图）
    
    参数:
    - data_dict: {算法名: (time_series, aggregated)}
    - output_path: 输出路径
    - smooth: 是否平滑
    - smooth_window: 平滑窗口
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Video Quality Multi-Metric Comparison', fontsize=20, fontweight='bold', y=0.995)
    
    # 定义要绘制的指标
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
    
    for idx, (metric_key, title_cn, ylabel) in enumerate(metrics_config):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for algo_name, (time_series, _) in data_dict.items():
            # 特殊处理 BWE，因为它有自己的时间轴
            if metric_key == 'bwe_mbps':
                time = time_series.get('bwe_time', [])
                data = time_series.get('bwe_mbps', [])
            else:
                time = time_series['time']
                data = time_series.get(metric_key, [])
            
            if len(data) == 0:
                continue
                
            if smooth and len(data) > smooth_window:
                # 确保数据长度足够进行平滑
                try:
                    data = smooth_data(data, smooth_window)
                except:
                    pass
            
            ax.plot(time, data, label=algo_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title_cn, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ 多指标对比图已保存: {output_path}")
    plt.close()


def plot_fps_comparison(data_dict, output_path, smooth=False, smooth_window=5):
    """绘制帧率对比图（网络FPS vs 渲染FPS）"""
    fig, axes = plt.subplots(1, len(data_dict), figsize=(6*len(data_dict), 5))
    
    if len(data_dict) == 1:
        axes = [axes]
    
    for idx, (algo_name, (time_series, _)) in enumerate(data_dict.items()):
        ax = axes[idx]
        time = time_series['time']
        
        network_fps = time_series['network_fps']
        render_fps = time_series['render_fps']
        decode_fps = time_series['decode_fps']
        
        if smooth:
            network_fps = smooth_data(network_fps, smooth_window)
            render_fps = smooth_data(render_fps, smooth_window)
            decode_fps = smooth_data(decode_fps, smooth_window)
        
        ax.plot(time, network_fps, label='Network FPS', linewidth=2, alpha=0.8)
        ax.plot(time, render_fps, label='Render FPS', linewidth=2, alpha=0.8)
        ax.plot(time, decode_fps, label='Decode FPS', linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frame Rate (FPS)', fontsize=12)
        ax.set_title(f'{algo_name} - Frame Rate Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ 帧率对比图已保存: {output_path}")
    plt.close()


def plot_resolution_timeline(data_dict, output_path, smooth=False, smooth_window=5):
    """绘制分辨率随时间变化的图表"""
    fig, axes = plt.subplots(1, len(data_dict), figsize=(6*len(data_dict), 5))
    
    if len(data_dict) == 1:
        axes = [axes]
    
    for idx, (algo_name, (time_series, aggregated)) in enumerate(data_dict.items()):
        ax = axes[idx]
        time = time_series['time']
        
        widths = time_series['width']
        heights = time_series['height']
        
        # 计算像素总数（百万像素）
        pixels = [w * h / 1e6 for w, h in zip(widths, heights)]
        
        if smooth and len(pixels) > smooth_window:
            pixels = smooth_data(pixels, smooth_window)
        
        ax.plot(time, pixels, label='Resolution (MPixels)', linewidth=2, alpha=0.8, color='purple')
        
        # 标注分辨率变化点
        resolution_changes = aggregated['resolution_changes']
        if resolution_changes > 0:
            for i in range(1, len(widths)):
                if widths[i] != widths[i-1] or heights[i] != heights[i-1]:
                    ax.axvline(x=time[i], color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax.text(time[i], max(pixels) * 0.9, f'{int(widths[i])}x{int(heights[i])}', 
                           rotation=90, va='top', fontsize=8, color='red')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Resolution (MPixels)', fontsize=12)
        ax.set_title(f'{algo_name} - Video Resolution\n(Changes: {resolution_changes})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 在图表上显示平均分辨率
        avg_res = aggregated['resolution']
        ax.text(0.02, 0.98, f'Avg: {avg_res}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ 分辨率时间线图已保存: {output_path}")
    plt.close()


def plot_qoe_metrics(data_dict, output_path):
    """绘制QoE（用户体验质量）指标柱状图"""
    algos = list(data_dict.keys())
    
    # 提取QoE相关指标
    freeze_counts = [data_dict[a][1]['total_freeze_count'] for a in algos]
    freeze_durations = [data_dict[a][1]['total_freeze_duration'] / 1000.0 for a in algos]  # 转换为秒
    freeze_rates = [data_dict[a][1]['freeze_rate'] for a in algos]  # 卡顿率
    packet_loss_rates = [data_dict[a][1]['packet_loss_rate'] for a in algos]  # 丢包率
    avg_render_fps = [data_dict[a][1]['avg_render_fps'] for a in algos]  # 平均渲染帧率
    avg_bitrates = [data_dict[a][1]['avg_bitrate'] for a in algos]  # 平均比特率
    p95_e2e_delays = [data_dict[a][1]['p95_e2e_delay'] if data_dict[a][1]['p95_e2e_delay'] > 0 else 0 for a in algos]  # P95端到端延迟
    p95_network_delays = [data_dict[a][1]['p95_network_delay'] if data_dict[a][1]['p95_network_delay'] > 0 else 0 for a in algos]  # P95网络延迟
    avg_pixels = [data_dict[a][1]['avg_pixels'] / 1e6 for a in algos]  # 平均像素数（百万）
    resolution_changes = [data_dict[a][1]['resolution_changes'] for a in algos]  # 分辨率变化次数
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Quality of Experience (QoE) Metrics Comparison (Delay: P95)', fontsize=20, fontweight='bold')
    
    # 第一行：卡顿相关指标
    # 1. 卡顿次数
    axes[0, 0].bar(algos, freeze_counts, color='coral', alpha=0.8)
    axes[0, 0].set_ylabel('Freeze Count', fontsize=12)
    axes[0, 0].set_title('Video Freeze Count', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 卡顿总时长
    axes[0, 1].bar(algos, freeze_durations, color='lightcoral', alpha=0.8)
    axes[0, 1].set_ylabel('Freeze Duration (s)', fontsize=12)
    axes[0, 1].set_title('Total Freeze Duration', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 卡顿率
    axes[0, 2].bar(algos, freeze_rates, color='salmon', alpha=0.8)
    axes[0, 2].set_ylabel('Freeze Rate (%)', fontsize=12)
    axes[0, 2].set_title('Freeze Rate (Freeze Time / Total Time)', fontsize=14, fontweight='bold')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # 第二行：丢包、帧率、比特率
    # 4. 丢包率
    axes[1, 0].bar(algos, packet_loss_rates, color='lightyellow', alpha=0.8, edgecolor='orange')
    axes[1, 0].set_ylabel('Packet Loss Rate (%)', fontsize=12)
    axes[1, 0].set_title('Packet Loss Rate', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 5. 平均渲染帧率
    axes[1, 1].bar(algos, avg_render_fps, color='plum', alpha=0.8)
    axes[1, 1].set_ylabel('Average FPS', fontsize=12)
    axes[1, 1].set_title('Average Render Frame Rate', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 6. 平均比特率
    axes[1, 2].bar(algos, avg_bitrates, color='lightcyan', alpha=0.8, edgecolor='teal')
    axes[1, 2].set_ylabel('Bitrate (Mbps)', fontsize=12)
    axes[1, 2].set_title('Average Video Bitrate', fontsize=14, fontweight='bold')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    # 第三行：两个延迟指标
    # 7. P95端到端延迟
    axes[2, 0].bar(algos, p95_e2e_delays, color='lavender', alpha=0.8, edgecolor='purple')
    axes[2, 0].set_ylabel('E2E Delay P95 (ms)', fontsize=12)
    axes[2, 0].set_title('P95 End-to-End Delay', fontsize=14, fontweight='bold')
    axes[2, 0].grid(axis='y', alpha=0.3)
    
    # 8. P95网络延迟
    axes[2, 1].bar(algos, p95_network_delays, color='peachpuff', alpha=0.8, edgecolor='darkorange')
    axes[2, 1].set_ylabel('Network Delay P95 (ms)', fontsize=12)
    axes[2, 1].set_title('P95 Network Delay', fontsize=14, fontweight='bold')
    axes[2, 1].grid(axis='y', alpha=0.3)
    
    # 9. 平均分辨率（百万像素）
    colors_res = ['lightblue' if rc == 0 else 'lightcoral' for rc in resolution_changes]
    bars = axes[2, 2].bar(algos, avg_pixels, color=colors_res, alpha=0.8, edgecolor='navy')
    axes[2, 2].set_ylabel('Resolution (MPixels)', fontsize=12)
    axes[2, 2].set_title('Average Video Resolution', fontsize=14, fontweight='bold')
    axes[2, 2].grid(axis='y', alpha=0.3)
    
    # 在柱状图上标注分辨率变化次数
    for i, (bar, changes) in enumerate(zip(bars, resolution_changes)):
        if changes > 0:
            height = bar.get_height()
            axes[2, 2].text(bar.get_x() + bar.get_width()/2., height,
                          f'Δ{changes}',
                          ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    
    for ax in axes.flat:
        if ax.get_visible():
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ QoE指标图已保存: {output_path}")
    plt.close()


def plot_custom_qoe(data_dict, output_path):
    """
    根据用户提供的公式绘制自定义 QoE 指标对比图
    
    公式:
    - QoE_recv = 100 * U (U = bitrate / capacity)
    - QoE_delay = 100 * (d_max - d_95th) / (d_max - d_min)
    - QoE_loss = 100 * (1 - L)
    - QoE_total = 0.2 * QoE_recv + 0.2 * QoE_delay + 0.3 * QoE_loss
    """
    algos = list(data_dict.keys())
    
    qoe_recv = [data_dict[a][1]['qoe_recv_rate'] for a in algos]
    qoe_delay = [data_dict[a][1]['qoe_delay'] for a in algos]
    qoe_loss = [data_dict[a][1]['qoe_loss'] for a in algos]
    qoe_total = [data_dict[a][1]['qoe_total'] for a in algos]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Custom QoE Metrics Comparison (Based on User Formulas)', fontsize=18, fontweight='bold')
    
    metrics = [
        (qoe_recv, 'QoE Recv Rate (100 * U)', 'Score', 'skyblue'),
        (qoe_delay, 'QoE Delay (100 * (d_max - d_95) / (d_max - d_min))', 'Score', 'lightgreen'),
        (qoe_loss, 'QoE Loss (100 * (1 - L))', 'Score', 'salmon'),
        (qoe_total, 'Total QoE (0.2*R + 0.2*D + 0.3*L)', 'Score', 'gold')
    ]
    
    for idx, (data, title, ylabel, color) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        bars = ax.bar(algos, data, color=color, alpha=0.8, edgecolor='black')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(data) * 1.1, 100) if data else 110)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ 自定义 QoE 对比图已保存: {output_path}")
    plt.close()


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
        f.write("\n数据来源说明:\n")
        f.write("  - 所有指标均来自: video_receive_stream2.cc 视频层统计日志\n")
        f.write("\n延迟指标说明:\n")
        f.write("  - P95处理延迟: 接收端处理延迟的 95 分位值 (已剔除 3.0*IQR 以外的极端异常点)\n")
        f.write("  - P95端到端延迟: 发送端到接收端的 95 分位完整延迟 (已剔除极端异常点)\n")
        f.write("  - P95网络延迟: 网络传输的 95 分位延迟 (已剔除极端异常点)\n")
        f.write("  - 注意: 所有的延迟指标现在均采用 95 分位值（P95），并预先通过 IQR 方法过滤了因系统卡顿或初始连接产生的极少数尖峰数据。\n")
        f.write("\n丢包率计算:\n")
        f.write("  - 公式: cum_loss / (cum_loss + packets_received) × 100%\n")
        f.write("  - cum_loss: 累计丢失的包数（视频层统计）\n")
        f.write("  - packets_received: 累计接收的包数（视频层统计）\n")
        f.write("\n评分说明:\n")
        f.write("  - QoE评分综合考虑: 帧率稳定性、延迟、卡顿次数/时长/率、丢包率、分辨率等因素\n")
        f.write("  - 评分维度: 帧率(25分) + 延迟(20分) + 卡顿(25分) + 丢包(20分) + 分辨率(10分)\n")
        f.write("  - 分辨率评分: 基于平均像素数(8分) + 分辨率稳定性(2分)\n")
        f.write("    * 720p及以上: 8分, 480p(VGA): 6分, 360p: 4分\n")
        f.write("    * 无分辨率变化: +2分, ≤2次变化: +1分\n")
        f.write("  - 评分范围: 0-100，分数越高表示用户体验越好\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ 统计报告已保存: {output_path}")


def calculate_qoe_score(aggregated):
    """
    计算简单的QoE评分 (0-100)
    
    评分维度:
    - 帧率 (25分): 接近30fps得分高
    - 延迟 (20分): 延迟越低得分越高
    - 卡顿 (25分): 无卡顿得满分
    - 丢包 (20分): 无丢包得满分
    - 分辨率 (10分): 分辨率越高得分越高，稳定性加分
    """
    score = 0.0
    
    # 1. 帧率评分 (满分25)
    fps = aggregated['avg_render_fps']
    if fps >= 30:
        score += 25
    elif fps >= 24:
        score += 22
    elif fps >= 20:
        score += 18
    elif fps >= 15:
        score += 13
    else:
        score += max(0, fps / 30 * 25)
    
    # 2. 延迟评分 (满分20) - 基于 P95 延迟
    delay = aggregated['p95_delay']
    if delay <= 50:
        score += 20
    elif delay <= 100:
        score += 16
    elif delay <= 150:
        score += 12
    elif delay <= 200:
        score += 8
    elif delay <= 300:
        score += 6
    else:
        score += max(0, 20 - (delay - 300) / 30)
    
    # 3. 卡顿评分 (满分25)
    freeze_count = aggregated['total_freeze_count']
    freeze_duration = aggregated['total_freeze_duration']
    freeze_rate = aggregated['freeze_rate']
    
    if freeze_count == 0:
        score += 25
    elif freeze_count <= 2:
        score += 20
    elif freeze_count <= 5:
        score += 15
    elif freeze_count <= 10:
        score += 10
    else:
        score += max(0, 25 - freeze_count * 2)
    
    # 考虑卡顿时长和卡顿率
    if freeze_duration > 5000:  # 超过5秒
        score -= 8
    if freeze_rate > 10:  # 卡顿率超过10%
        score -= 6
    elif freeze_rate > 5:  # 卡顿率超过5%
        score -= 3
    
    # 4. 丢包评分 (满分20) - 基于丢包率
    packet_loss_rate = aggregated['packet_loss_rate']
    if packet_loss_rate == 0:
        score += 20
    elif packet_loss_rate < 0.5:
        score += 18
    elif packet_loss_rate < 1.0:
        score += 15
    elif packet_loss_rate < 2.0:
        score += 12
    elif packet_loss_rate < 5.0:
        score += 8
    else:
        score += max(0, 20 - packet_loss_rate)
    
    # 5. 分辨率评分 (满分10)
    avg_pixels = aggregated['avg_pixels']
    resolution_changes = aggregated['resolution_changes']
    
    # 根据像素数评分 (8分)
    # 720p (1280x720 = 0.92M), 480p (640x480 = 0.31M), 360p (640x360 = 0.23M)
    if avg_pixels >= 0.9:  # >= 720p
        score += 8
    elif avg_pixels >= 0.6:  # 接近 720p
        score += 7
    elif avg_pixels >= 0.3:  # 480p (VGA)
        score += 6
    elif avg_pixels >= 0.2:  # 360p
        score += 4
    else:  # < 360p
        score += max(0, avg_pixels / 0.3 * 6)
    
    # 分辨率稳定性评分 (2分)
    if resolution_changes == 0:
        score += 2  # 分辨率完全稳定
    elif resolution_changes <= 2:
        score += 1  # 轻微变化
    # 否则不加分
    
    return min(100, max(0, score))


# ============================================
# 主程序
# ============================================

def main():
    parser = argparse.ArgumentParser(description='分析 VideoReceiveStream 统计数据')
    # --- 新增参数 ---
    parser.add_argument('--input', type=str, help='指定要分析的单个日志文件路径')
    # ----------------
    parser.add_argument('--smooth', action='store_true', help='启用平滑处理')
    parser.add_argument('--window', type=int, default=5, help='平滑窗口大小')
    parser.add_argument('--warmup', type=float, default=0, help='预热时间（秒），剔除开头的数据')
    args = parser.parse_args()

    # ... 省略中间的 print ...

    # --- 修改查找逻辑 ---
    if args.input:
        if os.path.exists(args.input):
            log_files = [args.input]
        else:
            print(f"错误: 找不到文件 {args.input}")
            return
    else:
        # 保持原有的逻辑：查找 OUTPUT_DIR 下的所有文件
        log_files = glob.glob(os.path.join(OUTPUT_DIR, '*_receiver.log'))
    
    if not log_files:
        print(f"错误: 在 {OUTPUT_DIR} 中没有找到日志文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件\n")
    
    # 解析所有日志
    data_dict = {}
    
    for log_file in sorted(log_files):
        algo_name = os.path.basename(log_file).replace('_receiver.log', '')
        print(f"处理: {algo_name}...")
        
        try:
            # 解析视频统计数据
            parser = VideoStatsParser(log_file)
            
            if not parser.video_stats:
                print(f"  警告: 未找到视频统计数据\n")
                continue
            
            # 计算指标（丢包率从视频层的 cum_loss 和 packets_received 计算）
            time_series, aggregated = VideoMetrics.calculate_metrics(parser.video_stats, parser.bwe_stats, args.warmup)
            
            if not time_series:
                continue
                
            data_dict[algo_name] = (time_series, aggregated)
            
            print(f"  平均比特率: {aggregated['avg_bitrate']:.3f} Mbps")
            print(f"  平均帧率: {aggregated['avg_render_fps']:.1f} FPS")
            print(f"  平均分辨率: {aggregated['resolution']} (变化次数: {aggregated['resolution_changes']})")
            print(f"  处理延迟: Avg: {aggregated['avg_delay']:.2f} ms / P95: {aggregated['p95_delay']:.2f} ms")
            
            # 显示端到端延迟和网络延迟
            if aggregated['p95_e2e_delay'] > 0:
                print(f"  端到端延迟: Avg: {aggregated['avg_e2e_delay']:.2f} ms / P95: {aggregated['p95_e2e_delay']:.2f} ms")
                print(f"  网络延迟: Avg: {aggregated['avg_network_delay']:.2f} ms / P95: {aggregated['p95_network_delay']:.2f} ms")
            else:
                print(f"  端到端延迟/网络延迟: 无有效数据")
            
            print(f"  卡顿次数: {aggregated['total_freeze_count']}, 卡顿率: {aggregated['freeze_rate']:.2f} %")
            print(f"  丢包: {aggregated['total_packet_loss']} / {aggregated['total_packet_loss'] + aggregated['total_packets_received']}, 丢包率: {aggregated['packet_loss_rate']:.2f} %\n")
            
        except Exception as e:
            print(f"  错误: {e}\n")
            continue
    
    if not data_dict:
        print("错误: 没有成功提取任何数据")
        return
    
    print(f"成功处理 {len(data_dict)} 个算法的数据\n")
    print("=" * 100)
    print("生成图表和报告...")
    print("=" * 100 + "\n")
    
    # 生成图表
    suffix = f"_smooth_w{args.window}" if args.smooth else ""
    
    # 1. 多指标对比图
    multi_metrics_path = os.path.join(RESULT_DIR, f"video_multi_metrics{suffix}.pdf")
    plot_multi_metrics(data_dict, multi_metrics_path, args.smooth, args.window)
    
    # 2. 帧率对比图
    fps_path = os.path.join(RESULT_DIR, f"video_fps_comparison{suffix}.pdf")
    plot_fps_comparison(data_dict, fps_path, args.smooth, args.window)
    
    # 3. 分辨率时间线图
    resolution_path = os.path.join(RESULT_DIR, f"video_resolution_timeline{suffix}.pdf")
    plot_resolution_timeline(data_dict, resolution_path, args.smooth, args.window)
    
    # 4. QoE指标图
    qoe_path = os.path.join(RESULT_DIR, "video_qoe_metrics.pdf")
    plot_qoe_metrics(data_dict, qoe_path)
    
    # 5. 自定义 QoE 对比图
    custom_qoe_path = os.path.join(RESULT_DIR, "video_custom_qoe.pdf")
    plot_custom_qoe(data_dict, custom_qoe_path)
    
    # 6. 生成报告
    report_path = os.path.join(RESULT_DIR, "video_quality_report.txt")
    generate_report(data_dict, report_path, args.warmup)
    
    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)
    print(f"\n结果保存在: {RESULT_DIR}/")
    print("  - video_quality_report.txt               # 视频质量统计报告")
    print(f"  - video_multi_metrics{suffix}.pdf            # 多指标对比图")
    print(f"  - video_fps_comparison{suffix}.pdf           # 帧率对比图")
    print(f"  - video_resolution_timeline{suffix}.pdf      # 分辨率时间线图")
    print("  - video_qoe_metrics.pdf                  # QoE指标图（含分辨率）")
    print("  - video_custom_qoe.pdf                   # 自定义 QoE 对比图")
    print("\n使用提示:")
    print("  python3 eval/eval_video_stats.py                    # 显示原始数据")
    print("  python3 eval/eval_video_stats.py --smooth           # 使用平滑处理")
    print("  python3 eval/eval_video_stats.py --smooth --window 10  # 自定义窗口")
    print()


if __name__ == '__main__':
    main()
