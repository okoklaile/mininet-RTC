#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 receiver.log 文件中的视频流统计数据（video_receive_stream2.cc:800）

功能:
- 提取 VideoReceiveStream stats 数据
- 从网络层提取丢包率数据
- 分析视频质量指标（比特率、帧率、延迟、卡顿、丢包等）
- 生成多维度对比图表
- 生成视频质量评估报告

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
        self.parse_video_stats()
    
    def parse_video_stats(self):
        """从日志中提取 VideoReceiveStream stats 数据"""
        if not os.path.exists(self.log_path):
            raise ValueError(f"日志文件不存在: {self.log_path}")
        
        # 正则表达式匹配 VideoReceiveStream stats 行
        pattern = re.compile(
            r'VideoReceiveStream stats: (\d+), \{(.+?)\}(?:, interframe_delay_max_ms: (\d+)\})?'
        )
        
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'VideoReceiveStream stats:' not in line:
                    continue
                
                try:
                    # 提取时间戳和统计数据
                    match = re.search(r'VideoReceiveStream stats: (\d+), \{(.+)\}', line)
                    if not match:
                        continue
                    
                    timestamp = int(match.group(1))
                    stats_str = match.group(2)
                    
                    # 解析键值对
                    stats = {'timestamp': timestamp}
                    
                    # 使用正则表达式提取所有键值对
                    kv_pattern = re.compile(r'(\w+):\s*(-?\d+(?:\.\d+)?)')
                    for kv_match in kv_pattern.finditer(stats_str):
                        key = kv_match.group(1)
                        value_str = kv_match.group(2)
                        
                        # 尝试转换为数值
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
                    continue
        
        print(f"  找到 {len(self.video_stats)} 条视频统计记录")


class NetworkStatsParser:
    """解析网络层统计日志，用于提取丢包率"""
    
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
    def calculate_metrics(video_stats, packet_loss_rate=None):
        """
        计算各种视频质量指标
        
        参数:
        - video_stats: 视频统计数据列表
        - packet_loss_rate: 网络层丢包率（百分比）
        
        返回:
        - 时间序列数据字典
        - 聚合统计指标
        """
        if not video_stats:
            return {}, {}
        
        # 时间序列数据
        timestamps = [s['timestamp'] for s in video_stats]
        base_time = timestamps[0]
        rel_times = [(t - base_time) / 1000.0 for t in timestamps]  # 转换为秒
        
        time_series = {
            'time': rel_times,
            'total_bps': [s.get('total_bps', 0) / 1e6 for s in video_stats],  # Mbps
            'render_fps': [s.get('render_fps', 0) for s in video_stats],
            'network_fps': [s.get('network_fps', 0) for s in video_stats],
            'decode_fps': [s.get('decode_fps', 0) for s in video_stats],
            'cur_delay_ms': [s.get('cur_delay_ms', 0) for s in video_stats],
            'jb_delay_ms': [s.get('jb_delay_ms', 0) for s in video_stats],
            'frames_dropped': [s.get('frames_dropped', 0) for s in video_stats],
            'freeze_cnt': [s.get('freeze_cnt', 0) for s in video_stats],
            'freeze_dur_ms': [s.get('freeze_dur_ms', 0) for s in video_stats],
            'cum_loss': [s.get('cum_loss', 0) for s in video_stats],
            'nack': [s.get('nack', 0) for s in video_stats],
            'decode_ms': [s.get('decode_ms', 0) for s in video_stats],
        }
        
        # 聚合统计
        total_packet_loss = time_series['cum_loss'][-1] if time_series['cum_loss'] else 0
        
        # 计算总播放时长（秒）
        total_duration_s = rel_times[-1] if rel_times else 0
        total_freeze_duration_ms = time_series['freeze_dur_ms'][-1] if time_series['freeze_dur_ms'] else 0
        freeze_rate = (total_freeze_duration_ms / 1000.0 / total_duration_s * 100) if total_duration_s > 0 else 0
        
        # 延迟统计增强
        delay_list = time_series['cur_delay_ms']
        min_delay = min(delay_list) if delay_list else 0
        max_delay = max(delay_list) if delay_list else 0
        p95_delay = np.percentile(delay_list, 95) if delay_list else 0
        p99_delay = np.percentile(delay_list, 99) if delay_list else 0
        
        # 自致延迟 = 当前延迟 - 最小延迟（反映拥塞造成的额外延迟）
        self_inflicted_delays = [d - min_delay for d in delay_list] if delay_list else []
        avg_self_inflicted = np.mean(self_inflicted_delays) if self_inflicted_delays else 0
        
        aggregated = {
            'avg_bitrate': np.mean(time_series['total_bps']),
            'avg_render_fps': np.mean(time_series['render_fps']),
            'avg_network_fps': np.mean(time_series['network_fps']),
            'avg_delay': np.mean(time_series['cur_delay_ms']),
            'min_delay': min_delay,
            'max_delay': max_delay,
            'p95_delay': p95_delay,
            'p99_delay': p99_delay,
            'avg_self_inflicted_delay': avg_self_inflicted,
            'avg_jb_delay': np.mean(time_series['jb_delay_ms']),
            'total_freeze_count': time_series['freeze_cnt'][-1] if time_series['freeze_cnt'] else 0,
            'total_freeze_duration': total_freeze_duration_ms,
            'freeze_rate': freeze_rate,
            'total_frames_dropped': time_series['frames_dropped'][-1] if time_series['frames_dropped'] else 0,
            'total_packet_loss': total_packet_loss,
            'packet_loss_rate': packet_loss_rate if packet_loss_rate is not None else 0.0,
            'total_nack': time_series['nack'][-1] if time_series['nack'] else 0,
            'avg_decode_time': np.mean(time_series['decode_ms']),
            'resolution': f"{video_stats[0].get('width', 0)}x{video_stats[0].get('height', 0)}",
            'total_duration': total_duration_s,
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
    绘制多指标对比图（2x3子图）
    
    参数:
    - data_dict: {算法名: (time_series, aggregated)}
    - output_path: 输出路径
    - smooth: 是否平滑
    - smooth_window: 平滑窗口
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Video Quality Multi-Metric Comparison', fontsize=20, fontweight='bold', y=0.995)
    
    # 定义要绘制的指标
    metrics_config = [
        ('total_bps', 'Video Bitrate', 'Bitrate (Mbps)'),
        ('render_fps', 'Render Frame Rate', 'Render FPS'),
        ('cur_delay_ms', 'End-to-End Delay', 'End-to-End Delay (ms)'),
        ('jb_delay_ms', 'Jitter Buffer Delay', 'Jitter Buffer Delay (ms)'),
        ('freeze_cnt', 'Freeze Count', 'Freeze Count'),
        ('cum_loss', 'Cumulative Packet Loss', 'Cumulative Packet Loss'),
    ]
    
    for idx, (metric_key, title_cn, ylabel) in enumerate(metrics_config):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for algo_name, (time_series, _) in data_dict.items():
            time = time_series['time']
            data = time_series[metric_key]
            
            if smooth and len(data) > 0:
                data = smooth_data(data, smooth_window)
            
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


def plot_qoe_metrics(data_dict, output_path):
    """绘制QoE（用户体验质量）指标柱状图"""
    algos = list(data_dict.keys())
    
    # 提取QoE相关指标
    freeze_counts = [data_dict[a][1]['total_freeze_count'] for a in algos]
    freeze_durations = [data_dict[a][1]['total_freeze_duration'] / 1000.0 for a in algos]  # 转换为秒
    freeze_rates = [data_dict[a][1]['freeze_rate'] for a in algos]  # 卡顿率
    frames_dropped = [data_dict[a][1]['total_frames_dropped'] for a in algos]
    avg_delays = [data_dict[a][1]['avg_delay'] for a in algos]
    packet_loss_rates = [data_dict[a][1]['packet_loss_rate'] for a in algos]  # 丢包率
    avg_render_fps = [data_dict[a][1]['avg_render_fps'] for a in algos]  # 平均渲染帧率
    avg_bitrates = [data_dict[a][1]['avg_bitrate'] for a in algos]  # 平均比特率
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Quality of Experience (QoE) Metrics Comparison', fontsize=20, fontweight='bold')
    
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
    
    # 4. 丢帧数
    axes[1, 0].bar(algos, frames_dropped, color='skyblue', alpha=0.8)
    axes[1, 0].set_ylabel('Frames Dropped', fontsize=12)
    axes[1, 0].set_title('Dropped Frames Count', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 5. 平均延迟
    axes[1, 1].bar(algos, avg_delays, color='lightgreen', alpha=0.8)
    axes[1, 1].set_ylabel('Average Delay (ms)', fontsize=12)
    axes[1, 1].set_title('Average End-to-End Delay', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 6. 丢包率
    axes[1, 2].bar(algos, packet_loss_rates, color='lightyellow', alpha=0.8, edgecolor='orange')
    axes[1, 2].set_ylabel('Packet Loss Rate (%)', fontsize=12)
    axes[1, 2].set_title('Packet Loss Rate', fontsize=14, fontweight='bold')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    # 7. 平均渲染帧率
    axes[2, 0].bar(algos, avg_render_fps, color='plum', alpha=0.8)
    axes[2, 0].set_ylabel('Average FPS', fontsize=12)
    axes[2, 0].set_title('Average Render Frame Rate', fontsize=14, fontweight='bold')
    axes[2, 0].grid(axis='y', alpha=0.3)
    
    # 8. 平均比特率
    axes[2, 1].bar(algos, avg_bitrates, color='lightcyan', alpha=0.8, edgecolor='teal')
    axes[2, 1].set_ylabel('Bitrate (Mbps)', fontsize=12)
    axes[2, 1].set_title('Average Video Bitrate', fontsize=14, fontweight='bold')
    axes[2, 1].grid(axis='y', alpha=0.3)
    
    # 9. 隐藏第三行第三个子图（保持布局对称）
    axes[2, 2].axis('off')
    
    for ax in axes.flat:
        if ax.get_visible():
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ QoE指标图已保存: {output_path}")
    plt.close()


# ============================================
# 统计报告
# ============================================

def generate_report(data_dict, output_path):
    """生成视频质量统计报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("视频流质量统计报告 (VideoReceiveStream Stats Analysis)\n")
        f.write("=" * 100 + "\n\n")
        
        # 按平均比特率排序
        sorted_algos = sorted(data_dict.items(), 
                            key=lambda x: x[1][1]['avg_bitrate'], 
                            reverse=True)
        
        for i, (algo_name, (time_series, aggregated)) in enumerate(sorted_algos, 1):
            f.write(f"{i}. {algo_name}\n")
            f.write("-" * 90 + "\n")
            
            # 基本信息
            f.write(f"  视频分辨率:              {aggregated['resolution']}\n")
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
            f.write("  【延迟指标】\n")
            f.write(f"    平均端到端延迟:        {aggregated['avg_delay']:.2f} ms\n")
            f.write(f"    最小端到端延迟:        {aggregated['min_delay']:.2f} ms\n")
            f.write(f"    最大端到端延迟:        {aggregated['max_delay']:.2f} ms\n")
            f.write(f"    95分位端到端延迟:      {aggregated['p95_delay']:.2f} ms\n")
            f.write(f"    99分位端到端延迟:      {aggregated['p99_delay']:.2f} ms\n")
            f.write(f"    平均自致延迟:          {aggregated['avg_self_inflicted_delay']:.2f} ms (相对于最小延迟)\n")
            f.write(f"    平均抖动缓冲延迟:      {aggregated['avg_jb_delay']:.2f} ms\n")
            f.write(f"    平均解码耗时:          {aggregated['avg_decode_time']:.2f} ms\n")
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
            f.write(f"    累计丢包数(视频层):    {aggregated['total_packet_loss']}\n")
            f.write(f"    丢包率(网络层):        {aggregated['packet_loss_rate']:.2f} %\n")
            f.write(f"    NACK请求次数:          {aggregated['total_nack']}\n")
            f.write("\n")
            
            # QoE评分（简单评分系统）
            qoe_score = calculate_qoe_score(aggregated)
            f.write(f"  【综合QoE评分】:         {qoe_score:.1f} / 100\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("\n数据来源说明:\n")
        f.write("  - 视频质量指标: video_receive_stream2.cc 视频层统计日志\n")
        f.write("  - 延迟指标: video_receive_stream2.cc 的 cur_delay_ms (真实端到端延迟)\n")
        f.write("  - 丢包率: remote_estimator_proxy.cc 网络层日志（通过序列号差值计算）\n")
        f.write("  - 累计丢包数: 视频层统计的 cum_loss 字段（可能与网络层不同）\n")
        f.write("\n评分说明:\n")
        f.write("  - QoE评分综合考虑: 帧率稳定性、延迟、卡顿次数/时长/率、丢包率等因素\n")
        f.write("  - 评分维度: 帧率(30分) + 延迟(25分) + 卡顿(25分) + 丢包(20分)\n")
        f.write("  - 评分范围: 0-100，分数越高表示用户体验越好\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ 统计报告已保存: {output_path}")


def calculate_qoe_score(aggregated):
    """
    计算简单的QoE评分 (0-100)
    
    评分维度:
    - 帧率 (30分): 接近30fps得分高
    - 延迟 (25分): 延迟越低得分越高
    - 卡顿 (25分): 无卡顿得满分
    - 丢包 (20分): 无丢包得满分
    """
    score = 0.0
    
    # 1. 帧率评分 (满分30)
    fps = aggregated['avg_render_fps']
    if fps >= 30:
        score += 30
    elif fps >= 24:
        score += 25
    elif fps >= 20:
        score += 20
    elif fps >= 15:
        score += 15
    else:
        score += max(0, fps / 30 * 30)
    
    # 2. 延迟评分 (满分25)
    delay = aggregated['avg_delay']
    if delay <= 50:
        score += 25
    elif delay <= 100:
        score += 20
    elif delay <= 150:
        score += 15
    elif delay <= 200:
        score += 10
    elif delay <= 300:
        score += 8
    else:
        score += max(0, 25 - (delay - 300) / 30)
    
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
    
    return min(100, max(0, score))


# ============================================
# 主程序
# ============================================

def main():
    parser = argparse.ArgumentParser(description='分析 VideoReceiveStream 统计数据')
    parser.add_argument('--smooth', action='store_true', help='启用平滑处理')
    parser.add_argument('--window', type=int, default=5, help='平滑窗口大小')
    args = parser.parse_args()
    
    print("=" * 100)
    print("视频流质量分析工具 (VideoReceiveStream Stats Analyzer)")
    if args.smooth:
        print(f"平滑处理: 已启用 (窗口大小={args.window})")
    else:
        print("平滑处理: 未启用")
    print("=" * 100 + "\n")
    
    # 确保结果目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 查找所有日志文件
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
            
            # 解析网络层数据以获取丢包率
            net_parser = NetworkStatsParser(log_file)
            
            # 计算指标（传入网络层丢包率）
            time_series, aggregated = VideoMetrics.calculate_metrics(
                parser.video_stats, 
                packet_loss_rate=net_parser.packet_loss_rate
            )
            data_dict[algo_name] = (time_series, aggregated)
            
            print(f"  平均比特率: {aggregated['avg_bitrate']:.3f} Mbps")
            print(f"  平均帧率: {aggregated['avg_render_fps']:.1f} FPS")
            print(f"  平均延迟: {aggregated['avg_delay']:.2f} ms")
            print(f"  卡顿次数: {aggregated['total_freeze_count']}")
            print(f"  卡顿率: {aggregated['freeze_rate']:.2f} %")
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
    print("=" * 100 + "\n")
    
    # 生成图表
    suffix = f"_smooth_w{args.window}" if args.smooth else ""
    
    # 1. 多指标对比图
    multi_metrics_path = os.path.join(RESULT_DIR, f"video_multi_metrics{suffix}.pdf")
    plot_multi_metrics(data_dict, multi_metrics_path, args.smooth, args.window)
    
    # 2. 帧率对比图
    fps_path = os.path.join(RESULT_DIR, f"video_fps_comparison{suffix}.pdf")
    plot_fps_comparison(data_dict, fps_path, args.smooth, args.window)
    
    # 3. QoE指标图
    qoe_path = os.path.join(RESULT_DIR, "video_qoe_metrics.pdf")
    plot_qoe_metrics(data_dict, qoe_path)
    
    # 4. 生成报告
    report_path = os.path.join(RESULT_DIR, "video_quality_report.txt")
    generate_report(data_dict, report_path)
    
    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)
    print(f"\n结果保存在: {RESULT_DIR}/")
    print("  - video_quality_report.txt          # 视频质量统计报告")
    print(f"  - video_multi_metrics{suffix}.pdf       # 多指标对比图")
    print(f"  - video_fps_comparison{suffix}.pdf      # 帧率对比图")
    print("  - video_qoe_metrics.pdf             # QoE指标图")
    print("\n使用提示:")
    print("  python3 eval/eval_video_stats.py                    # 显示原始数据")
    print("  python3 eval/eval_video_stats.py --smooth           # 使用平滑处理")
    print("  python3 eval/eval_video_stats.py --smooth --window 10  # 自定义窗口")
    print()


if __name__ == '__main__':
    main()
