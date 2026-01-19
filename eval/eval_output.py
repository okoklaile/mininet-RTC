#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 receiver.log 文件中的网络性能数据

功能:
- 从所有算法的 receiver.log 中提取网络数据
- 绘制吞吐量对比图
- 生成性能统计报告

使用方法:
    python3 eval/eval_output.py [--smooth] [--window WINDOW]
    
参数:
    --smooth         启用平滑处理
    --window WINDOW  平滑窗口大小（默认5）
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ============================================
# 配置
# ============================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # eval目录
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # 项目根目录
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')  # 日志文件位置
RESULT_DIR = os.path.join(ROOT_DIR, 'eval_results')  # 分析结果保存位置


# ============================================
# 数据解析
# ============================================

class NetInfo(object):
    """解析 WebRTC 网络日志文件"""
    def __init__(self, net_path):
        self.net_path = net_path
        self.net_data = None
        self.parse_net_log()

    def parse_net_log(self):
        """从日志文件中提取 JSON 格式的网络数据"""
        if not self.net_path or not os.path.exists(self.net_path):
            raise ValueError(f"日志文件不存在: {self.net_path}")

        ret = []
        with open(self.net_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                # 只处理包含 remote_estimator_proxy.cc 的行
                if "remote_estimator_proxy.cc" not in line:
                    continue
                try:
                    # 提取 JSON 部分
                    raw_json = line[line.index('{'):]
                    json_network = json.loads(raw_json)
                    ret.append(json_network)
                except (ValueError, json.JSONDecodeError):
                    # 无法解析的 JSON 行，跳过
                    pass
                except Exception as e:
                    print(f"  ⚠️ 解析 JSON 时出错: {e}")
                    
        self.net_data = ret


class NetEvalMethodExtension(object):
    """网络性能评估方法"""
    def __init__(self):
        self.eval_name = "extension"
    
    def eval(self, dst_net_info):
        """
        评估网络性能指标
        
        返回：
        - time_nbytes: 字典 {时间戳: 字节数}
        - mean_self_inflicted: 平均自致延迟（ms）- 排除基础延迟后的队列延迟
        - mean_delay_95: 95分位延迟（ms）- 包含基础网络传播延迟
        - sum_recv_rate: 总接收速率（Mbps）
        - loss_ratio: 丢包率
        
        注：
        - 自致延迟 = 当前延迟 - 最小延迟，反映拥塞造成的队列缓冲延迟
        - 95分位延迟 = 原始端到端延迟的95百分位数，反映整体延迟水平
        """
        net_data = dst_net_info.net_data
        
        # 处理空数据情况
        if not net_data or len(net_data) == 0:
            return ({}, 0.0, 0.0, 0.0, 0.0)
        
        ssrc_info = {}
        time_nbytes = {}  # 核心数据：时间戳 -> 字节数映射
        loss_count = 0
        last_seqNo = {}
        
        # 遍历所有网络数据包
        for item in net_data:
            packet_info = item["packetInfo"]
            
            # 适配新的日志结构（字段扁平化）
            ssrc = packet_info["ssrc"]
            sequence_number = packet_info["seqNum"]
            send_time_ms = packet_info["sendTimeMs"]
            arrival_time_ms = packet_info["arrivalTimeMs"]
            payload_size = packet_info["payloadSize"]
            
            # 计算延迟（如果日志中已包含 delayMs，也可以直接使用）
            tmp_delay = arrival_time_ms - send_time_ms
            timestamp = arrival_time_ms
            
            # 初始化该 SSRC 的统计信息
            if ssrc not in ssrc_info:
                ssrc_info[ssrc] = {
                    "time_delta": -tmp_delay,
                    "delay_list": [],
                    "received_nbytes": 0,
                    "start_recv_time": arrival_time_ms,
                    "avg_recv_rate": 0,
                }
            
            # 累计每个时间戳的字节数
            if timestamp not in time_nbytes:
                time_nbytes[timestamp] = payload_size
            else:
                time_nbytes[timestamp] += payload_size

            # 计算丢包数
            if ssrc in last_seqNo:
                loss_count += max(0, sequence_number - last_seqNo[ssrc] - 1)
            last_seqNo[ssrc] = sequence_number

            # 记录延迟和字节数
            ssrc_info[ssrc]["delay_list"].append(ssrc_info[ssrc]["time_delta"] + tmp_delay)
            ssrc_info[ssrc]["received_nbytes"] += payload_size
            if arrival_time_ms != ssrc_info[ssrc]["start_recv_time"]:
                ssrc_info[ssrc]["avg_recv_rate"] = ssrc_info[ssrc]["received_nbytes"] / \
                    (arrival_time_ms - ssrc_info[ssrc]["start_recv_time"])
        
        # 计算延迟指标
        all_self_inflicted_delays = []
        all_delay_percentile_95 = []
        for ssrc in ssrc_info:
            if len(ssrc_info[ssrc]["delay_list"]) > 0:
                min_delay = min(ssrc_info[ssrc]["delay_list"])
                self_inflicted_delay = [delay - min_delay for delay in ssrc_info[ssrc]["delay_list"]]
                all_self_inflicted_delays.extend(self_inflicted_delay)
                all_delay_percentile_95.append(np.percentile(ssrc_info[ssrc]["delay_list"], 95))
        
        # 计算接收速率（Mbps）
        recv_rate_list = [ssrc_info[ssrc]["avg_recv_rate"] * 8.0 / 1000.0 
                         for ssrc in ssrc_info if ssrc_info[ssrc]["avg_recv_rate"] > 0]

        # 防止空列表导致错误
        mean_self_inflicted = np.mean(all_self_inflicted_delays) if all_self_inflicted_delays else 0.0
        mean_delay_95 = np.mean(all_delay_percentile_95) if all_delay_percentile_95 else 0.0
        sum_recv_rate = np.sum(recv_rate_list) if recv_rate_list else 0.0
        
        # 防止除零错误
        total_packets = loss_count + len(net_data)
        loss_ratio = loss_count / total_packets if total_packets > 0 else 0.0
        
        return (time_nbytes, mean_self_inflicted, mean_delay_95, sum_recv_rate, loss_ratio)


# ============================================
# 平滑处理函数
# ============================================

def smooth_data(data, window_size=5, method='savgol'):
    """
    对数据进行平滑处理
    
    参数:
    - data: 原始数据列表
    - window_size: 平滑窗口大小
    - method: 平滑方法 ('savgol' 或 'moving_avg')
    
    返回:
    - 平滑后的数据
    """
    if len(data) < window_size:
        return data
    
    if method == 'savgol':
        # Savitzky-Golay 滤波器 (更好地保留峰值)
        # window_size 必须是奇数
        if window_size % 2 == 0:
            window_size += 1
        # polyorder 必须小于 window_size
        poly_order = min(3, window_size - 1)
        try:
            return savgol_filter(data, window_size, poly_order)
        except:
            # 如果失败，使用移动平均
            return smooth_data(data, window_size, method='moving_avg')
    
    elif method == 'moving_avg':
        # 移动平均
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        # 补齐开头部分
        prefix = data[:window_size//2]
        suffix = data[-(window_size//2):]
        return np.concatenate([prefix, smoothed, suffix])
    
    return data


# ============================================
# 绘图函数
# ============================================

def draw_goodput(time_nbytes_list, label_list, output_path, min_gap=500, duration=60, 
                 smooth=False, smooth_window=5):
    """
    绘制吞吐量（Goodput）随时间变化的曲线图
    
    参数：
    - time_nbytes_list: 列表，每个元素是一个 {时间戳: 字节数} 字典
    - label_list: 标签列表，与 time_nbytes_list 对应
    - output_path: 输出文件路径
    - min_gap: 统计窗口大小（毫秒）
    - duration: 绘图时长（秒）
    - smooth: 是否启用平滑处理
    - smooth_window: 平滑窗口大小
    """
    plt.figure(figsize=(12, 7))
    
    for idx, time_nbytes in enumerate(time_nbytes_list):
        timestamps = list(time_nbytes.keys())
        nbytes = list(time_nbytes.values())
        
        if len(timestamps) == 0:
            continue
        
        # 计算相对时间（从 0 开始）
        rel_stamps = [timestamps[i] - timestamps[0] for i in range(len(timestamps))]
        
        # 计算吞吐量（按 min_gap 窗口统计）
        goodput_list = []
        goodput_time = []
        prev_time = rel_stamps[0]
        goodput_gap = nbytes[0]
        
        for i in range(1, len(time_nbytes)):
            if rel_stamps[i] - prev_time < min_gap:
                # 累计字节数
                goodput_gap += nbytes[i]
            else:
                # 计算该窗口的吞吐量（Mbps）
                goodput_mbps = (goodput_gap * 8.0 / 1000.0) / (rel_stamps[i] - prev_time)
                goodput_list.append(goodput_mbps)
                goodput_time.append(rel_stamps[i])
                prev_time = rel_stamps[i]
                goodput_gap = nbytes[i]
        
        # 应用平滑处理
        if smooth and len(goodput_list) > 0:
            goodput_list_smooth = smooth_data(goodput_list, window_size=smooth_window)
            # 绘制平滑曲线
            plt.plot(goodput_time, goodput_list_smooth, label=label_list[idx], lw=2, alpha=0.8)
            # 可选：绘制原始数据的浅色背景
            # plt.plot(goodput_time, goodput_list, label=f'{label_list[idx]} (原始)', 
            #          lw=1, alpha=0.3, linestyle='--')
        else:
            # 绘制原始曲线
            plt.plot(goodput_time, goodput_list, label=label_list[idx], lw=2, alpha=0.8)
    
    # 设置 X 轴刻度（每 10 秒一个刻度）
    xticks = np.arange(0, duration * 1000 + 1, 10000)
    xtick_labels = (xticks / 1000).astype(int)
    plt.xticks(xticks, xtick_labels, fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.ylabel("Throughput (Mbps)", fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)
    
    # 标题根据是否平滑调整
    title = "Congestion Control Algorithm Throughput Comparison"
    if smooth:
        title += f" (Smoothed, window={smooth_window})"
    plt.title(title, fontsize=20, fontweight='bold')
    
    plt.legend(loc='best', ncol=min(3, len(label_list)), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}")
    plt.close()


# ============================================
# 统计报告
# ============================================

def save_statistics_report(stats_dict, output_path):
    """保存统计报告"""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("拥塞控制算法网络层性能统计报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("⚠️ 延迟指标说明（重要）:\n")
        f.write("  - 自致延迟: 排除基础传播延迟后的队列缓冲延迟 ✓ 准确（反映拥塞程度）\n")
        f.write("  - 95分位延迟: 相对于首包的延迟抖动95百分位数 ⚠️ 不是真实端到端延迟\n")
        f.write("  - 由于sendTimeMs和arrivalTimeMs时间基准不同，无法计算真实传输延迟\n")
        f.write("  - 真实端到端延迟请参考: eval_video_stats.py 生成的 video_quality_report.txt\n")
        f.write("\n")
        
        # 按接收速率排序
        sorted_algos = sorted(stats_dict.items(), 
                            key=lambda x: x[1]['recv_rate'], 
                            reverse=True)
        
        for i, (algo, stats) in enumerate(sorted_algos, 1):
            f.write(f"{i}. {algo}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  数据包数量:      {stats['packet_count']}\n")
            f.write(f"  接收速率:        {stats['recv_rate']:.3f} Mbps\n")
            f.write(f"  平均自致延迟:    {stats['avg_delay']:.2f} ms (队列延迟)\n")
            f.write(f"  95分位延迟:      {stats['p95_delay']:.2f} ms (端到端)\n")
            f.write(f"  丢包率:          {stats['loss_ratio']*100:.2f}%\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ 统计报告已保存: {output_path}")


# ============================================
# 主程序
# ============================================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析 receiver.log 中的网络性能数据')
    parser.add_argument('--smooth', action='store_true', 
                        help='启用平滑处理（使用 Savitzky-Golay 滤波）')
    parser.add_argument('--window', type=int, default=5,
                        help='平滑窗口大小（默认: 5）')
    args = parser.parse_args()
    
    print("=" * 80)
    print("分析 receiver.log 中的网络性能数据")
    if args.smooth:
        print(f"平滑处理: 已启用 (窗口大小={args.window})")
    else:
        print("平滑处理: 未启用（显示原始数据）")
    print("=" * 80 + "\n")
    
    # 确保结果目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 查找所有 receiver.log 文件
    log_pattern = os.path.join(OUTPUT_DIR, '*_receiver.log')
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"错误: 在 {OUTPUT_DIR} 中没有找到 *_receiver.log 文件")
        print("请先运行 multi_cc_test.py 生成测试数据")
        return
    
    print(f"找到 {len(log_files)} 个 receiver.log 文件\n")
    
    # 提取数据
    time_nbytes_list = []
    label_list = []
    stats_dict = {}
    max_duration = 0
    
    for log_file in sorted(log_files):
        # 从文件名提取算法名
        algo_name = os.path.basename(log_file).replace('_receiver.log', '')
        
        print(f"处理: {algo_name}...")
        
        try:
            # 解析网络日志
            net_info = NetInfo(log_file)
            
            if not net_info.net_data or len(net_info.net_data) == 0:
                print(f"  警告: 未找到网络数据")
                continue
            
            # 计算性能指标
            eval_method = NetEvalMethodExtension()
            result = eval_method.eval(net_info)
            time_nbytes, avg_delay, p95_delay, recv_rate, loss_ratio = result
            
            if not time_nbytes:
                print(f"  警告: 未提取到吞吐量数据")
                continue
            
            # 记录数据
            time_nbytes_list.append(time_nbytes)
            label_list.append(algo_name)
            
            # 统计信息
            stats_dict[algo_name] = {
                'packet_count': len(net_info.net_data),
                'recv_rate': recv_rate,
                'avg_delay': avg_delay,
                'p95_delay': p95_delay,
                'loss_ratio': loss_ratio,
            }
            
            # 计算持续时间
            timestamps = list(time_nbytes.keys())
            duration = (max(timestamps) - min(timestamps)) / 1000.0
            max_duration = max(max_duration, duration)
            
            print(f"  数据包: {len(net_info.net_data)}, 接收速率: {recv_rate:.3f} Mbps")
            print(f"  自致延迟(平均): {avg_delay:.2f} ms, 95分位延迟: {p95_delay:.2f} ms, 丢包率: {loss_ratio*100:.2f}%")
            
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    if not time_nbytes_list:
        print("\n错误: 没有成功提取任何数据")
        return
    
    print(f"\n成功处理 {len(time_nbytes_list)} 个算法的数据")
    print("\n" + "=" * 80)
    print("生成报告和图表...")
    print("=" * 80 + "\n")
    
    # 绘制吞吐量对比图
    duration_sec = int(max_duration) + 5
    if args.smooth:
        output_path = os.path.join(RESULT_DIR, f"throughput_comparison_smooth_w{args.window}.pdf")
    else:
        output_path = os.path.join(RESULT_DIR, "throughput_comparison.pdf")
    
    draw_goodput(time_nbytes_list, label_list, output_path, 
                 min_gap=200, duration=duration_sec, 
                 smooth=args.smooth, smooth_window=args.window)
    
    # 保存统计报告
    report_path = os.path.join(RESULT_DIR, 'statistics_report.txt')
    save_statistics_report(stats_dict, report_path)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n结果保存在: {RESULT_DIR}/")
    print("  - statistics_report.txt          # 统计报告")
    if args.smooth:
        print(f"  - throughput_comparison_smooth_w{args.window}.pdf  # 吞吐量对比图（平滑）")
    else:
        print("  - throughput_comparison.pdf      # 吞吐量对比图（原始）")
    print("\n使用提示:")
    print("  python3 eval/eval_output.py              # 显示原始数据")
    print("  python3 eval/eval_output.py --smooth     # 使用平滑处理（默认窗口=5）")
    print("  python3 eval/eval_output.py --smooth --window 10  # 自定义窗口大小")
    print()


if __name__ == '__main__':
    main()