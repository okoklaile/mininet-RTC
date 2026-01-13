import collections
import math

# PCC Vivace Constants
kMinDurationMicro = 2 * 1000 * 1000  # 最小 MI 持续时间 (us) - 实际上适配 PyRTC 可用 200ms
kInitialRate = 300000        # 初始带宽 300kbps
kMinRate = 150000            # 最小带宽
kAlpha = 0.9                 # 吞吐量效用指数
kLossCoefficient = 11.35     # 丢包惩罚系数
kLatencyCoefficient = 900    # 延迟梯度惩罚系数
kGradientStepSize = 0.05     # 每次调整的步长 (5%)
kMonitorIntervalMs = 200     # MI 时长 (ms)

class Estimator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.packets_list = []
        self.start_time_ms = -1
        self.current_rate = kInitialRate
        self.last_rate = kInitialRate
        
        # Vivace State
        self.utility_history = []  # 存储 (rate, utility)
        self.gradient = 1          # 初始梯度方向 (1: 增加, -1: 减少)
        self.step_size = kGradientStepSize
        self.mi_start_ms = -1
        
        # RTT Tracking for gradient
        self.prev_avg_rtt = 0
        self.curr_avg_rtt = 0

    def report_states(self, stats: dict):
        '''
        收集数据包统计信息
        '''
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        
        now_ms = packet_info.receive_timestamp
        if self.start_time_ms == -1:
            self.start_time_ms = now_ms
            self.mi_start_ms = now_ms

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        '''
        PCC 核心逻辑：在 Monitor Interval 结束时计算 Utility 并调整速率
        '''
        if not self.packets_list:
            return int(self.current_rate)

        now_ms = self.packets_list[-1].receive_timestamp
        duration = now_ms - self.mi_start_ms

        # 检查 Monitor Interval 是否结束 (例如 200ms 或 1个 RTT)
        # 这里为了简化适配，使用固定 200ms 作为决策周期
        if duration < kMonitorIntervalMs:
            return int(self.current_rate)

        # 1. 计算当前 MI 的统计数据
        throughput, loss_rate, avg_rtt = self.compute_stats(duration)
        
        # 2. 计算效用 (Utility)
        # 延迟梯度 (Latency Gradient): (CurrRTT - PrevRTT) / duration
        # 为了数值稳定性，单位统一处理
        rtt_gradient = 0
        if self.prev_avg_rtt > 0:
            rtt_gradient = (avg_rtt - self.prev_avg_rtt) / (duration / 1000.0) # ms/s
        
        # 简单的噪声过滤
        if abs(rtt_gradient) < 5: 
            rtt_gradient = 0

        # Vivace Utility Function
        # U = T^0.9 - 11.35 * T * L - 900 * T * rtt_gradient
        # T 单位: Mbps (避免数值过大)
        t_mbps = throughput / 1000000.0
        
        utility = (math.pow(t_mbps, kAlpha) 
                   - kLossCoefficient * t_mbps * loss_rate 
                   - kLatencyCoefficient * t_mbps * max(0, rtt_gradient))

        # 3. 速率调整 (Gradient Ascent 简化版)
        self.update_rate(utility)

        # 4. 更新状态以准备下一个 MI
        self.prev_avg_rtt = avg_rtt
        self.mi_start_ms = now_ms
        self.packets_list = [] # 清空当前 MI 数据
        
        return int(self.current_rate)

    def compute_stats(self, duration_ms):
        '''
        计算 MI 内的吞吐量、丢包率、平均 RTT
        '''
        total_bytes = 0
        rtt_sum = 0
        count = 0
        min_seq = float('inf')
        max_seq = float('-inf')
        
        valid_pkts = 0

        for pkt in self.packets_list:
            total_bytes += pkt.size
            rtt = pkt.receive_timestamp - pkt.send_timestamp
            rtt_sum += rtt
            count += 1
            
            if pkt.payload_type == 126: # 假设 126 是视频流
                valid_pkts += 1
                if pkt.sequence_number < min_seq: min_seq = pkt.sequence_number
                if pkt.sequence_number > max_seq: max_seq = pkt.sequence_number

        # Throughput (bps)
        throughput = (total_bytes * 8) / (duration_ms / 1000.0)
        
        # Avg RTT (ms)
        avg_rtt = rtt_sum / count if count > 0 else 0
        
        # Loss Rate
        loss_rate = 0
        if valid_pkts > 0 and (max_seq - min_seq) > 0:
            expected = max_seq - min_seq + 1
            loss_rate = 1.0 - (valid_pkts / expected)
            loss_rate = max(0.0, min(1.0, loss_rate))

        return throughput, loss_rate, avg_rtt

    def update_rate(self, current_utility):
        '''
        基于 Utility 变化调整速率
        '''
        if len(self.utility_history) > 0:
            prev_rate, prev_utility = self.utility_history[-1]
            
            # 如果 Utility 增加，继续沿当前梯度方向走
            if current_utility > prev_utility:
                # 保持方向，步长可能加速 (这里保持固定步长简化)
                pass 
            else:
                # Utility 减少，反转方向
                self.gradient *= -1
                # 可选：减小步长以收敛
                
        # 记录历史
        self.utility_history.append((self.current_rate, current_utility))
        if len(self.utility_history) > 10:
            self.utility_history.pop(0)

        # 应用调整
        # NewRate = OldRate * (1 + sign * step)
        change = self.gradient * self.step_size
        self.current_rate = self.current_rate * (1 + change)
        
        # 边界限制
        self.current_rate = max(self.current_rate, kMinRate)
        # PyRTC 模拟中通常不需要设上限，或设一个物理上限
        self.current_rate = min(self.current_rate, 100 * 1000 * 1000) # 100 Mbps max


class PacketInfo:
    def __init__(self):
        self.payload_type = None
        self.sequence_number = None
        self.send_timestamp = None
        self.ssrc = None
        self.padding_length = None
        self.header_length = None
        self.receive_timestamp = None
        self.payload_size = None
        self.size = None