import collections

# COPA Constants
kDefaultDelta = 0.5          # 默认 delta (0.5 为论文推荐值)
kMinBitrate = 300000         # 300 kbps
kInitBitrate = 1000000       # 1 Mbps
kMaxBitrate = 50 * 1000000   # 50 Mbps

class Estimator(object):
    def __init__(self):
        self.packets_list = []
        self.start_time = -1
        
        # COPA State variables
        self.current_bitrate = kInitBitrate
        self.delta = kDefaultDelta
        self.velocity = 1.0       # 速率调整速度因子 (v)
        self.direction = 0        # 0: None, 1: Increase, -1: Decrease
        self.same_direction_count = 0
        
        # RTT and Delay tracking
        self.min_rtt = float('inf')
        self.avg_rtt = 0
        self.queuing_delay = 0
        self.now_ms = 0
        
        # Timer delta for clock synchronization (抵消时钟偏移)
        self.timer_delta = None

    def reset(self):
        self.packets_list = []
        self.current_bitrate = kInitBitrate
        self.delta = kDefaultDelta
        self.velocity = 1.0
        self.direction = 0
        self.same_direction_count = 0
        self.min_rtt = float('inf')
        self.start_time = -1
        self.timer_delta = None

    def report_states(self, stats: dict):
        if stats.get("type") == "qoe":
            return
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        
        self.now_ms = packet_info.receive_timestamp
        if self.start_time == -1:
            self.start_time = self.now_ms

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        if not self.packets_list:
            return int(self.current_bitrate)

        # 1. 更新 RTT 和排队延迟
        self.update_rtt_stats()

        # 2. 计算目标速率 (Target Rate)
        # 公式: lambda_target = 1 / (delta * d_queue)
        # 为了单位适配 (bits/s), 我们假设一个平均包大小或直接使用比例系数
        # Target_Rate (bps) = (MTU_bits) / (delta * queuing_delay_sec)
        mtu_bits = 1200 * 8 # 假设平均包大小 1200 字节
        
        if self.queuing_delay <= 0.002: # 延迟极低 (2ms以下)
            # 几乎没有排队，允许倍增或最大化利用
            target_bitrate = self.current_bitrate * 2
        else:
            target_bitrate = mtu_bits / (self.delta * self.queuing_delay)

        # 3. 更新速率 (基于 Target Rate)
        self.update_rate(target_bitrate)

        # 4. 更新 Delta (竞争模式)
        self.update_delta()

        # 清空数据包缓存
        self.packets_list = []
        return int(self.current_bitrate)

    def update_rtt_stats(self):
        curr_rtt_sum = 0
        count = 0
        for pkt in self.packets_list:
            # 修改点：增加 payload_type 检查，仅使用视频包计算 RTT
            if pkt.payload_type == 125:
                # 使用抵消法计算延迟，消除时钟偏移影响
                if self.timer_delta is None:
                    # 第一个包：初始化 timer_delta
                    self.timer_delta = -(pkt.receive_timestamp - pkt.send_timestamp)
                
                rtt = self.timer_delta + pkt.receive_timestamp - pkt.send_timestamp
                if rtt > 0:
                    curr_rtt_sum += rtt
                    count += 1
                    # 维护最小 RTT (min_rtt)
                    if rtt < self.min_rtt:
                        self.min_rtt = rtt
        
        if count > 0:
            self.avg_rtt = curr_rtt_sum / count
            self.queuing_delay = (self.avg_rtt - self.min_rtt) / 1000.0
            
            # 简单的过期机制: 如果很久没更新 min_rtt，可能网络路径变了
            # 标准 Copa 不包含此逻辑，但在长期运行中推荐加入
            # 这里保持纯粹，不主动重置 min_rtt，依靠 Copa+ 来处理 Bufferbloat

    def update_rate(self, target_bitrate):
        # 类似 TCP 的 AI (Additive Increase) 和 MD (Multiplicative Decrease)
        # 但 Copa 使用速度因子 v 来加速收敛
        
        rate_change_step = (kInitBitrate * 0.05) * self.velocity
        
        if self.current_bitrate < target_bitrate:
            # 需要增加速率
            new_direction = 1
            self.current_bitrate += rate_change_step
        else:
            # 需要减少速率
            new_direction = -1
            self.current_bitrate = max(kMinBitrate, self.current_bitrate - rate_change_step)

        # 更新速度因子 v
        if new_direction == self.direction:
            self.same_direction_count += 1
            if self.same_direction_count > 3:
                # 连续同向调整 3 次，加速 (v *= 2 or v += 1)
                self.velocity = min(self.velocity * 2, 8.0)
                self.same_direction_count = 0 # 重置计数或保留取决于策略，这里重置以平滑
        else:
            # 方向反转，重置速度
            self.velocity = 1.0
            self.same_direction_count = 0
            self.direction = new_direction
            
        self.current_bitrate = min(self.current_bitrate, kMaxBitrate)

    def update_delta(self):
        """
        竞争模式: 动态调整 Delta
        如果排队延迟持续很高 (被 TCP 填满)，减小 Delta 以提高 Target Rate。
        如果排队延迟很低，增加 Delta 以降低延迟。
        """
        is_queue_busy = self.queuing_delay > 0.01 # >10ms
        
        if is_queue_busy:
            # 更加激进 (Competitve)
            self.delta = max(0.1, self.delta * 0.98)
        else:
            # 回归保守 (Delay-sensitive)
            self.delta = min(kDefaultDelta, self.delta * 1.05)

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