import collections

# COPA Constants
kDefaultDelta = 0.5
kMinBitrate = 300000
kInitBitrate = 1000000
kMaxBitrate = 50 * 1000000

# COPA+ Specific Constants
kProbeIntervalMs = 10000     # 每 10 秒探测一次
kProbeDurationMs = 250       # 探测持续 250ms (约 1-2 RTT)
kProbeRateScale = 0.5        # 探测期间速率降为 0.5 倍

class Estimator(object):
    def __init__(self):
        self.packets_list = []
        self.start_time = -1
        
        # COPA State
        self.current_bitrate = kInitBitrate
        self.delta = kDefaultDelta
        self.velocity = 1.0
        self.direction = 0
        self.same_direction_count = 0
        
        # RTT and Delay tracking
        self.min_rtt = float('inf')
        self.avg_rtt = 0
        self.queuing_delay = 0
        self.now_ms = 0
        
        # COPA+ Probing State
        self.last_probe_time = -1
        self.is_probing = False
        
        # Timer delta for clock synchronization (抵消时钟偏移)
        self.timer_delta = None

    def reset(self):
        self.packets_list = []
        self.current_bitrate = kInitBitrate
        self.delta = kDefaultDelta
        self.velocity = 1.0
        self.min_rtt = float('inf')
        self.start_time = -1
        self.last_probe_time = -1
        self.is_probing = False
        self.timer_delta = None

    def report_states(self, stats: dict):
        if stats.get("type") == "qoe":
            return
        # (同上，保持一致)
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
            self.last_probe_time = self.now_ms

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        if not self.packets_list:
            return int(self.current_bitrate)

        # 1. 更新 RTT 和排队延迟
        self.update_rtt_stats()

        # 2. COPA+ 核心: 周期性排空探测 (Probing Logic)
        if self.check_and_run_probing():
            # 如果正在探测期间，直接返回调整后的低速率，跳过标准更新
            self.packets_list = []
            return int(self.current_bitrate)

        # 3. 计算目标速率 (Standard Copa Logic)
        mtu_bits = 1200 * 8
        if self.queuing_delay <= 0.002:
            target_bitrate = self.current_bitrate * 2
        else:
            target_bitrate = mtu_bits / (self.delta * self.queuing_delay)

        # 4. 更新速率
        self.update_rate(target_bitrate)

        # 5. 更新 Delta
        self.update_delta()

        self.packets_list = []
        return int(self.current_bitrate)

    def check_and_run_probing(self):
        """
        管理 Copa+ 的 Probe 状态
        返回: True 如果正在 Probing, False 否则
        """
        # 初始化
        if self.last_probe_time == -1:
            self.last_probe_time = self.now_ms
            return False

        time_since_last = self.now_ms - self.last_probe_time

        # 触发 Probing
        if not self.is_probing and time_since_last > kProbeIntervalMs:
            self.is_probing = True
            self.last_probe_time = self.now_ms
            # 立即降低速率以排空队列
            self.current_bitrate = max(kMinBitrate, self.current_bitrate * kProbeRateScale)
            # 重置 velocity 以避免恢复时过度冲激
            self.velocity = 1.0
            return True

        # 检查 Probing 是否结束
        if self.is_probing:
            if self.now_ms - self.last_probe_time > kProbeDurationMs:
                self.is_probing = False
                # 退出 Probing 时，不立即恢复速率，而是让 Copa 逻辑根据
                # 刷新后的 min_rtt 自动爬升
            else:
                # 保持低速率
                pass 
            return True

        return False

    def update_rtt_stats(self):
        curr_rtt_sum = 0
        count = 0
        for pkt in self.packets_list:
            # 修改点：增加 payload_type 检查，仅使用视频包计算 RTT
            if pkt.payload_type == 98:
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

    def update_rate(self, target_bitrate):
        rate_change_step = (kInitBitrate * 0.05) * self.velocity
        
        if self.current_bitrate < target_bitrate:
            new_direction = 1
            self.current_bitrate += rate_change_step
        else:
            new_direction = -1
            self.current_bitrate = max(kMinBitrate, self.current_bitrate - rate_change_step)

        if new_direction == self.direction:
            self.same_direction_count += 1
            if self.same_direction_count > 3:
                self.velocity = min(self.velocity * 2, 8.0)
                self.same_direction_count = 0
        else:
            self.velocity = 1.0
            self.same_direction_count = 0
            self.direction = new_direction
            
        self.current_bitrate = min(self.current_bitrate, kMaxBitrate)

    def update_delta(self):
        is_queue_busy = self.queuing_delay > 0.01
        if is_queue_busy:
            self.delta = max(0.1, self.delta * 0.98)
        else:
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