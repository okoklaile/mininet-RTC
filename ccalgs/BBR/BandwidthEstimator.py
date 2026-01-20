import collections
import math

# BBR Constants
kProbeRTTInterval = 10000     # 10s
kProbeRTTDuration = 200       # 200ms
kMinRttWindow = 10000         # 10s
kBtlBwWindowMs = 10000        # 10s

# Gains
kHighGain = 2.885             # 2/ln(2)
kDrainGain = 1.0 / kHighGain  # ln(2)/2
kPacingGainCycle = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

kMinBitrate = 150000          # 150 kbps
kInitBitrate = 300000         # 300 kbps (降低初始值，给 Startup 留空间)
kMaxBitrate = 30 * 1000000    # 30 Mbps (防止仿真环境被撑爆)

class BBRState:
    STARTUP = 0
    DRAIN = 1
    PROBE_BW = 2
    PROBE_RTT = 3

class Estimator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.packets_list = []
        self.start_time = -1
        self.now_ms = 0

        # BBR State
        self.state = BBRState.STARTUP
        self.pacing_gain = kHighGain
        self.current_bitrate = kInitBitrate
        
        # BtlBw (Max Bandwidth)
        self.btl_bw = 0
        self.btl_bw_filter = WindowedMaxFilter(kBtlBwWindowMs)
        
        # RTprop (Min RTT)
        self.rt_prop = float('inf')
        self.rt_prop_stamp = -1 

        # Cycle logic
        self.cycle_idx = 0
        self.cycle_start_time = -1
        
        # Timer delta for clock synchronization (抵消时钟偏移)
        self.timer_delta = None
        
        # Startup logic
        self.full_bw_reached = False
        self.last_startup_bw = 0
        self.rounds_without_growth = 0
        
        # ProbeRTT logic
        self.probe_rtt_start_ms = -1

    def report_states(self, stats: dict):
        if stats.get("type") == "qoe":
            return
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        
        self.now_ms = packet_info.receive_timestamp
        if self.start_time == -1:
            self.start_time = self.now_ms
            self.rt_prop_stamp = self.now_ms
            self.cycle_start_time = self.now_ms

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        if not self.packets_list:
            return int(self.current_bitrate)

        # 1. 更新测量模型 (仅使用视频包)
        self.update_model_and_stats()

        # 2. 状态机更新
        self.update_control_state()

        # 3. 计算目标速率
        # 如果 BtlBw 未建立，使用当前速率作为基准进行增长
        ref_bw = self.btl_bw if self.btl_bw > 0 else self.current_bitrate
        target_rate = self.pacing_gain * ref_bw

        # 4. 边界限制
        self.current_bitrate = max(kMinBitrate, target_rate)
        self.current_bitrate = min(self.current_bitrate, kMaxBitrate)
        
        # ProbeRTT 期间限制 (Cwnd = 4 MSS, approx 0.5 * BtlBw)
        if self.state == BBRState.PROBE_RTT and self.btl_bw > 0:
            limit = max(kMinBitrate, self.btl_bw * 0.5)
            self.current_bitrate = min(self.current_bitrate, limit)

        # 清空列表，准备下一轮
        self.packets_list = []
        return int(self.current_bitrate)

    def update_model_and_stats(self):
        # 过滤视频包 (Payload 126)
        video_packets = [p for p in self.packets_list if p.payload_type == 125]
        if not video_packets:
            return

        # 更新 BtlBw
        if len(video_packets) >= 2:
            total_bytes = sum([p.size for p in video_packets]) * 8 
            duration = video_packets[-1].receive_timestamp - video_packets[0].receive_timestamp
            
            # 保护: 防止 duration 为 0 (同一 tick 到达)
            if duration <= 0:
                duration = 1 # 设为 1ms 避免除零

            sample_rate = total_bytes / (duration / 1000.0)
            self.btl_bw_filter.update(sample_rate, self.now_ms)
        
        self.btl_bw = self.btl_bw_filter.get_best()

        # 更新 RTprop (使用抵消法计算 RTT)
        for pkt in video_packets:
            # 使用抵消法计算延迟，消除时钟偏移影响
            if self.timer_delta is None:
                # 第一个包：初始化 timer_delta
                self.timer_delta = -(pkt.receive_timestamp - pkt.send_timestamp)
            
            rtt = self.timer_delta + pkt.receive_timestamp - pkt.send_timestamp
            if rtt > 0:
                if rtt <= self.rt_prop:
                    self.rt_prop = rtt
                    self.rt_prop_stamp = self.now_ms
                else:
                    # 窗口过期逻辑 (10s)
                    if self.now_ms - self.rt_prop_stamp > kMinRttWindow:
                        # 不在此处强制重置，依赖 ProbeRTT
                        pass
    
    def update_control_state(self):
        # 1. Check ProbeRTT Entry
        # 只有在有有效 RTprop 时才检查过期
        if self.rt_prop != float('inf'):
            if (self.state != BBRState.PROBE_RTT and 
                self.now_ms - self.rt_prop_stamp > kProbeRTTInterval):
                self.enter_probe_rtt()
                return

        # 2. Check ProbeRTT Exit
        if self.state == BBRState.PROBE_RTT:
            if self.now_ms - self.probe_rtt_start_ms > kProbeRTTDuration:
                self.exit_probe_rtt()
            return 

        # 3. State Transitions
        if self.state == BBRState.STARTUP:
            if self.check_full_bandwidth_reached():
                self.state = BBRState.DRAIN
                self.pacing_gain = kDrainGain
        
        elif self.state == BBRState.DRAIN:
            # 简化版 Drain 退出：当 pacing rate 降至 BtlBw 以下
            if self.current_bitrate <= self.btl_bw:
                self.state = BBRState.PROBE_BW
                self.pacing_gain = 1.0
                self.cycle_idx = 0
                self.cycle_start_time = self.now_ms

        elif self.state == BBRState.PROBE_BW:
            self.update_probe_bw_cycle()

    def enter_probe_rtt(self):
        self.state = BBRState.PROBE_RTT
        self.pacing_gain = 1.0
        self.probe_rtt_start_ms = self.now_ms

    def exit_probe_rtt(self):
        self.rt_prop_stamp = self.now_ms
        if self.full_bw_reached:
            self.state = BBRState.PROBE_BW
            self.pacing_gain = 1.0
            self.cycle_start_time = self.now_ms # 重置 cycle 计时
        else:
            self.state = BBRState.STARTUP
            self.pacing_gain = kHighGain

    def check_full_bandwidth_reached(self):
        # 如果 BtlBw 还没测出来，继续 Startup
        if self.btl_bw == 0:
            return False

        # 辅助退出条件：如果当前速率已经很大 (接近上限) 或者检测到明显丢包
        # 这里仅使用带宽平稳判定
        if self.last_startup_bw == 0:
            self.last_startup_bw = self.btl_bw
            return False
        
        # 增长阈值 25%
        if self.btl_bw >= self.last_startup_bw * 1.25:
            self.last_startup_bw = self.btl_bw
            self.rounds_without_growth = 0
            return False
        else:
            self.rounds_without_growth += 1
            # 连续 3 次检查 (每次 get_bwe 调用算一次检查，稍微有点频繁，但能防爆)
            if self.rounds_without_growth >= 3:
                self.full_bw_reached = True
                return True
        return False

    def update_probe_bw_cycle(self):
        # 保护: 如果 rt_prop 无效，默认 200ms
        rtt = self.rt_prop if self.rt_prop != float('inf') else 200
        phase_duration = max(rtt, 200)
        
        if self.now_ms - self.cycle_start_time > phase_duration:
            self.cycle_idx = (self.cycle_idx + 1) % len(kPacingGainCycle)
            self.cycle_start_time = self.now_ms
            self.pacing_gain = kPacingGainCycle[self.cycle_idx]


class WindowedMaxFilter:
    def __init__(self, window_len_ms):
        self.window_len = window_len_ms
        self.samples = collections.deque() # (value, time)

    def update(self, value, now_ms):
        while self.samples and (now_ms - self.samples[0][1] > self.window_len):
            self.samples.popleft()
        self.samples.append((value, now_ms))

    def get_best(self):
        if not self.samples:
            return 0
        return max(s[0] for s in self.samples)

class PacketInfo:
    def __init__(self):
        self.payload_type = None
        self.sequence_number = None
        self.send_timestamp = None
        self.receive_timestamp = None
        self.size = None