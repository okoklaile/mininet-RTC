"""
GCC (Google Congestion Control) 带宽估计器实现
基于延迟和丢包率进行带宽估计，用于实时通信场景
"""
import collections

# GCC算法相关常量定义
kMinNumDeltas = 60  # 最小延迟样本数量
threshold_gain_ = 4  # 阈值增益系数
kBurstIntervalMs = 5  # 数据包突发分组的时间间隔(毫秒)
kTrendlineWindowSize = 20  # 趋势线计算的窗口大小
kTrendlineSmoothingCoeff = 0.9  # 趋势线平滑系数
kOverUsingTimeThreshold = 10  # 过载检测时间阈值(毫秒)
kMaxAdaptOffsetMs = 15.0  # 最大自适应偏移量(毫秒)
eta = 1.08  # 乘性增长因子
alpha = 0.85  # 平滑系数
k_up_ = 0.0087  # 阈值上升速率
k_down_ = 0.039  # 阈值下降速率
Time_Interval = 200  # 时间间隔(毫秒)


class GCCEstimator(object):
    """
    GCC带宽估计器主类
    实现基于延迟梯度和丢包率的带宽估计算法
    """
    def __init__(self):
        # 数据包相关
        self.packets_list = []  # 接收到的数据包列表
        self.packet_group = []  # 数据包分组列表
        self.first_group_complete_time = -1  # 第一个数据包组完成时间

        # 延迟相关参数
        self.acc_delay = 0  # 累积延迟
        self.smoothed_delay = 0  # 平滑后的延迟
        self.acc_delay_list = collections.deque([])  # 累积延迟列表（用于趋势线计算）
        self.smoothed_delay_list = collections.deque([])  # 平滑延迟列表（用于趋势线计算）

        # 带宽估计相关
        self.state = 'Hold'  # 当前状态：'Hold', 'Increase', 'Decrease'
        self.last_bandwidth_estimation = 300 * 1000  # 上次带宽估计值(bps)，初始300kbps
        self.avg_max_bitrate_kbps_ = -1  # 平均最大比特率(kbps)
        self.var_max_bitrate_kbps_ = -1  # 最大比特率方差
        self.rate_control_region_ = "kRcMaxUnknown"  # 速率控制区域状态
        self.time_last_bitrate_change_ = -1  # 上次比特率变化的时间

        # 过载检测相关
        self.gamma1 = 12.5  # 动态阈值，用于判断网络过载/空闲
        self.num_of_deltas_ = 0  # 延迟梯度样本数量
        self.time_over_using = -1  # 过载持续时间
        self.prev_trend = 0.0  # 上一次的趋势值
        self.overuse_counter = 0  # 过载计数器
        self.overuse_flag = 'NORMAL'  # 过载标志：'NORMAL', 'OVERUSE', 'UNDERUSE'
        self.last_update_ms = -1  # 上次更新时间
        self.last_update_threshold_ms = -1  # 上次更新阈值的时间
        self.now_ms = -1  # 当前时间

    def reset(self):
        """
        重置估计器到初始状态
        用于开始新的估计周期或遇到异常情况时
        """
        self.packets_list = []
        self.packet_group = []
        self.first_group_complete_time = -1

        self.acc_delay = 0
        self.smoothed_delay = 0
        self.acc_delay_list = collections.deque([])
        self.smoothed_delay_list = collections.deque([])

        self.state = 'Hold'
        self.last_bandwidth_estimation = 300 * 1000
        self.avg_max_bitrate_kbps_ = -1
        self.var_max_bitrate_kbps_ = -1
        self.rate_control_region_ = "kRcMaxUnknown"
        self.time_last_bitrate_change_ = -1 

        self.gamma1 = 12.5
        self.num_of_deltas_ = 0
        self.time_over_using = -1
        self.prev_trend = 0.0 
        self.overuse_counter = 0
        self.overuse_flag = 'NORMAL'
        self.last_update_ms = -1 
        self.last_update_threshold_ms = -1
        self.now_ms = -1

    def report_states(self, stats: dict):
        """
        接收并存储数据包信息
        将接收到的数据包头部信息存储在packets_list中
        Args:
            stats: 包含数据包统计信息的字典
        """
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.padding_length = pkt["padding_length"]
        packet_info.header_length = pkt["header_length"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        packet_info.bandwidth_prediction = self.last_bandwidth_estimation
        self.now_ms = packet_info.receive_timestamp  # use the arrival time of the last packet as the system time

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self):
        """
        计算并返回最终的带宽估计值
        同时基于延迟和丢包率进行估计，取两者的最小值作为最终结果
        Returns:
            bandwidth_estimation: 带宽估计值(bps)
            overuse_flag: 网络状态标志
        """
        BWE_by_delay, flag = self.get_estimated_bandwidth_by_delay()  # 基于延迟的带宽估计
        BWE_by_loss = self.get_estimated_bandwidth_by_loss()  # 基于丢包率的带宽估计
        bandwidth_estimation = min(BWE_by_delay, BWE_by_loss)  # 取两者最小值，保守估计
        if flag == True:
            self.packets_list = []  # 清空已处理的数据包列表
        self.last_bandwidth_estimation = bandwidth_estimation
        return bandwidth_estimation,self.overuse_flag

    def get_inner_estimation(self):
        """
        获取内部估计值（用于调试和分析）
        分别返回基于延迟和基于丢包率的带宽估计值
        Returns:
            BWE_by_delay: 基于延迟的带宽估计
            BWE_by_loss: 基于丢包率的带宽估计
        """
        BWE_by_delay, flag = self.get_estimated_bandwidth_by_delay()
        BWE_by_loss = self.get_estimated_bandwidth_by_loss()
        bandwidth_estimation = min(BWE_by_delay, BWE_by_loss)
        if flag == True:
            self.packets_list = []
        return BWE_by_delay,BWE_by_loss

    def change_bandwidth_estimation(self,bandwidth_prediction):
        """
        手动设置带宽估计值
        Args:
            bandwidth_prediction: 新的带宽预测值(bps)
        """
        self.last_bandwidth_estimation = bandwidth_prediction

    def get_estimated_bandwidth_by_delay(self):
        """
        基于延迟的带宽估计
        通过分析数据包延迟梯度的趋势来判断网络拥塞状态，并调整带宽估计
        
        核心流程：
        1. 将数据包按发送时间分组
        2. 计算各组之间的延迟梯度
        3. 使用趋势线过滤器分析延迟变化趋势
        4. 检测网络过载/空闲状态
        5. 根据状态决定带宽调整方向
        6. 计算最终的带宽估计值
        
        Returns:
            bandwidth_estimation: 带宽估计值(bps)
            flag: 是否成功完成估计
        """
        if len(self.packets_list) == 0:  # 在此时间间隔内没有收到数据包
            return self.last_bandwidth_estimation, False

        # 1. 将数据包分组（根据发送时间将数据包分成burst组）
        pkt_group_list = self.divide_packet_group()
        if len(pkt_group_list) < 2:  # 至少需要2组才能计算延迟梯度
            return self.last_bandwidth_estimation, False

        # 2. 计算数据包组之间的延迟梯度
        send_time_delta_list, _, _, delay_gradient_list = self.compute_deltas_for_pkt_group(pkt_group_list)

        # 3. 计算趋势线（对延迟梯度进行线性回归）
        trendline = self.trendline_filter(delay_gradient_list, pkt_group_list)
        if trendline == None:  # 样本不足，无法计算趋势线
            return self.last_bandwidth_estimation, False

        # 4. 过载检测（根据趋势线判断当前网络状态）
        self.overuse_detector(trendline, sum(send_time_delta_list))

        # 5. 状态转换（确定带宽调整方向：增加/保持/减少）
        state = self.ChangeState()

        # 6. 根据状态调整带宽估计值
        bandwidth_estimation = self.rate_adaptation_by_delay(state)

        return bandwidth_estimation, True

    def get_estimated_bandwidth_by_loss(self) -> int:
        """
        基于丢包率的带宽估计
        当网络出现丢包时，说明带宽可能不足，需要降低发送速率
        
        Returns:
            bandwidth_estimation: 基于丢包率计算的带宽估计值(bps)
        """
        loss_rate = self.caculate_loss_rate()
        if loss_rate == -1:  # 无法计算丢包率
            return self.last_bandwidth_estimation

        bandwidth_estimation = self.rate_adaptation_by_loss(loss_rate)
        return bandwidth_estimation

    def caculate_loss_rate(self):
        """
        计算当前时间间隔内的丢包率
        通过序列号的连续性来判断丢包情况
        
        Returns:
            loss_rate: 丢包率 (0.0-1.0)，如果无法计算则返回-1
        """
        flag = False
        valid_packets_num = 0
        min_sequence_number, max_sequence_number = 0, 0
        if len(self.packets_list) == 0:  # 没有收到数据包
            return -1
        # 统计有效数据包（payload_type == 125）
        for i in range(len(self.packets_list)):
            if self.packets_list[i].payload_type == 125:
                if not flag:
                    min_sequence_number = self.packets_list[i].sequence_number
                    max_sequence_number = self.packets_list[i].sequence_number
                    flag = True
                valid_packets_num += 1
                min_sequence_number = min(min_sequence_number, self.packets_list[i].sequence_number)
                max_sequence_number = max(max_sequence_number, self.packets_list[i].sequence_number)
        if (max_sequence_number - min_sequence_number) == 0:
            return -1
        # 计算接收率和丢包率
        receive_rate = valid_packets_num / (max_sequence_number - min_sequence_number)
        loss_rate = 1 - receive_rate
        return loss_rate

    def rate_adaptation_by_loss(self, loss_rate) -> int:
        """
        基于丢包率调整带宽估计
        采用阈值策略：高丢包率时降低，低丢包率时提升
        
        Args:
            loss_rate: 丢包率 (0.0-1.0)
        Returns:
            bandwidth_estimation: 调整后的带宽估计值(bps)
        """
        bandwidth_estimation = self.last_bandwidth_estimation
        if loss_rate > 0.1:  # 丢包率超过10%，降低带宽
            bandwidth_estimation = self.last_bandwidth_estimation * (1 - 0.5 * loss_rate)
        elif loss_rate < 0.02:  # 丢包率低于2%，可以尝试增加带宽
            bandwidth_estimation = 1.05 * self.last_bandwidth_estimation
        return bandwidth_estimation

    def divide_packet_group(self):
        """
        数据包分组
        将发送时间接近的数据包（在kBurstIntervalMs内）分为一组
        这样可以减少单个数据包延迟抖动的影响
        
        Returns:
            pkt_group_list: 数据包组列表
        """
        pkt_group_list = []
        first_send_time_in_group = self.packets_list[0].send_timestamp

        pkt_group = [self.packets_list[0]]
        for pkt in self.packets_list[1:]:
            # 如果数据包发送时间与组内第一个包的时间差小于阈值，加入当前组
            if pkt.send_timestamp - first_send_time_in_group <= kBurstIntervalMs:
                pkt_group.append(pkt)
            else:
                # 否则，当前组结束，开始新组
                pkt_group_list.append(PacketGroup(pkt_group))
                if self.first_group_complete_time == -1:
                    self.first_group_complete_time = pkt_group[-1].receive_timestamp
                first_send_time_in_group = pkt.send_timestamp
                pkt_group = [pkt]
        return pkt_group_list

    def compute_deltas_for_pkt_group(self, pkt_group_list):
        """
        计算数据包组之间的时间差和延迟梯度
        延迟梯度 = 接收时间差 - 发送时间差
        如果延迟梯度为正，说明网络排队延迟在增加（可能拥塞）
        
        Args:
            pkt_group_list: 数据包组列表
        Returns:
            send_time_delta_list: 发送时间差列表
            arrival_time_delta_list: 到达时间差列表
            group_size_delta_list: 组大小差列表
            delay_gradient_list: 延迟梯度列表
        """
        send_time_delta_list, arrival_time_delta_list, group_size_delta_list, delay_gradient_list = [], [], [], []
        for idx in range(1, len(pkt_group_list)): 
            # 计算相邻两组之间的发送时间差
            send_time_delta = pkt_group_list[idx].send_time_list[-1] - pkt_group_list[idx - 1].send_time_list[-1]
            # 计算相邻两组之间的到达时间差
            arrival_time_delta = pkt_group_list[idx].arrival_time_list[-1] - pkt_group_list[idx - 1].arrival_time_list[-1]
            # 计算组大小差
            group_size_delta = pkt_group_list[idx].pkt_group_size - pkt_group_list[idx - 1].pkt_group_size
            # 计算延迟梯度（单向延迟变化）
            delay = arrival_time_delta - send_time_delta
            self.num_of_deltas_ += 1
            send_time_delta_list.append(send_time_delta)
            arrival_time_delta_list.append(arrival_time_delta)
            group_size_delta_list.append(group_size_delta)
            delay_gradient_list.append(delay)

        return send_time_delta_list, arrival_time_delta_list, group_size_delta_list, delay_gradient_list

    def trendline_filter(self, delay_gradient_list, pkt_group_list):
        '''
        Calculate the trendline from the delay gradient of the packet 
        '''
        for i, delay_gradient in enumerate(delay_gradient_list):
            accumulated_delay = self.acc_delay + delay_gradient
            smoothed_delay = kTrendlineSmoothingCoeff * self.smoothed_delay + (
                    1 - kTrendlineSmoothingCoeff) * accumulated_delay

            self.acc_delay = accumulated_delay
            self.smoothed_delay = smoothed_delay

            arrival_time_ms = pkt_group_list[i + 1].complete_time
            self.acc_delay_list.append(arrival_time_ms - self.first_group_complete_time)

            self.smoothed_delay_list.append(smoothed_delay)
            if len(self.acc_delay_list) > kTrendlineWindowSize:
                self.acc_delay_list.popleft()
                self.smoothed_delay_list.popleft()
        if len(self.acc_delay_list) == kTrendlineWindowSize:
            avg_acc_delay = sum(self.acc_delay_list) / len(self.acc_delay_list)
            avg_smoothed_delay = sum(self.smoothed_delay_list) / len(self.smoothed_delay_list)

            numerator = 0
            denominator = 0
            for i in range(kTrendlineWindowSize):
                numerator += (self.acc_delay_list[i] - avg_acc_delay) * (
                        self.smoothed_delay_list[i] - avg_smoothed_delay)
                denominator += (self.acc_delay_list[i] - avg_acc_delay) * (self.acc_delay_list[i] - avg_acc_delay)

            trendline = numerator / (denominator + 1e-05)
        else:
            trendline = None
            self.acc_delay_list.clear()
            self.smoothed_delay_list.clear()
            self.acc_delay = 0
            self.smoothed_delay = 0
        return trendline

    def overuse_detector(self, trendline, ts_delta):
        """
        过载检测器
        根据趋势线斜率判断当前网络状态：
        - OVERUSE: 延迟持续增长，网络过载
        - UNDERUSE: 延迟持续下降，网络空闲
        - NORMAL: 延迟稳定，网络正常
        
        Args:
            trendline: 趋势线斜率
            ts_delta: 时间间隔
        """
        now_ms = self.now_ms
        if self.num_of_deltas_ < 2:
            return

        # 修正趋势值：考虑样本数量和增益系数
        modified_trend = trendline * min(self.num_of_deltas_, kMinNumDeltas) * threshold_gain_

        if modified_trend > self.gamma1:  # 延迟增长超过阈值
            if self.time_over_using == -1:
                self.time_over_using = ts_delta / 2
            else:
                self.time_over_using += ts_delta
            self.overuse_counter += 1
            # 需要持续过载一段时间且趋势还在增长才判定为OVERUSE
            if self.time_over_using > kOverUsingTimeThreshold and self.overuse_counter > 1:
                if trendline > self.prev_trend:
                    self.time_over_using = 0
                    self.overuse_counter = 0
                    self.overuse_flag = 'OVERUSE'
        elif modified_trend < -self.gamma1:  # 延迟下降超过阈值
            self.time_over_using = -1
            self.overuse_counter = 0
            self.overuse_flag = 'UNDERUSE'
        else:  # 延迟变化在正常范围内
            self.time_over_using = -1
            self.overuse_counter = 0
            self.overuse_flag = 'NORMAL'

        self.prev_trend = trendline
        self.update_threthold(modified_trend, now_ms)

    def update_threthold(self, modified_trend, now_ms):
        """
        更新过载检测阈值
        阈值会根据网络状态动态调整：
        - 当趋势接近阈值时，阈值会逐渐靠近趋势（自适应）
        - 阈值保持在[6, 600]范围内
        
        Args:
            modified_trend: 修正后的趋势值
            now_ms: 当前时间戳(毫秒)
        """
        if self.last_update_threshold_ms == -1:
            self.last_update_threshold_ms = now_ms
        # 如果趋势远超阈值，不更新阈值
        if abs(modified_trend) > self.gamma1 + kMaxAdaptOffsetMs:
            self.last_update_threshold_ms = now_ms
            return
        # 根据趋势与阈值的关系选择调整速率
        if abs(modified_trend) < self.gamma1:
            k = k_down_  # 趋势低于阈值，阈值下降
        else:
            k = k_up_  # 趋势高于阈值，阈值上升
        kMaxTimeDeltaMs = 100
        time_delta_ms = min(now_ms - self.last_update_threshold_ms, kMaxTimeDeltaMs)
        # 指数平滑更新阈值
        self.gamma1 += k * (abs(modified_trend) - self.gamma1) * time_delta_ms
        # 限制阈值范围
        if (self.gamma1 < 6):
            self.gamma1 = 6
        elif (self.gamma1 > 600):
            self.gamma1 = 600
        self.last_update_threshold_ms = now_ms

    def state_transfer(self):
        """
        状态转换函数（较复杂的状态机版本）
        根据当前状态和过载标志确定下一个状态
        
        状态转换规则：
        - Decrease状态：持续过载继续Decrease，否则转Hold
        - Hold状态：过载转Decrease，正常转Increase，空闲保持Hold
        - Increase状态：过载转Decrease，正常继续Increase，空闲转Hold
        
        Returns:
            newstate: 新的状态
        """
        newstate = None
        overuse_flag = self.overuse_flag
        if self.state == 'Decrease' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Decrease' and (overuse_flag == 'NORMAL' or overuse_flag == 'UNDERUSE'):
            newstate = 'Hold'
        elif self.state == 'Hold' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Hold' and overuse_flag == 'NORMAL':
            newstate = 'Increase'
        elif self.state == 'Hold' and overuse_flag == 'UNDERUSE':
            newstate = 'Hold'
        elif self.state == 'Increase' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Increase' and overuse_flag == 'NORMAL':
            newstate = 'Increase'
        elif self.state == 'Increase' and overuse_flag == 'UNDERUSE':
            newstate = 'Hold'
        else:
            print('Wrong state!')
        self.state = newstate
        return newstate

    def ChangeState(self):
        """
        状态变更函数（简化版本）
        根据过载标志直接更新状态，逻辑更简洁
        
        - NORMAL: Hold状态下转为Increase
        - OVERUSE: 立即转为Decrease
        - UNDERUSE: 转为Hold
        
        Returns:
            self.state: 更新后的状态
        """
        overuse_flag = self.overuse_flag
        if overuse_flag == 'NORMAL':
            if self.state == 'Hold':
                self.state = 'Increase'
        elif overuse_flag == 'OVERUSE':
            if self.state != 'Decrease':
                self.state = 'Decrease'
        elif overuse_flag == 'UNDERUSE':
            self.state = 'Hold'
        return self.state

    def rate_adaptation_by_delay(self, state):
        """
        基于延迟的速率自适应算法
        根据状态（Increase/Hold/Decrease）调整带宽估计值
        
        核心思想：
        - Increase状态：采用AIMD（加性增/乘性增）策略
        - Decrease状态：快速降低带宽（乘性减）
        - Hold状态：保持当前带宽
        
        Args:
            state: 当前状态（'Increase', 'Hold', 'Decrease'）
        Returns:
            bandwidth_estimation: 调整后的带宽估计值(bps)
        """
        # 计算当前时间窗口内的实际吞吐量
        estimated_throughput = 0
        for pkt in self.packets_list:
            estimated_throughput += pkt.size
        if len(self.packets_list) == 0:
            estimated_throughput_bps = 0
        else:
            time_delta = self.now_ms - self.packets_list[0].receive_timestamp
            time_delta = max(time_delta , Time_Interval)
            estimated_throughput_bps = 1000 * 8 * estimated_throughput / time_delta
        estimated_throughput_kbps = estimated_throughput_bps / 1000
     
        # 基于吞吐量的带宽上限（防止过度估计）
        troughput_based_limit = 3 * estimated_throughput_bps + 10
        
        # 更新最大吞吐量的估计和方差
        self.UpdateMaxThroughputEstimate(estimated_throughput_kbps)
        # 计算最大比特率的标准差
        std_max_bit_rate = pow(self.var_max_bitrate_kbps_ * self.avg_max_bitrate_kbps_, 0.5)

        if state == 'Increase':
            # 如果当前吞吐量远超历史最大值，说明网络条件改善，重置最大值估计
            if self.avg_max_bitrate_kbps_ >= 0 and \
                    estimated_throughput_kbps > self.avg_max_bitrate_kbps_ + 3 * std_max_bit_rate:
                self.avg_max_bitrate_kbps_ = -1.0
                self.rate_control_region_ = "kRcMaxUnknown"

            if self.rate_control_region_ == "kRcNearMax":
                # 已接近最大值，采用加性增长（保守策略）
                additive_increase_bps = self.AdditiveRateIncrease(self.now_ms, self.time_last_bitrate_change_)
                bandwidth_estimation = self.last_bandwidth_estimation + additive_increase_bps
            elif self.rate_control_region_ == "kRcMaxUnknown":
                # 最大值未知，采用乘性增长（激进策略）
                multiplicative_increase_bps = self.MultiplicativeRateIncrease(self.now_ms,
                                                                              self.time_last_bitrate_change_)
                bandwidth_estimation = self.last_bandwidth_estimation + multiplicative_increase_bps
            else:
                print("error!")
            # 限制带宽估计不超过吞吐量上限
            bandwidth_estimation = min(bandwidth_estimation,troughput_based_limit)
            self.time_last_bitrate_change_ = self.now_ms
        elif state == 'Decrease':
            # 网络过载，需要快速降低带宽（乘性减少）
            beta = 0.85  # 衰减系数
            bandwidth_estimation = beta * estimated_throughput_bps + 0.5
            # 确保带宽估计不会反而增加
            if bandwidth_estimation > self.last_bandwidth_estimation:
                if self.rate_control_region_ != "kRcMaxUnknown":
                    bandwidth_estimation = (beta * self.avg_max_bitrate_kbps_ * 1000 + 0.5)
                bandwidth_estimation = min(bandwidth_estimation, self.last_bandwidth_estimation)
            # 标记进入接近最大值区域
            self.rate_control_region_ = "kRcNearMax"

            # 如果吞吐量大幅下降，重置最大值估计
            if estimated_throughput_kbps < self.avg_max_bitrate_kbps_-3*std_max_bit_rate:
                self.avg_max_bitrate_kbps_ = -1
            self.UpdateMaxThroughputEstimate(estimated_throughput_kbps)

            # 降低后转为Hold状态，观察网络反应
            self.state='Hold'
            self.time_last_bitrate_change_ = self.now_ms
        elif state == 'Hold':
            # 保持当前带宽不变
            bandwidth_estimation = self.last_bandwidth_estimation
        else:
            print('Wrong State!')
        return bandwidth_estimation

    def AdditiveRateIncrease(self, now_ms, last_ms):
        """
        加性增长算法（Additive Increase）
        当网络接近容量上限时，采用保守的线性增长策略
        类似TCP的拥塞避免阶段
        
        增长量与平均包大小和时间间隔成正比
        Args:
            now_ms: 当前时间戳(毫秒)
            last_ms: 上次带宽变更时间戳(毫秒)
        Returns:
            additive_increase_bps: 加性增长的带宽增量(bps)
        """
        sum_packet_size, avg_packet_size = 0, 0
        for pkt in self.packets_list:
            sum_packet_size += pkt.size
        # 计算平均包大小（转换为比特）
        avg_packet_size = 8 * sum_packet_size / len(self.packets_list)

        beta = 0.0
        RTT = 2 * (self.packets_list[-1].receive_timestamp - self.packets_list[-1].send_timestamp)
        response_time = 200  # 响应时间阈值(毫秒)

        # 根据距上次更新的时间计算增长系数
        if last_ms > 0:
            beta = min(((now_ms - last_ms) / response_time), 1.0)
        # 增长量：至少800bps，或者按平均包大小增长
        additive_increase_bps = max(800, beta * avg_packet_size)
        return additive_increase_bps

    def MultiplicativeRateIncrease(self, now_ms, last_ms):
        """
        乘性增长算法（Multiplicative Increase）
        当网络容量未知时，采用激进的指数增长策略快速探测可用带宽
        类似TCP慢启动阶段
        
        按比例增长，可快速达到网络容量
        Args:
            now_ms: 当前时间戳(毫秒)
            last_ms: 上次带宽变更时间戳(毫秒)
        Returns:
            multiplicative_increase_bps: 乘性增长的带宽增量(bps)
        """
        alpha = 1.08  # 基础增长因子（8%增长率）
        if last_ms > -1:
            # 根据时间间隔调整增长因子
            time_since_last_update_ms = min(now_ms - last_ms, 1000)
            alpha = pow(alpha, time_since_last_update_ms / 1000)
        # 计算增长量：当前带宽的(alpha-1)倍，至少1000bps
        multiplicative_increase_bps = max(self.last_bandwidth_estimation * (alpha - 1.0), 1000.0)
        return multiplicative_increase_bps

    def UpdateMaxThroughputEstimate(self, estimated_throughput_kbps):
        """
        更新最大吞吐量估计
        使用指数加权移动平均(EWMA)跟踪最大吞吐量的均值和方差
        方差用于判断网络的稳定性
        
        Args:
            estimated_throughput_kbps: 当前估计的吞吐量(kbps)
        """
        alpha = 0.05  # 平滑系数
        if self.avg_max_bitrate_kbps_ == -1:
            # 首次初始化
            self.avg_max_bitrate_kbps_ = estimated_throughput_kbps
        else:
            # 指数平滑更新平均值
            self.avg_max_bitrate_kbps_ = (1 - alpha) * self.avg_max_bitrate_kbps_ + alpha * estimated_throughput_kbps
        # 归一化因子（避免除零）
        norm = max(self.avg_max_bitrate_kbps_, 1.0)
        # 计算归一化的方差
        var_value = pow((self.avg_max_bitrate_kbps_ - estimated_throughput_kbps), 2) / norm
        # 指数平滑更新方差
        self.var_max_bitrate_kbps_ = (1 - alpha) * self.var_max_bitrate_kbps_ + alpha * var_value
        # 限制方差在合理范围内 [0.4, 2.5]
        if self.var_max_bitrate_kbps_ < 0.4:
            self.var_max_bitrate_kbps_ = 0.4
        if self.var_max_bitrate_kbps_ > 2.5:
            self.var_max_bitrate_kbps_ = 2.5


class PacketInfo:
    """
    数据包信息类
    存储单个数据包的所有相关信息，用于带宽估计
    """
    def __init__(self):
        self.payload_type = None  # 载荷类型
        self.sequence_number = None  # 序列号(int)，用于检测丢包
        self.send_timestamp = None  # 发送时间戳(int, ms)
        self.ssrc = None  # 同步源标识符(int)
        self.padding_length = None  # 填充长度(int, Bytes)
        self.header_length = None  # 头部长度(int, Bytes)
        self.receive_timestamp = None  # 接收时间戳(int, ms)
        self.payload_size = None  # 载荷大小(int, Bytes)
        self.bandwidth_prediction = None  # 带宽预测值(int, bps)

class PacketGroup:
    """
    数据包组类
    将发送时间接近的数据包分为一组，便于计算延迟梯度
    通过分组可以减少单个包的延迟抖动对估计的影响
    """
    def __init__(self, pkt_group):
        """
        初始化数据包组
        Args:
            pkt_group: 数据包列表
        """
        self.pkts = pkt_group  # 组内所有数据包
        self.arrival_time_list = [pkt.receive_timestamp for pkt in pkt_group]  # 到达时间列表
        self.send_time_list = [pkt.send_timestamp for pkt in pkt_group]  # 发送时间列表
        self.pkt_group_size = sum([pkt.size for pkt in pkt_group])  # 组的总大小(Bytes)
        self.pkt_num_in_group = len(pkt_group)  # 组内数据包数量
        self.complete_time = self.arrival_time_list[-1]  # 组完成时间（最后一个包到达时间）
        self.transfer_duration = self.arrival_time_list[-1] - self.arrival_time_list[0]  # 传输持续时间
