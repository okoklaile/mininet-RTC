#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class PacketRecord:
    # feature_interval can be modified
    def __init__(self, base_delay_ms=0):
        self.base_delay_ms = base_delay_ms
        self.reset()

    def reset(self):
        self.packet_num = 0
        self.packet_list = []  # ms
        self.last_seqNo = {}
        self.timer_delta = None  # ms
        self.min_seen_delay = self.base_delay_ms  # ms
        # ms, record the rtime of the last packet in last interval,
        self.last_interval_rtime = None

    def clear(self):
        self.packet_num = 0
        if self.packet_list:
            self.last_interval_rtime = self.packet_list[-1]['timestamp']
        self.packet_list = []

    def on_receive(self, packet_info):
        # 如果接收到乱序的包，直接忽略
        if (len(self.packet_list) > 0 
            and packet_info.receive_timestamp < self.packet_list[-1]['timestamp']):
            # 静默忽略乱序包
            return
        
        # Calculate the loss count
        loss_count = 0
        if packet_info.ssrc in self.last_seqNo:
            loss_count = max(0,
                packet_info.sequence_number - self.last_seqNo[packet_info.ssrc] - 1)
        self.last_seqNo[packet_info.ssrc] = packet_info.sequence_number

        # Calculate packet delay
        if self.timer_delta is None:
             # shift delay of the first packet to base delay
            self.timer_delta = self.base_delay_ms - (packet_info.receive_timestamp - packet_info.send_timestamp)
        delay = self.timer_delta + packet_info.receive_timestamp - packet_info.send_timestamp

        self.min_seen_delay = min(delay, self.min_seen_delay)
        
        # Check the last interval rtime
        if self.last_interval_rtime is None:
            self.last_interval_rtime = packet_info.receive_timestamp

        # Record result in current packet
        packet_result = {
            'timestamp': packet_info.receive_timestamp,  # ms
            'delay': delay,  # ms
            'payload_byte': packet_info.payload_size,  # B
            'loss_count': loss_count,  # p
            'bandwidth_prediction': packet_info.bandwidth_prediction,  # bps
            'payload_type': packet_info.payload_type  # 添加 payload_type 用于过滤
        }
        self.packet_list.append(packet_result)
        self.packet_num += 1

    def _get_result_list(self, interval, key, filter_payload_type=None):
        """
        获取结果列表
        Args:
            interval: 时间间隔(ms)，0表示使用所有数据
            key: 要提取的字段名
            filter_payload_type: 如果指定，只返回该 payload_type 的包（例如 125 表示视频包）
        """
        if self.packet_num == 0:
            return []

        result_list = []
        if interval == 0:
            interval = self.packet_list[-1]['timestamp'] -\
                self.last_interval_rtime
        start_time = self.packet_list[-1]['timestamp'] - interval
        index = self.packet_num - 1
        while index >= 0 and self.packet_list[index]['timestamp'] > start_time:
            # 如果指定了 filter_payload_type，只统计匹配的包
            if filter_payload_type is None or self.packet_list[index].get('payload_type') == filter_payload_type:
                result_list.append(self.packet_list[index][key])
            index -= 1
        return result_list

    def calculate_average_delay(self, interval=0, filter_payload_type=None):
        '''
        Calulate the average delay in the last interval time,
        interval=0 means based on the whole packets
        The unit of return value: ms
        Args:
            filter_payload_type: 如果指定，只统计该 payload_type 的包（例如 125 表示视频包）
        '''
        delay_list = self._get_result_list(interval=interval, key='delay', filter_payload_type=filter_payload_type)
        if delay_list:
            return np.mean(delay_list) - self.base_delay_ms
            # return np.mean(delay_list)
        else:
            return 0

    def calculate_loss_ratio(self, interval=0, filter_payload_type=None):
        '''
        Calulate the loss ratio in the last interval time,
        interval=0 means based on the whole packets
        The unit of return value: packet/packet
        Args:
            filter_payload_type: 如果指定，只统计该 payload_type 的包（例如 125 表示视频包）
        '''
        loss_list = self._get_result_list(interval=interval, key='loss_count', filter_payload_type=filter_payload_type)
        if loss_list:
            loss_num = np.sum(loss_list)
            received_num = len(loss_list)
            return loss_num / (loss_num + received_num)
        else:
            return 0

    def calculate_receiving_rate(self, interval=0, filter_payload_type=None):
        '''
        Calulate the receiving rate in the last interval time,
        interval=0 means based on the whole packets
        The unit of return value: bps
        Args:
            filter_payload_type: 如果指定，只统计该 payload_type 的包（例如 125 表示视频包）
        '''
        received_size_list = self._get_result_list(interval=interval, key='payload_byte', filter_payload_type=filter_payload_type)
        if received_size_list:
            received_nbytes = np.sum(received_size_list)
            if interval == 0:
                interval = self.packet_list[-1]['timestamp'] -\
                    self.last_interval_rtime
            return received_nbytes * 8 / interval * 1000
        else:
            return 0

    def calculate_latest_prediction(self):
        '''
        The unit of return value: bps
        '''
        if self.packet_num > 0:
            return self.packet_list[-1]['bandwidth_prediction']
        else:
            return 0
