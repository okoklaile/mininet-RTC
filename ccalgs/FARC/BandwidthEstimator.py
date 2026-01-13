#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FARC 带宽估计器
使用 Fast Actor and Not-So-Furious Critic 模型进行带宽预测
使用 ONNX Runtime 进行推理
"""
import os
import sys
import numpy as np
import warnings

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 导入 PacketInfo 和 PacketRecord
from packet_info import PacketInfo
from packet_record import PacketRecord

# 导入 onnxruntime
try:
    import onnxruntime as ort
    # 静默 ONNX Runtime 的 CUDA 警告（Docker 环境中常见）
    ort.set_default_logger_severity(3)  # 3 = ERROR level
except ImportError:
    raise ImportError("需要安装 onnxruntime: pip install onnxruntime-gpu 或 pip install onnxruntime")

# 调试标志 - 可以通过环境变量控制
DEBUG = os.getenv('FARC_DEBUG', '0') == '1'

def debug_print(msg):
    """调试输出"""
    if DEBUG:
        print(f"[FARC DEBUG] {msg}", file=sys.stderr, flush=True)


class Estimator:
    """
    FARC 带宽估计器
    使用 Fast Actor and Not-So-Furious Critic 模型进行带宽预测
    使用 ONNX Runtime 进行推理
    """
    def __init__(self, model_path=None, step_time=60):
        """
        初始化带宽估计器
        
        Args:
            model_path: ONNX 模型文件路径，默认使用当前目录下的 fast_and_furious_model.onnx
            step_time: 推理步长(毫秒)，默认60ms（FARC原始设计）
        """
        debug_print(f"初始化 FARC 估计器，step_time={step_time}ms")
        
        # 初始化隐藏状态
        self.hidden_size = 128  # FARC 使用 128 维隐藏状态
        self.input_size = 150   # 输入特征维度
        self.step_time = step_time  # 推理步长
        
        # 设置模型路径
        if model_path is None:
            model_path = os.path.join(current_dir, "fast_and_furious_model.onnx")
        
        if not model_path.endswith('.onnx'):
            raise ValueError(f"模型文件必须是 ONNX 格式 (.onnx)，当前路径: {model_path}")
        
        debug_print(f"模型路径: {model_path}")
        debug_print(f"模型文件存在: {os.path.exists(model_path)}")
        
        print(f"[FARC] 加载 ONNX 模型: {model_path}")
        
        # 加载 ONNX 模型
        self.model = self._load_onnx_model(model_path)
        
        # 初始化隐藏状态
        self.reset_states()
        
        # PacketRecord 用于统计特征
        self.packet_record = PacketRecord()
        
        # 特征提取参数（使用两个观察窗口：短期和长期）
        self.observation_windows_ms = [100, 500]  # 100ms短期窗口，500ms长期窗口
        
        # 初始化带宽预测值
        self.bandwidth_prediction = 1000000  # 初始值 1 Mbps
        
        # 历史数据记录（用于构建150维特征）
        self.feature_history = []  # 存储最近10个时间步的特征
        self.history_length = 10  # 保留10个时间步的历史
        
        # 统计信息
        self.report_count = 0  # 调用 report_states 的次数
        self.estimate_count = 0  # 调用 get_estimated_bandwidth 的次数
        
        print(f"[FARC] 模型加载成功！")
        debug_print("初始化完成")
    
    def _load_onnx_model(self, model_path):
        """
        加载 ONNX 模型
        
        Args:
            model_path: ONNX 模型文件路径
            
        Returns:
            ONNX Runtime InferenceSession
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建 ONNX Runtime 会话
        # 设置会话选项以使用 GPU（如果可用）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 检查可用的执行提供者
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        
        print(f"[FARC] 可用的执行提供者: {providers}")
        
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # 打印模型输入输出信息
        print(f"[FARC] 模型输入:")
        for inp in session.get_inputs():
            print(f"  - {inp.name}: {inp.shape} ({inp.type})")
        print(f"[FARC] 模型输出:")
        for out in session.get_outputs():
            print(f"  - {out.name}: {out.shape} ({out.type})")
        
        return session
    
    def reset_states(self):
        """重置隐藏状态"""
        self.hidden_state = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.cell_state = np.zeros((1, self.hidden_size), dtype=np.float32)
    
    def reset(self):
        """重置估计器（包括隐藏状态和 PacketRecord）"""
        self.reset_states()
        self.packet_record.reset()
        self.feature_history = []
        self.bandwidth_prediction = 1000000
    
    def report_states(self, stats: dict):
        """
        接收并记录数据包信息
        符合库的标准接口，接收数据包统计信息并更新内部状态
        
        Args:
            stats: 数据包统计信息字典，包含以下字段：
            {
                "send_time_ms": uint,        # 发送时间戳(毫秒)
                "arrival_time_ms": uint,     # 到达时间戳(毫秒)
                "payload_type": int,         # 载荷类型
                "sequence_number": uint,     # 序列号
                "ssrc": int,                 # 同步源标识符
                "padding_length": uint,      # 填充长度(字节)
                "header_length": uint,       # 头部长度(字节)
                "payload_size": uint         # 载荷大小(字节)
            }
        """
        self.report_count += 1
        
        if self.report_count <= 5 or self.report_count % 100 == 0:
            debug_print(f"report_states 调用#{self.report_count}: payload_size={stats.get('payload_size')}, " 
                       f"arrival_time={stats.get('arrival_time_ms')}")
        
        # 构造 PacketInfo 对象
        packet_info = PacketInfo()
        packet_info.payload_type = stats["payload_type"]
        packet_info.ssrc = stats["ssrc"]
        packet_info.sequence_number = stats["sequence_number"]
        packet_info.send_timestamp = stats["send_time_ms"]
        packet_info.receive_timestamp = stats["arrival_time_ms"]
        packet_info.padding_length = stats["padding_length"]
        packet_info.header_length = stats["header_length"]
        packet_info.payload_size = stats["payload_size"]
        packet_info.bandwidth_prediction = self.bandwidth_prediction
        
        # 更新 packet_record 用于统计网络指标
        self.packet_record.on_receive(packet_info)
        
        if self.report_count <= 5:
            debug_print(f"packet_record.packet_num={self.packet_record.packet_num}")
    
    def get_estimated_bandwidth(self) -> int:
        """
        计算并返回带宽估计值
        符合库的标准接口
        
        工作流程：
        1. 从 packet_record 中提取当前时间窗口的网络统计特征
        2. 构建150维观察特征向量
        3. 使用 FARC 模型进行推理
        4. 返回带宽预测值
        
        Returns:
            bandwidth_prediction: 带宽预测值(bps)
        """
        self.estimate_count += 1
        
        debug_print(f"=== get_estimated_bandwidth 调用#{self.estimate_count} ===")
        debug_print(f"当前packet_num={self.packet_record.packet_num}")
        debug_print(f"当前bandwidth_prediction={self.bandwidth_prediction}")
        
        # 如果数据包数量不足，返回当前估计值
        if self.packet_record.packet_num < 5:
            debug_print(f"数据包数量不足(<5)，返回默认值: {self.bandwidth_prediction}")
            return int(self.bandwidth_prediction)
        
        # 1. 从 packet_record 中提取网络统计特征
        receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
        delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        
        debug_print(f"网络统计: receiving_rate={receiving_rate:.0f} bps, delay={delay:.2f} ms, loss_ratio={loss_ratio:.4f}")
        
        # 计算排队延迟（当前延迟减去最小延迟）
        queuing_delay = max(0, delay - self.packet_record.min_seen_delay)
        
        # 2. 构建单个时间步的特征（15维）
        current_features = {
            'receiving_rate': receiving_rate,
            'packet_num': len([p for p in self.packet_record.packet_list 
                              if p['timestamp'] > self.packet_record.packet_list[-1]['timestamp'] - self.step_time]),
            'received_bytes': sum([p['payload_byte'] for p in self.packet_record.packet_list 
                                  if p['timestamp'] > self.packet_record.packet_list[-1]['timestamp'] - self.step_time]),
            'queuing_delay': queuing_delay,
            'delay': delay,
            'min_delay': self.packet_record.min_seen_delay,
            'delay_ratio': delay / self.packet_record.min_seen_delay if self.packet_record.min_seen_delay > 0 else 1.0,
            'delay_diff': queuing_delay,
            'interarrival_time': 0,  # 简化处理
            'jitter': 0,  # 简化处理
            'loss_ratio': loss_ratio,
            'avg_lost_packets': 0,  # 简化处理
            'video_prob': 0,  # 简化处理
            'audio_prob': 0,  # 简化处理
            'probing_prob': 0,  # 简化处理
        }
        
        # 添加到历史记录
        self.feature_history.append(current_features)
        if len(self.feature_history) > self.history_length:
            self.feature_history.pop(0)
        
        # 3. 构建150维特征向量（15个特征 x 10个时间步）
        obs = self._build_observation()
        
        if self.estimate_count <= 5:
            debug_print(f"观察向量: shape={obs.shape}, min={obs.min():.2f}, max={obs.max():.2f}, mean={obs.mean():.2f}")
        
        # 4. 使用 ONNX Runtime 进行推理
        obs_input = obs.reshape(1, 1, -1).astype(np.float32)
        
        feed_dict = {
            'obs': obs_input,
            'hidden_states': self.hidden_state,
            'cell_states': self.cell_state
        }
        
        debug_print("开始模型推理...")
        
        # 运行推理
        try:
            outputs = self.model.run(None, feed_dict)
            action, self.hidden_state, self.cell_state = outputs
            
            debug_print(f"模型输出: action.shape={action.shape}, action[0,0]={action[0,0]}")
            
            # 提取带宽预测（取第一个输出）
            bandwidth_prediction = float(action[0, 0, 0])
            
            debug_print(f"原始预测值: {bandwidth_prediction:.0f} bps ({bandwidth_prediction/1e6:.2f} Mbps)")
            
            # 限制范围（防止预测值过大或过小）
            bandwidth_prediction = max(80000, min(bandwidth_prediction, 20000000))  # 80kbps - 20Mbps
            
            debug_print(f"限制后预测值: {bandwidth_prediction:.0f} bps ({bandwidth_prediction/1e6:.2f} Mbps)")
            
            # 更新内部状态
            self.bandwidth_prediction = bandwidth_prediction
            
            # 清空当前时间窗口的数据包记录
            self.packet_record.clear()
            
            debug_print(f"返回带宽估计: {int(bandwidth_prediction)} bps")
            
            return int(bandwidth_prediction)
            
        except Exception as e:
            debug_print(f"模型推理出错: {e}")
            import traceback
            debug_print(traceback.format_exc())
            # 出错时返回上次的预测值
            return int(self.bandwidth_prediction)
    
    def _build_observation(self):
        """
        从历史特征中构建150维观察特征向量
        
        特征组成（每组10个时间步）：
        1. Receiving rate (10)
        2. Number of received packets (10)
        3. Received bytes (10)
        4. Queuing delay (10)
        5. Delay (10)
        6. Minimum seen delay (10)
        7. Delay ratio (10)
        8. Delay average minimum difference (10)
        9. Interarrival time (10)
        10. Jitter (10)
        11. Packet loss ratio (10)
        12. Average number of lost packets (10)
        13. Video packets probability (10)
        14. Audio packets probability (10)
        15. Probing packets probability (10)
        
        Returns:
            features: 150维特征向量
        """
        features = np.zeros(150, dtype=np.float32)
        
        # 如果历史不足10个，用最早的值填充前面的位置
        history_to_use = self.feature_history.copy()
        while len(history_to_use) < self.history_length:
            if len(history_to_use) > 0:
                history_to_use.insert(0, history_to_use[0])
            else:
                # 如果完全没有历史，使用零值
                history_to_use.append({
                    'receiving_rate': 0,
                    'packet_num': 0,
                    'received_bytes': 0,
                    'queuing_delay': 0,
                    'delay': 0,
                    'min_delay': 0,
                    'delay_ratio': 1.0,
                    'delay_diff': 0,
                    'interarrival_time': 0,
                    'jitter': 0,
                    'loss_ratio': 0,
                    'avg_lost_packets': 0,
                    'video_prob': 0,
                    'audio_prob': 0,
                    'probing_prob': 0,
                })
        
        # 按照特征类型填充（每个特征占用10个时间步）
        for t in range(self.history_length):
            feat = history_to_use[t]
            # 1. Receiving rate
            features[0 + t] = feat['receiving_rate']
            # 2. Number of received packets
            features[10 + t] = feat['packet_num']
            # 3. Received bytes
            features[20 + t] = feat['received_bytes']
            # 4. Queuing delay
            features[30 + t] = feat['queuing_delay']
            # 5. Delay
            features[40 + t] = feat['delay']
            # 6. Minimum seen delay
            features[50 + t] = feat['min_delay']
            # 7. Delay ratio
            features[60 + t] = feat['delay_ratio']
            # 8. Delay average minimum difference
            features[70 + t] = feat['delay_diff']
            # 9. Interarrival time
            features[80 + t] = feat['interarrival_time']
            # 10. Jitter
            features[90 + t] = feat['jitter']
            # 11. Packet loss ratio
            features[100 + t] = feat['loss_ratio']
            # 12. Average number of lost packets
            features[110 + t] = feat['avg_lost_packets']
            # 13. Video packets probability
            features[120 + t] = feat['video_prob']
            # 14. Audio packets probability
            features[130 + t] = feat['audio_prob']
            # 15. Probing packets probability
            features[140 + t] = feat['probing_prob']
        
        return features


# 为了向后兼容，提供 BandwidthEstimator 别名
BandwidthEstimator = Estimator


def main():
    """
    测试函数 - 符合库的接口规范
    """
    print("=" * 60)
    print("FARC 带宽估计器测试")
    print("=" * 60)
    
    # 创建估计器
    estimator = Estimator()
    
    print("\n模拟数据包接收...")
    
    # 模拟接收一系列数据包
    current_time = 0
    send_time = 0
    sequence_number = 0
    
    # 模拟60ms的数据包接收（FARC的默认步长）
    for i in range(20):
        # 模拟一个数据包
        packet_stats = {
            "send_time_ms": send_time,
            "arrival_time_ms": current_time,
            "payload_type": 125,  # 视频包
            "sequence_number": sequence_number,
            "ssrc": 12345,
            "padding_length": 0,
            "header_length": 12,
            "payload_size": 1200,  # 1200字节
        }
        
        # 报告数据包状态
        estimator.report_states(packet_stats)
        
        # 更新时间和序列号
        current_time += 20  # 每20ms一个包
        send_time += 20
        sequence_number += 1
        
        # 每隔60ms（3个包）获取一次带宽估计
        if (i + 1) % 3 == 0:
            bandwidth = estimator.get_estimated_bandwidth()
            print(f"  时刻 {current_time}ms: 带宽估计 = {bandwidth} bps ({bandwidth/1e6:.2f} Mbps)")
    
    print("\n最终带宽预测:")
    final_bandwidth = estimator.get_estimated_bandwidth()
    print(f"  带宽预测: {final_bandwidth} bps ({final_bandwidth/1e6:.2f} Mbps)")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

