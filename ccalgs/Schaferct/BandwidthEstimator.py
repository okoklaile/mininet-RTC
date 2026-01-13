#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schaferct Bandwidth Estimator

基于 IQL (Implicit Q-Learning) 的带宽估计器
使用 ONNX Runtime 进行推理
"""

import os
import sys
import numpy as np

# 尝试导入 onnxruntime
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False
    print("[Schaferct] 警告: 未安装 onnxruntime")

from packet_info import PacketInfo
from packet_record import PacketRecord


def debug_print(*args, **kwargs):
    """调试输出函数，通过环境变量 SCHAFERCT_DEBUG 控制"""
    if os.environ.get('SCHAFERCT_DEBUG', '0') == '1':
        import sys
        print("[Schaferct DEBUG]", *args, file=sys.stderr, flush=True, **kwargs)


class Estimator:
    """
    Schaferct 带宽估计器
    
    使用 IQL 模型进行带宽预测，基于历史网络统计信息
    """
    
    def __init__(self, model_path=None, step_time=60):
        """
        初始化估计器
        
        Args:
            model_path: ONNX 模型路径，默认使用同目录下的 Schaferct_model.onnx
            step_time: 时间步长（毫秒），默认 60ms
        """
        debug_print(f"初始化 Schaferct 估计器，step_time={step_time}ms")
        
        self.step_time = step_time
        self.call_count_report = 0
        self.call_count_estimate = 0
        
        # 模型配置
        self.state_dim = 150  # 观察维度
        self.history_length = 10  # 历史长度（150 = 15 features × 10 steps）
        
        # 加载 ONNX 模型
        if model_path is None:
            # 默认使用当前目录下的模型
            current_dir = os.path.dirname(os.path.abspath(__file__))
            #model_path = os.path.join(current_dir, "Schaferct_model.onnx")
            model_path = os.path.join(current_dir, "baseline.onnx")
        self.model_path = model_path
        debug_print(f"模型路径: {self.model_path}")
        debug_print(f"模型文件存在: {os.path.exists(self.model_path)}")
        
        if HAS_ONNXRUNTIME:
            self.ort_session = self._load_onnx_model(model_path)
        else:
            self.ort_session = None
            print("[Schaferct] 错误: 无法加载模型，onnxruntime 未安装")
        
        # 初始化 LSTM 状态 (1, 1)
        self.hidden_state = np.zeros((1, 1), dtype=np.float32)
        self.cell_state = np.zeros((1, 1), dtype=np.float32)
        
        # 初始化 packet record
        self.packet_record = PacketRecord()
        
        # 特征历史
        self.feature_history = []
        
        # 当前带宽预测值
        self.bandwidth_prediction = 1_000_000  # 默认 1 Mbps
        
        debug_print("初始化完成")
    
    def _load_onnx_model(self, model_path):
        """加载 ONNX 模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 尝试使用 CUDA，如果不可用则降级到 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, providers=providers)
        
        # 打印模型信息
        debug_print(f"模型加载成功")
        debug_print(f"输入:")
        for inp in session.get_inputs():
            debug_print(f"  {inp.name}: {inp.shape} ({inp.type})")
        debug_print(f"输出:")
        for out in session.get_outputs():
            debug_print(f"  {out.name}: {out.shape} ({out.type})")
        
        return session
    
    def reset_states(self):
        """重置 LSTM 状态"""
        self.hidden_state = np.zeros((1, 1), dtype=np.float32)
        self.cell_state = np.zeros((1, 1), dtype=np.float32)
        debug_print("LSTM 状态已重置")
    
    def reset(self):
        """重置所有状态"""
        self.reset_states()
        self.packet_record = PacketRecord()
        self.feature_history = []
        self.bandwidth_prediction = 1_000_000
        self.call_count_report = 0
        self.call_count_estimate = 0
        debug_print("估计器已完全重置")
    
    def report_states(self, stats: dict):
        """
        接收网络统计信息
        
        Args:
            stats: 包含网络统计的字典，格式参考 PacketInfo
        """
        self.call_count_report += 1
        debug_print(f"report_states 调用#{self.call_count_report}: payload_size={stats.get('payload_size', 0)}, arrival_time={stats.get('arrival_time_ms', 0)}")
        
        # 将 stats 转换为 PacketInfo 并添加到记录中
        packet_info = PacketInfo()
        packet_info.payload_type = stats.get("payload_type", 0)
        packet_info.ssrc = stats.get("ssrc", 0)
        packet_info.sequence_number = stats.get("sequence_number", 0)
        packet_info.send_timestamp = stats.get("send_time_ms", 0)
        packet_info.receive_timestamp = stats.get("arrival_time_ms", 0)
        packet_info.padding_length = stats.get("padding_length", 0)
        packet_info.header_length = stats.get("header_length", 0)
        packet_info.payload_size = stats.get("payload_size", 0)
        packet_info.bandwidth_prediction = self.bandwidth_prediction
        
        self.packet_record.on_receive(packet_info)
        debug_print(f"packet_record.packet_num={self.packet_record.packet_num}")
    
    def get_estimated_bandwidth(self) -> int:
        """
        获取当前的带宽估计值
        
        Returns:
            int: 带宽估计值（bps）
        """
        self.call_count_estimate += 1
        debug_print(f"=== get_estimated_bandwidth 调用#{self.call_count_estimate} ===")
        debug_print(f"当前packet_num={self.packet_record.packet_num}")
        debug_print(f"当前bandwidth_prediction={self.bandwidth_prediction}")
        
        # 如果没有足够的数据包，返回默认值
        if self.packet_record.packet_num < 5:
            debug_print(f"数据包数量不足(<5)，返回默认值: {self.bandwidth_prediction}")
            return self.bandwidth_prediction
        
        # 如果模型未加载，返回默认值
        if self.ort_session is None:
            debug_print("模型未加载，返回默认值")
            return self.bandwidth_prediction
        
        # 提取网络特征
        receiving_rate = self.packet_record.calculate_receiving_rate(interval=500)
        delay = self.packet_record.calculate_average_delay(interval=100)
        loss_ratio = self.packet_record.calculate_loss_ratio(interval=500)
        
        # 计算排队延迟
        base_delay = self.packet_record.min_seen_delay
        if self.packet_record.packet_num > 0:
            latest_packet = self.packet_record.packet_list[-1]
            queuing_delay = latest_packet['delay'] - base_delay
        else:
            queuing_delay = 0
        
        debug_print(f"网络统计: receiving_rate={receiving_rate} bps, delay={delay:.2f} ms, loss_ratio={loss_ratio:.4f}")
        
        # 构建当前时刻的特征（15维） - 参考FARC的特征定义
        # 1. 计算平均包大小和接收字节数
        if self.packet_record.packet_num > 0:
            total_bytes = sum(p['payload_byte'] for p in self.packet_record.packet_list)
            avg_packet_size = total_bytes / self.packet_record.packet_num
            received_bytes = total_bytes
        else:
            avg_packet_size = 0
            received_bytes = 0
        
        # 2. 计算时间间隔（interarrival time）
        if len(self.feature_history) > 0 and self.packet_record.packet_num > 0:
            current_time = self.packet_record.packet_list[-1]['timestamp']
            prev_features = self.feature_history[-1]
            interarrival_time = current_time - prev_features.get('last_timestamp', current_time)
        else:
            interarrival_time = self.step_time  # 默认时间步长
        
        # 3. 计算jitter（延迟抖动）
        if self.packet_record.packet_num > 1:
            delays = [p['delay'] for p in self.packet_record.packet_list]
            jitter = np.std(delays) if len(delays) > 1 else 0
        else:
            jitter = 0
        
        # 4. 延迟比率
        delay_ratio = delay / base_delay if base_delay > 0 else 1.0
        
        # 5. 丢包数
        loss_count = loss_ratio * self.packet_record.packet_num
        
        current_timestamp = self.packet_record.packet_list[-1]['timestamp'] if self.packet_record.packet_num > 0 else 0
        
        current_features = {
            'receiving_rate': receiving_rate,  # 特征0-9: bps
            'packet_count': self.packet_record.packet_num,  # 特征10-19: 包数量
            'received_bytes': received_bytes,  # 特征20-29: 字节数
            'queuing_delay': queuing_delay,  # 特征30-39: 排队延迟(ms)
            'delay': delay,  # 特征40-49: 延迟(ms)
            'base_delay': base_delay,  # 特征50-59: 基准延迟(ms)
            'delay_ratio': delay_ratio,  # 特征60-69: 延迟比率
            'delay_diff': queuing_delay,  # 特征70-79: 延迟差值(ms)
            'interarrival_time': interarrival_time,  # 特征80-89: 包间隔时间(ms)
            'jitter': jitter,  # 特征90-99: 抖动(ms)
            'loss_ratio': loss_ratio,  # 特征100-109: 丢包率
            'loss_count': loss_count,  # 特征110-119: 丢包数
            'avg_packet_size': avg_packet_size,  # 特征120-129: 平均包大小(bytes)
            'bandwidth_prediction': self.bandwidth_prediction,  # 特征130-139: 带宽预测(bps)
            'feature_15': 0.0,  # 特征140-149: 保留
            # 保存时间戳用于下次计算时间间隔
            'last_timestamp': current_timestamp,
        }
        
        # 添加到历史
        self.feature_history.append(current_features)
        if len(self.feature_history) > self.history_length:
            self.feature_history.pop(0)
        
        # 构建观察向量 (150维)
        obs = self._build_observation()
        debug_print(f"观察向量: shape={obs.shape}, min={obs.min():.2f}, max={obs.max():.2f}, mean={obs.mean():.2f}")
        
        # ONNX 推理
        try:
            debug_print("开始模型推理...")
            obs_input = obs.reshape(1, 1, -1).astype(np.float32)
            
            feed_dict = {
                'obs': obs_input,
                'hidden_states': self.hidden_state,
                'cell_states': self.cell_state
            }
            
            # 运行推理
            # 输出: [output(1,1,2), state_out(1,1), cell_out(1,1)]
            outputs = self.ort_session.run(None, feed_dict)
            output, self.hidden_state, self.cell_state = outputs
            
            debug_print(f"模型输出: output.shape={output.shape}, output[0,0,:]={output[0,0,:]}")
            
            # 提取预测值 (单位: bps) - 取第一个输出值
            bandwidth_prediction = float(output[0, 0, 0])
            debug_print(f"原始预测值: {bandwidth_prediction} bps ({bandwidth_prediction/1e6:.2f} Mbps)")
            
            # 限制在合理范围内 (80kbps ~ 20Mbps)
            bandwidth_prediction = max(80_000, min(20_000_000, bandwidth_prediction))
            debug_print(f"限制后预测值: {bandwidth_prediction} bps ({bandwidth_prediction/1e6:.2f} Mbps)")
            
            # 更新当前预测值
            self.bandwidth_prediction = int(bandwidth_prediction)
            
        except Exception as e:
            debug_print(f"推理失败: {e}")
            import traceback
            debug_print(traceback.format_exc())
        
        # 清空 packet record 为下一个时间窗口做准备
        self.packet_record.clear()
        
        debug_print(f"返回带宽估计: {self.bandwidth_prediction} bps")
        return self.bandwidth_prediction
    
    def _build_observation(self):
        """
        构建 150 维观察向量
        
        特征按类型分组，每个类型占10个连续位置（10个时间步）：
        1. Receiving rate (10)
        2. Delay (10)
        3. Loss ratio (10)
        4. Queuing delay (10)
        5. Avg packet size (10)
        6. Packet count (10)
        7. Time interval (10)
        8. Base delay (10)
        9. Bandwidth prediction (10)
        10-15. Reserved features (60)
        """
        obs = np.zeros(150, dtype=np.float32)
        
        # 如果历史不足10个，用最早的特征填充
        history_to_use = self.feature_history.copy()
        while len(history_to_use) < self.history_length:
            if len(history_to_use) > 0:
                history_to_use.insert(0, history_to_use[0])
            else:
                # 完全没有历史，使用默认值
                history_to_use.append({
                    'receiving_rate': 0,
                    'packet_count': 0,
                    'received_bytes': 0,
                    'queuing_delay': 0,
                    'delay': 0,
                    'base_delay': 0,
                    'delay_ratio': 1.0,
                    'delay_diff': 0,
                    'interarrival_time': self.step_time,
                    'jitter': 0,
                    'loss_ratio': 0,
                    'loss_count': 0,
                    'avg_packet_size': 0,
                    'bandwidth_prediction': 1000000,
                    'feature_15': 0,
                    'last_timestamp': 0,
                })
        
        # 按特征类型填充（每个特征占10个时间步）
        for t in range(self.history_length):
            feat = history_to_use[t]
            obs[0 + t] = feat['receiving_rate']  # 0-9: bps
            obs[10 + t] = feat['packet_count']  # 10-19: 包数量
            obs[20 + t] = feat['received_bytes']  # 20-29: 字节数
            obs[30 + t] = feat['queuing_delay']  # 30-39: 排队延迟
            obs[40 + t] = feat['delay']  # 40-49: 延迟
            obs[50 + t] = feat['base_delay']  # 50-59: 基准延迟
            obs[60 + t] = feat['delay_ratio']  # 60-69: 延迟比率
            obs[70 + t] = feat['delay_diff']  # 70-79: 延迟差值
            obs[80 + t] = feat['interarrival_time']  # 80-89: 包间隔
            obs[90 + t] = feat['jitter']  # 90-99: 抖动
            obs[100 + t] = feat['loss_ratio']  # 100-109: 丢包率
            obs[110 + t] = feat['loss_count']  # 110-119: 丢包数
            obs[120 + t] = feat['avg_packet_size']  # 120-129: 平均包大小
            obs[130 + t] = feat['bandwidth_prediction']  # 130-139: 带宽预测
            obs[140 + t] = feat['feature_15']  # 140-149: 保留
        
        return obs


# 测试代码
def main():
    """测试 Schaferct 估计器"""
    print("=" * 60)
    print("Schaferct Bandwidth Estimator 测试")
    print("=" * 60)
    
    # 创建估计器实例
    estimator = Estimator()
    
    # 模拟一些数据包
    for i in range(10):
        stats = {
            "payload_type": 0,
            "ssrc": 12345,
            "sequence_number": i,
            "send_time_ms": 1000 + i * 20,
            "arrival_time_ms": 1005 + i * 20,
            "padding_length": 0,
            "header_length": 12,
            "payload_size": 1200,
        }
        estimator.report_states(stats)
        
        # 每隔几个包获取一次带宽估计
        if (i + 1) % 5 == 0:
            bandwidth = estimator.get_estimated_bandwidth()
            print(f"Step {i+1}: 估计带宽 = {bandwidth/1e6:.2f} Mbps")
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

