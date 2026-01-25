#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BC-GCC BandwidthEstimator 测试脚本
"""
import sys
sys.path.insert(0, '/home/wyq/桌面/mininet-RTC/ccalgs/BC-GCC')
from BandwidthEstimator import Estimator

def test_estimator():
    """测试 BandwidthEstimator"""
    print("="*70)
    print("BC-GCC BandwidthEstimator 测试")
    print("="*70)
    
    # 初始化
    estimator = Estimator()
    
    # 模拟不同网络条件
    scenarios = [
        {
            "name": "良好网络 (1 Mbps, 低延迟)",
            "base_delay": 50,
            "delay_var": 5,
            "packet_size": 1200,
            "packet_interval": 10,  # 10ms发送间隔
            "num_packets": 30,
        },
        {
            "name": "中等网络 (500 kbps, 中延迟)",
            "base_delay": 100,
            "delay_var": 15,
            "packet_size": 1200,
            "packet_interval": 20,
            "num_packets": 30,
        },
        {
            "name": "拥塞网络 (200 kbps, 高延迟)",
            "base_delay": 200,
            "delay_var": 50,
            "packet_size": 1200,
            "packet_interval": 50,
            "num_packets": 30,
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"场景: {scenario['name']}")
        print(f"{'='*70}")
        
        estimator.reset()
        
        send_time = 0
        arrival_time = 0
        seq_num = 0
        
        # 发送数据包
        for i in range(scenario['num_packets']):
            import random
            
            # 计算延迟
            delay = scenario['base_delay'] + random.randint(-scenario['delay_var'], scenario['delay_var'])
            
            stats = {
                "send_time_ms": send_time,
                "arrival_time_ms": arrival_time + delay,
                "payload_type": 125,
                "sequence_number": seq_num,
                "ssrc": 12345,
                "padding_length": 0,
                "header_length": 12,
                "payload_size": scenario['packet_size'],
            }
            
            estimator.report_states(stats)
            
            # 每隔一段时间获取一次带宽估计
            if (i + 1) % 10 == 0:
                bandwidth = estimator.get_estimated_bandwidth()
                print(f"   包 {i+1}/{scenario['num_packets']}: 带宽估计 = {bandwidth/1e6:.3f} Mbps")
            
            send_time += scenario['packet_interval']
            arrival_time = send_time  # 基准到达时间
            seq_num += 1
        
        # 最终估计
        final_bandwidth = estimator.get_estimated_bandwidth()
        print(f"\n   ✅ 最终带宽估计: {final_bandwidth/1e6:.3f} Mbps ({final_bandwidth} bps)")
    
    print(f"\n{'='*70}")
    print("✅ 所有测试完成")
    print(f"{'='*70}")

if __name__ == '__main__':
    test_estimator()
