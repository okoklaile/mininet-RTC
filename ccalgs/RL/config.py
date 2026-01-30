"""
Configuration for Online RL Bandwidth Estimator
基于 BC-GCC 的配置，扩展支持 QoE 特征，并避免与本目录下 `config` 模块产生循环依赖
"""

import os
import importlib.util


def _load_base_config():
    """
    动态加载上一级目录 BC-GCC 下的 config.py，返回其中的 Config 类，
    避免使用裸模块名 `config` 导致循环 import。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bc_gcc_config_path = os.path.join(base_dir, "..", "BC-GCC", "config.py")

    spec = importlib.util.spec_from_file_location("bc_gcc_config", bc_gcc_config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 BC-GCC 配置文件: {bc_gcc_config_path}")

    bc_gcc_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc_gcc_config)
    return bc_gcc_config.Config


BaseConfig = _load_base_config()

# 为了兼容 BC-GCC 模型内部使用的 `from config import Config`，
# 这里显式暴露同名的 Config 引用指向 BaseConfig。
Config = BaseConfig


class Config32D(BaseConfig):
    """32维特征配置（用于trial3模型，包含QoE特征）"""
    TOTAL_FEATURE_DIM = 32
    
    # 扩展保留字段，包含 QoE 特征
    RESERVED_FEATURES = BaseConfig.RESERVED_FEATURES + [
        'render_fps',           # 渲染帧率 (FPS)
        'freeze_rate',          # 卡顿率 (%)
        'e2e_delay_ms',         # 端到端延迟 (ms)
        'custom_9',             # 保留字段
        'custom_10',            # 保留字段
        'custom_11',            # 保留字段
        'custom_12',            # 保留字段
        'custom_13',            # 保留字段
        'custom_14',            # 保留字段
        'custom_15',            # 保留字段
        'custom_16',            # 保留字段
    ]
    
    # 扩展归一化参数，添加 QoE 特征的归一化范围
    NORM_STATS = BaseConfig.NORM_STATS.copy()
    NORM_STATS.update({
        # QoE 特征归一化范围
        'render_fps': {'min': 0, 'max': 60},           # 渲染帧率 0-60 FPS
        'freeze_rate': {'min': 0, 'max': 100},         # 卡顿率 0-100%
        'e2e_delay_ms': {'min': 0, 'max': 10000},      # 端到端延迟 0-10秒
    })
