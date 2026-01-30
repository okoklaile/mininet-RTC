"""
LSTM-based model wrapper for Online RL Bandwidth Estimation
基于 BC-GCC 的模型封装，避免与本目录下的 `model` 模块产生循环依赖
"""

import os
import importlib.util


def _load_bc_gcc_model():
    """
    动态加载上一级目录下 BC-GCC 的 model.py，
    并返回其中的 GCCBC_LSTM 类，避免使用裸模块名 `model` 造成循环 import。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bc_gcc_model_path = os.path.join(base_dir, "..", "BC-GCC", "model.py")

    spec = importlib.util.spec_from_file_location("bc_gcc_model", bc_gcc_model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 BC-GCC 模型文件: {bc_gcc_model_path}")

    bc_gcc_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc_gcc_model)
    return bc_gcc_model.GCCBC_LSTM


# 对外暴露与原来相同的类名
GCCBC_LSTM = _load_bc_gcc_model()
