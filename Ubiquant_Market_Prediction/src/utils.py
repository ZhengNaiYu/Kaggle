"""通用工具函数"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem_usage(df):
    """降低DataFrame内存占用"""
    import pandas as pd

    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            else:
                for dtype in [np.float32, np.float64]:
                    if c_min > np.finfo(dtype).min and c_max < np.finfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory: {start_mem:.1f}MB -> {end_mem:.1f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    return df


def get_device(device_str: str = "cuda") -> torch.device:
    """获取计算设备"""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
