"""评估指标模块"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr


def pearson_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算Pearson相关系数"""
    return pearsonr(y_true, y_pred)[0]


def time_grouped_pearson(
    time_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """按time_id分组计算Pearson相关系数的均值（竞赛核心指标）"""
    df = pd.DataFrame({
        "time_id": time_ids,
        "target": y_true,
        "pred": y_pred,
    })
    scores = df.groupby("time_id")[["target", "pred"]].apply(
        lambda g: pearsonr(g["target"], g["pred"])[0] if len(g) > 1 else 0.0
    )
    return scores.mean()


class PearsonCorrLoss(torch.nn.Module):
    """基于Pearson相关系数的损失函数: 1 - corr(pred, target)"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze()
        target = target.squeeze()
        vx = pred - pred.mean()
        vy = target - target.mean()
        corr = (vx * vy).sum() / (
            torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + 1e-8
        )
        return 1.0 - corr


class CombinedLoss(torch.nn.Module):
    """MSE + PearsonCorr 混合损失"""

    def __init__(self, mse_weight: float = 0.5, corr_weight: float = 0.5):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.corr = PearsonCorrLoss()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(pred, target) + self.corr_weight * self.corr(pred, target)
