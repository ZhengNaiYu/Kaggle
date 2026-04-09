"""数据加载与预处理模块"""

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import DataConfig, TrainConfig
from src.utils import reduce_mem_usage


# ────────────────────────────── Dataset ──────────────────────────────


class UbiquantDataset(Dataset):
    """通用PyTorch Dataset，支持带/不带 investment_id"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        investment_ids: Optional[np.ndarray] = None,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.investment_ids = (
            torch.tensor(investment_ids, dtype=torch.long)
            if investment_ids is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        feat = self.features[idx]
        target = self.targets[idx]
        if self.investment_ids is not None:
            return feat, self.investment_ids[idx], target
        return feat, target


# ────────────────────────────── DataModule ──────────────────────────────


class UbiquantDataModule:
    """数据加载和交叉验证拆分"""

    def __init__(self, data_cfg: DataConfig, train_cfg: TrainConfig):
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.df: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        self.investment_id_map: Dict[int, int] = {}

    # ── 加载 ──

    def load(self) -> pd.DataFrame:
        path = self.data_cfg.data_dir / self.data_cfg.train_file
        print(f"Loading data from {path} ...")
        self.df = pd.read_parquet(path)
        if self.data_cfg.min_time_id is not None:
            self.df = self.df[self.df[self.data_cfg.time_col] >= self.data_cfg.min_time_id]
        self.df = reduce_mem_usage(self.df)

        # 随机采样（快速验证模式）
        if self.data_cfg.sample_frac < 1.0:
            n_before = len(self.df)
            self.df = self.df.sample(frac=self.data_cfg.sample_frac, random_state=42).reset_index(drop=True)
            print(f"[Fast-dev] Sampled {len(self.df):,} rows ({self.data_cfg.sample_frac*100:.3f}%) from {n_before:,}")

        # 构建 investment_id -> 连续整数 映射
        unique_ids = sorted(self.df[self.data_cfg.asset_col].unique())
        self.investment_id_map = {v: i + 1 for i, v in enumerate(unique_ids)}  # 0 留给 unknown
        self.num_investment_ids = len(unique_ids) + 1

        print(f"Data shape: {self.df.shape}")
        print(f"Unique investment_ids: {len(unique_ids)}, time_ids: {self.df[self.data_cfg.time_col].nunique()}")
        return self.df

    # ── 交叉验证拆分 ──

    def get_cv_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        assert self.df is not None, "Call load() first"
        n_folds = self.train_cfg.n_folds
        method = self.train_cfg.cv_method

        # Edge case: n_folds < 2 (e.g., --fast-dev)
        if n_folds < 2:
            n_samples = len(self.df)
            split_idx = int(0.8 * n_samples)  # 80/20 split
            train_idx = np.arange(split_idx)
            test_idx = np.arange(split_idx, n_samples)
            return [(train_idx, test_idx)]

        if method == "group":
            kf = GroupKFold(n_splits=n_folds)
            groups = self.df[self.data_cfg.time_col].values
            splits = list(kf.split(self.df, groups=groups))
        elif method == "time_series":
            kf = TimeSeriesSplit(n_splits=n_folds)
            splits = list(kf.split(self.df))
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.train_cfg.seed)
            splits = list(kf.split(self.df))
        return splits

    # ── 特征工程 ──

    def add_combination_features(self, df: pd.DataFrame, pairs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """添加特征交叉组合"""
        new_cols = []
        for pair in pairs:
            f1, f2 = pair.split("-")
            col_name = f"{f1}_plus_{f2}"
            df[col_name] = df[f1].values + df[f2].values
            new_cols.append(col_name)
        return df, new_cols

    # ── 构建 DataLoader ──

    def build_fold_data(
        self,
        fold_idx: int,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        extra_feature_cols: Optional[List[str]] = None,
        include_ids: bool = True,
        fit_scaler: bool = True,
    ) -> Dict:
        """获取指定fold的训练/验证数据

        Returns:
            dict with keys: X_train, X_val, y_train, y_val, time_val,
                            id_train, id_val (if include_ids)
        """
        train_idx, val_idx = splits[fold_idx]
        feat_cols = list(self.data_cfg.feature_cols)
        if extra_feature_cols:
            feat_cols += extra_feature_cols

        X_train = self.df.iloc[train_idx][feat_cols].values.astype(np.float32)
        X_val = self.df.iloc[val_idx][feat_cols].values.astype(np.float32)
        y_train = self.df.iloc[train_idx][self.data_cfg.target_col].values.astype(np.float32)
        y_val = self.df.iloc[val_idx][self.data_cfg.target_col].values.astype(np.float32)
        time_val = self.df.iloc[val_idx][self.data_cfg.time_col].values

        if fit_scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)

        result = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "time_val": time_val,
        }

        if include_ids:
            id_train = self.df.iloc[train_idx][self.data_cfg.asset_col].map(self.investment_id_map).values
            id_val = self.df.iloc[val_idx][self.data_cfg.asset_col].map(self.investment_id_map).values
            result["id_train"] = id_train
            result["id_val"] = id_val

        return result

    def make_dataloaders(
        self,
        fold_data: Dict,
        batch_size: int,
        include_ids: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """从fold数据构建PyTorch DataLoader"""
        train_ds = UbiquantDataset(
            fold_data["X_train"],
            fold_data["y_train"],
            fold_data.get("id_train") if include_ids else None,
        )
        val_ds = UbiquantDataset(
            fold_data["X_val"],
            fold_data["y_val"],
            fold_data.get("id_val") if include_ids else None,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader
