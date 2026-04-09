"""主训练入口

用法:
    python train.py --model all          # 训练全部模型
    python train.py --model dnn          # 只训练DNN
    python train.py --model lgbm         # 只训练LightGBM
    python train.py --model ensemble     # 只训练集成DNN
    python train.py --model transformer  # 只训练Transformer
    python train.py --model autoencoder  # 只训练AutoEncoder合成信号
    python train.py --model dnn lgbm     # 训练多个指定模型
    python train.py --folds 3            # 指定fold数
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import (
    AutoEncoderConfig,
    DataConfig,
    DNNConfig,
    EnsembleConfig,
    LGBMConfig,
    TrainConfig,
    TransformerConfig,
)
from src.data_loader import UbiquantDataModule
from src.metrics import time_grouped_pearson
from src.utils import seed_everything


# ────────────────────────────── 训练调度 ──────────────────────────────


def train_dnn(dm: UbiquantDataModule, splits, train_cfg: TrainConfig, fast_dev: bool = False) -> Dict:
    from src.models.dnn import DNNTrainer

    cfg = DNNConfig()
    if fast_dev:
        cfg.epochs = 2
    trainer = DNNTrainer(cfg, train_cfg, num_ids=dm.num_investment_ids, n_features=len(dm.data_cfg.feature_cols))
    results = {}
    for fold_idx in range(len(splits)):
        fold_data = dm.build_fold_data(fold_idx, splits, include_ids=True)
        result = trainer.train_fold(fold_idx, fold_data, dm)
        results[f"fold_{fold_idx}"] = result
    avg_pearson = np.mean([r["pearson"] for r in results.values()])
    print(f"\n[DNN] Average Pearson: {avg_pearson:.6f}")
    results["avg_pearson"] = avg_pearson
    return results


def train_lgbm(dm: UbiquantDataModule, splits, train_cfg: TrainConfig, fast_dev: bool = False) -> Dict:
    from src.models.lgbm_model import LGBMTrainer

    cfg = LGBMConfig()
    if fast_dev:
        cfg.n_estimators = 50
    extra_cols = []
    if cfg.add_combination_features:
        dm.df, extra_cols = dm.add_combination_features(dm.df, cfg.combination_pairs)
        print(f"Added {len(extra_cols)} combination features: {extra_cols}")

    trainer = LGBMTrainer(cfg, train_cfg)
    results = {}
    for fold_idx in range(len(splits)):
        fold_data = dm.build_fold_data(fold_idx, splits, extra_feature_cols=extra_cols, include_ids=False)
        result = trainer.train_fold(fold_idx, fold_data)
        results[f"fold_{fold_idx}"] = result
    avg_pearson = np.mean([r["pearson"] for r in results.values()])
    print(f"\n[LGBM] Average Pearson: {avg_pearson:.6f}")
    results["avg_pearson"] = avg_pearson

    # 特征重要性
    importance = trainer.get_feature_importance()
    feat_names = list(dm.data_cfg.feature_cols) + extra_cols
    top_k = 20
    indices = np.argsort(importance)[::-1][:top_k]
    print(f"\nTop {top_k} features by importance:")
    for rank, idx in enumerate(indices):
        print(f"  {rank+1:2d}. {feat_names[idx]:20s} = {importance[idx]:.1f}")

    return results


def train_ensemble(dm: UbiquantDataModule, splits, train_cfg: TrainConfig, fast_dev: bool = False) -> Dict:
    from src.models.ensemble import EnsembleTrainer

    cfg = EnsembleConfig()
    if fast_dev:
        cfg.epochs = 2
    trainer = EnsembleTrainer(cfg, train_cfg, num_ids=dm.num_investment_ids, n_features=len(dm.data_cfg.feature_cols))
    results = {}
    for fold_idx in range(len(splits)):
        fold_data = dm.build_fold_data(fold_idx, splits, include_ids=True)
        result = trainer.train_fold(fold_idx, fold_data, dm)
        results[f"fold_{fold_idx}"] = result
    print(f"\n[Ensemble] Training complete")
    return results


def train_transformer(dm: UbiquantDataModule, splits, train_cfg: TrainConfig, fast_dev: bool = False) -> Dict:
    from src.models.transformer import TransformerTrainer

    cfg = TransformerConfig()
    if fast_dev:
        cfg.epochs = 2
    trainer = TransformerTrainer(cfg, train_cfg, num_ids=dm.num_investment_ids, n_features=len(dm.data_cfg.feature_cols))
    results = {}
    for fold_idx in range(len(splits)):
        fold_data = dm.build_fold_data(fold_idx, splits, include_ids=True)
        result = trainer.train_fold(fold_idx, fold_data, dm)
        results[f"fold_{fold_idx}"] = result
    avg_pearson = np.mean([r["pearson"] for r in results.values()])
    print(f"\n[Transformer] Average Pearson: {avg_pearson:.6f}")
    results["avg_pearson"] = avg_pearson
    return results


def train_autoencoder(dm: UbiquantDataModule, splits, train_cfg: TrainConfig, fast_dev: bool = False) -> Dict:
    from src.models.autoencoder import AutoEncoderTrainer

    cfg = AutoEncoderConfig()
    if fast_dev:
        cfg.ae_epochs = 2
        cfg.pred_epochs = 2
    trainer = AutoEncoderTrainer(cfg, train_cfg, num_ids=dm.num_investment_ids, n_features=len(dm.data_cfg.feature_cols))
    results = {}
    for fold_idx in range(len(splits)):
        fold_data = dm.build_fold_data(fold_idx, splits, include_ids=True)
        result = trainer.train_fold(fold_idx, fold_data, dm)
        results[f"fold_{fold_idx}"] = result
    avg_pearson = np.mean([r["pearson"] for r in results.values()])
    print(f"\n[AutoEncoder] Average Pearson: {avg_pearson:.6f}")
    results["avg_pearson"] = avg_pearson
    return results


# ────────────────────────────── CLI ──────────────────────────────

MODEL_REGISTRY = {
    "dnn": train_dnn,
    "lgbm": train_lgbm,
    "ensemble": train_ensemble,
    "transformer": train_transformer,
    "autoencoder": train_autoencoder,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Ubiquant Market Prediction Training")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="要训练的模型",
    )
    parser.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cv", type=str, default="group", choices=["group", "time_series", "kfold"])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--min-time-id", type=int, default=None, help="最小time_id (截取训练集)")
    parser.add_argument(
        "--sample-frac", type=float, default=1.0,
        help="采样比例 (0~1)，例如 0.0005 表示仅加载 0.05%% 的数据用于快速验证",
    )
    parser.add_argument(
        "--fast-dev", action="store_true",
        help="快速验证模式: 采样 0.05%% 数据，仅训练 1 fold，运行 2 epochs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # fast-dev 覆盖
    if args.fast_dev:
        args.sample_frac = 0.0005
        args.folds = 1
        args.batch_size = 256
        print("[Fast-dev] sample_frac=0.05%, folds=1, batch_size=256, epochs capped at 2")

    # 配置
    data_cfg = DataConfig(min_time_id=args.min_time_id, sample_frac=args.sample_frac)
    train_cfg = TrainConfig(
        seed=args.seed,
        n_folds=args.folds,
        cv_method=args.cv,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=Path(args.output_dir),
    )
    seed_everything(train_cfg.seed)
    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # device
    print(f"Using device: {train_cfg.device}")

    # 数据
    dm = UbiquantDataModule(data_cfg, train_cfg)
    dm.load()
    splits = dm.get_cv_splits()
    print(f"\nCV method: {train_cfg.cv_method}, {len(splits)} folds")

    # 训练
    models_to_train = list(MODEL_REGISTRY.keys()) if "all" in args.model else args.model
    all_results = {}

    for model_name in models_to_train:
        print(f"\n{'='*80}")
        print(f"  Training: {model_name.upper()}")
        print(f"{'='*80}")
        train_fn = MODEL_REGISTRY[model_name]
        result = train_fn(dm, splits, train_cfg, fast_dev=args.fast_dev)
        all_results[model_name] = result

    # 保存结果摘要（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"results_summary_{timestamp}.json"
    summary_path = train_cfg.log_dir / summary_filename

    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(summary_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
