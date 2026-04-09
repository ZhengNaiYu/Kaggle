"""全局配置模块"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    data_dir: Path = Path("./data")
    train_file: str = "train.parquet"
    n_features: int = 300
    feature_cols: List[str] = field(default_factory=lambda: [f"f_{i}" for i in range(300)])
    target_col: str = "target"
    time_col: str = "time_id"
    asset_col: str = "investment_id"
    min_time_id: Optional[int] = None  # 用于截取训练数据
    sample_frac: float = 1.0           # 采样比例，<1.0时随机采样部分数据


@dataclass
class TrainConfig:
    seed: int = 42
    n_folds: int = 5
    cv_method: str = "group"  # "group" | "time_series"
    batch_size: int = 512
    num_workers: int = 4
    device: str = "cuda"  # "cuda" | "cpu"
    output_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")


@dataclass
class DNNConfig:
    embedding_dim: int = 32
    id_hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    feat_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    head_hidden_dims: List[int] = field(default_factory=lambda: [512, 128, 32])
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    patience: int = 10


@dataclass
class LGBMConfig:
    objective: str = "regression"
    metric: str = "mse"
    boosting_type: str = "gbdt"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 63
    subsample: float = 0.8
    colsample_bytree: float = 0.6
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_samples: int = 100
    early_stopping_rounds: int = 50
    verbose: int = 100
    add_combination_features: bool = True
    combination_pairs: List[str] = field(default_factory=lambda: [
        "f_0-f_1", "f_0-f_2", "f_1-f_2",
        "f_0-f_56", "f_1-f_56", "f_2-f_56",
    ])


@dataclass
class EnsembleConfig:
    """多DNN架构集成配置"""
    model_configs: List[dict] = field(default_factory=lambda: [
        {
            "name": "dnn_v1",
            "id_dims": [64, 64, 64],
            "feat_dims": [256, 256, 256],
            "head_dims": [512, 128, 32],
            "dropout": 0.1,
        },
        {
            "name": "dnn_v2",
            "id_dims": [64, 64, 64, 64],
            "feat_dims": [256, 256, 256, 256],
            "head_dims": [512, 128, 32, 32],
            "dropout": 0.65,
        },
        {
            "name": "dnn_v3",
            "id_dims": [64, 32],
            "feat_dims": [256, 128, 64],
            "head_dims": [64, 32, 16],
            "dropout": 0.5,
            "use_bn": True,
        },
    ])
    weights: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    patience: int = 10


@dataclass
class TransformerConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    patch_size: int = 10  # 将300个特征分成30个patch，每个patch 10维
    embedding_dim: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    patience: int = 10


@dataclass
class AutoEncoderConfig:
    """自编码器生成合成信号 + 预测头"""
    encoder_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    latent_dim: int = 10  # 10个合成信号
    decoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    predictor_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    ae_lr: float = 1e-3
    pred_lr: float = 1e-3
    weight_decay: float = 1e-5
    ae_epochs: int = 20
    pred_epochs: int = 30
    patience: int = 10
    recon_weight: float = 1.0
    pred_weight: float = 1.0
