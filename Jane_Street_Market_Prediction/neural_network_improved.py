"""
Jane Street Market Prediction — Neural Network Training
Models: Bidirectional LSTM  +  FT-Transformer (ensembled)

Artifacts saved to:  ./tmp/neural_network/
  - lstm_fold{n}.pth          LSTM checkpoints
  - transformer_fold{n}.pth   Transformer checkpoints
  - scaler_mean.npy / scaler_scale.npy
"""

import warnings
warnings.filterwarnings('ignore')

import os, gc, random, time, math, contextlib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================
TMP_DIR      = './tmp/neural_network'
os.makedirs(TMP_DIR, exist_ok=True)

BATCH_SIZE   = 2048
LR           = 1e-3
NUM_EPOCHS   = 10
NUM_FOLDS    = 5
PATIENCE     = 10
LABEL_SMOOTH = 0.01


# ============================================================
# Seed
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)


# ============================================================
# Preprocessing
# ============================================================
print('Loading data...')
train    = pd.read_csv('./data/train.csv')
features = [c for c in train.columns if 'feature' in c]

print('Filling missing values by date group (feature_1 onward)...')
for feat in features[1:]:
    train[feat] = (
        train.groupby('date')[feat]
             .transform(lambda x: x.fillna(x.mean()))
             .fillna(train[feat].mean())
    )

train = train.loc[train.weight > 0].reset_index(drop=True)
print(f'Shape after weight filter: {train.shape}')

# Remove highly correlated features
print('Removing highly correlated features (threshold=0.95)...')
corr_matrix = train[features].corr().abs()
upper_tri   = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
to_drop = []
for col in corr_matrix.columns:
    col_idx = corr_matrix.columns.get_loc(col)
    if (corr_matrix.iloc[col_idx][upper_tri[col_idx]] > 0.95).any():
        to_drop.append(col)
to_drop = list(set(to_drop))
features = [f for f in features if f not in to_drop]
print(f'Features: {len(features)} (removed {len(to_drop)} correlated)')

# Standardize
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
np.save(os.path.join(TMP_DIR, 'scaler_mean.npy'),  scaler.mean_)
np.save(os.path.join(TMP_DIR, 'scaler_scale.npy'), scaler.scale_)
print(f'Scalers saved to {TMP_DIR}/')

train['action'] = (train['resp'] > 0).astype('int')
print(f'Label distribution: {train["action"].value_counts(normalize=True).to_dict()}')
print('Preprocessing done.\n')


# ============================================================
# Models
# ============================================================

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM for tabular market data.

    The `input_dim` features are linearly projected into `seq_len` time steps
    of size `step_size` each, allowing the LSTM to capture inter-feature
    temporal dynamics.  A two-layer BiLSTM processes the sequence and the
    concatenated final hidden states are decoded by an MLP head.
    """
    def __init__(
        self,
        input_dim:  int,
        seq_len:    int   = 10,
        hidden_dim: int   = 256,
        num_layers: int   = 2,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len   = seq_len
        step_size      = math.ceil(input_dim / seq_len)
        padded_dim     = step_size * seq_len
        self.step_size = step_size

        self.input_bn   = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, padded_dim)

        self.lstm = nn.LSTM(
            input_size  = step_size,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.input_proj(x)                                 # (B, padded)
        x = x.view(x.size(0), self.seq_len, self.step_size)   # (B, seq, step)
        _, (h_n, _) = self.lstm(x)                            # h_n: (L*2, B, H)
        # Concatenate last forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)             # (B, 2H)
        return self.head(h)                                    # (B, 1)


class TransformerModel(nn.Module):
    """
    Feature Tokenizer Transformer (FT-Transformer).

    Each feature scalar is linearly projected to an `embed_dim`-dimensional
    token.  A learnable [CLS] token is prepended, and the full sequence is
    processed by a stack of Pre-LN Transformer encoder layers.  The CLS
    output is passed through a classification head.

    Reference: Gorishniy et al., "Revisiting Deep Learning Models for
    Tabular Data", NeurIPS 2021.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int   = 64,
        num_heads: int   = 4,
        num_layers: int  = 3,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        # Per-feature linear tokenizer: scalar → embed_dim
        self.tokenizer = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,   # Pre-LayerNorm improves stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B      = x.size(0)
        tokens = self.tokenizer(x.unsqueeze(-1))        # (B, D, E)
        cls    = self.cls_token.expand(B, -1, -1)       # (B, 1, E)
        tokens = torch.cat([cls, tokens], dim=1)        # (B, D+1, E)
        out    = self.encoder(tokens)                   # (B, D+1, E)
        return self.head(out[:, 0])                     # (B, 1)  — CLS token


class MLPModel(nn.Module):
    """
    Baseline MLP (original simple architecture) kept for comparison.
    BN → Dropout → 256 → 128 → 64 → 1, with ReLU activations.
    """
    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Dataset
# ============================================================
class MarketDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feats: list) -> None:
        self.X = torch.tensor(df[feats].values,          dtype=torch.float32)
        self.y = torch.tensor(df['action'].values,       dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(df['weight'].values,       dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.w[idx]


# ============================================================
# Training / Inference helpers
# ============================================================
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler,
    smoothing: float = 0.01,
) -> float:
    """One training epoch with weighted, label-smoothed BCE loss."""
    model.train()
    total, n = 0.0, 0
    for X, y, w in loader:
        X, y, w = X.to(device), y.to(device), w.to(device)
        y_smooth = y * (1.0 - smoothing) + 0.5 * smoothing
        logits   = model(X)
        loss     = (
            F.binary_cross_entropy_with_logits(logits, y_smooth, reduction='none')
             .squeeze()
             .mul(w.squeeze())
             .mean()
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total += loss.item() * len(X)
        n     += len(X)
    return total / n


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _, _ in loader:
            preds.append(torch.sigmoid(model(X.to(device))).cpu().numpy())
    return np.concatenate(preds).ravel()


# ============================================================
# Early Stopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience: int = 7, delta: float = 1e-5) -> None:
        self.patience   = patience
        self.delta      = delta
        self.best_score = -np.inf
        self.counter    = 0
        self.early_stop = False

    def step(self, score: float, model: nn.Module, path: str) -> None:
        if score > self.best_score + self.delta:
            self.best_score = score
            torch.save(model.state_dict(), path)
            print(f'    Checkpoint saved (auc={score:.5f})')
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ============================================================
# Utility Score (Jane Street competition metric)
# ============================================================
def utility_score(
    date:   np.ndarray,
    weight: np.ndarray,
    resp:   np.ndarray,
    action: np.ndarray,
) -> float:
    date    = date.astype(int)
    count_i = len(np.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t  = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    return float(np.clip(t, 0, 6) * np.sum(Pi))


# ============================================================
# Train one model for one fold
# ============================================================
def train_one(
    model:    nn.Module,
    name:     str,
    fold:     int,
    tr_df:    pd.DataFrame,
    va_df:    pd.DataFrame,
    feats:    list,
    device:   torch.device,
) -> np.ndarray:
    path  = os.path.join(TMP_DIR, f'{name}_fold{fold}.pth')
    tr_ld = DataLoader(MarketDataset(tr_df, feats), BATCH_SIZE, shuffle=True,  num_workers=0)
    va_ld = DataLoader(MarketDataset(va_df, feats), BATCH_SIZE, shuffle=False, num_workers=0)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, eta_min=1e-6)
    es  = EarlyStopping(patience=PATIENCE)

    for ep in range(NUM_EPOCHS):
        loss    = run_epoch(model, tr_ld, device, opt, sch, LABEL_SMOOTH)
        va_pred = predict(model, va_ld, device)
        auc     = roc_auc_score(va_df['action'].values, va_pred)
        ll      = log_loss(va_df['action'].values, va_pred)
        va_binary = (va_pred >= 0.5).astype(int)
        u = utility_score(
            va_df['date'].values, va_df['weight'].values,
            va_df['resp'].values, va_binary
        )
        print(f'  [{name}] ep={ep:2d}  loss={loss:.5f}  auc={auc:.5f}  u={u:8.2f}')
        es.step(auc, model, path)
        if es.early_stop:
            print(f'  [{name}] Early stopping at epoch {ep}')
            break

    # Load best checkpoint before returning predictions
    model.load_state_dict(torch.load(path, map_location=device))
    return predict(model, va_ld, device)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    device = (
        torch.device('cuda:0') if torch.cuda.is_available()  else
        torch.device('mps')    if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print(f'Device: {device}\n')

    gc.collect()
    start     = time.time()
    n         = len(train)
    oof_mlp   = np.zeros(n)
    oof_lstm  = np.zeros(n)
    oof_tfm   = np.zeros(n)

    gkf = GroupKFold(n_splits=NUM_FOLDS)

    for fold, (tr, va) in enumerate(
        gkf.split(train, train['action'], train['date'])
    ):
        print(f'\n{"="*62}')
        print(f' FOLD {fold}  (train={len(tr):,}, valid={len(va):,})')
        print(f'{"="*62}')

        tr_df = train.loc[tr].reset_index(drop=True)
        va_df = train.loc[va].reset_index(drop=True)

        # --- Baseline MLP ---
        print('[MLP]')
        mlp = MLPModel(input_dim=len(features), dropout=0.2)
        oof_mlp[va] = train_one(mlp, 'mlp', fold, tr_df, va_df, features, device)
        del mlp; gc.collect()

        # --- BiLSTM ---
        print('[LSTM]')
        lstm = LSTMModel(
            input_dim=len(features), seq_len=10,
            hidden_dim=256, num_layers=2, dropout=0.2,
        )
        oof_lstm[va] = train_one(lstm, 'lstm', fold, tr_df, va_df, features, device)
        del lstm; gc.collect()

        # --- FT-Transformer ---
        print('[Transformer]')
        tfm = TransformerModel(
            input_dim=len(features), embed_dim=64,
            num_heads=4, num_layers=3, dropout=0.1,
        )
        oof_tfm[va] = train_one(tfm, 'transformer', fold, tr_df, va_df, features, device)
        del tfm; gc.collect()

    # --- Ensemble (equal-weight across all three models) ---
    oof_ens = (oof_mlp + oof_lstm + oof_tfm) / 3.0
    labels  = train['action'].values

    u_mlp = utility_score(
        train['date'].values, train['weight'].values,
        train['resp'].values, (oof_mlp >= 0.5).astype(int),
    )
    u_ens = utility_score(
        train['date'].values, train['weight'].values,
        train['resp'].values, (oof_ens >= 0.5).astype(int),
    )

    print(f'\n{"="*62}')
    print(f' SUMMARY')
    print(f'{"="*62}')
    print(f'MLP         OOF AUC : {roc_auc_score(labels, oof_mlp):.5f}  Utility={u_mlp:.2f}')
    print(f'LSTM        OOF AUC : {roc_auc_score(labels, oof_lstm):.5f}')
    print(f'Transformer OOF AUC : {roc_auc_score(labels, oof_tfm):.5f}')
    print(f'Ensemble    OOF AUC : {roc_auc_score(labels, oof_ens):.5f}  Utility={u_ens:.2f}')
    print(f'Total time          : {(time.time() - start) / 60:.2f} min')