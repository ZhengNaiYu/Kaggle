import warnings
warnings.filterwarnings('ignore')

import os, gc, random, time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


# ============================================================
# Seed
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=42)


# ============================================================
# Preprocessing (Improved)
# ============================================================
print('Loading...')
train = pd.read_csv('./data/train.csv')
features = [c for c in train.columns if 'feature' in c]

print('Filling...')
print('Original shape:', train.shape)
print('features[1:] shape:', train[features[1:]].shape)

# P0 IMPROVEMENT: 按日期分组填充，避免信息泄露
print('Filling missing values by date...')
for feature in features[1:]:
    train[feature] = train.groupby('date')[feature].transform(
        lambda x: x.fillna(x.mean())
    ).fillna(train[feature].mean())

# 过滤权重 > 0 的样本
train = train.loc[train.weight > 0].reset_index(drop=True)

print('Original data shape after weight filter:', train.shape)

# P1 IMPROVEMENT: 特征去重 - 移除高相关性特征
print('Feature deduplication...')
corr_matrix = train[features].corr().abs()
upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
to_drop = []
for col in corr_matrix.columns:
    col_idx = corr_matrix.columns.get_loc(col)
    if (corr_matrix.iloc[col_idx][upper_tri[col_idx]] > 0.95).any():
        to_drop.append(col)

to_drop = list(set(to_drop))
features_dedup = [f for f in features if f not in to_drop]
print(f'Features before dedup: {len(features)}, after: {len(features_dedup)}')
features = features_dedup

# P1 IMPROVEMENT: 标准化特征
print('Standardizing features...')
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# 创建标签
train['action'] = (train['resp'] > 0).astype('int')

print('Converting...')
print('Features shape:', train[features].shape)
print('Train shape:', train.shape)
print('Label distribution:', train['action'].value_counts(normalize=True))
print('Weight statistics - mean: {:.5f}, std: {:.5f}'.format(train['weight'].mean(), train['weight'].std()))

print('Finish preprocessing.')


# ============================================================
# Model (Improved - Simpler Architecture)
# ============================================================
class ImprovedModel(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedModel, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(input_dim)
        self.dropout0 = nn.Dropout(0.1)

        # 简化架构，避免过拟合
        self.dense1 = nn.Linear(input_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)

        self.dense2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.dense3 = nn.Linear(128, 64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.dense4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.dense4(x)
        return x


# ============================================================
# Dataset (Improved - with weight support)
# ============================================================
class ImprovedMarketDataset:
    def __init__(self, df, features):
        self.features = df[features].values
        self.label = df['action'].values.reshape(-1, 1)
        self.weight = df['weight'].values.reshape(-1, 1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float),
            'weight': torch.tensor(self.weight[idx], dtype=torch.float)
        }


# ============================================================
# Train / Inference functions (Improved)
# ============================================================
def train_fn_weighted(model, optimizer, scheduler, loss_fn, dataloader, device):
    """P0 IMPROVEMENT: 使用样本权重的加权损失"""
    model.train()
    final_loss = 0
    total_weight = 0

    for data in dataloader:
        optimizer.zero_grad()
        feat = data['features'].to(device)
        label = data['label'].to(device)
        weight = data['weight'].to(device)
        
        outputs = model(feat)
        loss = loss_fn(outputs, label)
        
        # 加权损失
        weighted_loss = (loss.squeeze() * weight.squeeze()).mean()
        
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        final_loss += weighted_loss.item() * len(feat)
        total_weight += len(feat)

    final_loss /= total_weight
    return final_loss


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        feat = data['features'].to(device)
        with torch.no_grad():
            outputs = model(feat)
        preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)
    return preds


# ============================================================
# Loss
# ============================================================
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


# ============================================================
# Early Stopping (Improved)
# ============================================================
class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter % 5 == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(f'Validation score improved ({self.val_score:.5f} --> {epoch_score:.5f}). Saving model!')
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


# ============================================================
# Utility Score
# ============================================================
def utility_score_bincount(date, weight, resp, action):
    count_i = len(np.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u


# ============================================================
# Training (Improved)
# ============================================================
if __name__ == '__main__':
    batch_size = 2048
    label_smoothing = 1e-2
    learning_rate = 1e-3
    num_epochs = 15

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    gc.collect()

    start_time = time.time()
    oof = np.zeros(len(train['action']))
    gkf = GroupKFold(n_splits=5)

    fold_results = []

    for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
        print(f"{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        # P0 IMPROVEMENT: 分离train/valid数据集
        train_df = train.loc[tr].reset_index(drop=True)
        valid_df = train.loc[te].reset_index(drop=True)

        train_set = ImprovedMarketDataset(train_df, features)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        
        valid_set = ImprovedMarketDataset(valid_df, features)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

        model = ImprovedModel(len(features))
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-5)
        loss_fn = SmoothBCEwLogits(smoothing=label_smoothing)

        ckp_path = f'JSModel_improved_{fold}.pth'

        es = EarlyStopping(patience=15, mode="max", delta=1e-4)
        
        best_auc = 0
        for epoch in range(num_epochs):
            train_loss = train_fn_weighted(model, optimizer, scheduler, loss_fn, train_loader, device)
            valid_pred = inference_fn(model, valid_loader, device)
            
            auc_score = roc_auc_score(valid_df['action'].values, valid_pred)
            logloss_score = log_loss(valid_df['action'].values, valid_pred)
            
            valid_pred_binary = np.where(valid_pred >= 0.5, 1, 0).astype(int)
            u_score = utility_score_bincount(
                date=valid_df['date'].values,
                weight=valid_df['weight'].values,
                resp=valid_df['resp'].values,
                action=valid_pred_binary
            )

            elapsed_time = (time.time() - start_time) / 60
            print(
                f"FOLD{fold} EPOCH:{epoch:2d}, train_loss:{train_loss:.5f}, "
                f"u_score:{u_score:8.2f}, auc:{auc_score:.5f}, "
                f"logloss:{logloss_score:.5f}, time:{elapsed_time:6.2f}min"
            )

            es(auc_score, model, model_path=ckp_path)
            
            if auc_score > best_auc:
                best_auc = auc_score
            
            if es.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Fold {fold} best AUC: {best_auc:.5f}\n")
        fold_results.append(best_auc)

        # 释放内存
        del train_set, valid_set, train_loader, valid_loader, model, optimizer, scheduler
        gc.collect()

    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Fold AUCs: {[f'{x:.5f}' for x in fold_results]}")
    print(f"Mean AUC: {np.mean(fold_results):.5f}")
    print(f"Std AUC: {np.std(fold_results):.5f}")
    print(f"Total training time: {(time.time() - start_time) / 60:.2f}min")