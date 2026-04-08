import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = r'./data/train.parquet'

df = pd.read_parquet(file_path)

# 统计每个investment_id的观测数
obs_by_asset = df.groupby(['investment_id'])['target'].count()

print("="*80)
print("Investment ID - Observation Count Statistics")
print("="*80)
print("\nTotal unique investment_ids: {}".format(len(obs_by_asset)))
print("\nDescriptive Statistics:")
print(obs_by_asset.describe())

print("\nMin observations: {} (id={})".format(obs_by_asset.min(), obs_by_asset.idxmin()))
print("Max observations: {} (id={})".format(obs_by_asset.max(), obs_by_asset.idxmax()))

# 绘制直方图
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 方式1: 直方图（bins=50）
axes[0].hist(obs_by_asset, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Observation Count per Investment ID')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Observations by Investment ID')
axes[0].grid(True, alpha=0.3)

# 方式2: 排序后的柱状图（显示每个investment_id）
obs_sorted = obs_by_asset.sort_values(ascending=False)
if len(obs_sorted) <= 100:
    axes[1].bar(range(len(obs_sorted)), obs_sorted.values, color='coral', edgecolor='black')
    axes[1].set_xlabel('Investment ID (sorted)')
else:
    axes[1].bar(range(min(100, len(obs_sorted))), obs_sorted.values[:100], color='coral', edgecolor='black')
    axes[1].set_xlabel('Investment ID (top 100, sorted)')

axes[1].set_ylabel('Observation Count')
axes[1].set_title('Observation Count by Investment ID (Descending Order)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('investment_id_distribution.png', dpi=100, bbox_inches='tight')
print("\nHistogram saved as 'investment_id_distribution.png'")
plt.show()

# 打印前10个和后10个
print("\n" + "="*80)
print("Top 10 Investment IDs by Observation Count:")
print("="*80)
print(obs_sorted.head(10))

print("\n" + "="*80)
print("Bottom 10 Investment IDs by Observation Count:")
print("="*80)
print(obs_sorted.tail(10))