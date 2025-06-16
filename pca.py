import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


# CSVファイルの読み込み
df = pd.read_csv('sentence_analysis.csv')

# 必要な列が存在するか確認
required_columns = ['文字数', '単語数', '平均単語長', '感情スコア']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"CSVファイルに'{column}'列が見つかりません。")

# 特徴量行列を準備
features = df[required_columns]

# データの標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCAの実行
pca = PCA(n_components=2)  # 2次元に削減する
principal_components = pca.fit_transform(features_scaled)

# PCAスコアのデータフレーム作成
pca_scores = pd.DataFrame(principal_components, columns=['PCA1', 'PCA3'])

explained_variance = pca.explained_variance_ratio_
print(f"主成分1: {explained_variance[0] * 100:.1f}%")
print(f"主成分2: {explained_variance[1] * 100:.1f}%")

# バイプロットの作成
plt.figure(figsize=(10, 7))
for i in range(len(df)):
    plt.scatter(pca_scores['PCA1'][i], pca_scores['PCA3'][i], alpha=0.5)

# 特徴ベクトルのプロット
feature_vectors = pca.components_.T
explained_variance_sqrt = np.sqrt(pca.explained_variance_)
scaled_feature_vectors = feature_vectors * explained_variance_sqrt

# PCAスコアの保存
pca_scores.to_csv('pca_scores13.csv', index=False)

# スケーリングされた特徴ベクトルをデータフレームに変換
scaled_feature_vectors_df = pd.DataFrame(
    scaled_feature_vectors,
    columns=['PCA1', 'PCA3']  # 主成分の名前を列名に使用
)
scaled_feature_vectors_df['Feature'] = required_columns  # 特徴量の名前を追加

# 特徴ベクトルの保存
scaled_feature_vectors_df.to_csv('scaled_feature_vectors.csv', index=False)

required_columns = ['CC', 'WC', 'AWL', 'SS']

for i, v in enumerate(scaled_feature_vectors):
    # 矢印を描画
    plt.arrow(0, 0, v[0], v[1], color='r', alpha=0.8, head_width=0.1, head_length=0.15)
    # テキストを矢印の先端に配置
    plt.text(v[0], v[1], required_columns[i], color='r', ha='center', va='center', fontsize=12)

plt.xlabel('PCA1')
plt.ylabel('PCA3')
plt.title('Biplot')
plt.axhline(0, color='grey', linewidth=0.5, linestyle='--')
plt.axvline(0, color='grey', linewidth=0.5, linestyle='--')
plt.grid(True)
plt.show()

