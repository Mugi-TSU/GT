import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# CSVファイルの読み込み
df = pd.read_csv('sentence_analysis.csv')

# 必要な列が存在するか確認
required_columns = ['文字数', '単語数', '平均単語長']
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

# 主成分負荷量の確認
loadings = pd.DataFrame(pca.components_, columns=required_columns, index=['主成分1', '主成分2'])

# 負荷量の棒グラフを表示
plt.figure(figsize=(10, 6))

# 主成分1と主成分2の負荷量を並べた棒グラフにする
loadings.T.plot(kind='bar', figsize=(10, 6))
plt.title('主成分負荷量（PCA loadings）')
plt.xlabel('特徴量')
plt.ylabel('負荷量')
plt.xticks(rotation=0)
plt.grid(True)
plt.legend(title='主成分')

# グラフの表示
plt.tight_layout()
plt.show()
