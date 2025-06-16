import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('sentence_analysis.csv')

# 必要な列が存在するか確認
required_columns = ['文字数', '単語数', '平均単語長','感情スコア']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"CSVファイルに'{column}'列が見つかりません。")

# 特徴量行列を準備
features = df[required_columns]

# データの標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCAの実行
pca = PCA(n_components=3)  # 3次元に削減する
principal_components = pca.fit_transform(features_scaled)

# 結果をDataFrameに変換
pca_df = pd.DataFrame(data=principal_components, columns=['主成分1', '主成分2','主成分3'])


# 主成分分析の結果を表示
print("主成分分析の結果:")
print(pca_df)

# 主成分の寄与率を表示
explained_variance = pca.explained_variance_ratio_
print("\n主成分の寄与率:")
print(f"主成分1: {explained_variance[0]:.2f}")
print(f"主成分2: {explained_variance[1]:.2f}")
print(f"主成分3: {explained_variance[2]:.2f}")

# 主成分の分散と標準偏差を計算
explained_variance_values = pca.explained_variance_  # 主成分ごとの分散
explained_std_dev = explained_variance_values**0.5  # 主成分ごとの標準偏差

print("\n主成分の分散:")
print(f"主成分1: {explained_variance_values[0]:.2f}")
print(f"主成分2: {explained_variance_values[1]:.2f}")
print(f"主成分3: {explained_variance_values[2]:.2f}")

print("\n主成分の標準偏差:")
print(f"主成分1: {explained_std_dev[0]:.2f}")
print(f"主成分2: {explained_std_dev[1]:.2f}")
print(f"主成分3: {explained_std_dev[2]:.2f}")

# 主成分負荷量の確認
loadings = pd.DataFrame(pca.components_, columns=required_columns, index=['主成分1', '主成分2','主成分3'])

print("主成分負荷量:")
print(loadings)


# 結果のプロット
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['主成分1'], pca_df['主成分2'], c='blue', marker='o')

# グラフのタイトルと軸ラベル
plt.title('Figure 4: Soseki\' PCA score')
plt.xlabel('PC1')
plt.ylabel('PC2')

# グリッド表示
plt.grid(True)

# グラフ表示
plt.show()
