import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# 1. 資料載入與前處理 (Data Loading & Preprocessing)
# ============================================================
url = (
    "https://raw.githubusercontent.com/Avik-Jain/"
    "100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
)
df = pd.read_csv(url)

# 對類別變數 State 進行 One-Hot Encoding (drop_first=True 避免虛擬變數陷阱)
df_encoded = pd.get_dummies(df, drop_first=True, dtype=float)

X = df_encoded.drop(columns=["Profit"])
y = df_encoded["Profit"]

# 切分訓練集與測試集 (80% Training, 20% Testing, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ============================================================
# 2. 前向特徵選擇 (Forward Feature Selection)
# ============================================================
# 依序加入對測試集表現最好的特徵：
# 1 個特徵: R&D Spend
# 2 個特徵: R&D Spend + Marketing Spend
# 3 個特徵: R&D Spend + Marketing Spend + State_New York
# 4 個特徵: R&D Spend + Marketing Spend + State_New York + State_Florida
# 5 個特徵: 全部 5 個特徵 (+ Administration)

feature_steps = [
    ["R&D Spend"],
    ["R&D Spend", "Marketing Spend"],
    ["R&D Spend", "Marketing Spend", "State_New York"],
    ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida"],
    ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"]
]

num_features = []
rmse_list = []
r2_list = []

print("特徵數量與模型評估指標：")
print("-" * 60)
for i, features in enumerate(feature_steps, 1):
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])
    
    # 計算 RMSE 與 R-squared
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = r2_score(y_test, y_pred)
    
    num_features.append(i)
    rmse_list.append(rmse)
    r2_list.append(r2)
    print(f"特徵數 {i}: RMSE = {rmse:,.2f}, R^2 = {r2:.4f} | 使用特徵: {features}")

# ============================================================
# 3. 繪製評估圖表 (Plotting Performance by Number of Features)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 左圖：RMSE by Number of Features
ax1.plot(num_features, rmse_list, marker='o')
ax1.set_title("RMSE by Number of Features")
ax1.set_xlabel("Number of Features")
ax1.set_ylabel("RMSE")

# 右圖：R-squared by Number of Features
ax2.plot(num_features, r2_list, marker='o')
ax2.set_title("R-squared by Number of Features")
ax2.set_xlabel("Number of Features")
ax2.set_ylabel("R-squared")

plt.tight_layout()

# 儲存與顯示圖表
plt.savefig("features_performance.png", dpi=150)
print("\n圖表已儲存為: features_performance.png")
plt.show()
