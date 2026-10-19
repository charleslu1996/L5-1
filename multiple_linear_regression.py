# ============================================================
# Multiple Linear Regression: 50 Startups Dataset
# Methodology: CRISP-DM
# ============================================================

# STEP 1: BUSINESS UNDERSTANDING
# Goal: Predict startup Profit from spending and State.

# STEP 2: DATA UNDERSTANDING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset directly from GitHub
url = (
    "https://raw.githubusercontent.com/Avik-Jain/"
    "100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
)

df = pd.read_csv(url)

print("\n===== STEP 2: DATA UNDERSTANDING =====")
print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn data types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# STEP 3: DATA PREPARATION

# Separate independent variables (X) and target (y)
X = df.drop(columns=["Profit"])
y = df["Profit"]

numeric_features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]

categorical_features = ["State"]

# Encode State without assigning a false numeric order.
# Keep numeric spending variables unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== STEP 3: DATA PREPARATION =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 4: MODELING

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train the model
model.fit(X_train, y_train)

print("\n===== STEP 4: MODELING =====")
print("Model training completed.")

# Display regression coefficients
feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model.named_steps["regressor"].coef_
)

coefficient_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

print("\nIntercept:")
print(model.named_steps["regressor"].intercept_)

print("\nRegression coefficients:")
print(coefficient_table.to_string(index=False))

# STEP 5: EVALUATION

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== STEP 5: EVALUATION =====")
print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R^2  : {r2:.4f}")

# Compare actual and predicted profits
results = pd.DataFrame({
    "Actual Profit": y_test.to_numpy(),
    "Predicted Profit": y_pred,
    "Error": y_test.to_numpy() - y_pred
})

print("\nActual vs Predicted:")
print(results.round(2).to_string(index=False))

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.8)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("50 Startups: Actual vs Predicted Profit")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# STEP 6: DEPLOYMENT

# Save preprocessing + regression as one pipeline
joblib.dump(model, "startup_profit_model.joblib")

# Predict profit for a new startup
new_startup = pd.DataFrame([{
    "R&D Spend": 100000,
    "Administration": 120000,
    "Marketing Spend": 200000,
    "State": "California"
}])

predicted_profit = model.predict(new_startup)[0]

print("\n===== STEP 6: DEPLOYMENT =====")
print(f"Predicted profit: ${predicted_profit:,.2f}")

print("\nSaved model: startup_profit_model.joblib")
print("Saved plot: actual_vs_predicted.png")
