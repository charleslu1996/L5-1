# 50 Startups Profit Prediction & Feature Analysis

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success?style=for-the-badge&logo=github)](https://charleslu1996.github.io/L5-1/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning project implementing the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to model, evaluate, and predict startup profitability based on expenditure profiles and geographical location.

---

## 🚀 Live Demo

Experience the interactive model directly in your browser:

👉 **[Launch Interactive Web Demo (GitHub Pages)](https://charleslu1996.github.io/L5-1/)**

### Local Preview
You can also launch the web demo locally without external dependencies:

```bash
# Option 1: Using Python built-in HTTP server
python -m http.server 8000

# Then open http://localhost:8000 in your browser
```
*(Or simply open `index.html` directly in any web browser).*

**Key Features of the Live Demo:**
- **Real-Time Sliders**: Adjust R&D Spend, Marketing Spend, Administration budget, and Company State.
- **Instant Inference**: Computes predicted profit immediately using the fitted regression formula.
- **Financial Breakdown**: Visualizes the positive and negative dollar contributions of each spending variable.
- **Preset Scenarios**: Quick-test presets like *High-Growth Tech*, *Balanced Profile*, and *Early Bootstrapped*.

---

## Description

This project analyzes the **50 Startups** dataset to predict startup `Profit` using spending in Research & Development (`R&D Spend`), Administration expenses (`Administration`), Marketing budget (`Marketing Spend`), and company headquarters (`State`). 

Through exploratory data analysis, data preparation via scikit-learn pipelines, Multiple Linear Regression modeling, and iterative Forward Feature Selection, this project reveals key financial drivers of profitability. It demonstrates that R&D and Marketing investments serve as the primary profit drivers, achieving an $R^2$ of up to ~0.95 on held-out test data.

---

## Features

- **CRISP-DM Lifecycle Implementation**: Structured across all 6 phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.
- **Interactive Web Simulator**: Zero-dependency browser-based live demo for instant profit estimation and feature contribution analysis.
- **Robust Scikit-Learn Pipelines**: Uses `ColumnTransformer` and `OneHotEncoder` to encode categorical location data while preventing data leakage and dummy variable traps.
- **Forward Feature Selection Analysis**: Evaluates subset model performance across 1 to 5 features, identifying the optimal predictor combination that minimizes RMSE and maximizes $R^2$.
- **Model Evaluation & Residual Diagnostics**: Tracks standard regression metrics (MAE, MSE, RMSE, $R^2$) and outputs visual diagnostics comparing actual vs. predicted values.
- **Production Pipeline Serialization**: Persists the fully-fitted preprocessing and regression pipeline to disk via `joblib` for zero-friction inference on new data.
- **Automated Visualization**: Generates high-resolution performance and residual charts (`actual_vs_predicted.png` and `features_performance.png`).

---

## Tech Stack

- **Language**: Python 3.9+, JavaScript (ES6+), HTML5, CSS3
- **Machine Learning & Preprocessing**: [scikit-learn](https://scikit-learn.org/) (`LinearRegression`, `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `train_test_split`)
- **Data Manipulation & Analysis**: [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization**: [matplotlib](https://matplotlib.org/), [Chart.js](https://www.chartjs.org/)
- **Model Persistence**: [joblib](https://joblib.readthedocs.io/)
- **Version Control & Hosting**: Git, GitHub, GitHub Pages

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/charleslu1996/L5-1.git
cd L5-1
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

On Windows (Command Prompt / PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies

Install dependencies using `pip`:
```bash
pip install -r requirements.txt
```

Alternatively, install packages directly:
```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

---

## Usage

### 1. Run the Full CRISP-DM Multiple Linear Regression Pipeline

Execute `multiple_linear_regression.py` to train the model, evaluate metrics, generate residual plots, and export the serialized pipeline:

```bash
python multiple_linear_regression.py
```

**Sample Output:**
```text
===== STEP 4: MODELING =====
Intercept: 54343.30

Regression coefficients:
                      Feature  Coefficient
categorical__State_California  -315.260255
   categorical__State_Florida   623.532751
  categorical__State_New York  -308.272495
         remainder__R&D Spend     0.805630
    remainder__Administration    -0.068788
   remainder__Marketing Spend     0.029855

===== STEP 5: EVALUATION =====
MAE  : 6,961.48
MSE  : 82,010,363.05
RMSE : 9,055.96
R^2  : 0.8987

===== STEP 6: DEPLOYMENT =====
Predicted profit: $132,307.59
Saved model: startup_profit_model.joblib
Saved plot: actual_vs_predicted.png
```

---

### 2. Run Forward Feature Selection & Generate Performance Plots

Execute `plot_features_performance.py` to analyze model accuracy as a function of the number of features:

```bash
python plot_features_performance.py
```

This generates `features_performance.png`, comparing RMSE and $R^2$ trajectories across 1 to 5 features:

| Features Selected | Active Features | Test RMSE | Test $R^2$ |
| :---: | :--- | :---: | :---: |
| **1** | `R&D Spend` | 8,274.87 | 0.9465 |
| **2 (Optimal)** | `R&D Spend`, `Marketing Spend` | **8,198.80** | **0.9474** |
| **3** | `R&D Spend`, `Marketing Spend`, `State_New York` | 8,309.06 | 0.9460 |
| **4** | `R&D Spend`, `Marketing Spend`, `State_New York`, `State_Florida` | 8,409.92 | 0.9447 |
| **5** | All 5 Features (including `Administration`) | 9,137.99 | 0.9347 |

---

### 3. Making Predictions on New Data with the Saved Model

Load the exported pipeline (`startup_profit_model.joblib`) to predict profit for new startup ventures without manual preprocessing:

```python
import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load("startup_profit_model.joblib")

# Define new startup expenditures
new_data = pd.DataFrame([{
    "R&D Spend": 120000,
    "Administration": 105000,
    "Marketing Spend": 310000,
    "State": "New York"
}])

# Predict profit
prediction = model.predict(new_data)[0]
print(f"Predicted Startup Profit: ${prediction:,.2f}")
```
