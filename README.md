# 50 Startups Profit Prediction & Feature Analysis

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success?style=for-the-badge&logo=github)](https://charleslu1996.github.io/L5-1/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Vercel%20Serverless-black?style=flat&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application and data science project implementing the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to analyze, model, and predict startup profitability based on expenditure profiles (`R&D Spend`, `Administration`, `Marketing Spend`) and geographical location (`State`).

---

## 🚀 Live Demo

Experience the interactive web simulator and real-time inference model directly in your browser:

👉 **[Launch Interactive Web Demo (GitHub Pages)](https://charleslu1996.github.io/L5-1/)**

### Web Demo Highlights
- **Interactive Budget Sliders**: Fine-tune R&D, Marketing, and Administrative budgets in real time.
- **Instant Machine Learning Inference**: Real-time profit calculations driven by Multiple Linear Regression.
- **Dynamic Contribution Breakdown**: Visualizes the positive and negative dollar impacts of each budget component.
- **Forward Feature Selection Diagnostic**: Side-by-side interactive curves analyzing **RMSE** and **$R^2$** across 1 to 5 features.
- **Preset Scenarios**: One-click exploration of industry archetypes like *High-Growth Tech*, *Balanced Profile*, and *Early Bootstrapped*.

---

## 📖 Introduction

In venture capital and corporate finance, allocating capital effectively is critical to maximizing profitability. This project utilizes the **50 Startups dataset** to uncover how financial allocations impact net profit:
- **Primary Profit Drivers**: Pinpoints which expenditures generate statistically significant positive returns.
- **Feature Optimization**: Employs Forward Feature Selection to identify the most parsimonious model that minimizes prediction error (**RMSE**) while maximizing explanatory power (**$R^2$**).
- **Production-Ready Web Service**: Wraps the trained model in a modern Flask web server configured for zero-friction deployment to Vercel Serverless Functions.

---

## ✨ Features

- **Full CRISP-DM Lifecycle**: Structured rigorously across Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.
- **Interactive Glassmorphism Dashboard**: Premium dark-mode UI with Chart.js charts, responsive design, and real-time slider controls.
- **Forward Feature Selection Analysis**: Evaluates subset model performance across 1 to 5 features, identifying that a **2-feature subset (`R&D` + `Marketing`)** delivers optimal accuracy ($R^2 \approx 0.9474$, $\text{RMSE} \approx 8,198.80$).
- **Vercel Serverless & REST API**: Exposes `/api/predict` (POST/GET) and `/api/health` endpoints powered by Flask.
- **Robust Scikit-Learn Pipelines**: Uses `ColumnTransformer` and `OneHotEncoder` to encode categorical location features while avoiding data leakage and dummy variable traps.
- **Model Persistence**: Serializes the fitted pipeline to disk as `startup_profit_model.joblib` for instantaneous offline or online inference.
- **Automated Diagnostic Plots**: Generates publication-ready evaluation charts (`features_performance.png` and `actual_vs_predicted.png`).

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **In-Browser Python Engine** | [Pyodide](https://pyodide.org/) (CPython compiled to WebAssembly, 100% serverless for GitHub Pages) |
| **Backend & Web Service** | Python 3.9+, Flask, Vercel Serverless Functions, WSGI |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) (`LinearRegression`, `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `train_test_split`) |
| **Data Processing** | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Model Persistence** | [joblib](https://joblib.readthedocs.io/) |
| **Visualization** | [Chart.js](https://www.chartjs.org/), [matplotlib](https://matplotlib.org/) |
| **Frontend UI** | HTML5, CSS3 (Glassmorphism, Flexbox/Grid, CSS Variables) |
| **Deployment & CI/CD** | GitHub Pages, Vercel, Git |

---

## 💻 Installation (CLI)

Follow these steps to set up the project locally using the command line:

### 1. Clone the Repository
```bash
git clone https://github.com/charleslu1996/L5-1.git
cd L5-1
```

### 2. Create and Activate a Virtual Environment

- **On macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- **On Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

- **On Windows (Command Prompt)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Examples

### 1. Launch the Local Web Application & REST API
Run the Flask server locally:
```bash
python main.py
```
Open your browser and navigate to:
```text
http://localhost:5000
```
This serves the interactive user interface and provides live API endpoints.

---

### 2. Query the Prediction API (cURL / HTTP)
You can send HTTP POST requests to `/api/predict` or `/predict` to get instant predictions:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "R&D Spend": 100000,
    "Administration": 120000,
    "Marketing Spend": 200000,
    "State": "California"
  }'
```

**JSON Response:**
```json
{
  "status": "success",
  "predicted_profit": 133446.89,
  "inputs": {
    "rnd_spend": 100000.0,
    "administration": 120000.0,
    "marketing_spend": 200000.0,
    "state": "California"
  },
  "formula_breakdown": {
    "intercept": 54343.3,
    "rnd_impact": 80563.0,
    "admin_impact": -8254.56,
    "marketing_impact": 5971.0,
    "state_impact": -315.26
  }
}
```

---

### 3. Run the CRISP-DM Model Training Pipeline
Train the model, evaluate metrics, and export `startup_profit_model.joblib`:
```bash
python multiple_linear_regression.py
```

**Sample Terminal Output:**
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

### 4. Run Forward Feature Selection & Generate Diagnostic Charts
Run the feature selection script to evaluate model performance across 1 to 5 features:
```bash
python plot_features_performance.py
```

**Forward Feature Selection Results:**

| Features Selected | Active Features | Test RMSE | Test $R^2$ | Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | `R&D Spend` | 8,274.87 | 0.9465 | Strong baseline |
| **2 (Optimal)** | `R&D Spend`, `Marketing Spend` | **8,198.80** | **0.9474** | 🏆 Best performance |
| **3** | `R&D Spend`, `Marketing Spend`, `State_New York` | 8,309.06 | 0.9460 | Slight degradation |
| **4** | `R&D Spend`, `Marketing Spend`, `State_New York`, `State_Florida` | 8,409.92 | 0.9447 | Downward trend |
| **5** | All 5 Features (including `Administration`) | 9,137.99 | 0.9347 | Overfitting / Noise |

---

### 5. Python In-Code Inference with Saved Model
Load the serialized pipeline in any Python script:
```python
import joblib
import pandas as pd

# 1. Load fitted pipeline
model = joblib.load("startup_profit_model.joblib")

# 2. Provide new venture parameters
new_startup = pd.DataFrame([{
    "R&D Spend": 165349.20,
    "Administration": 136897.80,
    "Marketing Spend": 471784.10,
    "State": "New York"
}])

# 3. Predict profit
profit = model.predict(new_startup)[0]
print(f"Predicted Startup Profit: ${profit:,.2f}")
```

---

## 🌐 Deployment

### Deploying to Vercel (Serverless Function)
This repository is pre-configured for Vercel Python deployment:
1. Push your repository to GitHub.
2. Import the project in the [Vercel Dashboard](https://vercel.com/).
3. Vercel automatically detects `main.py` and `requirements.txt` to deploy the app with zero additional configuration.

### Deploying to GitHub Pages
The static interface can be hosted directly on GitHub Pages:
1. In your GitHub repository, navigate to **Settings** > **Pages**.
2. Under **Branch**, select `main` and set folder to `/(root)`.
3. Save to publish the live interactive demo at `https://<username>.github.io/<repo>/`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
