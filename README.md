# 50 Startups Profit Prediction & Feature Analysis

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success?style=for-the-badge&logo=github)](https://charleslu1996.github.io/L5-1/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Vercel%20Serverless-black?style=flat&logo=flask)](https://flask.palletsprojects.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white)](https://github.com/charleslu1996/L5-1/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application and data science project implementing the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to analyze, evaluate, and predict startup profitability based on expenditure profiles (`R&D Spend`, `Administration`, `Marketing Spend`) and geographic headquarters (`State`).

---

## 🚀 Live Demo

Experience the interactive model simulation directly in your web browser:

👉 **[Launch Interactive Web Demo (GitHub Pages)](https://charleslu1996.github.io/L5-1/)**

### Key Features of the Live Demo
- **Real-Time Sliders**: Adjust R&D Spend, Marketing Spend, Administration budget, and Company State dynamically.
- **Instant Inference**: Computes predicted profit immediately using the fitted regression model with sub-millisecond latency.
- **Financial Breakdown**: Visualizes positive and negative dollar contributions of each spending variable.
- **Default Matplotlib Feature Analysis Plot**: Displays the original high-resolution `features_performance.png` chart directly on page load.
- **Interactive Diagnostic Toggle**: Switch between the original Matplotlib chart, Chart.js interactive curves (RMSE & $R^2$), and the Actual vs. Predicted residual plot.
- **Preset Scenarios**: Quick-test presets including *High-Growth Tech*, *Balanced Profile*, *Early Bootstrapped*, and *Zero R&D*.

---

## 📖 Introduction

In venture capital and corporate financial planning, knowing where to deploy capital is fundamental to maximizing return on investment. This project analyzes the **50 Startups dataset** to uncover how financial allocations drive net profitability:

1. **Statistical Significance**: Identifies which expenditure categories serve as true profit drivers versus those that merely represent operational overhead.
2. **Feature Optimization**: Employs **Forward Feature Selection** across 1 to 5 features to identify the most parsimonious model that minimizes Root Mean Squared Error (**RMSE**) and maximizes explained variance (**$R^2$**).
3. **Production Web Application**: Provides a dual-hosting architecture: a client-side zero-dependency static web app for **GitHub Pages**, and a **Flask** application ready for **Vercel Serverless Functions** and REST API consumption.

---

## ✨ Features

- **CRISP-DM 6-Phase Lifecycle**: Structured rigorously across Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.
- **Interactive Glassmorphism Dashboard**: Modern dark-mode UI styled with CSS backdrop-filter blur, responsive CSS grid, and Chart.js visualizations.
- **Forward Feature Selection Analysis**: Demonstrates that a **2-feature subset (`R&D Spend` + `Marketing Spend`)** delivers the optimal test-set accuracy ($R^2 \approx 0.9474$, $\text{RMSE} \approx 8,198.80$), outperforming the full 5-feature model.
- **Automated GitHub Actions CI/CD Pipeline**: Automatically trains the model, executes feature selection, exports model metrics directly into **GitHub Actions Job Summary (`$GITHUB_STEP_SUMMARY`)**, uploads diagnostic plots as artifacts, and deploys to GitHub Pages.
- **RESTful API & Serverless Support**: Provides `/api/predict` (POST/GET) and `/api/health` endpoints powered by Flask for external integration.
- **Robust Scikit-Learn Pipelines**: Uses `ColumnTransformer` and `OneHotEncoder` to encode categorical location features without data leakage or dummy variable traps.
- **Model Serialization**: Persists the fitted model pipeline as `startup_profit_model.joblib` for instantaneous offline and online inference.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Backend & Web API** | Python 3.9+, Flask, WSGI, Vercel Serverless Functions |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) (`LinearRegression`, `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `train_test_split`) |
| **Data Manipulation** | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Model Persistence** | [joblib](https://joblib.readthedocs.io/) |
| **Data Visualization** | [matplotlib](https://matplotlib.org/), [Chart.js](https://www.chartjs.org/) |
| **Frontend UI** | HTML5, Vanilla CSS3 (Glassmorphism, CSS Grid & Flexbox, JetBrains Mono font) |
| **CI/CD & Hosting** | GitHub Actions, GitHub Pages, Vercel, Git |

---

## 💻 Installation (CLI)

Follow these steps to set up the project locally from the command line:

### 1. Clone the Repository
```bash
git clone https://github.com/charleslu1996/L5-1.git
cd L5-1
```

### 2. Create and Activate a Virtual Environment

- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

- **Windows (Command Prompt)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage (Examples)

### 1. Launch the Local Web Server & API
Start the Flask web server locally:
```bash
python main.py
```
Open your browser and navigate to:
```text
http://localhost:5000
```
This serves the interactive user interface and provides live API endpoints.

---

### 2. Query the Prediction REST API
Send an HTTP POST request to `/api/predict` or `/predict` to get model predictions:

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

**Sample JSON Response:**
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

### 3. Run the CRISP-DM Machine Learning Pipeline
Train the model, evaluate metrics, and export `startup_profit_model.joblib`:
```bash
python multiple_linear_regression.py
```

**Terminal Output:**
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

### 4. Run Forward Feature Selection & Generate Diagnostic Plots
Run the feature selection script to evaluate model performance across 1 to 5 features:
```bash
python plot_features_performance.py
```

**Forward Feature Selection Results:**

| Features Selected | Active Features | Test RMSE | Test $R^2$ | Status |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `R&D Spend` | 8,274.87 | 0.9465 | Strong baseline |
| **2 (Optimal)** | `R&D Spend`, `Marketing Spend` | **8,198.80** | **0.9474** | 🏆 **Optimal subset (lowest error, highest $R^2$)** |
| **3** | `R&D Spend`, `Marketing Spend`, `State_New York` | 8,309.06 | 0.9460 | Slight performance drop |
| **4** | `R&D Spend`, `Marketing Spend`, `State_New York`, `State_Florida` | 8,409.92 | 0.9447 | Continued degradation |
| **5** | All 5 Features (including `Administration`) | 9,137.99 | 0.9347 | Overfitting / Noise |

---

### 5. Python In-Code Inference with Serialized Model
Load the exported pipeline in your own Python script:
```python
import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load("startup_profit_model.joblib")

# Input expenditure data for a new venture
new_startup = pd.DataFrame([{
    "R&D Spend": 165349.20,
    "Administration": 136897.80,
    "Marketing Spend": 471784.10,
    "State": "New York"
}])

# Predict profit
predicted_profit = model.predict(new_startup)[0]
print(f"Predicted Startup Profit: ${predicted_profit:,.2f}")
```

---

## 🤖 GitHub Actions CI/CD Pipeline

The repository includes an automated workflow defined in [`.github/workflows/ml_pipeline.yml`](.github/workflows/ml_pipeline.yml). On every push to `main`:
1. **Automated Execution**: Runs `multiple_linear_regression.py` and `plot_features_performance.py`.
2. **Job Summary Report**: Renders an executive markdown summary with full metric tables, regression coefficients, and business insights directly on the GitHub Actions Run page.
3. **Artifact Archival**: Saves `features_performance.png`, `actual_vs_predicted.png`, and `startup_profit_model.joblib` as downloadable artifacts.
4. **GitHub Pages Deployment**: Deploys the static web application automatically to GitHub Pages.

---

## 🌐 Deployment Options

### 1. Deploying to GitHub Pages
The web simulator runs natively on GitHub Pages:
1. In your GitHub repository, navigate to **Settings** > **Pages**.
2. Under **Build and deployment** > **Source**, choose **GitHub Actions** (or select `main` branch with `/(root)`).
3. The automated workflow will publish the demo to `https://<username>.github.io/<repo>/`.

### 2. Deploying to Vercel
This repository includes [`pyproject.toml`](pyproject.toml) and [`main.py`](main.py) pre-configured for Vercel:
1. Push your changes to GitHub.
2. In the [Vercel Dashboard](https://vercel.com/), click **Add New Project** and import this repository.
3. Vercel automatically detects `main.py` and `requirements.txt` to deploy the app with zero additional configuration.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
