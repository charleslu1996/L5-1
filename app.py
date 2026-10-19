import os
import json
from flask import Flask, request, jsonify, send_from_directory

# Optional ML libraries (with graceful fallback if not present)
try:
    import numpy as np
    import pandas as pd
    import joblib
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    np = None
    pd = None
    joblib = None

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "startup_profit_model.joblib")

# Flask Web Application instance conforming to Vercel Serverless Function specification
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
handler = app  # Alias for Vercel WSGI handler

# Fitted formula parameters from multiple_linear_regression.py
INTERCEPT = 54343.2998
COEFS = {
    "rnd": 0.805630,
    "adm": -0.068788,
    "mkt": 0.029855,
    "state": {
        "California": -315.2603,
        "Florida": 623.5328,
        "New York": -308.2725
    }
}

# Lazy load model pipeline
model = None
if HAS_ML_DEPS and os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load model pipeline ({e}), using fitted formula.")


def predict_profit(rnd_spend: float, administration: float, marketing_spend: float, state: str) -> float:
    """Predict profit using the trained model or calibrated formula."""
    global model
    if model is not None and HAS_ML_DEPS and pd is not None:
        try:
            df_input = pd.DataFrame([{
                "R&D Spend": float(rnd_spend),
                "Administration": float(administration),
                "Marketing Spend": float(marketing_spend),
                "State": str(state)
            }])
            return float(model.predict(df_input)[0])
        except Exception as e:
            print(f"Pipeline predict error ({e}), falling back to linear formula.")

    # Calibrated regression formula fallback
    state_coef = COEFS["state"].get(state, 0.0)
    profit = (
        INTERCEPT
        + (COEFS["rnd"] * float(rnd_spend))
        + (COEFS["adm"] * float(administration))
        + (COEFS["mkt"] * float(marketing_spend))
        + state_coef
    )
    return float(profit)


@app.route("/", methods=["GET"])
def home():
    """Serve the interactive web simulator."""
    index_file = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_file):
        return send_from_directory(BASE_DIR, "index.html")
    return jsonify({
        "message": "50 Startups Profit Predictor API is running.",
        "endpoints": {
            "predict": "/api/predict (POST)",
            "health": "/api/health (GET)"
        }
    })


@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    """Serve static files like images and stylesheets."""
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(BASE_DIR, filename)
    return jsonify({"error": f"File '{filename}' not found"}), 404


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for Vercel monitoring."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "ml_dependencies_ready": HAS_ML_DEPS
    })


@app.route("/api/predict", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Predict startup profit based on R&D Spend, Administration, Marketing Spend, and State.
    Supports both JSON POST body and GET query parameters.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form.to_dict() or {}
    else:
        data = request.args.to_dict()

    try:
        rnd = float(data.get("R&D Spend") or data.get("rnd_spend") or data.get("rnd") or 0.0)
        adm = float(data.get("Administration") or data.get("admin") or data.get("adm") or 0.0)
        mkt = float(data.get("Marketing Spend") or data.get("marketing_spend") or data.get("mkt") or 0.0)
        state = str(data.get("State") or data.get("state") or "California")

        predicted = predict_profit(rnd, adm, mkt, state)

        return jsonify({
            "status": "success",
            "predicted_profit": round(predicted, 2),
            "inputs": {
                "rnd_spend": rnd,
                "administration": adm,
                "marketing_spend": mkt,
                "state": state
            },
            "formula_breakdown": {
                "intercept": INTERCEPT,
                "rnd_impact": round(rnd * COEFS["rnd"], 2),
                "admin_impact": round(adm * COEFS["adm"], 2),
                "marketing_impact": round(mkt * COEFS["mkt"], 2),
                "state_impact": round(COEFS["state"].get(state, 0.0), 2)
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
