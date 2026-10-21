
from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Load trained pipelines once when the API starts
MODELS = {
    "random_forest": joblib.load(BASE_DIR / "random_forest.joblib"),
    "xgboost": joblib.load(BASE_DIR / "xgboost.joblib"),
}

# These must match the training features exactly.
FEATURES = [
    "brand",
    "model_name",
    "variant_trim",
    "vehicle_age_years",
    "km_driven",
    "fuel_type",
    "transmission_type",
    "seller_type",
    "owner_type",
    "city",
    "mileage_kmpl",
    "engine_cc",
    "max_power_bhp",
    "seats",
    "specs_imputed",
]

NUMERIC_FEATURES = {
    "vehicle_age_years",
    "km_driven",
    "mileage_kmpl",
    "engine_cc",
    "max_power_bhp",
    "seats",
    "specs_imputed",
}

CATEGORICAL_FEATURES = set(FEATURES) - NUMERIC_FEATURES


@app.get("/")
def home():
    return jsonify({
        "message": "Indian Used Car Price Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        },
        "models": list(MODELS.keys())
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(MODELS.keys())
    })


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Request body must be a JSON object."
            }), 400

        # Select model; default to Random Forest
        model_name = data.get("model", "random_forest")

        if model_name not in MODELS:
            return jsonify({
                "error": "Invalid model.",
                "available_models": list(MODELS.keys())
            }), 400

        # Reject misspelled or unsupported feature names
        allowed_keys = set(FEATURES) | {"model"}
        unknown_keys = set(data.keys()) - allowed_keys

        if unknown_keys:
            return jsonify({
                "error": "Unknown input fields.",
                "fields": sorted(unknown_keys)
            }), 400

        # Construct features in the exact training-column order.
        # Missing values are handled by the saved preprocessing pipeline.
        row = {}

        for col in FEATURES:
            value = data.get(col)

            if value is None:
                row[col] = None
                continue

            if col in NUMERIC_FEATURES:
                if isinstance(value, bool):
                    return jsonify({
                        "error": f"{col} must be numeric."
                    }), 400

                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return jsonify({
                        "error": f"{col} must be numeric."
                    }), 400

                if not pd.notna(number) or abs(number) == float("inf"):
                    return jsonify({
                        "error": f"{col} must be a finite number."
                    }), 400

                row[col] = number

            elif isinstance(value, str):
                row[col] = value

            else:
                return jsonify({
                    "error": f"{col} must be a string."
                }), 400

        input_df = pd.DataFrame([row], columns=FEATURES)

        # Predict using the saved preprocessing + trained model
        prediction = float(
            MODELS[model_name].predict(input_df)[0]
        )

        if not pd.notna(prediction) or abs(prediction) == float("inf"):
            raise ValueError("Model returned an invalid prediction.")

        return jsonify({
            "model": model_name,
            "predicted_price_inr": round(prediction, 2),
            "currency": "INR"
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check server logs."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
