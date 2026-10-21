```python
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost.joblib")
model = joblib.load(MODEL_PATH)

# Must match the exact feature columns used during training.
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

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Car Price Prediction API",
        "model": "XGBoost",
        "endpoints": ["/health", "/predict"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "XGBoost"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Send a valid JSON object"}), 400

        unknown = set(data.keys()) - set(FEATURES)
        if unknown:
            return jsonify({
                "error": "Unknown feature(s)",
                "features": sorted(unknown)
            }), 400

        # Missing fields are handled by the fitted preprocessing pipeline,
        # if the training pipeline was configured with imputers.
        input_df = pd.DataFrame(
            [[data.get(feature) for feature in FEATURES]],
            columns=FEATURES
        )

        prediction = float(model.predict(input_df)[0])

        return jsonify({
            "model": "XGBoost",
            "predicted_price_inr": round(prediction, 2),
            "currency": "INR"
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
```
