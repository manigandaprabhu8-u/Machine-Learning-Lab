import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from database import clear_predictions, get_predictions, initialize_database, save_prediction

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "used_cars.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
MODEL_INFO_PATH = PROJECT_ROOT / "models" / "model_info.json"

app = Flask(__name__)
app.secret_key = "used-car-price-prediction-project"


def normalize_column_name(value):
    cleaned = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return cleaned


def load_dataset_options():
    options = {
        "brand": ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Hyundai", "Kia", "Nissan", "Volkswagen", "Audi"],
        "model": ["Corolla", "Civic", "Focus", "3 Series", "C-Class", "Elantra", "Sportage", "Altima", "Golf", "A4"],
        "fuel_type": ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"],
        "transmission": ["Manual", "Automatic"],
        "engine": ["1.4L", "1.6L", "1.8L", "2.0L", "2.4L", "3.0L"]
    }

    if not DATASET_PATH.exists():
        return options

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception:
        return options

    if df.empty:
        return options

    df.columns = [str(col).strip() for col in df.columns]
    renamed = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=renamed)

    for key, values in {"brand": ["brand"], "model": ["model"], "fuel_type": ["fuel_type", "fuel", "fueltype"], "transmission": ["transmission", "gearbox"], "engine": ["engine", "engine_size", "enginesize"]}.items():
        for alias in values:
            if alias in df.columns:
                unique_values = [str(v).strip() for v in df[alias].dropna().unique() if str(v).strip()]
                if unique_values:
                    options[key] = unique_values[:20]
                    break

    if "model_year" in df.columns:
        options["model_year"] = sorted({int(v) for v in pd.to_numeric(df["model_year"], errors="coerce").dropna().unique() if pd.notna(v)})

    if "mileage" not in df.columns and "milage" in df.columns:
        options["mileage"] = []

    return options


def run_training_if_needed():
    if MODEL_PATH.exists() and MODEL_INFO_PATH.exists():
        return True
    if not DATASET_PATH.exists():
        return False
    try:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "train_model.py")], check=True, cwd=str(PROJECT_ROOT))
        return MODEL_PATH.exists() and MODEL_INFO_PATH.exists()
    except Exception:
        return False


def get_model_info():
    if not MODEL_INFO_PATH.exists():
        return None
    try:
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def format_currency(value):
    try:
        amount = float(value)
        return f"${amount:,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    options = load_dataset_options()

    if request.method == "POST":
        brand = request.form.get("brand", "").strip()
        model = request.form.get("model", "").strip()
        model_year = request.form.get("model_year", "").strip()
        mileage = request.form.get("mileage", "").strip()
        fuel_type = request.form.get("fuel_type", "").strip()
        transmission = request.form.get("transmission", "").strip()
        engine = request.form.get("engine", "").strip()

        missing_fields = [
            field for field, value in {
                "Brand": brand,
                "Model": model,
                "Model Year": model_year,
                "Mileage": mileage,
                "Fuel Type": fuel_type,
                "Transmission": transmission,
                "Engine": engine,
            }.items() if not value
        ]
        if missing_fields:
            flash(f"Please complete all required fields: {', '.join(missing_fields)}.")
            return render_template("predict.html", options=options)

        try:
            model_year = int(float(model_year))
            mileage_value = float(mileage)
        except ValueError:
            flash("Model year and mileage must be numeric values.")
            return render_template("predict.html", options=options)

        if not MODEL_PATH.exists():
            if not DATASET_PATH.exists():
                flash("used_cars.csv not found. Please place the dataset in the project folder.")
                return render_template("predict.html", options=options)
            trained = run_training_if_needed()
            if not trained:
                flash("The model is not available yet. Please train the model first.")
                return render_template("predict.html", options=options)

        try:
            model_pipeline = joblib.load(MODEL_PATH)
            prediction_input = pd.DataFrame([
                {
                    "brand": brand,
                    "model": model,
                    "model_year": model_year,
                    "mileage": mileage_value,
                    "fuel_type": fuel_type,
                    "transmission": transmission,
                    "engine": engine,
                }
            ])
            prediction_input = prediction_input[["brand", "model", "model_year", "mileage", "fuel_type", "transmission", "engine"]]
            prediction = float(model_pipeline.predict(prediction_input)[0])
            save_prediction(brand, model, model_year, mileage_value, fuel_type, transmission, prediction)
            return render_template(
                "result.html",
                brand=brand,
                model=model,
                model_year=model_year,
                mileage=mileage_value,
                fuel_type=fuel_type,
                transmission=transmission,
                prediction=prediction,
                formatted_prediction=format_currency(prediction),
            )
        except Exception:
            flash("Prediction failed. Please make sure the model and dataset are available and valid.")
            return render_template("predict.html", options=options)

    return render_template("predict.html", options=options)


@app.route("/results")
def results():
    info = get_model_info()
    graph_paths = {
        "price_distribution": PROJECT_ROOT / "static" / "results" / "price_distribution.png",
        "actual_vs_predicted": PROJECT_ROOT / "static" / "results" / "actual_vs_predicted.png",
        "model_comparison": PROJECT_ROOT / "static" / "results" / "model_comparison.png",
        "error_distribution": PROJECT_ROOT / "static" / "results" / "error_distribution.png",
        "feature_importance": PROJECT_ROOT / "static" / "results" / "feature_importance.png",
    }
    available_graphs = {
        key: "results/" + path.name
        for key, path in graph_paths.items()
        if path.exists()
    }
    return render_template("results.html", info=info, graphs=available_graphs)


@app.route("/history")
def history():
    predictions = get_predictions()
    return render_template("history.html", predictions=predictions)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    clear_predictions()
    flash("Prediction history cleared successfully.")
    return redirect(url_for("history"))


@app.errorhandler(404)
def page_not_found(error):
    return render_template("index.html", message="Page not found."), 404


@app.errorhandler(500)
def server_error(error):
    return render_template("index.html", message="An internal server error occurred. Please try again later."), 500


initialize_database()

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
