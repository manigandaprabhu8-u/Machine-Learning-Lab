import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "used_cars.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"
RESULTS_DIR = PROJECT_ROOT / "static" / "results"


def normalize_column_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def find_matching_column(columns, aliases):
    normalized = {normalize_column_name(col): col for col in columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def clean_numeric_series(series):
    cleaned = series.astype(str).str.replace(r"[$,\s\u20B9\u00A3\u20AC\u00A5]", "", regex=True)
    cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned = pd.to_numeric(cleaned, errors="coerce")
    return cleaned


def clean_mileage_series(series):
    cleaned = series.astype(str)
    numbers = cleaned.str.findall(r"\d+(?:\.\d+)?")
    extracted = numbers.map(lambda items: float(items[0]) if items else np.nan)
    return extracted


def print_dataset_columns(df):
    print("Actual CSV columns:")
    print(list(df.columns))


def load_and_clean_dataset():
    if not DATASET_PATH.exists():
        print("used_cars.csv not found. Please place the dataset in the project folder.")
        return None

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        print("The dataset is empty.")
        return None

    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns={col: normalize_column_name(col) for col in df.columns})

    if "fuel" in df.columns and "fuel_type" not in df.columns:
        df.rename(columns={"fuel": "fuel_type"}, inplace=True)
    if "km_driven" in df.columns and "mileage" not in df.columns:
        df.rename(columns={"km_driven": "mileage"}, inplace=True)
    if "milage" in df.columns and "mileage" not in df.columns:
        df.rename(columns={"milage": "mileage"}, inplace=True)
    if "model_year" not in df.columns and "year" in df.columns:
        df.rename(columns={"year": "model_year"}, inplace=True)
    if "selling_price" in df.columns and "price" not in df.columns:
        df.rename(columns={"selling_price": "price"}, inplace=True)

    print_dataset_columns(df)

    df = df.drop_duplicates().copy()
    df = df.reset_index(drop=True)

    for col in df.columns:
        if col in {"model_year", "mileage", "price"}:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    price_col = find_matching_column(df.columns, ["price", "selling_price", "sellingprice"])
    year_col = find_matching_column(df.columns, ["model_year", "year", "modelyear"])
    mileage_col = find_matching_column(df.columns, ["mileage", "milage"])
    brand_col = find_matching_column(df.columns, ["brand"])
    model_col = find_matching_column(df.columns, ["model"])
    fuel_col = find_matching_column(df.columns, ["fuel_type", "fuel", "fueltype"])
    transmission_col = find_matching_column(df.columns, ["transmission", "gearbox"])
    engine_col = find_matching_column(df.columns, ["engine", "engine_size", "enginesize"])

    if not price_col:
        raise ValueError("A valid price column is required for training.")

    df[price_col] = clean_numeric_series(df[price_col])
    if mileage_col:
        df[mileage_col] = clean_mileage_series(df[mileage_col])
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    numeric_cols = []
    for col in df.columns:
        if col == price_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    for col in numeric_cols:
        if col in {"model_year", "mileage"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[price_col]).copy()
    if mileage_col:
        df[mileage_col] = pd.to_numeric(df[mileage_col], errors="coerce")
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[price_col]).copy()

    if not brand_col:
        brand_col = "brand"
        if brand_col not in df.columns:
            df[brand_col] = "Unknown"
    if not model_col:
        model_col = "model"
        if model_col not in df.columns:
            df[model_col] = "Unknown"
    else:
        df.rename(columns={model_col: "model"}, inplace=True)
        model_col = "model"
    if not fuel_col:
        fuel_col = "fuel_type"
        if fuel_col not in df.columns:
            df[fuel_col] = "Unknown"
    else:
        df.rename(columns={fuel_col: "fuel_type"}, inplace=True)
        fuel_col = "fuel_type"
    if not transmission_col:
        transmission_col = "transmission"
        if transmission_col not in df.columns:
            df[transmission_col] = "Unknown"
    if not engine_col:
        engine_col = "engine"
        if engine_col not in df.columns:
            df[engine_col] = "Unknown"
    if not year_col:
        year_col = "model_year"
        if year_col not in df.columns:
            df[year_col] = 2020
    else:
        df.rename(columns={year_col: "model_year"}, inplace=True)
        year_col = "model_year"
    if not mileage_col:
        mileage_col = "mileage"
        if mileage_col not in df.columns:
            df[mileage_col] = 0
    else:
        df.rename(columns={mileage_col: "mileage"}, inplace=True)
        mileage_col = "mileage"

    if "brand" not in df.columns:
        df["brand"] = "Unknown"
    if "model" not in df.columns:
        df["model"] = "Unknown"
    if "fuel_type" not in df.columns:
        df["fuel_type"] = "Unknown"
    if "transmission" not in df.columns:
        df["transmission"] = "Unknown"
    if "engine" not in df.columns:
        df["engine"] = "Unknown"
    if "model_year" not in df.columns:
        df["model_year"] = 2020
    if "mileage" not in df.columns:
        df["mileage"] = 0

    feature_candidates = [
        brand_col,
        model_col,
        year_col,
        mileage_col,
        fuel_col,
        transmission_col,
        engine_col,
    ]

    feature_columns = [col for col in feature_candidates if col in df.columns]
    if not feature_columns:
        feature_columns = [c for c in df.columns if c != price_col]

    text_columns = [c for c in feature_columns if c != year_col and c != mileage_col and c not in numeric_cols]
    for col in text_columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    return df, feature_columns, price_col


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features)
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_model(model_name, pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return {
        "model_name": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
    }


def save_model_comparison_chart(results):
    model_names = [item["model_name"] for item in results]
    r2_scores = [item["r2"] for item in results]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=r2_scores, palette="Blues_d")
    plt.title("Model Comparison")
    plt.ylabel("R² Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()


def save_price_distribution(y):
    plt.figure(figsize=(8, 5))
    sns.histplot(y, bins=25, kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "price_distribution.png", dpi=150)
    plt.close()


def save_actual_vs_predicted(y_test, predictions):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=predictions)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title("Actual vs Predicted Price")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def save_error_distribution(predictions, y_test):
    errors = y_test - predictions
    plt.figure(figsize=(8, 5))
    sns.histplot(errors, bins=25, kde=True)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Residual Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "error_distribution.png", dpi=150)
    plt.close()


def save_feature_importance(best_pipeline, feature_cols):
    final_model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    transformed_names = preprocessor.get_feature_names_out()

    if hasattr(final_model, "feature_importances_"):
        importance_values = final_model.feature_importances_
        ordered = pd.DataFrame({"feature": transformed_names, "importance": importance_values})
        ordered = ordered.sort_values("importance", ascending=False).head(15)
    elif hasattr(final_model, "coef_"):
        importance_values = np.abs(final_model.coef_)
        ordered = pd.DataFrame({"feature": transformed_names, "importance": importance_values})
        ordered = ordered.sort_values("importance", ascending=False).head(15)
    else:
        ordered = pd.DataFrame({"feature": transformed_names[:10], "importance": np.ones(10)})

    plt.figure(figsize=(10, 6))
    sns.barplot(data=ordered, x="importance", y="feature", palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def train_and_save_model():
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_result = load_and_clean_dataset()
    if dataset_result is None:
        return False

    df, feature_columns, target_col = dataset_result
    X = df[feature_columns]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=200),
    }

    results = []
    for model_name, model in models.items():
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        result = evaluate_model(model_name, pipeline, X_test, y_test)
        results.append(result)

    best_result = max(results, key=lambda item: item["r2"])
    best_model_name = best_result["model_name"]
    best_pipeline = None

    for model_name, model in models.items():
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        if model_name == best_model_name:
            best_pipeline = pipeline
            break

    if best_pipeline is None:
        raise RuntimeError("Unable to determine the best model.")

    save_price_distribution(y)
    save_actual_vs_predicted(y_test, best_result["predictions"])
    save_model_comparison_chart(results)
    save_error_distribution(best_result["predictions"], y_test)
    save_feature_importance(best_pipeline, feature_columns)

    print("\nModel performance summary:")
    print("Model | MAE | MSE | RMSE | R2 Score")
    for item in results:
        print(
            f"{item['model_name']} | {item['mae']:.2f} | {item['mse']:.2f} | {item['rmse']:.2f} | {item['r2']:.4f}"
        )
    print(f"\nBest model: {best_model_name}")

    model_info = {
        "best_model_name": best_model_name,
        "mae": round(best_result["mae"], 4),
        "mse": round(best_result["mse"], 4),
        "rmse": round(best_result["rmse"], 4),
        "r2_score": round(best_result["r2"], 4),
        "feature_names": feature_columns,
    }

    joblib.dump(best_pipeline, MODEL_PATH)
    with MODEL_INFO_PATH.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Model info saved to: {MODEL_INFO_PATH}")
    return True


if __name__ == "__main__":
    try:
        train_and_save_model()
    except FileNotFoundError:
        print("used_cars.csv not found. Please place the dataset in the project folder.")
    except ValueError as exc:
        print(str(exc))
    except Exception as exc:
        print(f"Training failed: {exc}")
