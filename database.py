import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "car_predictions.db"


def initialize_database():
    DATABASE_DIR.mkdir(exist_ok=True, parents=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            model TEXT,
            model_year INTEGER,
            mileage REAL,
            fuel_type TEXT,
            transmission TEXT,
            predicted_price REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(brand, model, model_year, mileage, fuel_type, transmission, predicted_price):
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO predictions (brand, model, model_year, mileage, fuel_type, transmission, predicted_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (brand, model, model_year, mileage, fuel_type, transmission, predicted_price),
    )
    conn.commit()
    conn.close()


def get_predictions():
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, brand, model, model_year, mileage, fuel_type, transmission, predicted_price, created_at
        FROM predictions
        ORDER BY id DESC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def clear_predictions():
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
