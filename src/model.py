import os
import shutil
import sqlite3
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing import build_retraining_dataset

DB_PATH = "models/mlops_registry.db"
MODEL_PATH = "models/crop_classifier_model.h5"
ARCHIVE_DIR = "models/archive"

def init_db():
    """Initializes SQLite tables for uploaded sample logging and retraining audit history."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                file_path TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                used_in_retrain INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retraining_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_at TEXT NOT NULL,
                epochs INTEGER NOT NULL,
                new_samples_count INTEGER NOT NULL,
                train_accuracy REAL,
                val_accuracy REAL,
                val_precision REAL,
                val_recall REAL,
                val_f1 REAL,
                backup_path TEXT
            )
        """)

def log_uploaded_sample(filename: str, label: str, file_path: str):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO uploaded_samples (filename, label, file_path, uploaded_at) VALUES (?, ?, ?, ?)",
            (filename, label, file_path, datetime.utcnow().isoformat())
        )

def get_database_summary() -> dict:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM uploaded_samples WHERE used_in_retrain = 0")
        pending_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM uploaded_samples")
        total_uploaded = cursor.fetchone()[0]
        cursor.execute(
            "SELECT run_id, triggered_at, epochs, new_samples_count, train_accuracy, "
            "val_accuracy, val_precision, val_recall, val_f1 "
            "FROM retraining_runs ORDER BY run_id DESC LIMIT 10"
        )
        runs = [
            {
                "run_id": r[0], "triggered_at": r[1], "epochs": r[2],
                "new_samples": r[3], "train_acc": r[4], "test_acc": r[5],
                "test_precision": r[6], "test_recall": r[7], "test_f1": r[8]
            }
            for r in cursor.fetchall()
        ]
    return {
        "pending_retrain_samples": pending_count,
        "total_uploaded_samples": total_uploaded,
        "retraining_history": runs
    }

def evaluate_in_production(model, test_dir: str = "data/test") -> dict:
    """Evaluates the retrained model on the holdout test split in production."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=16,
        label_mode="categorical",
        shuffle=False
    )
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    return {
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "test_precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
    }

def trigger_model_retraining(train_dir: str = "data/train", test_dir: str = "data/test", epochs: int = 3) -> dict:
    """Loads models/crop_classifier_model.h5 as a pre-trained model, fine-tunes on updated data, and evaluates in production."""
    init_db()

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(ARCHIVE_DIR, f"model_backup_{timestamp_str}.h5")
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, backup_path)

    train_ds = build_retraining_dataset(train_dir, batch_size=16)

    # Load existing custom model as pre-trained base
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(train_ds, epochs=epochs)
    train_acc = round(float(history.history["accuracy"][-1]), 4)

    # Evaluate retrained model in production
    prod_eval = evaluate_in_production(model, test_dir=test_dir)

    # Atomically write updated weights to trigger hot-reload across API containers
    temp_model_path = f"{MODEL_PATH}.tmp.h5"
    model.save(temp_model_path)
    os.replace(temp_model_path, MODEL_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM uploaded_samples WHERE used_in_retrain = 0")
        new_samples = cursor.fetchone()[0]
        cursor.execute("UPDATE uploaded_samples SET used_in_retrain = 1 WHERE used_in_retrain = 0")
        cursor.execute(
            """INSERT INTO retraining_runs 
               (triggered_at, epochs, new_samples_count, train_accuracy, val_accuracy, val_precision, val_recall, val_f1, backup_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(), epochs, new_samples, train_acc,
                prod_eval["test_accuracy"], prod_eval["test_precision"],
                prod_eval["test_recall"], prod_eval["test_f1"], backup_path
            )
        )

    return {
        "status": "Retraining & Production Evaluation Complete",
        "newly_integrated_samples": new_samples,
        "train_accuracy": train_acc,
        "production_test_metrics": prod_eval,
        "archived_previous_model": backup_path
    }
