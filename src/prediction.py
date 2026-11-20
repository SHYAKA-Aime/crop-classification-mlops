import json
import os
import numpy as np
import tensorflow as tf
from src.preprocessing import preprocess_image_bytes

MODEL_PATH = "models/crop_classifier_model.h5"
CLASSES_PATH = "models/class_names.json"

_MODEL_INSTANCE = None
_CLASS_NAMES = None
_MODEL_MTIME = 0.0

def get_valid_classes() -> list:
    """Returns canonical ordered crop class names."""
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists("data/train"):
        return sorted([
            d for d in os.listdir("data/train")
            if os.path.isdir(os.path.join("data/train", d)) and not d.startswith(".")
        ])
    return []

def get_model_and_classes():
    """Caches model in RAM and automatically hot-reloads if models/crop_classifier_model.h5 updates on disk."""
    global _MODEL_INSTANCE, _CLASS_NAMES, _MODEL_MTIME

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Execute notebook/crop_classification_mlops.ipynb first.")

    current_mtime = os.path.getmtime(MODEL_PATH)
    if _MODEL_INSTANCE is None or current_mtime > _MODEL_MTIME:
        _MODEL_INSTANCE = tf.keras.models.load_model(MODEL_PATH)
        _MODEL_MTIME = current_mtime
        _CLASS_NAMES = get_valid_classes()

    return _MODEL_INSTANCE, _CLASS_NAMES

def make_prediction(image_bytes: bytes) -> dict:
    """Runs single crop image inference and returns predicted crop class, confidence, and probability breakdown."""
    model, class_names = get_model_and_classes()
    tensor = preprocess_image_bytes(image_bytes)
    probabilities = model.predict(tensor, verbose=0)[0]
    best_idx = int(np.argmax(probabilities))

    return {
        "predicted_class": class_names[best_idx],
        "confidence_percent": round(float(probabilities[best_idx]) * 100, 2),
        "model_version_timestamp": _MODEL_MTIME,
        "all_probabilities": {
            class_names[i]: round(float(prob), 4) for i, prob in enumerate(probabilities)
        }
    }
