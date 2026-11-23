import io
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)
STATS_CSV_PATH = "data/feature_stats.csv"

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Resizes raw image bytes to 224x224 RGB and returns a (1, 224, 224, 3) float32 tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)

def extract_and_append_feature_stats(image_bytes: bytes, filename: str, label: str):
    """Computes RGB channel means, brightness, and contrast variance and appends to data/feature_stats.csv."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    new_row = pd.DataFrame([{
        "filename": filename,
        "label": label,
        "mean_red": float(np.mean(arr[:, :, 0])),
        "mean_green": float(np.mean(arr[:, :, 1])),
        "mean_blue": float(np.mean(arr[:, :, 2])),
        "brightness": float(np.mean(arr)),
        "contrast_std": float(np.std(arr))
    }])
    os.makedirs(os.path.dirname(STATS_CSV_PATH), exist_ok=True)
    if os.path.exists(STATS_CSV_PATH):
        new_row.to_csv(STATS_CSV_PATH, mode="a", header=False, index=False)
    else:
        new_row.to_csv(STATS_CSV_PATH, mode="w", header=True, index=False)

def build_retraining_dataset(data_dir: str = "data/train", batch_size: int = 16):
    """Builds a prefetched TensorFlow dataset from the crop image training directories."""
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
