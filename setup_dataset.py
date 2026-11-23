import os
import glob
import random
import shutil
import argparse
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def clear_existing_data():
    for split in ["train", "test"]:
        split_dir = os.path.join("data", split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

def build_from_source(source_dir: str, train_count: int = 70, test_count: int = 20):
    """Samples, validates, resizes (224x224 RGB), and splits real crop images into data/train and data/test."""
    clear_existing_data()
    classes = [
        d for d in sorted(os.listdir(source_dir))
        if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith(".")
    ]
    if len(classes) < 2:
        raise ValueError(f"Expected at least 2 crop class folders inside {source_dir}, found: {classes}")

    for cls in classes:
        clean_cls = cls.strip().replace(" ", "_")
        os.makedirs(os.path.join("data", "train", clean_cls), exist_ok=True)
        os.makedirs(os.path.join("data", "test", clean_cls), exist_ok=True)

        all_files = glob.glob(os.path.join(source_dir, cls, "*.*"))
        random.seed(42)
        random.shuffle(all_files)

        valid_saved = 0
        for fpath in all_files:
            if valid_saved >= (train_count + test_count):
                break
            try:
                img = Image.open(fpath).convert("RGB").resize(IMG_SIZE)
                split = "train" if valid_saved < train_count else "test"
                out_name = f"{clean_cls}_{valid_saved:04d}.jpg"
                img.save(os.path.join("data", split, clean_cls, out_name), "JPEG", quality=90)
                valid_saved += 1
            except Exception:
                continue
        print(f"Crop Class '{clean_cls}': {min(valid_saved, train_count)} train, {max(0, valid_saved - train_count)} test images.")

def generate_crop_demo_dataset(train_count: int = 60, test_count: int = 20):
    """Generates a 5-class crop canopy dataset with distinct chlorophyll RGB & texture contrast profiles."""
    clear_existing_data()
    crop_profiles = {
        "Canola":   ([195, 185, 55],  28),
        "Corn":     ([55,  145, 50],  32),
        "Oats":     ([155, 170, 95],  20),
        "Soybeans": ([75,  165, 85],  16),
        "Wheat":    ([205, 175, 90],  24),
    }
    np.random.seed(42)
    for cls, (rgb_mean, contrast_std) in crop_profiles.items():
        for split, count in [("train", train_count), ("test", test_count)]:
            out_dir = os.path.join("data", split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(count):
                arr = np.random.normal(loc=rgb_mean, scale=contrast_std, size=(224, 224, 3))
                row_wave = (np.sin(np.linspace(0, 12 * np.pi, 224)) * 12).reshape(224, 1, 1)
                arr = np.clip(arr + row_wave, 0, 255).astype(np.uint8)
                Image.fromarray(arr, mode="RGB").save(os.path.join(out_dir, f"{cls}_{i:04d}.jpg"), "JPEG", quality=90)
        print(f"Generated Crop Class '{cls}': {train_count} train, {test_count} test images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Path to extracted crop image folders")
    parser.add_argument("--demo", action="store_true", help="Generate 5-class crop canopy dataset")
    args = parser.parse_args()

    if args.source:
        build_from_source(args.source)
    else:
        generate_crop_demo_dataset()
