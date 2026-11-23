import io
import os
import time
import socket
import zipfile
from typing import List
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.prediction import make_prediction, get_model_and_classes, get_valid_classes
from src.preprocessing import extract_and_append_feature_stats
from src.model import log_uploaded_sample, trigger_model_retraining, get_database_summary, init_db

app = FastAPI(
    title="Crop Classification MLOps API",
    description="End-to-End Image Classification, SQLite Sample Registry, and Continuous Retraining API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

def validate_image_bytes(content: bytes) -> bool:
    """Verifies that uploaded bytes form a valid, uncorrupted image."""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False

@app.on_event("startup")
def startup_event():
    init_db()
    try:
        get_model_and_classes()
    except Exception as e:
        print(f"Startup warning: {e}")

@app.get("/health")
def health_check():
    """Returns live API/model uptime, container hostname, valid crop classes, and SQLite audit summary."""
    uptime_seconds = int(time.time() - START_TIME)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "status": "Online & Healthy",
        "container_id": socket.gethostname(),
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": f"{hours}h {minutes}m {seconds}s",
        "valid_classes": get_valid_classes(),
        "database_summary": get_database_summary()
    }

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Accepts a single crop image and returns the predicted crop class and confidence."""
    content = await file.read()
    if not validate_image_bytes(content):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file uploaded.")
    try:
        result = make_prediction(content)
        result["served_by_container"] = socket.gethostname()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_bulk_retrain_data(
    label: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Validates bulk crop images, saves to data/train/{label}/, logs to SQLite, and updates feature_stats.csv."""
    valid_classes = get_valid_classes()
    if label not in valid_classes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid crop class '{label}'. Must be one of {valid_classes}."
        )

    target_dir = os.path.join("data", "train", label)
    os.makedirs(target_dir, exist_ok=True)
    saved_count, skipped_count = 0, 0

    for upload in files:
        content = await upload.read()
        if not validate_image_bytes(content):
            skipped_count += 1
            continue

        safe_name = f"{int(time.time() * 1000)}_{os.path.basename(upload.filename)}"
        file_path = os.path.join(target_dir, safe_name)
        with open(file_path, "wb") as f:
            f.write(content)

        log_uploaded_sample(safe_name, label, file_path)
        extract_and_append_feature_stats(content, safe_name, label)
        saved_count += 1

    return {
        "message": f"Saved {saved_count} verified '{label}' images to disk & SQLite database (skipped {skipped_count}).",
        "files_saved": saved_count,
        "files_skipped": skipped_count
    }

@app.post("/upload-zip")
async def upload_zip_dataset(file: UploadFile = File(...)):
    """Extracts a .zip archive containing crop class subfolders (e.g., cotton/, maize/) and logs each image to SQLite."""
    valid_classes = get_valid_classes()
    zip_bytes = await file.read()
    saved_count, skipped_count = 0, 0

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for member in z.infolist():
                if member.is_dir() or "__MACOSX" in member.filename:
                    continue
                parts = os.path.normpath(member.filename).split(os.sep)
                matched_class = next((p for p in parts if p in valid_classes), None)
                if not matched_class:
                    skipped_count += 1
                    continue

                content = z.read(member)
                if not validate_image_bytes(content):
                    skipped_count += 1
                    continue

                target_dir = os.path.join("data", "train", matched_class)
                os.makedirs(target_dir, exist_ok=True)
                safe_name = f"{int(time.time() * 1000)}_{os.path.basename(member.filename)}"
                file_path = os.path.join(target_dir, safe_name)
                with open(file_path, "wb") as f:
                    f.write(content)

                log_uploaded_sample(safe_name, matched_class, file_path)
                extract_and_append_feature_stats(content, safe_name, matched_class)
                saved_count += 1
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid .zip archive.")

    return {
        "message": f"Extracted & logged {saved_count} verified crop images across {valid_classes} (skipped {skipped_count}).",
        "files_saved": saved_count
    }

@app.post("/retrain")
def retrain_endpoint(epochs: int = Form(3)):
    """Triggers fine-tuning on the custom pre-trained model and runs production evaluation on data/test/."""
    try:
        return trigger_model_retraining(train_dir="data/train", test_dir="data/test", epochs=epochs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
