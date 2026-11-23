import os
import glob
from locust import HttpUser, task, between

PREFERRED_SAMPLE = os.path.join("data", "test", "cotton", "cotton_0004.jpg")
fallback_samples = glob.glob("data/test/*/*.jpg")
SAMPLE_IMAGE_PATH = PREFERRED_SAMPLE if os.path.exists(PREFERRED_SAMPLE) else (fallback_samples[0] if fallback_samples else None)

class CropMLOpsLoadUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(4)
    def predict_crop_image(self):
        if SAMPLE_IMAGE_PATH and os.path.exists(SAMPLE_IMAGE_PATH):
            with open(SAMPLE_IMAGE_PATH, "rb") as img_file:
                self.client.post(
                    "/predict",
                    files={"file": ("cotton_0004.jpg", img_file, "image/jpeg")},
                    name="/predict"
                )

    @task(1)
    def check_model_uptime(self):
        self.client.get("/health", name="/health")
