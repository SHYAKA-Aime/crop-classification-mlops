# Crop Classification MLOps

An end-to-end Machine Learning Operations (MLOps) pipeline for crop classification using computer vision and deep learning.

## Overview
This repository contains the architecture, training pipelines, API serving, UI dashboard, and deployment configurations for automated crop classification.

## Project Structure
```
crop-classification-mlops/
├── data/                  # Train and test dataset splits
├── models/                # Trained model weights & registry
├── notebook/              # Exploratory data analysis & experiments
├── src/                   # Core application logic & preprocessing
├── app.py                 # FastAPI backend application
├── ui.py                  # Streamlit frontend dashboard
├── setup_dataset.py       # Dataset preparation script
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-container orchestration
├── nginx.conf             # Reverse proxy configuration
├── locustfile.py          # Load and performance testing
└── requirements.txt       # Project dependencies
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SHYAKA-Aime/crop-classification-mlops.git
   cd crop-classification-mlops
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize dataset:**
   ```bash
   python setup_dataset.py --demo
   ```
