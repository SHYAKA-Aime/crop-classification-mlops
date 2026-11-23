import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Crop MLOps Command Center", layout="wide")
st.title("End-to-End MLOps Crop Classification & Continuous Retraining Platform")
st.caption("Extending Tabular Crop Mapping (WinnipegDataset) to 2D RGB Crop Imagery with MobileNetV2, FastAPI, SQLite & Docker")

# --- Top Status Bar: Model Uptime & SQLite Database Metrics ---
col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("Refresh System Status"):
        st.rerun()

health = None
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Status", health["status"])
    c2.metric("Model Uptime", health["uptime_formatted"])
    c3.metric("Pending Retrain Samples (SQLite)", health["database_summary"]["pending_retrain_samples"])
    c4.metric("Total Uploaded Samples (SQLite)", health["database_summary"]["total_uploaded_samples"])
except Exception:
    st.error(f"FastAPI Backend unreachable at {API_URL}. Start uvicorn in a separate terminal first.")

tab1, tab2, tab3 = st.tabs([
    "1. Data Visualizations & Feature Stories",
    "2. Single Crop Image Prediction",
    "3. Bulk Upload & Trigger Retraining"
])

# --- TAB 1: 3 Feature Visualizations & Interpretations ---
with tab1:
    st.subheader("Visual Feature Analysis & Domain Interpretations (Crop Imagery Dataset)")
    stats_path = "data/feature_stats.csv"

    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)

        # Feature 1: Class Distribution Balance
        st.markdown("#### Feature 1: Crop Class Distribution Balance (`label`)")
        class_counts = df["label"].value_counts().reset_index()
        class_counts.columns = ["Crop Class", "Image Count"]
        fig1 = px.bar(
            class_counts, x="Crop Class", y="Image Count", color="Crop Class",
            text="Image Count", title="Training Image Count Across the 5 Crop Categories"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.info(
            "**What Story Does It Tell?** Just as the Intro to ML study on the Winnipeg optical-radar dataset "
            "revealed class distribution skew across crop types, monitoring sample counts per crop class ensures "
            "no single crop dominates gradient updates. Balanced representation explains why the MobileNetV2 model "
            "achieves 0.9111 Weighted F1-Score alongside 0.9091 Accuracy, and lets operators spot underrepresented "
            "crops that need additional training uploads in Tab 3."
        )

        # Feature 2: RGB Chromatic Chlorophyll & Canopy Signature
        st.markdown("#### Feature 2: RGB Chromatic Channel Intensity (`mean_red`, `mean_green`, `mean_blue`)")
        df_melted = df.melt(
            id_vars=["label"], value_vars=["mean_red", "mean_green", "mean_blue"],
            var_name="Color Channel", value_name="Normalized Intensity"
        )
        fig2 = px.box(
            df_melted, x="label", y="Normalized Intensity", color="Color Channel",
            color_discrete_map={"mean_red": "#e74c3c", "mean_green": "#2ecc71", "mean_blue": "#3498db"},
            title="RGB Spectral Intensity Distribution by Crop Species"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.info(
            "**What Story Does It Tell?** Each crop exhibits a distinct chromatic reflectance profile: "
            "leafy broadleaf crops like maize and soyabean show strong green chlorophyll dominance, whereas "
            "mature wheat and cotton bolls shift toward higher red and blue luminance. This proves that preserving "
            "full 3-channel RGB color information during preprocessing (rather than converting to grayscale) "
            "provides critical discriminative signal for MobileNetV2's early convolutional layers."
        )

        # Feature 3: Spatial Contrast & Canopy Texture Variance
        st.markdown("#### Feature 3: Canopy Illumination vs. Spatial Contrast Variance (`brightness` vs. `contrast_std`)")
        fig3 = px.scatter(
            df, x="brightness", y="contrast_std", color="label",
            hover_data=["filename"],
            title="Image Brightness vs. Pixel Contrast Standard Deviation Across Crops"
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.info(
            "**What Story Does It Tell?** Pixel contrast standard deviation captures canopy texture granularity—"
            "separating high-contrast crops (such as white cotton bolls against dark foliage or structured maize rows) "
            "from smoother, uniform canopies like soyabean. Meanwhile, the wide horizontal spread in brightness across "
            "field photos justifies the Rescaling (`1/255`) and BatchNormalization layers so outdoor lighting changes "
            "do not degrade prediction reliability."
        )
    else:
        st.warning("`data/feature_stats.csv` not found. Execute `notebook/crop_classification_mlops.ipynb` first.")

# --- TAB 2: Single Crop Image Prediction ---
with tab2:
    st.subheader("Predict Crop Class from a Single Image Data Point")
    uploaded_img = st.file_uploader("Upload a crop image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        left_col, right_col = st.columns([1, 1])
        with left_col:
            st.image(uploaded_img, caption=f"Input Image: {uploaded_img.name}", width=320)
        with right_col:
            if st.button("Predict Crop Class", type="primary"):
                with st.spinner("Running MobileNetV2 inference via FastAPI..."):
                    res = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded_img.name, uploaded_img.getvalue(), uploaded_img.type)}
                    )
                    if res.status_code == 200:
                        pred = res.json()
                        st.success(f"**Predicted Crop:** `{pred['predicted_class'].upper()}`")
                        m1, m2 = st.columns(2)
                        m1.metric("Confidence Score", f"{pred['confidence_percent']}%")
                        m2.metric("Served By Node", pred.get("served_by_container", "local"))

                        st.write("**Class Probability Distribution:**")
                        prob_df = pd.DataFrame(
                            list(pred["all_probabilities"].items()),
                            columns=["Crop Class", "Probability"]
                        ).sort_values("Probability", ascending=False)
                        fig_prob = px.bar(
                            prob_df, x="Crop Class", y="Probability", color="Crop Class",
                            text_auto=".2%", title="Predicted Class Probability Breakdown"
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                    else:
                        st.error(f"Prediction failed: {res.text}")

# --- TAB 3: Bulk Data Upload & Retraining Trigger ---
with tab3:
    valid_classes = health["valid_classes"] if health and "valid_classes" in health else ["cotton", "maize", "rice", "soyabean", "wheat"]
    col_upload, col_retrain = st.columns(2)

    with col_upload:
        st.subheader("Step 1: Upload Bulk Crop Data to SQLite & Training Store")
        mode = st.radio("Upload Format", ["Multiple Images for One Crop Class", "Multi-Class .zip Archive"])

        if mode == "Multiple Images for One Crop Class":
            selected_crop = st.selectbox("Select Target Crop Class", options=valid_classes)
            bulk_files = st.file_uploader(
                "Select multiple crop images (.jpg, .jpeg, .png)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            if st.button("Upload Bulk Images & Save to Database"):
                if not bulk_files:
                    st.warning("Please select at least one image file before uploading.")
                else:
                    payload = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_files]
                    res = requests.post(f"{API_URL}/upload-data", data={"label": selected_crop}, files=payload)
                    if res.status_code == 200:
                        st.success(res.json()["message"])
                    else:
                        st.error(f"Upload failed: {res.text}")
        else:
            st.caption("Upload a .zip file containing subfolders named after valid crops (e.g., `cotton/`, `maize/`).")
            zip_upload = st.file_uploader("Select .zip dataset archive", type=["zip"])
            if st.button("Extract .zip & Log to Database"):
                if not zip_upload:
                    st.warning("Please select a .zip archive first.")
                else:
                    res = requests.post(
                        f"{API_URL}/upload-zip",
                        files={"file": (zip_upload.name, zip_upload.getvalue(), "application/zip")}
                    )
                    if res.status_code == 200:
                        st.success(res.json()["message"])
                    else:
                        st.error(f"Zip extraction failed: {res.text}")

    with col_retrain:
        st.subheader("Step 2: Trigger Model Retraining & Production Evaluation")
        st.write(
            "Loads `models/crop_classifier_model.h5` as a pre-trained model, fine-tunes on the updated "
            "training directory, evaluates against `data/test/`, and hot-reloads weights into production."
        )
        epochs_val = st.slider("Fine-Tuning Epochs", min_value=1, max_value=10, value=3)

        if st.button("Trigger Model Retraining Now", type="primary"):
            with st.spinner("Fine-tuning pre-trained MobileNetV2 & evaluating on holdout test split..."):
                res = requests.post(f"{API_URL}/retrain", data={"epochs": epochs_val}, timeout=300)
                if res.status_code == 200:
                    out = res.json()
                    st.success("Retraining & Production Evaluation Complete!")
                    st.json(out)
                else:
                    st.error(f"Retraining error: {res.text}")

    if health and health["database_summary"]["retraining_history"]:
        st.markdown("---")
        st.subheader("SQLite Production Retraining Audit Log")
        st.dataframe(pd.DataFrame(health["database_summary"]["retraining_history"]), use_container_width=True)


