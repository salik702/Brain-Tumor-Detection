"""API for serving the brain tumor classifier."""

from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import base64
import sys


def _prepend_local_venv_site_packages():
    project_root = Path(__file__).resolve().parent
    candidate_site_packages = [
        project_root / ".venv" / "Lib" / "site-packages",
        project_root / "brain_tumor_env" / "Lib" / "site-packages",
    ]

    for venv_site_packages in candidate_site_packages:
        if venv_site_packages.exists():
            site_packages_path = str(venv_site_packages)
            if site_packages_path not in sys.path:
                sys.path.insert(0, site_packages_path)


_prepend_local_venv_site_packages()

try:
    from flask import Flask, json, request
    from flask_cors import CORS, cross_origin

    IS_FLASK = True
except Exception:
    Flask = None
    json = None
    request = None
    CORS = None
    cross_origin = None
    IS_FLASK = False

try:
    import cv2

    IS_CV2_AVAILABLE = True
except Exception:
    cv2 = None
    IS_CV2_AVAILABLE = False

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from streamlit.runtime.scriptrunner import get_script_run_ctx

app = Flask(__name__) if IS_FLASK else None
if IS_FLASK:
    cors = CORS(app)
    app.config["CORS_HEADERS"] = "Content-Type"
STREAMLIT_CONTEXT = False

try:
    STREAMLIT_CONTEXT = get_script_run_ctx() is not None
except Exception:
    STREAMLIT_CONTEXT = False

MODEL_INPUT_SIZE = (224, 224)
LABEL_NAMES = ["No Tumor", "Pituitary Tumor"]
SUPPORTED_IMAGE_TYPES = {"png", "jpg", "jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _build_fallback_model():
    from tensorflow.keras.applications import vgg16
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    vgg = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in vgg.layers:
        layer.trainable = False

    top_model = vgg.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation="relu")(top_model)
    top_model = Dense(1024, activation="relu")(top_model)
    top_model = Dense(512, activation="relu")(top_model)
    top_model = Dense(2, activation="softmax")(top_model)
    model = Model(inputs=vgg.input, outputs=top_model)
    model.load_weights("model.h5")
    return model


@lru_cache(maxsize=1)
def load_model():
    import tensorflow as tf

    try:
        return tf.keras.models.load_model("model_complete.keras")
    except Exception:
        try:
            with open("model.json", "r") as json_file:
                loaded_model_json = json_file.read()
            model = tf.keras.models.model_from_json(loaded_model_json)
            model.load_weights("model.h5")
            return model
        except Exception:
            return _build_fallback_model()


def preprocess_image(image):
    if hasattr(image, "convert"):
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = np.array(image)

    if IS_CV2_AVAILABLE:
        image = cv2.resize(image, MODEL_INPUT_SIZE)
    else:
        image = Image.fromarray(np.asarray(image).astype("uint8")).resize(
            MODEL_INPUT_SIZE, Image.BILINEAR
        )
        image = np.array(image)

    return image.astype("float32") / 255.0


def predict_images(images):
    prediction_input = np.array(
        [preprocess_image(image) for image in images], dtype="float32"
    )
    model = load_model()
    return model.predict(prediction_input, verbose=0)


def predict_image_result(image):
    raw_prediction = np.asarray(predict_images([image])[0], dtype="float32").reshape(-1)

    if raw_prediction.size == 1:
        tumor_probability = float(np.clip(raw_prediction[0], 0.0, 1.0))
        probabilities = np.array(
            [1.0 - tumor_probability, tumor_probability], dtype="float32"
        )
    else:
        probabilities = np.asarray(raw_prediction[:2], dtype="float32")
        total = float(np.sum(probabilities))
        if total > 0:
            probabilities = probabilities / total

    label_index = int(np.argmax(probabilities))
    confidence = float(probabilities[label_index])

    return {
        "label_index": label_index,
        "label": LABEL_NAMES[label_index],
        "confidence": confidence,
        "probabilities": probabilities,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(",")[1]

    if IS_CV2_AVAILABLE:
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_data = BytesIO(base64.b64decode(encoded_data))
    pil_img = Image.open(image_data)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return np.array(pil_img)


def get_image_from_base64_string(b64str):
    encoded_data = b64str.split(",")[1]
    image_data = BytesIO(base64.b64decode(encoded_data))
    img = Image.open(image_data)
    return img


def home():
    return "Hello World"


def read_root():
    data = json.loads(request.data)
    predict_img = [get_cv2_image_from_base64_string(item) for item in data["image"]]
    prediction = predict_images(predict_img)
    result = np.argmax(prediction, axis=1)

    # make the probablity frtom prediction
    # print(prediction[:,1])
    # print(result)

    return {"result": prediction[:, 1].tolist()}


if IS_FLASK:

    @app.route("/home", methods=["GET"])
    def home():
        return "Hello World"

    @app.route("/", methods=["POST"])
    def read_root():
        data = json.loads(request.data)
        predict_img = [get_cv2_image_from_base64_string(item) for item in data["image"]]
        prediction = predict_images(predict_img)
        result = np.argmax(prediction, axis=1)

        # make the probablity frtom prediction
        # print(prediction[:,1])
        # print(result)

        return {"result": prediction[:, 1].tolist()}


if STREAMLIT_CONTEXT:
    import streamlit as st

    def initialize_state():
        st.session_state.setdefault("ui_theme", "Auto")
        st.session_state.setdefault("prediction_history", [])
        st.session_state.setdefault("latest_prediction", None)

    def clear_prediction_history():
        st.session_state["prediction_history"] = []

    def human_file_size(size_in_bytes):
        size = float(size_in_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0 or unit == "GB":
                return f"{size:.1f} {unit}"
            size /= 1024.0

    def theme_tokens(mode):
        light = {
            "bg": "#f4f7fb",
            "bg_alt": "#ffffff",
            "panel": "rgba(255, 255, 255, 0.82)",
            "panel_soft": "rgba(255, 255, 255, 0.62)",
            "border": "rgba(15, 23, 42, 0.10)",
            "text": "#0f172a",
            "muted": "#64748b",
            "primary": "#2563eb",
            "primary_soft": "rgba(37, 99, 235, 0.14)",
            "accent": "#14b8a6",
            "success": "#16a34a",
            "warning": "#d97706",
            "danger": "#dc2626",
            "shadow": "0 24px 60px rgba(15, 23, 42, 0.10)",
            "page_scheme": "light",
        }
        dark = {
            "bg": "#07111f",
            "bg_alt": "#0b1729",
            "panel": "rgba(15, 23, 42, 0.82)",
            "panel_soft": "rgba(15, 23, 42, 0.62)",
            "border": "rgba(148, 163, 184, 0.18)",
            "text": "#e5eefc",
            "muted": "#94a3b8",
            "primary": "#60a5fa",
            "primary_soft": "rgba(96, 165, 250, 0.16)",
            "accent": "#2dd4bf",
            "success": "#4ade80",
            "warning": "#f59e0b",
            "danger": "#f87171",
            "shadow": "0 28px 70px rgba(2, 6, 23, 0.45)",
            "page_scheme": "dark",
        }

        if mode == "Dark":
            return dark
        return light

    def theme_css(mode):
        light = theme_tokens("Light")
        dark = theme_tokens("Dark")
        tokens = theme_tokens(mode)

        base_css = f"""
        <style>
        :root {{
            color-scheme: {tokens["page_scheme"]};
            --app-bg: {tokens["bg"]};
            --app-bg-alt: {tokens["bg_alt"]};
            --app-panel: {tokens["panel"]};
            --app-panel-soft: {tokens["panel_soft"]};
            --app-border: {tokens["border"]};
            --app-text: {tokens["text"]};
            --app-muted: {tokens["muted"]};
            --app-primary: {tokens["primary"]};
            --app-primary-soft: {tokens["primary_soft"]};
            --app-accent: {tokens["accent"]};
            --app-success: {tokens["success"]};
            --app-warning: {tokens["warning"]};
            --app-danger: {tokens["danger"]};
            --app-shadow: {tokens["shadow"]};
        }}

        {"@media (prefers-color-scheme: dark) { :root {" if mode == "Auto" else ""}
        {"--app-bg: " + dark["bg"] + ";" if mode == "Auto" else ""}
        {"--app-bg-alt: " + dark["bg_alt"] + ";" if mode == "Auto" else ""}
        {"--app-panel: " + dark["panel"] + ";" if mode == "Auto" else ""}
        {"--app-panel-soft: " + dark["panel_soft"] + ";" if mode == "Auto" else ""}
        {"--app-border: " + dark["border"] + ";" if mode == "Auto" else ""}
        {"--app-text: " + dark["text"] + ";" if mode == "Auto" else ""}
        {"--app-muted: " + dark["muted"] + ";" if mode == "Auto" else ""}
        {"--app-primary: " + dark["primary"] + ";" if mode == "Auto" else ""}
        {"--app-primary-soft: " + dark["primary_soft"] + ";" if mode == "Auto" else ""}
        {"--app-accent: " + dark["accent"] + ";" if mode == "Auto" else ""}
        {"--app-success: " + dark["success"] + ";" if mode == "Auto" else ""}
        {"--app-warning: " + dark["warning"] + ";" if mode == "Auto" else ""}
        {"--app-danger: " + dark["danger"] + ";" if mode == "Auto" else ""}
        {"--app-shadow: " + dark["shadow"] + ";" if mode == "Auto" else ""}
        {"color-scheme: dark;" if mode == "Auto" else ""}
        {"}" if mode == "Auto" else ""}
        """

        if mode == "Auto":
            base_css += """
        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg: #07111f;
                --app-bg-alt: #0b1729;
                --app-panel: rgba(15, 23, 42, 0.82);
                --app-panel-soft: rgba(15, 23, 42, 0.62);
                --app-border: rgba(148, 163, 184, 0.18);
                --app-text: #e5eefc;
                --app-muted: #94a3b8;
                --app-primary: #60a5fa;
                --app-primary-soft: rgba(96, 165, 250, 0.16);
                --app-accent: #2dd4bf;
                --app-success: #4ade80;
                --app-warning: #f59e0b;
                --app-danger: #f87171;
                --app-shadow: 0 28px 70px rgba(2, 6, 23, 0.45);
                color-scheme: dark;
            }
        }
        """

        base_css += """
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.14), transparent 24%),
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.14), transparent 20%),
                linear-gradient(180deg, var(--app-bg) 0%, var(--app-bg-alt) 100%);
            color: var(--app-text);
            font-family: "Segoe UI Variable", "Segoe UI", "Aptos", "Helvetica Neue", sans-serif;
        }

        [data-testid="stHeader"], [data-testid="stToolbar"], footer {
            visibility: hidden;
            height: 0;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.25rem;
            max-width: 1320px;
        }

        .hero-card,
        .panel-card,
        .stat-card,
        .history-card,
        .upload-card,
        .model-card {
            background: var(--app-panel);
            border: 1px solid var(--app-border);
            box-shadow: var(--app-shadow);
            border-radius: 24px;
            backdrop-filter: blur(18px);
        }

        .hero-card {
            padding: 1.5rem 1.5rem 1.25rem;
            margin-bottom: 1rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.85fr);
            gap: 1rem;
            align-items: stretch;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            border-radius: 999px;
            padding: 0.4rem 0.75rem;
            background: var(--app-primary-soft);
            color: var(--app-primary);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .hero-title {
            margin: 0.85rem 0 0.35rem;
            font-size: clamp(2rem, 4vw, 3.55rem);
            line-height: 1.02;
            letter-spacing: -0.04em;
            color: var(--app-text);
        }

        .hero-copy {
            margin: 0;
            font-size: 1rem;
            line-height: 1.7;
            color: var(--app-muted);
            max-width: 58rem;
        }

        .hero-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .metric-chip {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            border: 1px solid var(--app-border);
            background: var(--app-panel-soft);
        }

        .metric-label {
            color: var(--app-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.3rem;
        }

        .metric-value {
            color: var(--app-text);
            font-size: 1.05rem;
            font-weight: 700;
        }

        .section-shell {
            padding: 1rem 0 0;
        }

        .panel-card,
        .upload-card,
        .history-card,
        .model-card {
            padding: 1.15rem;
            height: 100%;
        }

        .section-title {
            margin: 0 0 0.35rem;
            color: var(--app-text);
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .section-subtitle {
            margin: 0 0 1rem;
            color: var(--app-muted);
            font-size: 0.93rem;
            line-height: 1.6;
        }

        .image-frame {
            border-radius: 22px;
            overflow: hidden;
            border: 1px solid var(--app-border);
            background: rgba(15, 23, 42, 0.04);
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            border: 1px solid var(--app-border);
            color: var(--app-text);
            background: var(--app-panel-soft);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .status-badge strong {
            color: var(--app-primary);
        }

        .confidence-card {
            border-radius: 20px;
            padding: 1rem;
            border: 1px solid var(--app-border);
            background: var(--app-panel-soft);
            margin-bottom: 1rem;
        }

        .confidence-head {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .confidence-title {
            margin: 0;
            color: var(--app-text);
            font-size: 1.05rem;
            font-weight: 700;
        }

        .confidence-score {
            color: var(--app-muted);
            font-size: 0.92rem;
        }

        .confidence-track {
            width: 100%;
            height: 14px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.18);
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--app-primary), var(--app-accent));
        }

        .probability-grid {
            display: grid;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .probability-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.75rem;
            align-items: center;
        }

        .probability-name {
            color: var(--app-text);
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .probability-bar {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.18);
            overflow: hidden;
        }

        .probability-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--app-accent), var(--app-primary));
        }

        .probability-value {
            color: var(--app-muted);
            font-size: 0.9rem;
            font-variant-numeric: tabular-nums;
        }

        .tip-list {
            margin: 0.75rem 0 0;
            padding-left: 1.15rem;
            color: var(--app-muted);
        }

        .tip-list li {
            margin-bottom: 0.4rem;
        }

        .stFileUploader {
            border-radius: 20px;
        }

        div[data-testid="stFileUploaderDropzone"] {
            border: 1px dashed var(--app-border) !important;
            background: var(--app-panel-soft) !important;
            border-radius: 20px !important;
        }

        button[kind="primary"] {
            border-radius: 999px !important;
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.24) !important;
        }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--app-primary), var(--app-accent));
        }

        .stDataFrame,
        .stTable {
            border-radius: 18px;
            overflow: hidden;
        }

        @media (max-width: 900px) {
            .hero-grid,
            .hero-metrics {
                grid-template-columns: 1fr;
            }

            .probability-row {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 640px) {
            .hero-card,
            .panel-card,
            .upload-card,
            .history-card,
            .model-card {
                padding: 1rem;
                border-radius: 20px;
            }

            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
        }
        </style>
        """
        return base_css

    def image_error(uploaded_file):
        suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
        if suffix and suffix not in SUPPORTED_IMAGE_TYPES:
            return "Please upload a PNG, JPG, or JPEG image."
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            return (
                "The uploaded image is too large. Please use a file smaller than 10 MB."
            )
        return None

    def load_uploaded_image(uploaded_file):
        return ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")

    def format_probability_rows(probabilities):
        rows = []
        for label, probability in zip(LABEL_NAMES, probabilities):
            rows.append(f"""
                <div class=\"probability-row\">
                    <div>
                        <div class=\"probability-name\">{label}</div>
                        <div class=\"probability-bar\"><div class=\"probability-fill\" style=\"width: {probability * 100:.2f}%\"></div></div>
                    </div>
                    <div class=\"probability-value\">{probability * 100:.1f}%</div>
                </div>
                """)
        return "".join(rows)

    def build_history_entry(uploaded_file, image, result):
        return {
            "timestamp": result["timestamp"],
            "file_name": uploaded_file.name,
            "file_size": human_file_size(uploaded_file.size),
            "dimensions": f"{image.width} x {image.height}",
            "prediction": result["label"],
            "confidence": round(result["confidence"] * 100, 2),
            "no_tumor": round(float(result["probabilities"][0]) * 100, 2),
            "pituitary_tumor": round(float(result["probabilities"][1]) * 100, 2),
        }

    def push_history(entry):
        history = st.session_state["prediction_history"]
        history.insert(0, entry)
        st.session_state["prediction_history"] = history[:10]

    def render_sidebar():
        st.sidebar.markdown("### Control Panel")
        st.sidebar.selectbox(
            "Theme",
            ["Auto", "Light", "Dark"],
            key="ui_theme",
            help="Choose the interface theme for the dashboard.",
        )
        st.sidebar.markdown(
            """
            <div class="status-badge" style="width: 100%; justify-content: space-between; margin: 0.75rem 0;">
                <span>Model</span>
                <strong>Loaded on demand</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.metric("Input", "224 x 224")
        st.sidebar.metric("Classes", "2")
        st.sidebar.metric("History", str(len(st.session_state["prediction_history"])))
        if st.sidebar.button("Clear prediction history", use_container_width=True):
            clear_prediction_history()
            st.session_state["latest_prediction"] = None
            st.rerun()
        st.sidebar.caption("For analysis support only.")

    def render_hero():
        st.markdown(
            f"""
            <section class="hero-card">
                <div class="hero-grid">
                    <div>
                        <div class="hero-kicker">AI MRI Analysis Dashboard</div>
                        <h1 class="hero-title">Brain Tumor Detection</h1>
                        <p class="hero-copy">Upload a scan, run inference, and review the result.</p>
                    </div>
                    <div>
                        <div class="metric-chip">
                            <div class="metric-label">Workflow</div>
                            <div class="metric-value">Upload → Predict → Review</div>
                        </div>
                        <div class="metric-chip" style="margin-top: 0.75rem;">
                            <div class="metric-label">Model</div>
                            <div class="metric-value">TensorFlow classifier</div>
                        </div>
                        <div class="metric-chip" style="margin-top: 0.75rem;">
                            <div class="metric-label">Response</div>
                            <div class="metric-value">Responsive across devices</div>
                        </div>
                    </div>
                </div>
                <div class="hero-metrics">
                    <div class="metric-chip">
                        <div class="metric-label">Classes</div>
                        <div class="metric-value">No Tumor / Pituitary</div>
                    </div>
                    <div class="metric-chip">
                        <div class="metric-label">History</div>
                        <div class="metric-value">Session only</div>
                    </div>
                    <div class="metric-chip">
                        <div class="metric-label">Formats</div>
                        <div class="metric-value">PNG, JPG, JPEG</div>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    def render_upload_tab():
        left, right = st.columns([1.1, 1.2], gap="large")

        with left:
            st.markdown(
                '<div class="section-title">Upload MRI image</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">Drop one scan and get a result.</div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Choose an MRI image",
                type=["png", "jpg", "jpeg"],
                help="Upload a single MRI image for analysis.",
            )

            image = None
            upload_error = None
            if uploaded_file is not None:
                if (
                    st.session_state.get("last_uploaded_file_name")
                    != uploaded_file.name
                ):
                    st.session_state["latest_prediction"] = None
                upload_error = image_error(uploaded_file)
                if upload_error is None:
                    try:
                        image = load_uploaded_image(uploaded_file)
                        st.session_state["last_uploaded_file_name"] = uploaded_file.name
                        st.image(image, use_container_width=True)
                        meta_left, meta_right, meta_third = st.columns(3)
                        meta_left.metric(
                            "Format", uploaded_file.name.split(".")[-1].upper()
                        )
                        meta_right.metric("Width", str(image.width))
                        meta_third.metric("Height", str(image.height))
                        st.caption(
                            f"{human_file_size(uploaded_file.size)} • {uploaded_file.name}"
                        )
                    except Exception as exc:
                        upload_error = (
                            f"The selected file could not be opened as an image: {exc}"
                        )
                if upload_error:
                    st.session_state["latest_prediction"] = None
                    st.error(upload_error)
            else:
                st.session_state["last_uploaded_file_name"] = None
                st.session_state["latest_prediction"] = None
                st.info("Upload an image to begin.")

        with right:
            st.markdown(
                '<div class="section-title">Prediction dashboard</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">Result, confidence, and class scores.</div>',
                unsafe_allow_html=True,
            )

            can_predict = (
                uploaded_file is not None and image is not None and upload_error is None
            )
            predict_clicked = st.button(
                "Run prediction",
                type="primary",
                use_container_width=True,
                disabled=not can_predict,
            )

            if predict_clicked and can_predict:
                with st.spinner(
                    "Analyzing the MRI scan and generating confidence scores..."
                ):
                    progress = st.progress(10)
                    try:
                        progress.progress(35)
                        result = predict_image_result(image)
                        progress.progress(75)
                        entry = build_history_entry(uploaded_file, image, result)
                        push_history(entry)
                        st.session_state["latest_prediction"] = result | {
                            "file_name": uploaded_file.name
                        }
                        progress.progress(100)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                        st.session_state["latest_prediction"] = None
                    finally:
                        progress.empty()

            latest_prediction = st.session_state.get("latest_prediction")
            if latest_prediction and can_predict:
                confidence = latest_prediction["confidence"]
                confidence_percent = confidence * 100
                status_label = (
                    "High"
                    if confidence >= 0.85
                    else "Moderate" if confidence >= 0.65 else "Review"
                )
                status_color = (
                    "var(--app-success)"
                    if confidence >= 0.85
                    else (
                        "var(--app-warning)"
                        if confidence >= 0.65
                        else "var(--app-danger)"
                    )
                )

                st.markdown(
                    f"""
                    <div class="confidence-card">
                        <div class="confidence-head">
                            <div>
                                <p class="confidence-title">{latest_prediction["label"]}</p>
                                <div class="confidence-score">Confidence breakdown</div>
                            </div>
                            <div class="status-badge"><span style="color: {status_color};">●</span> {status_label}</div>
                        </div>
                        <div class="confidence-track">
                            <div class="confidence-fill" style="width: {confidence_percent:.2f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted class", latest_prediction["label"])
                c2.metric("Confidence", f"{confidence_percent:.1f}%")
                c3.metric(
                    "Session history", str(len(st.session_state["prediction_history"]))
                )

                st.markdown("### Probability chart")
                st.caption("Class scores")
                for label, probability in zip(
                    LABEL_NAMES, latest_prediction["probabilities"]
                ):
                    left_col, right_col = st.columns([8, 2])
                    with left_col:
                        st.markdown(f"**{label}**")
                        st.progress(float(probability))
                    with right_col:
                        st.markdown(
                            f"<div style='padding-top: 1.55rem; text-align: right;'>{probability * 100:.1f}%</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    """
                    <div class="confidence-card">
                        <div class="confidence-head">
                            <div>
                                <p class="confidence-title">No prediction yet</p>
                                <div class="confidence-score">Upload and predict.</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def render_history_tab():
        st.markdown(
            '<div class="section-title">Prediction history</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">Last 10 results.</div>',
            unsafe_allow_html=True,
        )
        history = st.session_state["prediction_history"]
        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download history as CSV",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="brain_tumor_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No predictions have been recorded in this session yet.")

    def render_model_tab():
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(
                '<div class="section-title">Model information</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">Core model specs.</div>',
                unsafe_allow_html=True,
            )
            st.metric("Architecture", "TensorFlow / Keras")
            st.metric("Input resolution", "224 x 224 RGB")
            st.metric("Output classes", "2")
            st.metric("Preprocessing", "Resize + normalize")
        with right:
            st.markdown(
                '<div class="section-title">Operational notes</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">Fast, cached, validated.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="status-badge" style="margin-top: 0.25rem;">Cached model · Normalized input · Session history</div>',
                unsafe_allow_html=True,
            )

    st.set_page_config(
        page_title="Brain Tumor Detection AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_state()
    st.markdown(theme_css(st.session_state["ui_theme"]), unsafe_allow_html=True)

    render_sidebar()
    render_hero()

    analyze_tab, history_tab, model_tab = st.tabs(["Analyze", "History", "Model Info"])
    with analyze_tab:
        render_upload_tab()

    with history_tab:
        render_history_tab()

    with model_tab:
        render_model_tab()


if __name__ == "__main__" and not STREAMLIT_CONTEXT:
    app.run(port=5000)
