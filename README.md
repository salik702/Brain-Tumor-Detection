<div align="center">

<!-- Animated Banner -->
<img src="https://capsule-render.vercel.app/api?type=shark&color=0:07111f,50:0f4c75,100:07111f&height=200&section=header&text=BRAIN%20TUMOR%20DETECTION&fontSize=62&fontColor=ffffff&animation=fadeIn&fontAlignY=45&desc=Deep%20Learning%20MRI%20Classifier&descAlignY=68&descSize=20&descColor=93c5fd" width="100%"/>

<!-- Typing Animation -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=800&size=22&duration=2600&pause=700&color=60A5FA&center=true&vCenter=true&repeat=true&width=820&height=55&lines=%F0%9F%A7%A0+MRI+Brain+Tumor+Classification+System;%F0%9F%9A%80+Streamlit+UI+%2B+Flask+API+Hybrid;%F0%9F%8E%AF+Predicts+No+Tumor+or+Pituitary+Tumor;%F0%9F%A7%AC+TensorFlow+Keras+Computer+Vision+Pipeline" alt="Typing SVG" />
</a>

<br/>

<!-- Badges -->
<p>
  <img src="https://img.shields.io/badge/MODEL-CNN%20%2F%20Keras-2563eb?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=07111f" />
  <img src="https://img.shields.io/badge/UI-STREAMLIT-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=07111f" />
  <img src="https://img.shields.io/badge/API-FLASK-000000?style=for-the-badge&logo=flask&logoColor=white&labelColor=07111f" />
  <img src="https://img.shields.io/badge/STATUS-READY-22c55e?style=for-the-badge&logo=statuspage&logoColor=white&labelColor=07111f" />
</p>

<p>
  <img src="https://img.shields.io/badge/IMAGES-MRI%20SCANS-0ea5e9?style=for-the-badge&logo=googlecloudstorage&logoColor=white&labelColor=07111f" />
  <img src="https://img.shields.io/badge/OUTPUT-NO%20TUMOR%20%7C%20PITUITARY%20TUMOR-f97316?style=for-the-badge&logo=brain&logoColor=white&labelColor=07111f" />
  <img src="https://img.shields.io/badge/DEPLOYMENT-LOCAL%20%2F%20RENDER-14b8a6?style=for-the-badge&logo=render&logoColor=white&labelColor=07111f" />
</p>

<br/>

<!-- Live Preview Button -->
<a href="https://brain-tumor-detection-salik702.streamlit.app/" target="_blank">
  <img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-CLICK%20TO%20LAUNCH%20APP-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=07111f" />
</a>

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> LIVE.PREVIEW — TRY IT NOW`

<div align="center">

<a href="https://brain-tumor-detection-salik702.streamlit.app/" target="_blank">
  <img src="https://img.shields.io/badge/%E2%9A%A1%20STREAMLIT%20APP-brain--tumor--detection--salik702.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=07111f" />
</a>

<br/><br/>

> 🧠 **Upload any MRI brain scan and get an instant prediction — No Tumor or Pituitary Tumor — directly in your browser. No installation required.**

| `ACTION` | `LINK` |
| :------: | :----- |
| 🌐 Open Live App | [brain-tumor-detection-salik702.streamlit.app](https://brain-tumor-detection-salik702.streamlit.app/) |
| 📤 Upload MRI | Drag & drop your `.jpg` / `.png` MRI scan |
| 🎯 Get Prediction | Receive class label + confidence score instantly |

</div>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>



https://github.com/user-attachments/assets/1a58b731-90c2-4537-9a2f-6690e5b69752



## `> SYSTEM.INIT — WHAT IS THIS PROJECT?`

**Brain Tumor Detection** is a deep learning web application for classifying MRI images into two categories: **No Tumor** and **Pituitary Tumor**. The project combines a trained Keras model with a polished Streamlit interface and a Flask-based prediction API, so it can be used both as an interactive app and as a backend service.

The application loads the saved model artifacts from the repository root and returns a prediction score with confidence for each uploaded MRI image.

<br/>

<div align="center">

|       `MODULE`       | `ROLE`                                                                     |   `STATE`   |
| :------------------: | :------------------------------------------------------------------------- | :---------: |
|   🧠 Model Loader    | Loads `model_complete.keras`, then falls back to `model.json` + `model.h5` | `✅ READY`  |
|  🖼️ Image Pipeline   | Accepts MRI uploads, resizes to model input, normalizes pixels             | `✅ ACTIVE` |
| 📈 Prediction Engine | Produces class probabilities and confidence scores                         | `🟢 ONLINE` |
|   🎛️ Streamlit UI    | User-facing interface for image upload and result display                  |  `⚡ LIVE`  |
|     🌐 Flask API     | `/home` health route and `/` prediction route                              |  `⚡ LIVE`  |

</div>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> PROJECT.PREVIEW — WHAT YOU GET`

This project includes:

- A trained brain tumor classifier for MRI scans
- A Streamlit-based app UI with custom styling
- A Flask API for programmatic predictions
- Saved model artifacts for direct loading and fallback support
- Training notebooks and dataset folders for experimentation

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> DETECTION.PIPELINE — HOW IT WORKS`

```
╔══════════════════════════════════════════════════════════════════╗
║               BRAIN TUMOR DETECTION EXECUTION FLOW              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   [01]  🖼️  MRI IMAGE UPLOAD                                     ║
║         └─► User uploads one or more MRI scans                   ║
║                                                                  ║
║   [02]  🔧  IMAGE PREPROCESSING                                   ║
║         └─► Exif transpose, RGB conversion, resize to 224x224    ║
║         └─► Pixel values normalized to the 0–1 range             ║
║                                                                  ║
║   [03]  🧠  MODEL LOADING                                         ║
║         └─► Prefer `model_complete.keras`                        ║
║         └─► Fallback to `model.json` + `model.h5`                ║
║         └─► Final fallback uses a VGG16-based architecture       ║
║                                                                  ║
║   [04]  📊  PREDICTION                                            ║
║         └─► `model.predict(...)` returns class probabilities     ║
║         └─► Output mapped to No Tumor / Pituitary Tumor         ║
║                                                                  ║
║   [05]  ✅  RESULT RENDERING                                      ║
║         └─► Confidence and probability values displayed in UI    ║
║         └─► API returns scores for integration use               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> MODEL.SPEC — ARCHITECTURE DETAILS`

<div align="center">

| `PARAMETER`          |                               `VALUE`                               |
| :------------------- | :-----------------------------------------------------------------: |
| 🧪 Task              |                   Brain MRI image classification                    |
| 🚀 Backend           |                         TensorFlow / Keras                          |
| 📐 Input Size        |                           `224 x 224 x 3`                           |
| 🏷️ Output Classes    |                    `No Tumor`, `Pituitary Tumor`                    |
| 📈 Training Accuracy |                                `98%`                                |
| 🧪 Testing Accuracy  |                                `86%`                                |
| 🧯 Fallback Strategy | `model_complete.keras` → `model.json` + `model.h5` → VGG16 fallback |
| 🌐 Interfaces        |                      Streamlit UI + Flask API                       |

</div>

<br/>

### `> LABEL MAPPING`

- `0` → `No Tumor`
- `1` → `Pituitary Tumor`

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> STACK.LOAD — TECHNOLOGIES`

<div align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />

</div>

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> LOCAL.SETUP — RUN IT ON YOUR MACHINE`

```bash
# Clone the repository
 git clone <your-repo-url>
 cd Brain-Tumor-Detection

# Create a virtual environment
 python -m venv venv

# Activate on Windows
 .\venv\Scripts\activate

# Activate on macOS/Linux
 source venv/bin/activate

# Install the packages used by the app
 pip install streamlit tensorflow flask flask-cors opencv-python numpy pandas pillow pydantic

# Start the app
 streamlit run app.py
```

If you are using the provided `brain_tumor_env` folder, activate it instead of creating a new environment:

```bash
# Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\brain_tumor_env\Scripts\Activate.ps1
streamlit run app.py
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> API.ROUTES — BACKEND ENDPOINTS`

| `METHOD` | `ROUTE` | `DESCRIPTION`                                               |
| :------: | :------ | :---------------------------------------------------------- |
|  `GET`   | `/home` | Health check route that returns `Hello World`               |
|  `POST`  | `/`     | Accepts base64 encoded images and returns prediction scores |

<br/>

### Example Request

```json
{
  "image": ["data:image/png;base64,..."]
}
```

### Example Response

```json
{
  "result": [0.0183]
}
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> PROJECT.STRUCTURE — REPO LAYOUT`

```text
Brain Tumor Detection/
├── app.py
├── Brain Tumor.ipynb
├── model_complete.keras
├── model.h5
├── model.json
├── model.weights.h5
├── brain_tumor_model.pkl
├── Brain_Tumor_Data/
├── client/
├── Procfile
└── README.md
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> DEPLOYMENT.NOTES — IMPORTANT`

- Keep the trained model files in the repository root so the loader can find them.
- The app prefers `model_complete.keras`, then tries `model.json` with `model.h5`, and finally falls back to a VGG16-based builder.
- If you retrain the model, update the saved artifacts before deploying.
- Large dataset folders are useful for training, but you do not need them for inference-only deployment.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

## `> CONTRIBUTION.PROTOCOL — GET INVOLVED`

Contributions are welcome. Useful improvements include better preprocessing, expanded class support, evaluation metrics, UI refinements, and deployment hardening.

```bash
git checkout -b feature/your-improvement
git add .
git commit -m "feat: describe your change"
git push origin feature/your-improvement
```

### Good contribution ideas

- Add more tumor classes
- Improve model accuracy and evaluation reporting
- Add Grad-CAM or explainability overlays
- Add batch upload support
- Improve deployment docs and environment pinning

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/fire.png" width="100%"/>

<div align="center">

<!-- Animated Name + Role Waving Footer Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:07111f,40:0d2137,70:0f4c75,100:07111f&height=200&section=footer&text=SALIK%20AHMAD&fontSize=52&fontColor=ffffff&animation=twinkling&fontAlignY=45&desc=AI%20%2F%20ML%20Engineer%20%E2%80%A2%20Deep%20Learning%20Developer&descAlignY=68&descSize=16&descColor=93c5fd" width="100%"/>

<br/>

<!-- Animated Typing Footer -->
<a href="https://salikahmad.vercel.app/">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=16&duration=3000&pause=900&color=60A5FA&center=true&vCenter=true&repeat=true&width=820&height=45&lines=Building+AI+that+sees+what+humans+miss.;MRI+tumor+detection+via+deep+learning.;Streamlit+UI+%2B+Flask+API+%2B+Keras+CNN.;Visit+%E2%86%92+salikahmad.vercel.app" alt="Footer Typing" />
</a>

<br/><br/>

<!-- Skill Capsules — always render, no username dependency -->
<img src="https://capsule-render.vercel.app/api?type=soft&color=0:07111f,100:0f4c75&height=60&text=Python%20%20%7C%20%20TensorFlow%20%20%7C%20%20Keras%20%20%7C%20%20OpenCV&fontSize=18&fontColor=93c5fd&animation=fadeIn" width="80%"/>

<br/>

<img src="https://capsule-render.vercel.app/api?type=soft&color=0:07111f,100:0f4c75&height=60&text=Streamlit%20%20%7C%20%20Flask%20%20%7C%20%20Deep%20Learning%20%20%7C%20%20Computer%20Vision&fontSize=18&fontColor=93c5fd&animation=fadeIn" width="80%"/>

<br/><br/>

<!-- Social Links — shields.io only, always reliable -->
<a href="https://salikahmad.vercel.app/" target="_blank">
  <img src="https://img.shields.io/badge/🌐%20Website-salikahmad.vercel.app-38bdf8?style=for-the-badge&labelColor=07111f&color=0f4c75" />
</a>
&nbsp;
<a href="https://www.linkedin.com/in/salik-ahmad-programmer/" target="_blank">
  <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=07111f" />
</a>
&nbsp;
<a href="https://www.kaggle.com/salikahmad702" target="_blank">
  <img src="https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white&labelColor=07111f" />
</a>
&nbsp;
<a href="https://github.com/SalikAhmad702" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-Profile-ffffff?style=for-the-badge&logo=github&logoColor=black&labelColor=07111f" />
</a>

<br/><br/>

<!-- Static info badges — shields.io, no external API needed -->
<img src="https://img.shields.io/badge/FOCUS-Deep%20Learning%20%2F%20AI-60a5fa?style=for-the-badge&labelColor=07111f" />
&nbsp;
<img src="https://img.shields.io/badge/DOMAIN-Medical%20Imaging-f97316?style=for-the-badge&labelColor=07111f" />
&nbsp;
<img src="https://img.shields.io/badge/STACK-Python%20%2F%20TensorFlow-22c55e?style=for-the-badge&labelColor=07111f" />

<br/><br/>

<sub>⭐ Star this repo if it helped you — it keeps the project alive and visible.</sub>

</div>
