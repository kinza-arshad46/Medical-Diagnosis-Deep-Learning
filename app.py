"""
==========================================================
Medical Diagnosis AI Dashboard
==========================================================

Author : Kinza Arshad

Description:
Professional Streamlit Application for
Chest X-ray Medical Diagnosis.
==========================================================
"""

import streamlit as st
from PIL import Image
import pandas as pd
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.predict import Predictor
from src.visualization.gradcam import GradCAM

# --------------------------------------------------------
# Page Configuration
# --------------------------------------------------------

st.set_page_config(
    page_title="Medical Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------
# Custom CSS
# --------------------------------------------------------

st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:bold;
    color:#1565C0;
    text-align:center;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}

.prediction-card{
    padding:20px;
    border-radius:12px;
    background:#F5F5F5;
    box-shadow:0px 2px 8px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Load Models
# --------------------------------------------------------

@st.cache_resource
def load_models():
    predictor = Predictor()
    gradcam = GradCAM()
    return predictor, gradcam

predictor, gradcam = load_models()

# --------------------------------------------------------
# Header
# --------------------------------------------------------

st.markdown(
    "<h1 class='main-title'>🩺 Medical Diagnosis AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Deep Learning Based Chest X-ray Pneumonia Detection System</p>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------

with st.sidebar:

    st.header("Project Information")

    st.success("Model : ResNet50")
    st.success("Classes : 2")
    st.success("Framework : PyTorch")
    st.success("Deployment : Streamlit")

    st.markdown("---")

    st.write("Created by")
    st.info("Kinza Arshad")

# --------------------------------------------------------
# Upload Image
# --------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# IMPORTANT:
# This prevents NameError
predict_button = False

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    predict_button = st.button("🔍 Predict")

# --------------------------------------------------------
# Prediction
# --------------------------------------------------------

prediction = None
top_predictions = None
gradcam_result = None

if predict_button:

    with st.spinner("Analyzing Chest X-ray..."):

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".png"
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            # Prediction
            prediction = predictor.predict_single(
                temp_path
            )

            # Top-K Predictions
            top_predictions = predictor.predict_top_k(
                temp_path,
                top_k=2
            )

            # Grad-CAM
            gradcam_result = gradcam.overlay_heatmap(
                temp_path
            )

            st.success("Prediction completed successfully.")

            st.divider()

            col1, col2 = st.columns(2)

            # -------------------------
            # Left Column
            # -------------------------

            with col1:

                st.subheader("Prediction")

                st.metric(
                    "Diagnosis",
                    prediction["predicted_class"]
                )

                st.metric(
                    "Confidence",
                    f"{prediction['confidence']*100:.2f}%"
                )

                st.metric(
                    "Inference Time",
                    f"{prediction['prediction_time']:.3f} sec"
                )

            # -------------------------
            # Right Column
            # -------------------------

            with col2:

                st.subheader("Prediction Probabilities")

                dataframe = pd.DataFrame(top_predictions)

                dataframe = dataframe.rename(
                    columns={
                        "class": "Class",
                        "probability": "Probability"
                    }
                )

                st.bar_chart(
                    dataframe.set_index("Class")
                )

            st.divider()

            st.subheader("Explainable AI (Grad-CAM)")

            fig = plt.figure(figsize=(8, 8))

            plt.imshow(
                gradcam_result["overlay"]
            )

            plt.axis("off")

            st.pyplot(fig)
            # --------------------------------------------------------
            # Download Results
            # --------------------------------------------------------

            st.divider()
            st.subheader("Download Results")

            prediction_dataframe = pd.DataFrame([prediction])

            csv_data = prediction_dataframe.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📄 Download Prediction Report (CSV)",
                data=csv_data,
                file_name="prediction_report.csv",
                mime="text/csv"
            )

            if (
                isinstance(gradcam_result, dict)
                and "output_path" in gradcam_result
                and Path(gradcam_result["output_path"]).exists()
            ):

                with open(
                    gradcam_result["output_path"],
                    "rb"
                ) as file:

                    st.download_button(
                        label="🔥 Download Grad-CAM Image",
                        data=file,
                        file_name="gradcam_result.png",
                        mime="image/png"
                    )

            st.divider()

            st.success(
                "Analysis completed successfully."
            )

        except Exception as error:

            st.error(
                f"Prediction Failed\n\n{error}"
            )

        finally:

            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()

# --------------------------------------------------------
# Session Prediction History
# --------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file is not None and prediction is not None:

    st.session_state.history.append(
        {
            "Image": uploaded_file.name,
            "Prediction": prediction["predicted_class"],
            "Confidence": prediction["confidence"],
            "Time (sec)": prediction["prediction_time"]
        }
    )

if len(st.session_state.history) > 0:

    st.divider()

    st.subheader("Prediction History")

    history_dataframe = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_dataframe,
        use_container_width=True
    )

# --------------------------------------------------------
# About Project
# --------------------------------------------------------

st.divider()

with st.expander("ℹ About This Project"):

    st.markdown(
        """
### Medical Diagnosis AI

This application uses a Deep Learning model trained on Chest X-ray images to detect Pneumonia.

### Features

- Deep Learning (ResNet50 / DenseNet121)
- Explainable AI using Grad-CAM
- Confidence Score
- Prediction History
- Downloadable Reports
- Streamlit Dashboard

### Disclaimer

This application is intended for educational and research purposes only.

It must not replace professional medical advice or diagnosis.
"""
    )

# --------------------------------------------------------
# Footer
# --------------------------------------------------------

st.markdown(
    """
---

<div style="text-align:center">

<b>Developed by Kinza Arshad</b><br><br>

🩺 Medical Diagnosis AI using Deep Learning<br>

Powered by PyTorch • Streamlit • OpenCV

</div>
""",
    unsafe_allow_html=True
)
