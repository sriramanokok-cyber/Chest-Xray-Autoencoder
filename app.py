
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Dual-Bottleneck Autoencoder",
    page_icon="🫁",
    layout="wide"
)


# ==================================================
# 2. CUSTOM UI DESIGN
# ==================================================

st.markdown("""
<style>

/* Application background */
.stApp {
    background: linear-gradient(135deg, #0b1220 0%, #111c30 100%);
    color: #f1f5f9;
}

/* Full-width layout */
.block-container {
    max-width: 100% !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
}

/* Main title */
h1 {
    color: #e2e8f0 !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
}

/* Section headings */
h2 {
    color: #60a5fa !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

h3 {
    color: #cbd5e1 !important;
    font-size: 1.15rem !important;
    font-weight: 650 !important;
}

/* General text */
p, label, .stCaption {
    color: #cbd5e1 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.8rem !important;
    font-weight: 750 !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.65);
    border: 1px dashed #60a5fa;
    border-radius: 14px;
    padding: 15px;
}

/* Image containers */
[data-testid="stImage"] {
    background: #172337;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 6px;
    overflow: hidden;
}

/* Status messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #334155 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #3b82f6;
    background: #2563eb;
    color: white;
    font-weight: 600;
}

/* Chart container */
[data-testid="stPyplot"] {
    background: rgba(30, 41, 59, 0.65);
    border-radius: 14px;
    padding: 10px;
}

/* Footer */
footer {
    visibility: hidden;
}

/* Responsive layout */
@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    h1 {
        font-size: 1.65rem !important;
    }

    h2 {
        font-size: 1.3rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. CUSTOM ATTENTION LAYER
# ==================================================

@tf.keras.utils.register_keras_serializable()
class AttentionSlice(tf.keras.layers.Layer):

    def __init__(self, index, **kwargs):
        kwargs.pop("mask", None)
        super().__init__(**kwargs)
        self.index = index
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return inputs[:, self.index:self.index + 1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


# ==================================================
# 4. APP TITLE
# ==================================================

st.title("🫁 Chest X-ray Reconstruction System")

st.markdown("""
### Dual-Bottleneck Autoencoder with Attention-Based Feature Fusion

An interactive demonstration of chest X-ray reconstruction
using a baseline autoencoder and a proposed dual-bottleneck
attention autoencoder.
""")

st.divider()


# ==================================================
# 5. MODEL FILE PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_model_file(possible_names):

    for filename in possible_names:

        path = os.path.join(BASE_DIR, filename)

        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            return path

    available_files = os.listdir(BASE_DIR)

    raise FileNotFoundError(
        "Model file not found.\n"
        f"Expected one of: {possible_names}\n"
        f"Files available: {available_files}"
    )


# ==================================================
# 6. LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():

    baseline_path = find_model_file([
        "baseline_autoencoder_retrained.keras",
        "baseline_autoencoder.keras",
        "baseline_autoencoder .keras",
        "baseline_autoencoder (1).keras"
    ])

    proposed_path = find_model_file([
        "dual_bottleneck_attention.keras"
    ])

    baseline = tf.keras.models.load_model(
        baseline_path,
        compile=False
    )

    proposed = tf.keras.models.load_model(
        proposed_path,
        compile=False,
        custom_objects={
            "AttentionSlice": AttentionSlice,
            "Custom>AttentionSlice": AttentionSlice
        }
    )

    attention = tf.keras.Model(
        inputs=proposed.input,
        outputs=proposed.get_layer(
            "attention_weights"
        ).output
    )

    return baseline, proposed, attention


# ==================================================
# 7. MODEL LOADING STATUS
# ==================================================

try:

    baseline_model, proposed_model, attention_model = load_models()

    st.success("Both autoencoder models loaded successfully!")

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# ==================================================
# 8. IMAGE UPLOAD
# ==================================================

st.header("1. Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        # ------------------------------------------
        # IMAGE PREPROCESSING
        # ------------------------------------------

        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original)

        image = original.convert("L").resize((128, 128))

        image_array = (
            np.array(image).astype("float32") / 255.0
        )

        input_image = np.expand_dims(
            image_array,
            axis=(0, -1)
        )

        # ------------------------------------------
        # RECONSTRUCTION
        # ------------------------------------------

        st.header("2. Reconstruction Comparison")

        with st.spinner("Reconstructing chest X-ray..."):

            baseline_reconstruction = baseline_model.predict(
                input_image,
                verbose=0
            )

            proposed_reconstruction = proposed_model.predict(
                input_image,
                verbose=0
            )

            attention_weights = attention_model.predict(
                input_image,
                verbose=0
            )[0]

        baseline_image = baseline_reconstruction[0].squeeze()

        proposed_image = proposed_reconstruction[0].squeeze()

        # ------------------------------------------
        # DISPLAY IMAGES
        # ------------------------------------------

        col1, col2, col3 = st.columns([1, 1.3, 1])

        with col1:

            st.subheader("Original X-ray")

            st.image(
                image_array,
                use_container_width=True
            )

        with col2:

            st.subheader("Baseline Autoencoder")

            st.image(
                baseline_image,
                use_container_width=True
            )

        with col3:

            st.subheader("Proposed Dual-Bottleneck Model")

            st.image(
                proposed_image,
                use_container_width=True
            )

        st.divider()

        # ==================================================
        # 9. RECONSTRUCTION METRICS
        # ==================================================

        st.header("3. Reconstruction Performance")

        original_tensor = tf.convert_to_tensor(
            input_image,
            dtype=tf.float32
        )

        baseline_tensor = tf.convert_to_tensor(
            baseline_reconstruction,
            dtype=tf.float32
        )

        proposed_tensor = tf.convert_to_tensor(
            proposed_reconstruction,
            dtype=tf.float32
        )

        # ------------------------------------------
        # METRIC CALCULATION
        # ------------------------------------------

        def calculate_metrics(original, reconstructed):

            mse = float(
                tf.reduce_mean(
                    tf.square(original - reconstructed)
                ).numpy()
            )

            ssim = float(
                tf.reduce_mean(
                    tf.image.ssim(
                        original,
                        reconstructed,
                        max_val=1.0
                    )
                ).numpy()
            )

            psnr = float(
                tf.reduce_mean(
                    tf.image.psnr(
                        original,
                        reconstructed,
                        max_val=1.0
                    )
                ).numpy()
            )

            return mse, ssim, psnr

        baseline_metrics = calculate_metrics(
            original_tensor,
            baseline_tensor
        )

        proposed_metrics = calculate_metrics(
            original_tensor,
            proposed_tensor
        )

        # ------------------------------------------
        # DISPLAY METRICS
        # ------------------------------------------

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.subheader("Baseline Model")

            st.metric(
                "MSE",
                f"{baseline_metrics[0]:.5f}",
                help="Mean Squared Error - lower is better"
            )

            st.metric(
                "SSIM",
                f"{baseline_metrics[1]:.5f}",
                help="Structural Similarity - higher is better"
            )

            st.metric(
                "PSNR",
                f"{baseline_metrics[2]:.2f} dB",
                help="Peak Signal-to-Noise Ratio - higher is better"
            )

        with metric_col2:

            st.subheader("Proposed Model")

            st.metric(
                "MSE",
                f"{proposed_metrics[0]:.5f}",
                help="Mean Squared Error - lower is better"
            )

            st.metric(
                "SSIM",
                f"{proposed_metrics[1]:.5f}",
                help="Structural Similarity - higher is better"
            )

            st.metric(
                "PSNR",
                f"{proposed_metrics[2]:.2f} dB",
                help="Peak Signal-to-Noise Ratio - higher is better"
            )

        st.divider()

        # ==================================================
        # 10. ATTENTION VISUALIZATION
        # ==================================================

        st.header("4. Attention-Based Feature Fusion")

        global_weight = float(attention_weights[0])

        local_weight = float(attention_weights[1])

        att_col1, att_col2 = st.columns(2)

        with att_col1:

            st.metric(
                "Global Attention",
                f"{global_weight:.2%}"
            )

        with att_col2:

            st.metric(
                "Local Attention",
                f"{local_weight:.2%}"
            )

        # ------------------------------------------
        # ATTENTION BAR CHART
        # ------------------------------------------

        fig, ax = plt.subplots(figsize=(8, 4))

        fig.patch.set_facecolor("#172337")
        ax.set_facecolor("#172337")

        bars = ax.bar(
            ["Global Features", "Local Features"],
            [global_weight, local_weight],
            color=["#38bdf8", "#818cf8"],
            width=0.55
        )

        ax.set_ylabel(
            "Attention Weight",
            color="#cbd5e1"
        )

        ax.set_title(
            "Global vs Local Attention",
            color="#f1f5f9",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_ylim(0, 1)

        ax.tick_params(colors="#cbd5e1")

        for spine in ax.spines.values():
            spine.set_color("#475569")

        ax.grid(
            axis="y",
            alpha=0.15
        )

        for bar, value in zip(
            bars,
            [global_weight, local_weight]
        ):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 0.97),
                f"{value:.2%}",
                ha="center",
                color="white",
                fontweight="bold"
            )

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        st.divider()

        # ==================================================
        # 11. EVALUATION INFORMATION
        # ==================================================

        st.header("5. Evaluation Information")

        st.caption(
            "The metrics displayed above are calculated for "
            "the currently uploaded image. They are not "
            "aggregate test-dataset results."
        )

        st.info(
            "This application demonstrates image reconstruction "
            "and attention-based feature fusion. It is not a "
            "pneumonia diagnosis system."
        )

        st.success(
            "Image reconstruction and attention analysis completed!"
        )

    except Exception as e:

        st.error(
            f"Error processing the uploaded image: {e}"
        )


else:

    st.info(
        "Upload a chest X-ray image to begin the demonstration."
    )


# ==================================================
# 12. FOOTER
# ==================================================

st.divider()

st.caption(
    "Dual-Bottleneck Autoencoder with Attention-Based "
    "Feature Fusion | Academic Project"
)
