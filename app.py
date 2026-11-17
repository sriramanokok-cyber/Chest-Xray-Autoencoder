import os
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps


# ==================================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==================================================

st.set_page_config(
    page_title="Dual-Bottleneck Autoencoder Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 10px 20px;
        color: #c9d1d9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 2. CUSTOM ATTENTION LAYER
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
# 3. MODEL FILE PATHS
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
# 4. LOAD MODELS
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

    baseline = tf.keras.models.load_model(baseline_path, compile=False)
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
        outputs=proposed.get_layer("attention_weights").output
    )

    return baseline, proposed, attention


# ==================================================
# 5. SIDEBAR CONTROL PANEL
# ==================================================

with st.sidebar:
    st.title("🫁 Control Panel")
    st.markdown("---")
    
    # Model Loading Status Check inside sidebar
    try:
        baseline_model, proposed_model, attention_model = load_models()
        st.success("Models Loaded Successfully")
    except Exception as e:
        st.error(f"Model Load Failed: {e}")
        st.stop()

    st.markdown("### Input Data")
    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg", "jpeg", "png"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Dual-Bottleneck Autoencoder with Attention-Based "
        "Feature Fusion | Academic Project"
    )


# ==================================================
# 6. MAIN APP HEADER
# ==================================================

st.title("🫁 Chest X-ray Reconstruction System")
st.markdown(
    "Interactive evaluation dashboard comparing baseline and dual-bottleneck "
    "attention autoencoder models for medical image reconstruction."
)
st.divider()


# ==================================================
# 7. PROCESSING & DASHBOARD DISPLAY
# ==================================================

if uploaded_file is not None:
    try:
        # Image Preprocessing
        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original)
        image = original.convert("L").resize((128, 128))
        image_array = np.array(image).astype("float32") / 255.0
        input_image = np.expand_dims(image_array, axis=(0, -1))

        # Run Predictions
        with st.spinner("Processing network predictions..."):
            baseline_reconstruction = baseline_model.predict(input_image, verbose=0)
            proposed_reconstruction = proposed_model.predict(input_image, verbose=0)
            attention_weights = attention_model.predict(input_image, verbose=0)[0]

        baseline_image = baseline_reconstruction[0].squeeze()
        proposed_image = proposed_reconstruction[0].squeeze()

        # Calculate Tensors & Metrics
        original_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
        baseline_tensor = tf.convert_to_tensor(baseline_reconstruction, dtype=tf.float32)
        proposed_tensor = tf.convert_to_tensor(proposed_reconstruction, dtype=tf.float32)

        def calculate_metrics(original, reconstructed):
            mse = float(tf.reduce_mean(tf.square(original - reconstructed)).numpy())
            ssim = float(tf.reduce_mean(tf.image.ssim(original, reconstructed, max_val=1.0)).numpy())
            psnr = float(tf.reduce_mean(tf.image.psnr(original, reconstructed, max_val=1.0)).numpy())
            return mse, ssim, psnr

        baseline_metrics = calculate_metrics(original_tensor, baseline_tensor)
        proposed_metrics = calculate_metrics(original_tensor, proposed_tensor)

        # ==============================================
        # TABBED LAYOUT FOR CLEAN VIEWPORT FIT
        # ==============================================
        tab1, tab2, tab3 = st.tabs([
            "📊 Reconstruction & Performance", 
            "🔬 Attention Feature Fusion", 
            "ℹ️ Evaluation Guidelines"
        ])

        with tab1:
            st.subheader("Image Reconstruction Comparison")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Original X-ray**")
                st.image(image_array, use_container_width=True)

            with col2:
                st.markdown("**Baseline Model**")
                st.image(baseline_image, use_container_width=True)

            with col3:
                st.markdown("**Proposed Model**")
                st.image(proposed_image, use_container_width=True)

            st.divider()
            
            st.subheader("Quantitative Metrics Comparison")
            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.markdown("### Baseline Model")
                st.metric("MSE", f"{proposed_metrics[0]:.5f}", help="Mean Squared Error - lower is better")
                st.metric("SSIM", f"{proposed_metrics[1]:.5f}", help="Structural Similarity - higher is better")
                st.metric("PSNR", f"{proposed_metrics[2]:.2f} dB", help="Peak Signal-to-Noise Ratio - higher is better")

            with metric_col2:
                st.markdown("### Proposed Model")
                st.metric("MSE", f"{baseline_metrics[0]:.5f}", help="Mean Squared Error - lower is better")
                st.metric("SSIM", f"{baseline_metrics[1]:.5f}", help="Structural Similarity - higher is better")
                st.metric("PSNR", f"{baseline_metrics[2]:.2f} dB", help="Peak Signal-to-Noise Ratio - higher is better")

        with tab2:
            st.subheader("Attention-Based Feature Fusion Weights")
            global_weight = float(attention_weights[0])
            local_weight = float(attention_weights[1])

            att_col1, att_col2 = st.columns(2)
            with att_col1:
                st.metric("Global Attention", f"{global_weight:.2%}")
            with att_col2:
                st.metric("Local Attention", f"{local_weight:.2%}")

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(["Global Features", "Local Features"], [global_weight, local_weight], color=['#1f6feb', '#238636'])
            ax.set_ylabel("Attention Weight")
            ax.set_title("Global vs Local Attention Distribution")
            ax.set_ylim(0, 1)

            for i, value in enumerate([global_weight, local_weight]):
                ax.text(i, min(value + 0.03, 0.95), f"{value:.2%}", ha="center", fontweight='bold')

            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            st.subheader("Evaluation Information & Disclaimer")
            st.caption(
                "The metrics displayed above are calculated dynamically for "
                "the currently uploaded image sample. They do not represent "
                "aggregate dataset performance benchmarks."
            )
            st.info(
                "This application demonstrates neural network image reconstruction "
                "and attention-based feature fusion routing. It is strictly for academic "
                "demonstration and is not an active clinical diagnostic tool."
            )
            st.success("Reconstruction pipeline executed successfully.")

    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")

else:
    # Empty State Callout
    st.info("👈 **Get Started:** Please upload a chest X-ray image using the sidebar control panel to run the reconstruction analysis.")
