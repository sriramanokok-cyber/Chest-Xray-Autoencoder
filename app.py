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
# 2. STYLING
# ==================================================

NAVY = "#12304A"
TEAL = "#1F7A8C"
SLATE = "#5B6B7A"
LINE = "#DCE3E9"
PAPER = "#F5F8FA"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stMetric {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

.block-container {{
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}

/* Hero banner */
.hero {{
    background: linear-gradient(120deg, {NAVY} 0%, #1B4E68 100%);
    border-radius: 14px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.5rem;
}}
.hero h1 {{
    color: #FFFFFF;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 .4rem 0;
    padding: 0;
    text-align: left;
}}
.hero .sub {{
    color: #BFD6E2;
    font-size: 1.02rem;
    font-weight: 500;
    margin: 0 0 .6rem 0;
}}
.hero .desc {{
    color: #E3EEF4;
    font-size: .95rem;
    line-height: 1.55;
    max-width: 760px;
    margin: 0;
}}

/* Section headings */
h2 {{
    color: {NAVY};
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    border-left: 4px solid {TEAL};
    padding: .1rem 0 .1rem .75rem !important;
    margin-top: .5rem !important;
}}

/* Panel titles above images / metrics */
.panel-title {{
    text-align: center;
    color: {NAVY};
    font-weight: 600;
    font-size: 1.02rem;
    padding: .55rem .4rem;
    background: {PAPER};
    border: 1px solid {LINE};
    border-radius: 8px 8px 0 0;
    margin-bottom: .6rem;
}}
.panel-title.proposed {{
    background: {TEAL};
    border-color: {TEAL};
    color: #FFFFFF;
}}

/* Images centered and equal size */
div[data-testid="stImage"] {{
    display: flex;
    justify-content: center;
}}
div[data-testid="stImage"] img {{
    border-radius: 6px;
    border: 1px solid {LINE};
}}

/* Metric cards */
div[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: .9rem 1.1rem;
    text-align: center;
}}
div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"] {{
    justify-content: center;
    color: {SLATE};
}}
div[data-testid="stMetricValue"] {{
    justify-content: center;
    color: {NAVY};
    font-weight: 600;
}}

/* Divider spacing */
hr {{
    margin: 1.6rem 0 !important;
    border-color: {LINE} !important;
}}

/* Footer */
.footer {{
    text-align: center;
    color: {SLATE};
    font-size: .85rem;
    padding-top: .4rem;
}}
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

st.markdown("""
<div class="hero">
    <h1>🫁 Chest X-ray Reconstruction System</h1>
    <p class="sub">Dual-Bottleneck Autoencoder with Attention-Based Feature Fusion</p>
    <p class="desc">
        An interactive demonstration of chest X-ray reconstruction comparing a
        baseline autoencoder with a proposed dual-bottleneck attention autoencoder.
    </p>
</div>
""", unsafe_allow_html=True)


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

    # Prefer the retrained baseline if it is included
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

    st.success("Both autoencoder models loaded successfully.")

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# ==================================================
# 8. IMAGE UPLOAD
# ==================================================

st.header("1. Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image (JPG, JPEG or PNG)",
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

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown(
                '<div class="panel-title">Original X-ray</div>',
                unsafe_allow_html=True
            )
            st.image(image_array, width=300, clamp=True)

        with col2:
            st.markdown(
                '<div class="panel-title">Baseline Autoencoder</div>',
                unsafe_allow_html=True
            )
            st.image(baseline_image, width=300, clamp=True)

        with col3:
            st.markdown(
                '<div class="panel-title proposed">Proposed Dual-Bottleneck Model</div>',
                unsafe_allow_html=True
            )
            st.image(proposed_image, width=300, clamp=True)

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
        # DISPLAY METRICS (each model shows its own results)
        # ------------------------------------------

        metric_col1, metric_col2 = st.columns(2, gap="large")

        with metric_col1:

            st.markdown(
                '<div class="panel-title">Baseline Model</div>',
                unsafe_allow_html=True
            )

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "MSE",
                f"{baseline_metrics[0]:.5f}",
                help="Mean Squared Error - lower is better"
            )
            m2.metric(
                "SSIM",
                f"{baseline_metrics[1]:.5f}",
                help="Structural Similarity - higher is better"
            )
            m3.metric(
                "PSNR",
                f"{baseline_metrics[2]:.2f} dB",
                help="Peak Signal-to-Noise Ratio - higher is better"
            )

        with metric_col2:

            st.markdown(
                '<div class="panel-title proposed">Proposed Model</div>',
                unsafe_allow_html=True
            )

            p1, p2, p3 = st.columns(3)
            p1.metric(
                "MSE",
                f"{proposed_metrics[0]:.5f}",
                help="Mean Squared Error - lower is better"
            )
            p2.metric(
                "SSIM",
                f"{proposed_metrics[1]:.5f}",
                help="Structural Similarity - higher is better"
            )
            p3.metric(
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

        att_col1, att_col2 = st.columns(2, gap="large")

        with att_col1:
            st.metric("Global Attention", f"{global_weight:.2%}")

        with att_col2:
            st.metric("Local Attention", f"{local_weight:.2%}")

        # ------------------------------------------
        # ATTENTION BAR CHART
        # ------------------------------------------

        fig, ax = plt.subplots(figsize=(8, 3.6))

        bars = ax.bar(
            ["Global Features", "Local Features"],
            [global_weight, local_weight],
            color=[NAVY, TEAL],
            width=0.45
        )

        ax.set_ylabel("Attention Weight", color=SLATE)
        ax.set_title("Global vs Local Attention", color=NAVY,
                     fontweight="bold", pad=12)
        ax.set_ylim(0, 1)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(LINE)
        ax.tick_params(colors=SLATE)
        ax.yaxis.grid(True, color=LINE, linewidth=0.8)
        ax.set_axisbelow(True)

        for i, value in enumerate([global_weight, local_weight]):
            ax.text(
                i,
                min(value + 0.025, 0.97),
                f"{value:.2%}",
                ha="center",
                color=NAVY,
                fontweight="semibold"
            )

        fig.tight_layout()

        chart_left, chart_mid, chart_right = st.columns([1, 2, 1])
        with chart_mid:
            st.pyplot(fig)

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
            "Image reconstruction and attention analysis completed."
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

st.markdown(
    '<div class="footer">Dual-Bottleneck Autoencoder with '
    'Attention-Based Feature Fusion &nbsp;|&nbsp; Academic Project</div>',
    unsafe_allow_html=True
)
