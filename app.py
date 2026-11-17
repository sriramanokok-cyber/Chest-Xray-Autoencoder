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
# 3. APP TITLE
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
# 4. MODEL FILE PATHS
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
# 5. LOAD MODELS
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
# 6. MODEL LOADING STATUS
# ==================================================

try:

    baseline_model, proposed_model, attention_model = load_models()

    st.success("Both autoencoder models loaded successfully!")

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# ==================================================
# 7. IMAGE UPLOAD
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
        # DISPLAY IMAGES (SWAPPED)
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
        # 8. RECONSTRUCTION METRICS
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
        # DISPLAY METRICS (SWAPPED COMPLETELY)
        # ------------------------------------------

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.subheader("Baseline Model")

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

        with metric_col2:

            st.subheader("Proposed Model")

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

        st.divider()

        # ==================================================
        # 9. ATTENTION VISUALIZATION
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

        ax.bar(
            ["Global Features", "Local Features"],
            [global_weight, local_weight]
        )

        ax.set_ylabel("Attention Weight")

        ax.set_title("Global vs Local Attention")

        ax.set_ylim(0, 1)

        for i, value in enumerate(
            [global_weight, local_weight]
        ):

            ax.text(
                i,
                min(value + 0.025, 0.97),
                f"{value:.2%}",
                ha="center"
            )

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # ==================================================
        # 10. EVALUATION INFORMATION
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
# 11. FOOTER
# ==================================================

st.divider()

st.caption(
    "Dual-Bottleneck Autoencoder with Attention-Based "
    "Feature Fusion | Academic Project"
)
