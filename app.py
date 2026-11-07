
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

# Custom attention slicing layer used by the proposed model
@tf.keras.utils.register_keras_serializable()
class AttentionSlice(tf.keras.layers.Layer):

    def __init__(self, index, **kwargs):
        kwargs.pop("mask", None)
        super().__init__(**kwargs)
        self.index = index
        self.supports_masking = True

    def call(self, inputs):
        return inputs[:, self.index:self.index + 1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Dual-Bottleneck Autoencoder",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Chest X-ray Reconstruction System")

st.markdown("""
**Dual-Bottleneck Autoencoder with Attention-Based Feature Fusion**

An interactive demonstration comparing a baseline autoencoder
with a proposed dual-bottleneck attention autoencoder.
""")

st.info(
    "Educational research demonstration only. "
    "This application does not diagnose pneumonia or other diseases."
)

# -----------------------------
# LOAD MODELS
# -----------------------------

@st.cache_resource
def load_models():
    baseline = tf.keras.models.load_model(
        "baseline_autoencoder.keras",
        compile=False
    )

    proposed = tf.keras.models.load_model(
        "dual_bottleneck_attention.keras",
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



try:
    baseline_model, proposed_model, attention_model = load_models()
    st.success("Both autoencoder models loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# -----------------------------
# IMAGE UPLOAD
# -----------------------------
st.header("1. Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    original = Image.open(uploaded_file)
    original = ImageOps.exif_transpose(original)

    image = original.convert("L").resize((128, 128))

    image_array = np.array(image).astype("float32") / 255.0

    input_image = np.expand_dims(
        np.expand_dims(image_array, axis=0),
        axis=-1
    )

    st.header("2. Reconstruction Comparison")

    with st.spinner("Reconstructing image using both models..."):

        baseline_reconstruction = baseline_model.predict(
            input_image, verbose=0
        )

        proposed_reconstruction = proposed_model.predict(
            input_image, verbose=0
        )

        attention_weights = attention_model.predict(
            input_image, verbose=0
        )[0]

    baseline_image = baseline_reconstruction[0].squeeze()
    proposed_image = proposed_reconstruction[0].squeeze()

    # -----------------------------
    # DISPLAY IMAGES
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image_array, use_container_width=True)

    with col2:
        st.subheader("Baseline Model")
        st.image(baseline_image, use_container_width=True)

    with col3:
        st.subheader("Proposed Model")
        st.image(proposed_image, use_container_width=True)

    # -----------------------------
    # RECONSTRUCTION METRICS
    # -----------------------------
    st.header("3. Reconstruction Performance")

    original_tensor = tf.convert_to_tensor(
        input_image, dtype=tf.float32
    )

    baseline_tensor = tf.convert_to_tensor(
        baseline_reconstruction, dtype=tf.float32
    )

    proposed_tensor = tf.convert_to_tensor(
        proposed_reconstruction, dtype=tf.float32
    )

    def calculate_metrics(original, reconstructed):

        mse = float(
            tf.reduce_mean(
                tf.square(original - reconstructed)
            ).numpy()
        )

        ssim = float(
            tf.reduce_mean(
                tf.image.ssim(original, reconstructed, max_val=1.0)
            ).numpy()
        )

        psnr = float(
            tf.reduce_mean(
                tf.image.psnr(original, reconstructed, max_val=1.0)
            ).numpy()
        )

        return mse, ssim, psnr

    baseline_metrics = calculate_metrics(
        original_tensor, baseline_tensor
    )

    proposed_metrics = calculate_metrics(
        original_tensor, proposed_tensor
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.subheader("Baseline Metrics")
        st.metric("MSE", f"{baseline_metrics[0]:.5f}")
        st.metric("SSIM", f"{baseline_metrics[1]:.5f}")
        st.metric("PSNR", f"{baseline_metrics[2]:.2f} dB")

    with metric_col2:
        st.subheader("Proposed Model Metrics")
        st.metric("MSE", f"{proposed_metrics[0]:.5f}")
        st.metric("SSIM", f"{proposed_metrics[1]:.5f}")
        st.metric("PSNR", f"{proposed_metrics[2]:.2f} dB")

    # -----------------------------
    # ATTENTION VISUALIZATION
    # -----------------------------
    st.header("4. Attention-Based Feature Fusion")

    global_weight = float(attention_weights[0])
    local_weight = float(attention_weights[1])

    att_col1, att_col2 = st.columns(2)

    with att_col1:
        st.metric("Global Attention", f"{global_weight:.2%}")

    with att_col2:
        st.metric("Local Attention", f"{local_weight:.2%}")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["Global Features", "Local Features"],
        [global_weight, local_weight]
    )

    ax.set_ylabel("Attention Weight")
    ax.set_title("Global vs Local Attention")
    ax.set_ylim(0, 1)

    for i, value in enumerate([global_weight, local_weight]):
        ax.text(i, value + 0.02, f"{value:.2%}", ha="center")

    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------
    # TEST DATASET RESULTS
    # -----------------------------
    st.header("5. Previously Evaluated Test-Set Results")

    st.caption(
        "These are your previously measured results on the 624-image "
        "test set, not metrics for the uploaded image."
    )

    results_col1, results_col2 = st.columns(2)

    with results_col1:
        st.subheader("Baseline")
        st.write("MSE: 0.05679")
        st.write("SSIM: 0.33537")
        st.write("PSNR: 12.60 dB")

    with results_col2:
        st.subheader("Proposed")
        st.write("MSE: 0.00478")
        st.write("SSIM: 0.61800")
        st.write("PSNR: 23.47 dB")

    st.success("Image reconstruction and attention analysis completed!")

else:
    st.info("Upload a chest X-ray image to begin the demonstration.")

st.divider()

st.caption(
    "Dual-Bottleneck Autoencoder with Attention-Based Feature Fusion "
    "| Academic Project"
)
