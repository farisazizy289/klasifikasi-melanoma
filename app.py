import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 300
DEFAULT_THRESHOLD = 0.47   # bisa kamu ubah ke 0.45 jika ingin recall lebih tinggi

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_model_finetuned_databalance.keras",
        compile=False,
        custom_objects={
            "tf": tf
        }
    )

model = load_model()

# ==============================
# PREPROCESS IMAGE
# ==============================
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32")
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# ==============================
# STREAMLIT UI
# ==============================
st.title("Klasifikasi Penyakit Kulit Melanoma")

st.markdown("""
Aplikasi ini merupakan **implementasi model EfficientNet-B3**  
untuk klasifikasi **Melanoma vs Non-Melanoma**.

⚠️ **Bukan alat diagnosis medis**  
Digunakan hanya sebagai **alat bantu dan demonstrasi penelitian**.
""")

st.divider()

# Upload image
uploaded_file = st.file_uploader(
    "Upload citra kulit (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# Threshold slider
threshold = st.slider(
    "Threshold Klasifikasi",
    min_value=0.1,
    max_value=0.9,
    value=DEFAULT_THRESHOLD,
    step=0.05
)

# ==============================
# PREDICTION
# ==============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Citra Input", use_column_width=True)

    with col2:
        with st.spinner("Melakukan prediksi..."):
            X = preprocess_image(image)
            prob = float(model.predict(X)[0][0])

            pred_class = "Melanoma" if prob >= threshold else "Non-Melanoma"

        st.subheader("Hasil Prediksi")
        st.write(f"**Kelas:** {pred_class}")
        st.write(f"**Probabilitas Melanoma:** {prob:.3f}")
        st.write(f"**Threshold:** {threshold}")

        if pred_class == "Melanoma":
            st.error("⚠️ Terindikasi Melanoma")
        else:
            st.success("✅ Non-Melanoma")

st.divider()

st.markdown("""
### Catatan
- Model dilatih menggunakan dataset citra penyakit kulit
- Menggunakan **fine-tuning EfficientNet-B3**
- Dilakukan **robustness testing** (noise, blur, brightness, cutout)
- Hasil ini bersifat **pendukung keputusan**, bukan diagnosis akhir
""")
