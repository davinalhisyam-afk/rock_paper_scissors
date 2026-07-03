import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================================
# 1. KONFIGURASI HALAMAN & LOAD MODEL
# ==========================================
st.set_page_config(page_title="Kalsifikasi Gunting Batu Kertas", layout="centered")

st.title("✌️ Rock-Paper-Scissors Classifier")
st.write("Unggah foto tangan Anda untuk diprediksi oleh model Deep Learning.")

# Fungsi cache agar model tidak di-load ulang setiap kali user klik tombol
@st.cache_resource
def load_my_model():
    # Pastikan file 'rock_paper_scissors_model.h5' berada di folder yang sama dengan file app.py ini
    return load_model('rock_paper_scissors_model.h5')

try:
    model = load_my_model()
    st.success("🤖 Model Berhasil Dimuat!")
except Exception as e:
    st.error("❌ Gagal memuat model. Pastikan file 'rock_paper_scissors_model.h5' berada di folder yang sama.")
    st.stop()

classes = ['paper', 'rock', 'scissors']

# ==========================================
# 2. FITUR UNGGAH GAMBAR
# ==========================================
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diunggah', use_container_width=True)
    
    st.write("🔄 Sedang memproses dan memprediksi...")
    
    # Preprocessing Gambar sesuai kebutuhan MobileNetV2
    # 1. Ubah ukuran ke 150x150
    img_resized = img.resize((150, 150))
    # 2. Ubah ke numpy array
    img_array = image.img_to_array(img_resized)
    # 3. Normalisasi (rescale 1./255)
    img_array = img_array / 255.0
    # 4. Tambah dimensi batch [1, 150, 150, 3]
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    predictions = model.predict(img_array)
    score = predictions[0]
    class_idx = np.argmax(score)
    result_class = classes[class_idx]
    confidence = score[class_idx] * 100
    
    # ==========================================
    # 3. TAMPILKAN HASILNYA DI STREAMLIT
    # ==========================================
    st.subheader("📊 Hasil Analisis Model:")
    
    # Tampilkan hasil utama dengan teks besar (Metric)
    st.metric(label="Prediksi Tangan", value=result_class.upper(), delta=f"{confidence:.2f}% Akurasi")
    
    # Tampilkan detail probabilitas untuk semua kelas dalam bentuk progress bar
    st.write("---")
    st.write("**Detail Probabilitas:**")
    for i, c in enumerate(classes):
        st.write(f"{c.upper()}")
        st.progress(float(score[i]))
        st.caption(f"Persentase: {score[i]*100:.2f}%")