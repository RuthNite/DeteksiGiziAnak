import streamlit as st
st.set_page_config(page_title="Prediksi Gizi Balita", layout="centered")

import pickle
import numpy as np
import pandas as pd
import base64

# -------------------------------
# Fungsi untuk encode gambar lokal jadi base64
# -------------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# -------------------------------
# Tambah Wallpaper Gambar Lokal
# -------------------------------
img_base64 = get_base64("posyandu.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"], [data-testid="stToolbar"] {{
    background-color: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# -------------------------------
# Load Model LightGBM
# -------------------------------
with open('lgbm_terlatih_80.pkl', 'rb') as f:
    model = pickle.load(f)

# -------------------------------
# Judul Aplikasi
# -------------------------------
st.title("💡 Prediksi Status BB/U Anak")
st.markdown("Aplikasi ini memprediksi status **Berat Badan menurut Umur (BB/U)** anak berdasarkan data antropometri.")

# -------------------------------
# Input Data Pengguna
# -------------------------------
st.header("🧒 Input Data Balita")

jk = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
usia = st.number_input("Usia Saat Ukur (bulan) min:0, max:60", min_value=0, max_value=60, value=12)
berat = st.number_input("Berat Badan (kg) min:0, max:30", min_value=0.0, max_value=30.0, step=0.1, value=8.0, format="%.1f")
tinggi = st.number_input("Tinggi Badan (cm) min:30, max:150", min_value=30.0, max_value=150.0, step=0.1, value=70.0, format="%.1f")

jk_encoded = 1 if jk == "Laki-laki" else 0
input_data = pd.DataFrame([[jk_encoded, usia, berat, tinggi]],
                          columns=['JK', 'Usia Saat Ukur', 'Berat', 'Tinggi'])

# -------------------------------
# Fungsi Prediksi dan Tampilan
# -------------------------------
if st.button("🔍 Prediksi"):
    prediction = model.predict(input_data)
    status = prediction[0]
    st.success(f"Hasil Prediksi Status BB/U: **{status}**")

    # Kelompok Usia
    def kategori_umur(usia):
        if usia <= 12:
            return "0-12"
        elif usia <= 24:
            return "13-24"
        elif usia <= 36:
            return "25-36"
        elif usia <= 48:
            return "37-48"
        else:
            return "49-60"

    kategori = kategori_umur(usia)

    # Tabel Ideal Berat dan Tinggi
    data_laki = {
        "Rentang Umur (bln)": ["0-12", "13-24", "25-36", "37-48", "49-60"],
        "Berat Ideal (kg)": [7.0, 9.5, 12.0, 14.0, 16.0],
        "Tinggi Ideal (cm)": [70, 80, 88, 95, 102],
        "Kategori": ["Normal"] * 5
    }

    data_perempuan = {
        "Rentang Umur (bln)": ["0-12", "13-24", "25-36", "37-48", "49-60"],
        "Berat Ideal (kg)": [6.5, 9.0, 11.0, 13.0, 15.0],
        "Tinggi Ideal (cm)": [68, 78, 86, 93, 100],
        "Kategori": ["Normal"] * 5
    }

    df_ideal = pd.DataFrame(data_laki if jk == "Laki-laki" else data_perempuan)

    # Tampilkan Informasi Tambahan
    st.markdown("### ℹ️ Informasi Tambahan")
    st.write(f"📌 **Kelompok usia anak Anda: `{kategori}` bulan**")
    st.markdown("### 📋 Tabel Ideal Berdasarkan Jenis Kelamin")
    st.dataframe(df_ideal, use_container_width=True)

    # Rekomendasi Berdasarkan Status
    st.markdown("### 🧾 Rekomendasi Gizi")
    if status == "Kurang":
        st.warning(
            "Anak Anda termasuk kategori **gizi kurang**. Pastikan makan 3–5 kali sehari dengan porsi kecil, beri camilan sehat, dan nutrisi cukup. "
            "Pastikan anak anda makan 3x sehari. Jika porsi makan sedikit, ubah jadwal makannya menjadi 4-5x sehari dengan porsi yang kecil. "
            "Kenalkan anak dengan konsep kenyang dan lapar dengan cara membuat jadwal makan, contohnya memberikan jeda makan selama 2 jam. " 
            "Misalnya, anak makan jam 12.00 maka berikan camilan pada jam 14.00. "
            "Berikan camilan sehat untuk anak sebanyak 1-2x sehari. "
            "Berikan makan dan minuman yang kaya nutrisi, seperti susu. "
            "Hindari minuman yang mengandung gula tinggi, seperti sirup dan minuman bersoda. "
            "Hindari memberikan minum terlalu banyak karena mudah menyebabkan kenyang. "
            "Jika diperlukan, segera konsultasikan dengan tenaga kesehatan atau dokter gizi."
        )
    elif status == "Lebih":
        st.warning(
            "Terapkan pola makan sehat, seperti mengonsumsi buah, sayur, dan susu tanpa lemak serta perbanyak asupan air putih. "
            "Kurangi konsumsi minuman dan makanan yang mengandung tinggi lemak dan manis. "
            "Membimbing anak untuk melakukan aktivitas fisik seperti berjalan-jalan atau olahraga ringan. "
            "Mengurangi waktu pasif anak dengan ajak melakukan aktivitas aktif, seperti membaca buku dan bermain bersama. "
            "Diskusikan dengan ahli gizi jika diperlukan."
        )
    else:
        st.success(
            "Status gizi anak Anda tergolong **normal**. Lanjutkan pemantauan berkala dan pastikan kebutuhan gizinya tetap terpenuhi."
        )