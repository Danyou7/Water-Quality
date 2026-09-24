import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Kualitas Air",
    page_icon="💧",
    layout="centered"
)

# 2. Memuat Model dan Scaler (Cache agar tidak di-load berulang)
@st.cache_resource
def load_components():
    # Pastikan nama file sesuai dengan yang Anda simpan di tahap pra-pemrosesan
    model = joblib.load('model_random_forest_water_potability.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_components()

# 3. Antarmuka Pengguna (UI) Utama
st.title("💧 Aplikasi Prediksi Kelayakan Air Minum")
st.write("""
Masukkan metrik kualitas air pada form di bawah ini untuk mengetahui apakah air tersebut 
**Layak Minum (Potable)** atau **Tidak Layak Minum (Not Potable)** berdasarkan standar WHO.
""")

st.markdown("---")

# 4. Membuat Form Input
col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH (0 - 14)", min_value=0.0, max_value=14.0, value=7.08)
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=196.36)
    solids = st.number_input("Solids / TDS (ppm)", min_value=0.0, value=22014.09)

with col2:
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.12)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=333.77)
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=426.20)

with col3:
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=14.28)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=66.39)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=3.96)

st.markdown("---")

# 5. Tombol Prediksi
if st.button("🔍 Prediksi Kualitas Air", use_container_width=True):
    # Menyusun data input menjadi DataFrame sesuai urutan fitur saat training
    input_data = pd.DataFrame([[
        ph, hardness, solids, chloramines, sulfate, 
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]], columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])
    
    # Menerapkan Scaling pada data input baru
    input_scaled = scaler.transform(input_data)

    # Melakukan prediksi menggunakan data yang sudah di-scale
    prediction = model.predict(input_scaled)
    
    # Menampilkan hasil
    st.subheader("Hasil Analisis:")
    if prediction[0] == 1:
        st.success("✅ **Air ini LAYAK MINUM (Potable)**. Parameter berada dalam batas aman.")
        st.balloons()
    else:
        st.error("❌ **Air ini TIDAK LAYAK MINUM (Not Potable)**. Konsumsi dapat membahayakan kesehatan.")