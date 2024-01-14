import streamlit as st
import pandas as pd
import numpy as np

# Dictionary Tabel Gejala 
gejala = {
    'G1': 'Nyeri dada',
    'G2': 'Diare selama beberapa hari',
    'G3': 'Sakit kepala',
    'G4': 'Jantung berdebar',
    'G5': 'Lelah',
    'G6': 'Sukar tidur',
    'G7': 'Cepat marah',
    'G8': 'Ingatan melemah',
    'G9': 'Tak mampu berkonsentrasi',
    'G10': 'Daya kemampuan berkurang',
    'G11': 'Tidak tahan terhadap suara atau gangguan lain',
    'G12': 'Emosi tidak terkendali'
}

# Tabel Jenis-jenis stres
jenis_stres = {
    'JK1': 'Stres Ringan',
    'JK2': 'Stres Sedang',
    'JK3': 'Stres Berat'
}

# Tabel Kaidah Produksi
kaidah_produksi = pd.DataFrame({
    'Kode Gejala': list(gejala.keys()),
    'JK1': [0.9, 0.8, 0.6, 0.8, 0.6, 0.3, 0.5, 0.7, 0.7, 0.7, 0.6, 0.9],
    'JK2': [0.7, 0.6, 0.7, 0.6, 0.7, 0.8, 0.6, 0.5, 0.9, 0.3, 0.6, 0.7],
    'JK3': [0.6, 0.9, 0.7, 0.9, 0.5, 0.7, 0.9, 0.9, 0.6, 0.8, 0.8, 0.9]
})

# Tabel Jenis-jenis stres dan Nilai Bayes
nilai_bayes = {
    'JK1': 1.70,
    'JK2': 4.10,
    'JK3': 3.10
}

# Naive Bayes Model
def predict_stress(gejala_input):
    evidence = kaidah_produksi.loc[kaidah_produksi['Kode Gejala'].isin(gejala_input)]#Mencari baris di kaidah_produksi yang sesuai dengan gejala yang dipilih.
    result = {}

    for jenis in jenis_stres:
        prob = np.prod(evidence[jenis])#Menghitung probabilitas dari gejala yang dipilih berdasarkan jenis stres saat ini.
        result[jenis] = prob * nilai_bayes[jenis]#Menggabungkan probabilitas dengan nilai Bayes untuk mendapatkan probabilitas akhir.

    # Normalisasi probabilitas
    total_prob = sum(result.values())#Menghitung total probabilitas dari semua jenis stres.
    for jenis in jenis_stres:
        result[jenis] /= total_prob#Mengnormalisasi probabilitas untuk memastikan total probabilitas adalah 1.

    return max(result, key=result.get), result#Mengembalikan jenis stres dengan probabilitas tertinggi dan semua probabilitas.

# Streamlit App
st.title('Aplikasi Klasifikasi Tingkat Stres Naive Bayes')
st.title('Nama Kelompok:')
st.title('5210411260 Aldianto Dickyu Septian')
st.title('5210411262 Yusuf Ashari')
st.title('5210411270 Fajar Ramadhan')
st.title('5210411276 Naninda Uswatun Hasanah')
st.title('5210411281 Rayhan Dzikri Fauzi')

st.write('Masukkan gejala yang dialami oleh pasien untuk mendapatkan prediksi tingkat stres.')




input_gejala = []
for k, v in gejala.items():
    if st.checkbox(v):
        input_gejala.append(k)

if st.button('Prediksi'):
    if not input_gejala:
        st.warning('Pilih setidaknya satu gejala.')
    else:
        predicted_stress, confidence = predict_stress(input_gejala)
        solusi = {
            'JK1': 'Atur pola hidup sehat',
            'JK2': 'Atur pola hidup sehat, perbanyak ibadah, dan istirahat',
            'JK3': 'Atur pola hidup sehat, perbanyak ibadah, istirahat, dan segera konseling ke dokter psikologi'
        }

        st.success(f'Tingkat stres yang diprediksi: {jenis_stres[predicted_stress]}')
        st.success(f'Tingkat keyakinan: {confidence[predicted_stress] * 100:.2f}%')
        st.info(f'Solusi: {solusi[predicted_stress]}')
