import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Nilai Ujian", layout="wide")

# Judul dan Deskripsi
st.title("🎓 Aplikasi Prediksi Nilai Ujian Siswa")
st.markdown("""
Aplikasi ini memprediksi skor ujian akhir berdasarkan input data siswa menggunakan 
model **Linear Regression** yang telah dilatih sebelumnya.
""")

# --- FUNGSI LOAD MODEL & OPSI ---
@st.cache_resource
def load_resources():
    # 1. Load Model
    try:
        model = joblib.load('Linear_Regression_(Default)_model.joblib')
    except FileNotFoundError:
        st.error("⚠️ File model 'Linear_Regression_(Default)_model.joblib' tidak ditemukan. Pastikan file berada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat model: {e}")
        st.stop()

    # 2. Load Data untuk Opsi Dropdown (Agar pilihan sesuai data asli)
    try:
        df = pd.read_csv('Exam_Score_Prediction.csv')
        options = {
            'gender': sorted(df['gender'].unique().tolist()),
            'course': sorted(df['course'].unique().tolist()),
            'internet_access': sorted(df['internet_access'].unique().tolist()),
            'study_method': sorted(df['study_method'].unique().tolist()),
            'sleep_quality': ['poor', 'average', 'good'],      # Urutan logis
            'facility_rating': ['low', 'medium', 'high'],    # Urutan logis
            'exam_difficulty': ['easy', 'moderate', 'hard']    # Urutan logis (sesuaikan jika data asli menggunakan 'low')
        }
    except FileNotFoundError:
        # Fallback jika CSV tidak ada, gunakan opsi default umum
        options = {
            'gender': ['male', 'female', 'other'],
            'course': ['b.tech', 'b.sc', 'b.com', 'ba', 'bca', 'diploma'], # Contoh umum
            'internet_access': ['yes', 'no'],
            'study_method': ['self-study', 'group study', 'online videos', 'coaching'],
            'sleep_quality': ['poor', 'average', 'good'],
            'facility_rating': ['low', 'moderate', 'high'],
            'exam_difficulty': ['easy', 'moderate', 'hard']
        }
    
    return model, options

# Memuat resources
pipeline, options = load_resources()

# --- FORM INPUT DATA (SIDEBAR) ---
st.sidebar.header("📝 Input Data Siswa")
st.sidebar.info("Sesuaikan input di bawah ini:")

with st.sidebar.form("prediction_form"):
    
    # Kelompok 1: Demografi
    st.subheader("Data Diri")
    age = st.number_input("Usia (Age)", min_value=15, max_value=40, value=18, step=1)
    gender = st.selectbox("Jenis Kelamin", options['gender'])
    course = st.selectbox("Jurusan (Course)", options['course'])
    
    # Kelompok 2: Kebiasaan Belajar
    st.subheader("Akademik & Belajar")
    study_hours = st.number_input("Jam Belajar/Hari", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
    class_attendance = st.number_input("Kehadiran Kelas (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    study_method = st.selectbox("Metode Belajar", options['study_method'])
    
    # Kelompok 3: Gaya Hidup & Lingkungan
    st.subheader("Faktor Lainnya")
    sleep_hours = st.number_input("Jam Tidur/Hari", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    sleep_quality = st.selectbox("Kualitas Tidur", options['sleep_quality'])
    internet_access = st.selectbox("Akses Internet", options['internet_access'])
    facility_rating = st.selectbox("Fasilitas Sekolah", options['facility_rating'])
    exam_difficulty = st.selectbox("Tingkat Kesulitan Ujian", options['exam_difficulty'])

    st.markdown("---")
    submit_button = st.form_submit_button(label='🔍 Prediksi Nilai')

# --- BAGIAN HASIL PREDIKSI ---
if submit_button:
    # 1. Membuat DataFrame Single Row dari Input
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'course': [course],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_access],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality],
        'study_method': [study_method],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty]
    })

    # 2. Tampilkan Input User (Opsional, agar user yakin)
    st.subheader("Data yang Anda Masukkan:")
    st.dataframe(input_data)

    # 3. Prediksi
    try:
        with st.spinner('Menghitung prediksi...'):
            prediction = pipeline.predict(input_data)[0]
        
        # 4. Tampilkan Output
        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="Prediksi Skor Akhir", value=f"{prediction:.2f}")
        
        with col_res2:
            if prediction >= 80:
                st.success("🌟 **Luar Biasa!** Potensi nilai sangat tinggi.")
            elif prediction >= 60:
                st.info("✅ **Bagus.** Nilai diprediksi lulus dengan baik.")
            else:
                st.warning("⚠️ **Perhatian.** Nilai diprediksi rendah, pertimbangkan untuk menambah jam belajar.")
                
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi. Pastikan format data model sesuai.\nError: {e}")