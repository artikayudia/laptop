from sklearn.preprocessing import LabelEncoder
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("laptop_prices.csv")

# Menyiapkan fitur dan target
features = ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq']
categorical_features = ['Company', 'Product', 'TypeName', 'OS', 'CPU_company', 'CPU_model', 'PrimaryStorageType', 'SecondaryStorageType', 'GPU_company', 'GPU_model']
target = 'Price_euros'

# Encode fitur kategorikal
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

def safe_transform(encoder, label):
    try:
        # Pastikan label yang diteruskan adalah string dan dalam format yang benar
        return encoder.transform([str(label)])[0]
    except ValueError:  # Jika label tidak ditemukan, tambahkan label baru
        encoder.classes_ = encoder.classes_.tolist() + [str(label)]
        return encoder.transform([str(label)])[0]

X = data[features + categorical_features]
y = data[target]

# Membagi data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Halaman aplikasi Streamlit
st.title("Prediksi Harga Laptop")
st.sidebar.header("Masukkan Spesifikasi Laptop")

# Fungsi untuk memvalidasi input
def transform_input(value, col):
    if value in label_encoders[col].classes_:
        return label_encoders[col].transform([value])[0]
    else:
        st.error(f"Label '{value}' tidak ditemukan pada kolom '{col}'. Gunakan nilai yang valid.")
        st.stop()

# Input pengguna melalui sidebar
company = st.sidebar.selectbox("Perusahaan", label_encoders['Company'].classes_)
product = st.sidebar.selectbox("Produk", label_encoders['Product'].classes_)
typename = st.sidebar.selectbox("Tipe Laptop", label_encoders['TypeName'].classes_)
os = st.sidebar.selectbox("Sistem Operasi", label_encoders['OS'].classes_)
inches = st.sidebar.slider("Ukuran Layar (Inci)", float(data['Inches'].min()), float(data['Inches'].max()), step=0.1)
ram = st.sidebar.selectbox("RAM (GB)", sorted(data['Ram'].unique()))
weight = st.sidebar.slider("Berat (Kg)", float(data['Weight'].min()), float(data['Weight'].max()), step=0.1)
screenw = st.sidebar.slider("Resolusi Layar Lebar (pixel)", int(data['ScreenW'].min()), int(data['ScreenW'].max()), step=1)
screenh = st.sidebar.slider("Resolusi Layar Tinggi (pixel)", int(data['ScreenH'].min()), int(data['ScreenH'].max()), step=1)
cpu_freq = st.sidebar.slider("Frekuensi CPU (GHz)", float(data['CPU_freq'].min()), float(data['CPU_freq'].max()), step=0.1)

# Nilai tukar Euro ke Rupiah
EXCHANGE_RATE = 16000  # Sesuaikan nilai tukar dengan kondisi saat ini

# Fungsi untuk melakukan transformasi input pengguna menjadi data yang bisa digunakan oleh model
def transform_input_data():
    transformed_data = {
        'Inches': inches,
        'Ram': ram,
        'Weight': weight,
        'ScreenW': screenw,
        'ScreenH': screenh,
        'CPU_freq': cpu_freq,
        'Company': transform_input(company, 'Company'),
        'Product': transform_input(product, 'Product'),
        'TypeName': transform_input(typename, 'TypeName'),
        'OS': transform_input(os, 'OS'),
        'CPU_company': 0,  # Tambahkan kolom lain dengan default jika tidak digunakan
        'CPU_model': 0,
        'PrimaryStorageType': 0,
        'SecondaryStorageType': 0,
        'GPU_company': 0,
        'GPU_model': 0
    }
    return pd.DataFrame([transformed_data])

# Menggunakan input pengguna untuk prediksi
if st.sidebar.button("Prediksi Harga"):
    # Transformasikan input pengguna
    input_data = transform_input_data()
    try:
        # Prediksi harga menggunakan model
        prediction_euro = model.predict(input_data)[0]
        prediction_idr = prediction_euro * EXCHANGE_RATE  # Konversi ke Rupiah
        st.success(f"Prediksi harga laptop adalah: Rp{prediction_idr:,.0f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
