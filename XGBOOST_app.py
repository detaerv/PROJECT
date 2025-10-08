import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Jumlah Pembelian",
    page_icon="🛒",
    layout="wide"
)

# Judul aplikasi
st.title("🛒 Prediksi Jumlah Pembelian dengan XGBoost")
st.markdown("---")

# Load model dan encoder
@st.cache_resource
def load_model():
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, le, feature_columns
    except FileNotFoundError:
        st.error("Model belum tersedia. Silakan latih model terlebih dahulu.")
        return None, None, None

model, le, feature_columns = load_model()

# Sidebar untuk input
st.sidebar.header("📝 Input Data Pelanggan")

# Input features
age = st.sidebar.slider("Usia", 18, 70, 35)
previous_purchases = st.sidebar.number_input("Jumlah Pembelian Sebelumnya", 0, 50, 10)
review_rating = st.sidebar.slider("Rating Review", 1.0, 5.0, 4.0, 0.1)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
item_purchased = st.sidebar.selectbox("Item yang Dibeli", [
    "Blouse", "Dress", "Jacket", "Jeans", "Pants", "Shirt", "Shoes", 
    "Shorts", "Skirt", "Sweater", "T-shirt", "Other"
])
category = st.sidebar.selectbox("Kategori", [
    "Accessories", "Clothing", "Footwear", "Outerwear"
])
location = st.sidebar.selectbox("Lokasi", [
    "Alabama", "Alaska", "Arizona", "California", "Colorado", 
    "Florida", "Georgia", "Illinois", "Indiana", "Kentucky",
    "Louisiana", "Massachusetts", "Michigan", "Montana", "Nevada",
    "New York", "Ohio", "Oregon", "Pennsylvania", "Texas"
])
size = st.sidebar.selectbox("Ukuran", ["S", "M", "L", "XL"])
color = st.sidebar.selectbox("Warna", [
    "Beige", "Black", "Blue", "Brown", "Charcoal", "Cyan", 
    "Gold", "Gray", "Green", "Lavender", "Magenta", "Maroon",
    "Olive", "Orange", "Peach", "Pink", "Purple", "Red",
    "Silver", "Teal", "Turquoise", "White", "Yellow"
])
season = st.sidebar.selectbox("Musim", ["Fall", "Winter", "Spring", "Summer"])
payment_method = st.sidebar.selectbox("Metode Pembayaran", [
    "Credit Card", "Cash", "Debit Card", "PayPal", "Venmo", "Bank Transfer"
])
shipping_type = st.sidebar.selectbox("Tipe Pengiriman", [
    "Express", "Free Shipping", "Next Day Air", "Standard", "Store Pickup", "2-Day Shipping"
])
frequency_of_purchases = st.sidebar.selectbox("Frekuensi Pembelian", [
    "Weekly", "Fortnightly", "Monthly", "Quarterly", "Annually", "Bi-Weekly", "Every 3 Months"
])

subscription_status = st.sidebar.radio("Status Langganan", ["Yes", "No"])
discount_applied = st.sidebar.radio("Diskon Diterapkan", ["Yes", "No"])
promo_code_used = st.sidebar.radio("Kode Promo Digunakan", ["Yes", "No"])

# Tombol prediksi
if st.sidebar.button("🔮 Prediksi Jumlah Pembelian", type="primary"):
    if model is not None:
        # Konversi input ke format yang sesuai
        input_data = {
            'age': age,
            'previous_purchases': previous_purchases,
            'review_rating': review_rating,
            'subscription_status': 1 if subscription_status == "Yes" else 0,
            'discount_applied': 1 if discount_applied == "Yes" else 0,
            'promo_code_used': 1 if promo_code_used == "Yes" else 0,
        }
        
        # One-hot encoding untuk variabel kategorikal
        categorical_features = {
            'gender': gender,
            'item_purchased': item_purchased,
            'category': category,
            'location': location,
            'size': size,
            'color': color,
            'season': season,
            'payment_method': payment_method,
            'frequency_of_purchases': frequency_of_purchases
        }
        
        # Buat DataFrame dengan semua kolom yang diperlukan
        df_input = pd.DataFrame([input_data])
        
        # Tambahkan kolom kategorikal untuk one-hot encoding
        for key, value in categorical_features.items():
            df_input[key] = value
        
        # One-hot encoding
        df_encoded = pd.get_dummies(df_input, columns=list(categorical_features.keys()), drop_first=True)
        
        # Label encoding untuk shipping_type
        df_encoded['shipping_type'] = shipping_type
        if le is not None:
            df_encoded['shipping_type'] = le.transform([shipping_type])[0]
        
        # Pastikan semua kolom sesuai dengan training
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Urutkan kolom sesuai dengan training
        df_encoded = df_encoded[feature_columns]
        
        # Prediksi
        prediction = model.predict(df_encoded)[0]
        
        # Tampilkan hasil
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💰 Prediksi Jumlah Pembelian", f"${prediction:.2f}")
        
        with col2:
            confidence = "Tinggi" if 20 <= prediction <= 80 else "Sedang"
            st.metric("📊 Confidence Level", confidence)
        
        with col3:
            category_avg = {"Accessories": 45, "Clothing": 55, "Footwear": 65, "Outerwear": 70}
            diff = prediction - category_avg.get(category, 50)
            st.metric("📈 vs Rata-rata Kategori", f"${diff:+.2f}")
        
        # Visualisasi
        st.markdown("---")
        st.subheader("📊 Analisis Prediksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Jumlah Pembelian (USD)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart perbandingan
            comparison_data = pd.DataFrame({
                'Kategori': ['Prediksi Anda', 'Rata-rata Kategori', 'Rata-rata Keseluruhan'],
                'Jumlah': [prediction, category_avg.get(category, 50), 50]
            })
            fig2 = px.bar(comparison_data, x='Kategori', y='Jumlah', 
                         title='Perbandingan dengan Rata-rata',
                         color='Jumlah', color_continuous_scale='blues')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Insight
        st.markdown("---")
        st.subheader("💡 Insight")
        
        insights = []
        if prediction > 70:
            insights.append("✅ Pelanggan ini termasuk dalam kategori **high-value customer**")
        elif prediction < 30:
            insights.append("⚠️ Pelanggan ini termasuk dalam kategori **low-value customer**")
        else:
            insights.append("ℹ️ Pelanggan ini termasuk dalam kategori **medium-value customer**")
        
        if subscription_status == "Yes":
            insights.append("📧 Pelanggan berlangganan - kemungkinan pembelian berulang tinggi")
        
        if discount_applied == "Yes" or promo_code_used == "Yes":
            insights.append("🎁 Pelanggan responsif terhadap promosi dan diskon")
        
        if previous_purchases > 20:
            insights.append("⭐ Pelanggan loyal dengan riwayat pembelian tinggi")
        
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.error("Model belum tersedia!")

# Informasi model
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    ### Model XGBoost untuk Prediksi Jumlah Pembelian
    
    Model ini dibangun menggunakan algoritma **XGBoost (Extreme Gradient Boosting)** untuk memprediksi 
    jumlah pembelian pelanggan berdasarkan berbagai fitur seperti:
    
    - Demografi pelanggan (usia, gender, lokasi)
    - Karakteristik produk (kategori, ukuran, warna)
    - Perilaku pembelian (frekuensi, riwayat pembelian)
    - Status promosi (diskon, kode promo, langganan)
    
    **Metrik Evaluasi:**
    - R-squared (R²): Mengukur seberapa baik model menjelaskan variabilitas data
    - RMSE: Root Mean Squared Error untuk mengukur akurasi prediksi
    - MAE: Mean Absolute Error untuk mengukur rata-rata kesalahan absolut
    
    **Cara Menggunakan:**
    1. Isi semua input di sidebar kiri
    2. Klik tombol "Prediksi Jumlah Pembelian"
    3. Lihat hasil prediksi dan analisis yang ditampilkan
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Dibuat dengan ❤️ menggunakan Streamlit & XGBoost</p>
    </div>
    """, 
    unsafe_allow_html=True
)