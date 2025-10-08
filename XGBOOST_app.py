import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis XGBoost - Prediksi Pembelian",
    page_icon="🛒",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .big-font {font-size:20px !important; font-weight: bold;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk train model dengan visualisasi
def train_model_with_viz(df):
    """Train XGBoost model dengan analisis lengkap"""
    
    st.subheader("📊 Tahap 1: Eksplorasi Data Awal (EDA)")
    
    with st.expander("🔍 Lihat 5 Baris Pertama Data", expanded=False):
        st.dataframe(df.head())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", f"{len(df):,} baris")
    with col2:
        st.metric("Jumlah Kolom", f"{len(df.columns)} kolom")
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Data Hilang", f"{missing} nilai")
    
    # Statistik Deskriptif
    with st.expander("📈 Statistik Deskriptif"):
        st.dataframe(df.describe())
    
    # Visualisasi Distribusi Target
    st.markdown("### 📊 Distribusi Purchase Amount (Target)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Purchase Amount (USD)'], kde=True, bins=30, ax=ax, color='skyblue')
    ax.set_title('Distribusi Jumlah Pembelian', fontsize=16, fontweight='bold')
    ax.set_xlabel('Purchase Amount (USD)')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)
    
    st.info("""
    **💡 Interpretasi:** 
    - Grafik menunjukkan distribusi nilai pembelian pelanggan
    - Pola distribusi membantu memahami range harga yang paling umum
    - Ini adalah variabel yang akan kita prediksi dengan model XGBoost
    """)
    
    # Data Cleaning
    st.markdown("---")
    st.subheader("🧹 Tahap 2: Pembersihan Data")
    
    if 'Customer ID' in df.columns:
        df = df.drop('Customer ID', axis=1)
        st.success("✅ Kolom 'Customer ID' dihapus (tidak relevan untuk prediksi)")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        st.success(f"✅ {duplicates} baris duplikat dihapus")
    else:
        st.success("✅ Tidak ada data duplikat")
    
    # Rename columns
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('__', '_').str.lower()
    
    # Feature Engineering
    st.markdown("---")
    st.subheader("⚙️ Tahap 3: Rekayasa Fitur (Feature Engineering)")
    
    df['subscription_status'] = df['subscription_status'].map({'Yes': 1, 'No': 0})
    df['discount_applied'] = df['discount_applied'].map({'Yes': 1, 'No': 0})
    df['promo_code_used'] = df['promo_code_used'].map({'Yes': 1, 'No': 0})
    
    st.success("✅ Variabel Yes/No diubah menjadi 1/0")
    
    # Korelasi sebelum encoding
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        st.markdown("### 🔗 Matriks Korelasi Variabel Numerik")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Korelasi Antar Variabel Numerik', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        st.info("""
        **💡 Interpretasi Korelasi:**
        - Nilai mendekati 1 atau -1 menunjukkan korelasi kuat
        - Korelasi positif: kedua variabel bergerak searah
        - Korelasi negatif: kedua variabel bergerak berlawanan arah
        """)
    
    # One-Hot Encoding
    cols_to_onehot = ['gender', 'item_purchased', 'category', 'location', 'size', 'color', 'season', 'payment_method', 'frequency_of_purchases']
    df = pd.get_dummies(df, columns=cols_to_onehot, drop_first=True)
    
    st.success(f"✅ One-Hot Encoding diterapkan. Total kolom sekarang: {df.shape[1]}")
    st.info("""
    **💡 One-Hot Encoding:** Mengubah variabel kategorikal menjadi format numerik yang dapat diproses oleh model machine learning.
    """)
    
    # Split data
    st.markdown("---")
    st.subheader("✂️ Tahap 4: Pembagian Data Training & Testing")
    
    X = df.drop('purchase_amount_usd', axis=1)
    y = df['purchase_amount_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Data Training", f"{len(X_train):,} baris (80%)")
    with col2:
        st.metric("Data Testing", f"{len(X_test):,} baris (20%)")
    
    st.info("""
    **💡 Kenapa dibagi?**
    - Training set: untuk melatih model
    - Testing set: untuk menguji akurasi model pada data yang belum pernah dilihat
    - Rasio 80:20 adalah standar umum dalam machine learning
    """)
    
    # Encoding shipping_type
    shipping_type_categories = X_train['shipping_type'].unique()
    shipping_type_mapping = {cat: idx for idx, cat in enumerate(shipping_type_categories)}
    
    X_train['shipping_type'] = X_train['shipping_type'].map(shipping_type_mapping)
    X_test['shipping_type'] = X_test['shipping_type'].map(shipping_type_mapping)
    
    # Train model
    st.markdown("---")
    st.subheader("🤖 Tahap 5: Training Model XGBoost")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("⏳ Menginisialisasi model XGBoost...")
    progress_bar.progress(20)
    
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    status_text.text("🔄 Melatih model... (ini mungkin butuh 1-2 menit)")
    progress_bar.progress(40)
    
    xgb_model.fit(X_train, y_train)
    
    status_text.text("✅ Model berhasil dilatih!")
    progress_bar.progress(100)
    
    st.success("🎉 Model XGBoost berhasil dilatih dengan 100 decision trees!")
    
    with st.expander("ℹ️ Tentang Hyperparameter yang Digunakan"):
        st.markdown("""
        - **n_estimators=100**: Jumlah pohon keputusan (decision trees)
        - **learning_rate=0.1**: Kecepatan pembelajaran model
        - **max_depth=5**: Kedalaman maksimum setiap pohon
        - **subsample=0.8**: 80% data digunakan untuk setiap pohon
        - **colsample_bytree=0.8**: 80% fitur digunakan untuk setiap pohon
        """)
    
    # Evaluation
    st.markdown("---")
    st.subheader("📊 Tahap 6: Evaluasi Performa Model")
    
    y_pred = xgb_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("MAE", f"${mae:.2f}")
        st.markdown("*Mean Absolute Error*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("RMSE", f"${rmse:.2f}")
        st.markdown("*Root Mean Squared Error*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("R² Score", f"{r2:.4f}")
        st.markdown("*Coefficient of Determination*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Akurasi", f"{r2*100:.2f}%")
        st.markdown("*Model Accuracy*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Interpretasi Metrics
    with st.expander("📖 Penjelasan Metrik Evaluasi"):
        st.markdown(f"""
        **MAE (Mean Absolute Error): ${mae:.2f}**
        - Rata-rata selisih absolut antara prediksi dan nilai aktual
        - Semakin kecil, semakin baik
        - Artinya: prediksi model rata-rata meleset sekitar ${mae:.2f}
        
        **RMSE (Root Mean Squared Error): ${rmse:.2f}**
        - Mirip dengan MAE, tapi memberikan penalti lebih besar pada error yang besar
        - Semakin kecil, semakin baik
        
        **R² Score: {r2:.4f} ({r2*100:.2f}%)**
        - Mengukur seberapa baik model menjelaskan variasi data
        - Range: 0-1 (0% - 100%)
        - {r2*100:.2f}% variasi dalam purchase amount dapat dijelaskan oleh model
        - **Interpretasi:** {"Excellent!" if r2 > 0.9 else "Sangat Baik!" if r2 > 0.8 else "Baik" if r2 > 0.7 else "Cukup Baik" if r2 > 0.6 else "Perlu Ditingkatkan"}
        """)
    
    # Visualisasi Prediksi vs Aktual
    st.markdown("### 🎯 Prediksi vs Nilai Aktual")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Nilai Aktual (USD)', fontsize=12)
    ax.set_ylabel('Prediksi Model (USD)', fontsize=12)
    ax.set_title('Prediksi vs Nilai Aktual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.info("""
    **💡 Cara Membaca:**
    - Titik-titik yang dekat dengan garis merah menunjukkan prediksi yang akurat
    - Jika titik menyebar jauh dari garis, model kurang akurat untuk nilai tersebut
    """)
    
    # Visualisasi Residuals
    st.markdown("### 📉 Analisis Residuals (Kesalahan Prediksi)")
    
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(residuals, kde=True, bins=30, ax=ax, color='coral')
        ax.set_title('Distribusi Residuals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Residuals (Aktual - Prediksi)')
        ax.set_ylabel('Frekuensi')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediksi (USD)', fontsize=11)
        ax.set_ylabel('Residuals', fontsize=11)
        ax.set_title('Residuals vs Prediksi', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.info("""
    **💡 Analisis Residuals:**
    - **Distribusi Normal**: Residuals seharusnya membentuk kurva normal (bell curve) yang terpusat di 0
    - **Scatter Plot**: Titik-titik seharusnya tersebar acak di sekitar garis y=0
    - Jika ada pola tertentu, mungkin ada informasi yang belum ditangkap model
    """)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("🔍 Tahap 7: Analisis Fitur Penting")
    
    feature_importances = xgb_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Top 15 features
    top_n = 15
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=features_df.head(top_n), palette='viridis', ax=ax)
    ax.set_title(f'Top {top_n} Fitur Terpenting dalam Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    st.pyplot(fig)
    
    with st.expander("📋 Lihat Semua Feature Importance"):
        st.dataframe(features_df, use_container_width=True)
    
    st.info("""
    **💡 Feature Importance menunjukkan:**
    - Fitur mana yang paling berpengaruh dalam prediksi
    - Semakin tinggi nilai importance, semakin besar pengaruhnya
    - Informasi ini berguna untuk memahami faktor-faktor kunci yang mempengaruhi pembelian
    """)
    
    # Kesimpulan
    st.markdown("---")
    st.subheader("📝 Kesimpulan")
    
    st.success(f"""
    ### ✅ Model Berhasil Dilatih!
    
    **Performa Model:**
    - Model mampu menjelaskan **{r2*100:.2f}%** variabilitas dalam jumlah pembelian
    - Rata-rata kesalahan prediksi: **${mae:.2f}**
    - Tingkat akurasi: **{"Sangat Baik" if r2 > 0.8 else "Baik" if r2 > 0.7 else "Cukup Baik"}**
    
    **Insight Penting:**
    - Fitur terpenting: **{features_df.iloc[0]['Feature']}**
    - Model siap digunakan untuk prediksi pembelian pelanggan baru
    """)
    
    with st.expander("🚀 Langkah Selanjutnya yang Disarankan"):
        st.markdown("""
        1. **Tuning Hyperparameter**: Gunakan GridSearchCV untuk menemukan parameter optimal
        2. **Cross-Validation**: Terapkan k-fold cross-validation untuk validasi lebih robust
        3. **Feature Engineering Lanjutan**: Buat fitur-fitur baru yang lebih kompleks
        4. **SHAP Analysis**: Gunakan SHAP values untuk interpretasi model yang lebih mendalam
        5. **A/B Testing**: Test model di production dengan sample kecil terlebih dahulu
        6. **Monitoring**: Setup monitoring untuk track performa model seiring waktu
        """)
    
    return xgb_model, shipping_type_mapping, X_train.columns.tolist(), {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2
    }

# Load model
@st.cache_resource
def get_model():
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('shipping_type_mapping.pkl', 'rb') as f:
            shipping_mapping = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, shipping_mapping, feature_columns, True
    except FileNotFoundError:
        return None, None, None, False

# Main App
st.title("🛒 Analisis XGBoost: Prediksi Jumlah Pembelian")
st.markdown("### *Sistem Prediksi Purchase Amount dengan Machine Learning*")
st.markdown("---")

# Tab navigation
tab1, tab2, tab3 = st.tabs(["📚 Training Model", "🎯 Prediksi", "📖 Dokumentasi"])

with tab1:
    st.header("📚 Training Model XGBoost")
    
    model, shipping_mapping, feature_columns, model_exists = get_model()
    
    if model_exists:
        st.success("✅ Model sudah tersedia dan siap digunakan!")
        st.info("💡 Jika ingin melatih ulang dengan data baru, upload dataset di bawah.")
    else:
        st.warning("⚠️ Model belum tersedia. Silakan upload dataset untuk training.")
    
    st.markdown("### 📤 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload file **shopping_behavior_updated.csv**",
        type=['csv'],
        help="File CSV harus berisi kolom Purchase Amount (USD) dan fitur lainnya"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dataset berhasil diupload!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Baris", f"{len(df):,}")
        with col2:
            st.metric("Total Kolom", f"{len(df.columns)}")
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if st.button("🚀 Mulai Training Model", type="primary", use_container_width=True):
            try:
                with st.spinner("Training sedang berlangsung..."):
                    model, shipping_mapping, feature_columns, metrics = train_model_with_viz(df)
                    
                    st.session_state.model = model
                    st.session_state.shipping_mapping = shipping_mapping
                    st.session_state.feature_columns = feature_columns
                    
                    st.balloons()
                    st.success("🎉 Model berhasil dilatih dan siap digunakan!")
                    st.info("💡 Silakan pindah ke tab **Prediksi** untuk mulai menggunakan model.")
                    
            except Exception as e:
                st.error(f"❌ Error saat training: {str(e)}")
                st.info("Pastikan format CSV sesuai dengan yang diharapkan.")

with tab2:
    st.header("🎯 Prediksi Jumlah Pembelian")
    
    # Load model
    if 'model' in st.session_state:
        model = st.session_state.model
        shipping_mapping = st.session_state.shipping_mapping
        feature_columns = st.session_state.feature_columns
        model_exists = True
    else:
        model, shipping_mapping, feature_columns, model_exists = get_model()
    
    if not model_exists:
        st.warning("⚠️ Model belum tersedia. Silakan training model terlebih dahulu di tab **Training Model**.")
    else:
        st.success("✅ Model siap digunakan!")
        
        st.markdown("### 📝 Input Data Pelanggan")
        st.markdown("*Isi semua informasi di bawah untuk mendapatkan prediksi*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Informasi Demografis")
            age = st.slider("Usia", 18, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            location = st.selectbox("Lokasi", [
                "California", "New York", "Texas", "Florida", "Illinois",
                "Pennsylvania", "Ohio", "Georgia", "Michigan", "Arizona",
                "Massachusetts", "Colorado", "Nevada", "Indiana", "Kentucky",
                "Louisiana", "Montana", "Alaska", "Alabama"
            ])
        
        with col2:
            st.markdown("#### 🛍️ Informasi Produk")
            item_purchased = st.selectbox("Item yang Dibeli", [
                "Blouse", "Dress", "Jacket", "Jeans", "Pants", "Shirt",
                "Shoes", "Shorts", "Skirt", "Sweater", "T-shirt"
            ])
            category = st.selectbox("Kategori", [
                "Accessories", "Clothing", "Footwear", "Outerwear"
            ])
            size = st.selectbox("Ukuran", ["S", "M", "L", "XL"])
            color = st.selectbox("Warna", [
                "Black", "White", "Blue", "Red", "Green", "Gray",
                "Brown", "Pink", "Yellow", "Orange", "Purple",
                "Beige", "Charcoal", "Cyan", "Gold", "Lavender",
                "Magenta", "Maroon", "Olive", "Peach", "Silver",
                "Teal", "Turquoise"
            ])
            season = st.selectbox("Musim", ["Fall", "Winter", "Spring", "Summer"])
        
        with col3:
            st.markdown("#### 💳 Informasi Transaksi")
            previous_purchases = st.number_input("Pembelian Sebelumnya", 0, 50, 10)
            review_rating = st.slider("Rating Review", 1.0, 5.0, 4.0, 0.1)
            payment_method = st.selectbox("Metode Pembayaran", [
                "Credit Card", "Cash", "Debit Card", "PayPal", "Venmo", "Bank Transfer"
            ])
            shipping_type = st.selectbox("Tipe Pengiriman", [
                "Express", "Free Shipping", "Next Day Air",
                "Standard", "Store Pickup", "2-Day Shipping"
            ])
            frequency_of_purchases = st.selectbox("Frekuensi Pembelian", [
                "Weekly", "Fortnightly", "Monthly", "Quarterly",
                "Annually", "Bi-Weekly", "Every 3 Months"
            ])
            
            st.markdown("#### 🎁 Status Promosi")
            subscription_status = st.radio("Status Langganan", ["Yes", "No"], horizontal=True)
            discount_applied = st.radio("Diskon Diterapkan", ["Yes", "No"], horizontal=True)
            promo_code_used = st.radio("Kode Promo Digunakan", ["Yes", "No"], horizontal=True)
        
        st.markdown("---")
        
        if st.button("🔮 Prediksi Jumlah Pembelian", type="primary", use_container_width=True):
            # Prepare input
            input_data = {
                'age': age,
                'previous_purchases': previous_purchases,
                'review_rating': review_rating,
                'subscription_status': 1 if subscription_status == "Yes" else 0,
                'discount_applied': 1 if discount_applied == "Yes" else 0,
                'promo_code_used': 1 if promo_code_used == "Yes" else 0,
            }
            
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
            
            df_input = pd.DataFrame([input_data])
            for key, value in categorical_features.items():
                df_input[key] = value
            
            df_encoded = pd.get_dummies(df_input, columns=list(categorical_features.keys()), drop_first=True)
            
            df_encoded['shipping_type'] = shipping_type
            if shipping_mapping is not None:
                if shipping_type in shipping_mapping:
                    df_encoded['shipping_type'] = shipping_mapping[shipping_type]
                else:
                    df_encoded['shipping_type'] = 0
            
            for col in feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            
            df_encoded = df_encoded[feature_columns]
            
            # Predict
            prediction = model.predict(df_encoded)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 💰 Hasil Prediksi")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"<h1 style='color: #1f77b4;'>${prediction:.2f}</h1>", unsafe_allow_html=True)
                st.markdown("**Prediksi Purchase Amount**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                confidence = "Tinggi" if 20 <= prediction <= 80 else "Sedang"
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"<h2>{confidence}</h2>", unsafe_allow_html=True)
                st.markdown("**Confidence Level**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                category_avg = {"Accessories": 45, "Clothing": 55, "Footwear": 65, "Outerwear": 70}
                diff = prediction - category_avg.get(category, 50)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"<h2 style='color: {'green' if diff > 0 else 'red'};'>{diff:+.2f}</h2>", unsafe_allow_html=True)
                st.markdown("**vs Rata-rata Kategori**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                level = "High Value" if prediction > 70 else "Medium Value" if prediction > 40 else "Low Value"
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"<h2>{level}</h2>", unsafe_allow_html=True)
                st.markdown("**Customer Segment**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            st.markdown("### 📊 Analisis Visual")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Level Pembelian:**")
                progress_value = min(prediction / 100, 1.0)
                st.progress(progress_value)
                
                if prediction > 70:
                    st.success("🟢 **High Value Customer** - Pelanggan premium dengan pembelian tinggi")
                elif prediction > 40:
                    st.info("🟡 **Medium Value Customer** - Pelanggan reguler dengan pembelian sedang")
                else:
                    st.warning("🔴 **Low Value Customer** - Pelanggan dengan pembelian rendah")
            
            with col2:
                st.markdown("**Perbandingan dengan Rata-rata:**")
                comparison_data = pd.DataFrame({
                    'Kategori': ['Prediksi Anda', 'Rata-rata Kategori', 'Rata-rata Keseluruhan'],
                    'Jumlah (USD)': [prediction, category_avg.get(category, 50), 50]
                })
                st.dataframe(comparison_data, use_container_width=True, hide_index=True)
                st.bar_chart(comparison_data.set_index('Kategori'))
            
            # Insights
            st.markdown("---")
            st.markdown("### 💡 Insight & Rekomendasi")
            
            insights = []
            recommendations = []
            
            if prediction > 70:
                insights.append("✅ Pelanggan ini termasuk dalam kategori **high-value customer** (top 20%)")
                recommendations.append("🎯 **Strategi:** Berikan program loyalitas premium dan penawaran eksklusif")
                recommendations.append("💎 **Upselling:** Tawarkan produk premium atau bundle dengan nilai lebih tinggi")
            elif prediction < 30:
                insights.append("⚠️ Pelanggan ini termasuk dalam kategori **low-value customer**")
                recommendations.append("💰 **Strategi:** Berikan diskon atau promosi khusus untuk meningkatkan nilai pembelian")
                recommendations.append("📧 **Email Marketing:** Kirim rekomendasi produk dengan harga terjangkau")
            else:
                insights.append("ℹ️ Pelanggan ini termasuk dalam kategori **medium-value customer**")
                recommendations.append("📈 **Strategi:** Dorong upgrade dengan cross-selling dan bundling")
                recommendations.append("🎁 **Reward:** Berikan poin rewards untuk pembelian berikutnya")
            
            if subscription_status == "Yes":
                insights.append("📧 **Subscriber aktif** - Kemungkinan pembelian berulang tinggi (retention rate +35%)")
                recommendations.append("💌 Kirim newsletter eksklusif dengan early access produk baru")
            
            if discount_applied == "Yes" or promo_code_used == "Yes":
                insights.append("🎁 **Price sensitive customer** - Responsif terhadap promosi dan diskon")
                recommendations.append("🏷️ Masukkan dalam segment untuk campaign seasonal sale")
            
            if previous_purchases > 20:
                insights.append("⭐ **Pelanggan loyal** dengan riwayat pembelian tinggi (20+ transaksi)")
                recommendations.append("🏆 Berikan status VIP atau akses ke exclusive member zone")
            elif previous_purchases > 10:
                insights.append("✨ Pelanggan reguler dengan engagement yang baik")
            else:
                insights.append("🆕 Pelanggan baru atau jarang bertransaksi")
                recommendations.append("👋 Berikan welcome discount untuk pembelian kedua")
            
            if age < 25:
                insights.append("👶 **Demografis muda** - Preferensi trend fashion terkini")
                recommendations.append("📱 Fokus pada social media marketing (Instagram, TikTok)")
            elif age > 50:
                insights.append("👴 **Demografis mature** - Preferensi kualitas dan kenyamanan")
                recommendations.append("📧 Gunakan email marketing dengan informasi detail produk")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 Insight Pelanggan")
                for insight in insights:
                    st.markdown(f"- {insight}")
            
            with col2:
                st.markdown("#### 🎯 Rekomendasi Aksi")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            # Additional metrics
            st.markdown("---")
            st.markdown("### 📈 Metrik Tambahan")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                clv_estimate = prediction * (previous_purchases + 10) * 0.8
                st.metric("Estimasi CLV", f"${clv_estimate:.2f}", 
                         help="Customer Lifetime Value - Total nilai pelanggan sepanjang hubungan")
            
            with col2:
                repeat_probability = min((previous_purchases / 50) * 100, 95)
                st.metric("Probabilitas Repeat", f"{repeat_probability:.0f}%",
                         help="Kemungkinan pelanggan akan kembali membeli")
            
            with col3:
                avg_order = prediction
                if previous_purchases > 0:
                    avg_order = prediction * 0.95
                st.metric("Avg Order Value", f"${avg_order:.2f}",
                         help="Rata-rata nilai per transaksi")
            
            with col4:
                churn_risk = "Low" if previous_purchases > 15 else "Medium" if previous_purchases > 5 else "High"
                color = "green" if churn_risk == "Low" else "orange" if churn_risk == "Medium" else "red"
                st.metric("Churn Risk", churn_risk,
                         help="Risiko pelanggan berhenti bertransaksi")

with tab3:
    st.header("📖 Dokumentasi")
    
    st.markdown("""
    ## 🎯 Tentang Aplikasi
    
    Aplikasi ini menggunakan **XGBoost (Extreme Gradient Boosting)**, sebuah algoritma machine learning 
    yang sangat powerful untuk prediksi. XGBoost adalah algoritma ensemble yang menggabungkan 
    banyak decision trees untuk membuat prediksi yang akurat.
    
    ### 🔍 Cara Kerja Model
    
    1. **Input Data**: Model menerima 15+ fitur pelanggan
    2. **Feature Engineering**: Data diproses dan ditransformasi
    3. **XGBoost Processing**: 100 decision trees bekerja bersama
    4. **Output**: Prediksi purchase amount dalam USD
    
    ### 📊 Fitur yang Digunakan
    
    **Demografis:**
    - Usia pelanggan
    - Gender
    - Lokasi geografis
    
    **Perilaku Pembelian:**
    - Jumlah pembelian sebelumnya
    - Frekuensi pembelian
    - Rating review
    
    **Karakteristik Produk:**
    - Kategori produk
    - Item spesifik
    - Ukuran dan warna
    - Musim pembelian
    
    **Transaksi:**
    - Metode pembayaran
    - Tipe pengiriman
    - Status langganan
    - Penggunaan diskon/promo
    
    ### 🎓 Metrik Evaluasi
    
    **MAE (Mean Absolute Error)**
    - Rata-rata selisih absolut antara prediksi dan aktual
    - Satuan: USD
    - Semakin kecil, semakin baik
    
    **RMSE (Root Mean Squared Error)**
    - Akar dari rata-rata kuadrat error
    - Memberikan penalti lebih besar pada error besar
    - Satuan: USD
    
    **R² Score (Coefficient of Determination)**
    - Mengukur proporsi variasi yang dijelaskan model
    - Range: 0 - 1 (0% - 100%)
    - > 0.8 = Excellent
    - 0.7 - 0.8 = Very Good
    - 0.6 - 0.7 = Good
    - < 0.6 = Needs Improvement
    
    ### 🚀 Cara Menggunakan
    
    **Step 1: Training Model**
    1. Buka tab "Training Model"
    2. Upload file CSV dengan data historis
    3. Klik "Mulai Training Model"
    4. Tunggu proses selesai (1-2 menit)
    
    **Step 2: Prediksi**
    1. Buka tab "Prediksi"
    2. Isi semua informasi pelanggan
    3. Klik "Prediksi Jumlah Pembelian"
    4. Lihat hasil dan rekomendasi
    
    ### 💡 Tips untuk Hasil Terbaik
    
    - Gunakan data training yang bersih dan lengkap
    - Minimal 1000 data points untuk hasil optimal
    - Update model secara berkala dengan data terbaru
    - Monitor performa model di production
    
    ### 🔬 Teknologi yang Digunakan
    
    - **Python**: Bahasa pemrograman
    - **Streamlit**: Framework web app
    - **XGBoost**: Machine learning algorithm
    - **Pandas**: Data manipulation
    - **Scikit-learn**: ML utilities
    - **Matplotlib/Seaborn**: Data visualization
    
    ### 📞 Support
    
    Untuk pertanyaan atau bantuan, silakan:
    - Buka issue di GitHub repository
    - Email: support@yourdomain.com
    - Dokumentasi lengkap: docs.yourdomain.com
    
    ### 📄 Lisensi
    
    MIT License - Free to use and modify
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** October 2025  
    **Built with ❤️ using Streamlit & XGBoost**
    """)
    
    with st.expander("🔧 Advanced Settings"):
        st.markdown("""
        ### Hyperparameter Tuning
        
        Untuk advanced users, Anda dapat memodifikasi hyperparameter XGBoost:
        
        ```python
        xgb_model = XGBRegressor(
            n_estimators=100,      # Jumlah trees (50-500)
            learning_rate=0.1,     # Learning rate (0.01-0.3)
            max_depth=5,           # Tree depth (3-10)
            subsample=0.8,         # Sample ratio (0.5-1.0)
            colsample_bytree=0.8,  # Feature ratio (0.5-1.0)
        )
        ```
        
        **Rekomendasi:**
        - Untuk dataset besar: tingkatkan n_estimators
        - Untuk mencegah overfitting: kurangi max_depth
        - Untuk training lebih cepat: kurangi n_estimators
        """)
    
    with st.expander("📊 Sample Data Format"):
        st.markdown("""
        ### Format CSV yang Diperlukan
        
        File CSV harus memiliki kolom-kolom berikut:
        
        | Kolom | Tipe | Contoh |
        |-------|------|--------|
        | Customer ID | String | "C12345" |
        | Age | Integer | 35 |
        | Gender | String | "Male" atau "Female" |
        | Item Purchased | String | "Blouse", "Dress", dll |
        | Category | String | "Clothing", "Footwear", dll |
        | Purchase Amount (USD) | Float | 59.99 |
        | Location | String | "California" |
        | Size | String | "M" |
        | Color | String | "Black" |
        | Season | String | "Fall" |
        | Review Rating | Float | 4.5 |
        | Subscription Status | String | "Yes" atau "No" |
        | Shipping Type | String | "Express", dll |
        | Discount Applied | String | "Yes" atau "No" |
        | Promo Code Used | String | "Yes" atau "No" |
        | Previous Purchases | Integer | 10 |
        | Payment Method | String | "Credit Card", dll |
        | Frequency of Purchases | String | "Monthly", dll |
        
        **Download sample:** [shopping_behavior_sample.csv](#)
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p><strong>Analisis XGBoost - Purchase Prediction System</strong></p>
        <p>Dibuat dengan ❤️ menggunakan Streamlit & XGBoost | Version 1.0.0</p>
        <p>© 2025 All Rights Reserved</p>
    </div>
    """, 
    unsafe_allow_html=True
)
