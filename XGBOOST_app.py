# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Analisis XGBoost", page_icon="🛒", layout="wide")

# Import dependencies
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor
    SKLEARN_OK = True
except ImportError as e:
    SKLEARN_OK = False
    st.error(f"Error importing sklearn/xgboost: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    MATPLOTLIB_OK = True
except ImportError as e:
    MATPLOTLIB_OK = False
    st.warning(f"Matplotlib tidak tersedia: {e}")

# CSS
st.markdown("""
<style>
.big-font {font-size:20px !important; font-weight: bold;}
.info-box {
    background-color: #e7f3ff;
    border-left: 5px solid #2196F3;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.success-box {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.warning-box {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def train_model_with_viz(df):
    """Train model dengan analisis lengkap dan penjelasan"""
    
    st.markdown("---")
    st.markdown("## 📊 TAHAP 1: Eksplorasi Data Awal (EDA)")
    st.markdown("""
    <div class='info-box'>
    <b>💡 Apa itu EDA?</b><br>
    Exploratory Data Analysis (EDA) adalah proses menganalisis data untuk memahami karakteristik, 
    pola, dan anomali sebelum membangun model. Ini langkah penting untuk memastikan kualitas data!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📝 Total Baris", f"{len(df):,}", help="Jumlah data pelanggan yang akan dianalisis")
    col2.metric("📋 Kolom", len(df.columns), help="Jumlah variabel/fitur dalam dataset")
    col3.metric("❓ Missing", df.isnull().sum().sum(), help="Jumlah data yang hilang/kosong")
    
    with st.expander("👁️ Preview 10 Baris Pertama Data"):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption("Preview ini menampilkan sample data untuk memastikan format sudah benar")
    
    with st.expander("📊 Statistik Deskriptif"):
        st.dataframe(df.describe(), use_container_width=True)
        st.caption("Statistik ini memberikan gambaran distribusi data numerik (mean, median, min, max, dll)")
    
    # Visualisasi distribusi
    if MATPLOTLIB_OK:
        st.markdown("### 📊 Distribusi Purchase Amount (Target Variable)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df['Purchase Amount (USD)'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Purchase Amount (USD)', fontsize=12)
            ax.set_ylabel('Frekuensi', fontsize=12)
            ax.set_title('Distribusi Jumlah Pembelian Pelanggan', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            stats = df['Purchase Amount (USD)'].describe()
            st.markdown("**📈 Statistik Kunci:**")
            st.metric("Rata-rata", f"${stats['mean']:.2f}")
            st.metric("Median", f"${stats['50%']:.2f}")
            st.metric("Min - Max", f"${stats['min']:.0f} - ${stats['max']:.0f}")
        
        st.markdown("""
        <div class='info-box'>
        <b>💡 Cara Membaca Histogram:</b><br>
        - Sumbu X: Nilai pembelian dalam USD<br>
        - Sumbu Y: Berapa banyak pelanggan yang melakukan pembelian dengan nilai tersebut<br>
        - Histogram ini menunjukkan pola pembelian pelanggan - apakah tersebar merata atau terpusat di nilai tertentu<br>
        - Ini adalah variabel <b>target</b> yang akan kita prediksi menggunakan XGBoost
        </div>
        """, unsafe_allow_html=True)
    
    # Data Cleaning
    st.markdown("---")
    st.markdown("## 🧹 TAHAP 2: Pembersihan Data")
    st.markdown("""
    <div class='info-box'>
    <b>💡 Kenapa Data Cleaning Penting?</b><br>
    Data kotor atau duplikat dapat menyebabkan model belajar pola yang salah. 
    Proses ini memastikan data berkualitas tinggi sebelum masuk ke model.
    </div>
    """, unsafe_allow_html=True)
    
    if 'Customer ID' in df.columns:
        df = df.drop('Customer ID', axis=1)
        st.success("✅ **Customer ID dihapus** - ID tidak relevan untuk prediksi karena bersifat unik per pelanggan")
    
    dup = df.duplicated().sum()
    if dup > 0:
        df.drop_duplicates(inplace=True)
        st.success(f"✅ **{dup} baris duplikat dihapus** - Data duplikat dapat menyebabkan overfitting")
    else:
        st.success("✅ **Tidak ada duplikat** - Data sudah bersih!")
    
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('__', '_').str.lower()
    st.info("✅ Nama kolom diseragamkan (lowercase, underscore) untuk kemudahan pemrosesan")
    
    # Feature Engineering
    st.markdown("---")
    st.markdown("## ⚙️ TAHAP 3: Feature Engineering")
    st.markdown("""
    <div class='info-box'>
    <b>💡 Apa itu Feature Engineering?</b><br>
    Proses mengubah data mentah menjadi format yang dapat dipahami oleh model machine learning. 
    Ini termasuk encoding variabel kategorikal menjadi angka.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔄 Binary Encoding (Yes/No → 1/0)")
    
    df['subscription_status'] = df['subscription_status'].map({'Yes': 1, 'No': 0})
    df['discount_applied'] = df['discount_applied'].map({'Yes': 1, 'No': 0})
    df['promo_code_used'] = df['promo_code_used'].map({'Yes': 1, 'No': 0})
    
    st.success("""
    ✅ **Variabel binary dikonversi:**
    - Subscription Status: Yes=1, No=0
    - Discount Applied: Yes=1, No=0  
    - Promo Code Used: Yes=1, No=0
    
    Model tidak bisa memproses teks, jadi kita ubah ke angka!
    """)
    
    # Correlation
    if MATPLOTLIB_OK:
        st.markdown("### 🔗 Analisis Korelasi Antar Variabel")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, center=0)
            ax.set_title('Matriks Korelasi', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            <div class='info-box'>
            <b>💡 Cara Membaca Heatmap Korelasi:</b><br>
            - <span style='color: red;'><b>Merah</b></span>: Korelasi positif (nilai naik bersama)<br>
            - <span style='color: blue;'><b>Biru</b></span>: Korelasi negatif (nilai berkebalikan)<br>
            - <b>Putih</b>: Tidak ada korelasi<br>
            - Angka menunjukkan kekuatan korelasi (-1 sampai +1)<br>
            - Korelasi kuat dengan Purchase Amount menunjukkan fitur penting untuk prediksi
            </div>
            """, unsafe_allow_html=True)
    
    # One-Hot Encoding
    st.markdown("### 🎯 One-Hot Encoding untuk Variabel Kategorikal")
    
    cols_to_encode = ['gender', 'item_purchased', 'category', 'location', 'size', 
                      'color', 'season', 'payment_method', 'frequency_of_purchases']
    
    original_cols = df.shape[1]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    new_cols = df.shape[1]
    
    st.success(f"✅ **One-Hot Encoding selesai:** {original_cols} kolom → **{new_cols} kolom**")
    
    st.markdown("""
    <div class='info-box'>
    <b>💡 Apa itu One-Hot Encoding?</b><br>
    Teknik untuk mengubah variabel kategorikal (seperti warna, ukuran) menjadi format binary.<br><br>
    <b>Contoh:</b><br>
    • Gender: "Male", "Female" → gender_Male: 0 atau 1<br>
    • Color: "Red", "Blue", "Green" → color_Red, color_Blue (0 atau 1)<br><br>
    Setiap kategori menjadi kolom terpisah dengan nilai 1 (jika kategori tersebut) atau 0 (jika bukan).
    </div>
    """, unsafe_allow_html=True)
    
    # Split
    st.markdown("---")
    st.markdown("## ✂️ TAHAP 4: Pembagian Data (Train-Test Split)")
    st.markdown("""
    <div class='info-box'>
    <b>💡 Kenapa Dibagi Training dan Testing?</b><br>
    - <b>Training Set (80%):</b> Digunakan untuk mengajarkan model mengenali pola<br>
    - <b>Testing Set (20%):</b> Digunakan untuk menguji akurasi model pada data yang belum pernah dilihat<br><br>
    Ini seperti latihan soal (training) sebelum ujian (testing)!
    </div>
    """, unsafe_allow_html=True)
    
    X = df.drop('purchase_amount_usd', axis=1)
    y = df['purchase_amount_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🎓 Training Set", f"{len(X_train):,} baris", "80% data")
    col2.metric("🧪 Testing Set", f"{len(X_test):,} baris", "20% data")
    col3.metric("📊 Total Fitur", X_train.shape[1], "Variabel prediksi")
    
    # PERBAIKAN: Encode shipping_type SEBELUM split untuk konsistensi
    # Buat mapping dari seluruh data
    all_shipping_types = df['shipping_type'].unique() if 'shipping_type' in df.columns else X['shipping_type'].unique()
    ship_map = {cat: idx for idx, cat in enumerate(all_shipping_types)}
    
    # Apply mapping
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    if 'shipping_type' in X_train_copy.columns:
        X_train_copy['shipping_type'] = X_train_copy['shipping_type'].map(ship_map)
        X_test_copy['shipping_type'] = X_test_copy['shipping_type'].map(ship_map)
        
        # Handle any NaN from mapping
        X_train_copy['shipping_type'].fillna(0, inplace=True)
        X_test_copy['shipping_type'].fillna(0, inplace=True)
    
    st.info(f"✅ Label Encoding untuk 'shipping_type': {len(ship_map)} kategori pengiriman")
    
    # Validasi: Pastikan tidak ada NaN
    if X_train_copy.isnull().sum().sum() > 0 or X_test_copy.isnull().sum().sum() > 0:
        st.error("⚠️ Ditemukan nilai NaN setelah encoding! Membersihkan...")
        X_train_copy.fillna(0, inplace=True)
        X_test_copy.fillna(0, inplace=True)
    
    # Train
    st.markdown("---")
    st.markdown("## 🤖 TAHAP 5: Training Model XGBoost")
    st.markdown("""
    <div class='success-box'>
    <b>🎯 Apa itu XGBoost?</b><br>
    <b>XGBoost (Extreme Gradient Boosting)</b> adalah algoritma machine learning yang sangat powerful!<br><br>
    <b>Cara Kerja:</b><br>
    • Membangun banyak "decision trees" (pohon keputusan) secara bertahap<br>
    • Setiap tree baru belajar dari kesalahan tree sebelumnya<br>
    • Hasil akhir adalah kombinasi dari semua trees (ensemble)<br>
    • Seperti konsultasi ke 100 ahli sekaligus!<br><br>
    <b>Keunggulan XGBoost:</b><br>
    ✅ Sangat akurat (sering menang di kompetisi Kaggle)<br>
    ✅ Cepat dan efisien<br>
    ✅ Handle missing values dengan baik<br>
    ✅ Mencegah overfitting dengan regularization
    </div>
    """, unsafe_allow_html=True)
    
    progress = st.progress(0)
    status = st.empty()
    
    status.text("⏳ Menginisialisasi 100 decision trees...")
    progress.progress(30)
    
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    status.text("🔄 Training model... (mungkin butuh 1-2 menit)")
    progress.progress(60)
    
    model.fit(X_train_copy, y_train)
    
    status.text("✅ Training selesai!")
    progress.progress(100)
    
    st.success("🎉 **Model XGBoost berhasil dilatih dengan 100 decision trees!**")
    
    with st.expander("🎛️ Hyperparameter yang Digunakan"):
        st.markdown("""
        - **n_estimators=100**: Jumlah decision trees (lebih banyak = lebih akurat tapi lebih lambat)
        - **learning_rate=0.1**: Kecepatan belajar (lebih kecil = lebih stabil)
        - **max_depth=5**: Kedalaman maksimum tree (mencegah overfitting)
        - **subsample=0.8**: Gunakan 80% data per tree (regularization)
        - **colsample_bytree=0.8**: Gunakan 80% fitur per tree (regularization)
        """)
    
    # Evaluate
    st.markdown("---")
    st.markdown("## 📊 TAHAP 6: Evaluasi Performa Model")
    st.markdown("""
    <div class='info-box'>
    <b>💡 Mengukur Seberapa Baik Model Bekerja</b><br>
    Kita gunakan data testing (yang belum pernah dilihat model) untuk menguji akurasi prediksi.
    </div>
    """, unsafe_allow_html=True)
    
    y_pred = model.predict(X_test_copy)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Validasi R² - jika negatif, ada masalah!
    if r2 < 0:
        st.error(f"""
        ⚠️ **PERINGATAN: R² Score Negatif ({r2:.4f})**
        
        Ini menunjukkan model performanya lebih buruk dari prediksi rata-rata sederhana!
        
        **Kemungkinan Penyebab:**
        - Feature mismatch antara train dan test
        - Data terlalu sedikit atau tidak representatif
        - Fitur tidak informatif untuk prediksi
        
        **Rekomendasi:**
        - Periksa kembali data input
        - Pastikan data training cukup banyak (minimal 500 rows)
        - Feature engineering mungkin perlu diperbaiki
        """)
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error - Rata-rata selisih prediksi")
    col2.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error - Penalti lebih besar untuk error besar")
    col3.metric("R² Score", f"{r2:.4f}", help="Proporsi variasi yang dijelaskan model (0-1)")
    
    # Interpretasi
    accuracy_pct = r2 * 100
    if r2 > 0.9:
        perf = "🌟 EXCELLENT"
        color = "success"
    elif r2 > 0.8:
        perf = "✅ SANGAT BAIK"
        color = "success"
    elif r2 > 0.7:
        perf = "👍 BAIK"
        color = "info"
    elif r2 > 0.6:
        perf = "📊 CUKUP BAIK"
        color = "warning"
    else:
        perf = "⚠️ PERLU DITINGKATKAN"
        color = "warning"
    
    if color == "success":
        st.success(f"""
        ### {perf}
        
        **Interpretasi Metrik:**
        
        📊 **R² Score: {r2:.4f} ({accuracy_pct:.1f}%)**
        - Model dapat menjelaskan **{accuracy_pct:.1f}%** variasi dalam purchase amount
        - Artinya: Model sangat baik dalam memprediksi jumlah pembelian!
        
        💵 **MAE: ${mae:.2f}**
        - Rata-rata, prediksi model meleset sekitar **${mae:.2f}** dari nilai sebenarnya
        - Ini cukup akurat untuk keperluan bisnis
        
        📈 **RMSE: ${rmse:.2f}**
        - Model cukup konsisten, tidak ada error yang terlalu ekstrem
        """)
    else:
        st.info(f"""
        ### {perf}
        
        **Interpretasi Metrik:**
        
        📊 **R² Score: {r2:.4f} ({accuracy_pct:.1f}%)**
        - Model dapat menjelaskan **{accuracy_pct:.1f}%** variasi dalam purchase amount
        
        💵 **MAE: ${mae:.2f}**
        - Rata-rata kesalahan prediksi sekitar **${mae:.2f}**
        
        📈 **RMSE: ${rmse:.2f}**
        """)
    
    # Visualizations
    if MATPLOTLIB_OK:
        st.markdown("### 🎯 Visualisasi: Prediksi vs Nilai Aktual")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter
        ax1.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Prediksi Sempurna')
        ax1.set_xlabel('Nilai Aktual (USD)', fontsize=11)
        ax1.set_ylabel('Prediksi Model (USD)', fontsize=11)
        ax1.set_title('Perbandingan Prediksi vs Aktual', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        ax2.hist(residuals, bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax2.set_xlabel('Residuals (Aktual - Prediksi)', fontsize=11)
        ax2.set_ylabel('Frekuensi', fontsize=11)
        ax2.set_title('Distribusi Error Prediksi', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        <div class='info-box'>
        <b>💡 Cara Membaca Visualisasi:</b><br><br>
        
        <b>Grafik Kiri (Scatter Plot):</b><br>
        • Setiap titik = 1 prediksi<br>
        • Garis merah putus-putus = prediksi sempurna (aktual = prediksi)<br>
        • Titik dekat garis merah = prediksi akurat ✅<br>
        • Titik jauh dari garis = prediksi kurang akurat ❌<br><br>
        
        <b>Grafik Kanan (Histogram Error):</b><br>
        • Menunjukkan distribusi kesalahan prediksi<br>
        • Idealnya berbentuk kurva bell (normal) terpusat di 0<br>
        • Jika terpusat di 0 = model tidak bias<br>
        • Jika menyebar lebar = model kurang konsisten
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Importance
        st.markdown("---")
        st.markdown("### 🔍 Analisis Feature Importance")
        st.markdown("""
        <div class='success-box'>
        <b>💡 Fitur Mana yang Paling Penting?</b><br>
        Feature Importance menunjukkan variabel mana yang paling berpengaruh dalam prediksi purchase amount.
        Semakin tinggi nilai importance, semakin besar pengaruhnya terhadap keputusan model.
        </div>
        """, unsafe_allow_html=True)
        
        feat_imp = pd.DataFrame({
            'Feature': X_train_copy.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feat_imp)))
        bars = ax.barh(range(len(feat_imp)), feat_imp['Importance'], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(feat_imp)))
        ax.set_yticklabels(feat_imp['Feature'], fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title('Top 15 Fitur Terpenting dalam Model', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add values
        for i, (bar, val) in enumerate(zip(bars, feat_imp['Importance'])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        <div class='info-box'>
        <b>💡 Interpretasi Feature Importance:</b><br>
        • <b>Fitur teratas</b> memiliki pengaruh terbesar terhadap prediksi<br>
        • Bisnis dapat fokus pada fitur-fitur ini untuk meningkatkan penjualan<br>
        • Contoh: Jika "previous_purchases" tinggi, fokus pada retention program<br>
        • Fitur dengan importance rendah mungkin bisa diabaikan untuk simplifikasi
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.success(f"""
    ### ✅ Kesimpulan Training
    
    Model XGBoost berhasil dilatih dengan performa {perf}!
    - Akurasi: {accuracy_pct:.1f}%
    - Rata-rata error: ${mae:.2f}
    - Model siap digunakan untuk prediksi
    """)
    
    return model, ship_map, X_train_copy.columns.tolist(), {'MAE': mae, 'RMSE': rmse, 'R2': r2}

@st.cache_resource
def get_model():
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('shipping_type_mapping.pkl', 'rb') as f:
            ship_map = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feat_cols = pickle.load(f)
        return model, ship_map, feat_cols, True
    except:
        return None, None, None, False

# Main
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🛒 Analisis XGBoost: Prediksi Purchase Amount</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #666;'>Sistem Prediksi Jumlah Pembelian Pelanggan dengan Machine Learning</h3>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box' style='margin-top: 20px;'>
<b>🎯 Tentang Aplikasi Ini:</b><br>
Aplikasi ini menggunakan <b>XGBoost (Extreme Gradient Boosting)</b>, algoritma machine learning 
yang sangat powerful untuk memprediksi jumlah pembelian pelanggan berdasarkan berbagai faktor seperti 
demografi, produk, dan perilaku transaksi.<br><br>

<b>Manfaat Bisnis:</b><br>
• Prediksi nilai pembelian pelanggan secara akurat<br>
• Identifikasi faktor-faktor yang mempengaruhi pembelian<br>
• Segmentasi pelanggan (High/Medium/Low value)<br>
• Rekomendasi strategi marketing yang tepat<br>
• Data-driven decision making
</div>
""", unsafe_allow_html=True)

if not SKLEARN_OK:
    st.error("❌ Required libraries not available!")
    st.stop()

if MATPLOTLIB_OK:
    st.success("✅ Semua library tersedia - Visualisasi lengkap aktif!")
else:
    st.warning("⚠️ Matplotlib unavailable - Visualisasi dibatasi")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📚 Training Model", "🎯 Prediksi", "📖 Panduan"])

with tab1:
    st.header("📚 Training Model XGBoost")
    st.markdown("""
    Di tab ini Anda akan:
    1. Upload dataset historis pembelian pelanggan
    2. Melihat analisis data lengkap step-by-step
    3. Melatih model XGBoost
    4. Melihat performa dan akurasi model
    """)
    
    model, ship_map, feat_cols, exists = get_model()
    
    if exists:
        st.success("✅ Model sudah tersedia dan siap digunakan untuk prediksi!")
        st.info("💡 Jika ingin melatih ulang dengan data baru, upload dataset di bawah")
    else:
        st.warning("⚠️ Model belum tersedia. Upload dataset untuk memulai training")
    
    file = st.file_uploader("📤 Upload CSV Dataset", type=['csv'],
                           help="File harus berisi kolom Purchase Amount (USD) dan fitur lainnya")
    
    if file:
        df = pd.read_csv(file)
        st.success(f"✅ Dataset berhasil diupload: **{len(df):,} baris** x **{len(df.columns)} kolom**")
        
        with st.expander("👁️ Preview Dataset"):
            st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("---")
        
        if st.button("🚀 Mulai Training Model", type="primary", use_container_width=True):
            try:
                with st.spinner("Training sedang berlangsung... Mohon tunggu"):
                    model, ship_map, feat_cols, metrics = train_model_with_viz(df)
                    
                    st.session_state.model = model
                    st.session_state.ship_map = ship_map
                    st.session_state.feat_cols = feat_cols
                    
                    st.balloons()
                    st.success("🎉 Training berhasil! Pindah ke tab 'Prediksi' untuk menggunakan model")
            except Exception as e:
                st.error(f"❌ Error: {e}")

with tab2:
    st.header("🎯 Prediksi Purchase Amount")
    st.markdown("""
    <div class='info-box'>
    <b>📝 Cara Menggunakan:</b><br>
    1. Isi semua informasi pelanggan di formulir bawah<br>
    2. Klik tombol "Prediksi"<br>
    3. Lihat hasil prediksi dan rekomendasi bisnis<br>
    4. Gunakan insight untuk strategi marketing yang tepat
    </div>
    """, unsafe_allow_html=True)
    
    if 'model' in st.session_state:
        model = st.session_state.model
        ship_map = st.session_state.ship_map
        feat_cols = st.session_state.feat_cols
        exists = True
    else:
        model, ship_map, feat_cols, exists = get_model()
    
    if not exists:
        st.warning("⚠️ Model belum tersedia. Silakan train model terlebih dahulu di tab **Training Model**")
    else:
        st.success("✅ Model siap digunakan untuk prediksi!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Informasi Demografis")
            age = st.slider("Usia", 18, 70, 35, help="Usia pelanggan dalam tahun")
            gender = st.selectbox("Gender", ["Male", "Female"])
            location = st.selectbox("Lokasi", ["California", "New York", "Texas", "Florida", "Illinois"])
        
        with col2:
            st.markdown("#### 🛍️ Informasi Produk")
            item = st.selectbox("Item", ["Blouse", "Dress", "Jacket", "Jeans", "Pants", "Shirt", "Shoes"])
            category = st.selectbox("Kategori", ["Clothing", "Footwear", "Accessories", "Outerwear"])
            size = st.selectbox("Ukuran", ["S", "M", "L", "XL"])
            color = st.selectbox("Warna", ["Black", "White", "Blue", "Red", "Green", "Gray"])
            season = st.selectbox("Musim", ["Fall", "Winter", "Spring", "Summer"])
        
        with col3:
            st.markdown("#### 💳 Informasi Transaksi")
            prev = st.number_input("Pembelian Sebelumnya", 0, 50, 10, 
                                   help="Total transaksi yang pernah dilakukan")
            rating = st.slider("Rating Review", 1.0, 5.0, 4.0, 
                              help="Rating rata-rata dari pelanggan")
            payment = st.selectbox("Metode Pembayaran", ["Credit Card", "Cash", "Debit Card", "PayPal"])
            shipping = st.selectbox("Tipe Pengiriman", ["Express", "Standard", "Free Shipping"])
            freq = st.selectbox("Frekuensi Pembelian", ["Weekly", "Monthly", "Quarterly", "Annually"])
            
            st.markdown("**Status Promosi:**")
            subs = st.radio("Langganan Newsletter?", ["Yes", "No"], horizontal=True)
            disc = st.radio("Ada Diskon?", ["Yes", "No"], horizontal=True)
            promo = st.radio("Pakai Promo Code?", ["Yes", "No"], horizontal=True)
        
        st.markdown("---")
        
        if st.button("🔮 PREDIKSI SEKARANG", type="primary", use_container_width=True):
            data = {
                'age': age, 'previous_purchases': prev, 'review_rating': rating,
                'subscription_status': 1 if subs == "Yes" else 0,
                'discount_applied': 1 if disc == "Yes" else 0,
                'promo_code_used': 1 if promo == "Yes" else 0,
            }
            
            cat_feat = {
                'gender': gender, 'item_purchased': item, 'category': category,
                'location': location, 'size': size, 'color': color,
                'season': season, 'payment_method': payment,
                'frequency_of_purchases': freq
            }
            
            df_in = pd.DataFrame([data])
            for k, v in cat_feat.items():
                df_in[k] = v
            
            df_enc = pd.get_dummies(df_in, columns=list(cat_feat.keys()), drop_first=True)
            df_enc['shipping_type'] = ship_map.get(shipping, 0)
            
            for col in feat_cols:
                if col not in df_enc.columns:
                    df_enc[col] = 0
            
            df_enc = df_enc[feat_cols]
            pred = model.predict(df_enc)[0]
            pred = float(pred)
            
            st.markdown("---")
            st.markdown("## 💰 Hasil Prediksi")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("💵 Purchase Amount", f"${pred:.2f}", 
                       help="Prediksi jumlah pembelian pelanggan ini")
            
            level = "High Value" if pred > 70 else "Medium Value" if pred > 40 else "Low Value"
            emoji = "🟢" if level == "High Value" else "🟡" if level == "Medium Value" else "🔴"
            col2.metric(f"{emoji} Customer Segment", level,
                       help="Kategori nilai pelanggan berdasarkan prediksi")
            
            col3.metric("⭐ Confidence Score", f"{int(min(pred, 100))}/100",
                       help="Skor kepercayaan prediksi")
            
            progress_val = float(max(0.0, min(pred / 100.0, 1.0)))
            st.progress(progress_val)
            
            st.markdown("---")
            st.markdown("### 📊 Analisis Detail")
            
            if pred > 70:
                st.markdown("""
                <div class='success-box'>
                <h4>🟢 HIGH VALUE CUSTOMER</h4>
                <b>Prediksi: ${:.2f}</b><br><br>
                
                <b>Karakteristik:</b><br>
                • Pelanggan premium dengan potensi pembelian tinggi<br>
                • Termasuk dalam top 20% customer base<br>
                • Lifetime value sangat tinggi<br><br>
                
                <b>🎯 Rekomendasi Strategi Bisnis:</b><br>
                ✅ <b>VIP Treatment:</b> Berikan akses ke program loyalitas premium<br>
                ✅ <b>Personal Touch:</b> Dedicated customer service & personal shopper<br>
                ✅ <b>Exclusive Access:</b> Early access produk baru & limited edition<br>
                ✅ <b>Premium Upselling:</b> Tawarkan produk premium atau bundle high-value<br>
                ✅ <b>Retention Focus:</b> Prioritas tinggi untuk customer retention program<br>
                ✅ <b>Referral Program:</b> Incentive untuk merekomendasikan ke teman
                </div>
                """.format(pred), unsafe_allow_html=True)
                
            elif pred > 40:
                st.markdown("""
                <div class='info-box'>
                <h4>🟡 MEDIUM VALUE CUSTOMER</h4>
                <b>Prediksi: ${:.2f}</b><br><br>
                
                <b>Karakteristik:</b><br>
                • Pelanggan reguler dengan potensi pertumbuhan<br>
                • Backbone dari revenue stream<br>
                • Ada peluang untuk upgrade ke high-value<br><br>
                
                <b>🎯 Rekomendasi Strategi Bisnis:</b><br>
                📈 <b>Growth Strategy:</b> Fokus pada increasing purchase frequency<br>
                🎁 <b>Cross-Selling:</b> Tawarkan produk komplementer<br>
                📦 <b>Bundling:</b> Paket produk dengan diskon menarik<br>
                💳 <b>Payment Options:</b> Installment atau buy-now-pay-later<br>
                🎖️ <b>Loyalty Points:</b> Reward program untuk pembelian berikutnya<br>
                📧 <b>Email Marketing:</b> Regular newsletter dengan special offers
                </div>
                """.format(pred), unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class='warning-box'>
                <h4>🔴 LOW VALUE CUSTOMER</h4>
                <b>Prediksi: ${:.2f}</b><br><br>
                
                <b>Karakteristik:</b><br>
                • Pelanggan dengan pembelian rendah<br>
                • Mungkin price-sensitive atau new customer<br>
                • Butuh strategi aktivasi dan engagement<br><br>
                
                <b>🎯 Rekomendasi Strategi Bisnis:</b><br>
                💰 <b>Price Strategy:</b> Berikan diskon atau voucher khusus<br>
                🎉 <b>Welcome Campaign:</b> Special offer untuk pembelian kedua (15-20% off)<br>
                📱 <b>Social Proof:</b> Tunjukkan review & testimoni positif<br>
                🔄 <b>Re-engagement:</b> Email campaign untuk customer yang tidak aktif<br>
                📧 <b>Newsletter:</b> Subscribe dengan benefit menarik<br>
                🎁 <b>Free Shipping:</b> Minimal purchase untuk free shipping
                </div>
                """.format(pred), unsafe_allow_html=True)
            
            # Additional Insights
            st.markdown("---")
            st.markdown("### 💡 Insight Tambahan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Analisis Karakteristik:**")
                insights = []
                
                if prev > 20:
                    insights.append("⭐ **Super Loyal Customer** - 20+ transaksi sebelumnya!")
                    insights.append("→ Rekomendasikan untuk brand ambassador program")
                elif prev > 10:
                    insights.append("✨ **Loyal Customer** - Engagement tinggi dengan 10+ transaksi")
                    insights.append("→ Naikkan tier loyalty untuk benefit lebih")
                else:
                    insights.append("🆕 **New/Occasional Customer** - Butuh strategi aktivasi")
                    insights.append("→ Fokus pada first impression & onboarding experience")
                
                if subs == "Yes":
                    insights.append("📧 **Newsletter Subscriber** - Retention rate +35%")
                    insights.append("→ Kirim konten eksklusif & early bird offers")
                else:
                    insights.append("📨 **Non-Subscriber** - Peluang untuk subscribe")
                    insights.append("→ Tawarkan incentive untuk subscribe (discount code)")
                
                if disc == "Yes" or promo == "Yes":
                    insights.append("🎁 **Price Sensitive** - Sangat responsif terhadap promosi")
                    insights.append("→ Target untuk seasonal sale & flash deals")
                
                for insight in insights:
                    st.markdown(f"- {insight}")
            
            with col2:
                st.markdown("**🎯 Action Items Priority:**")
                
                actions = []
                if pred > 70:
                    actions.append("1️⃣ **HIGH PRIORITY:** Setup VIP account")
                    actions.append("2️⃣ Assign dedicated account manager")
                    actions.append("3️⃣ Send exclusive product catalog")
                    actions.append("4️⃣ Invite to loyalty premium tier")
                elif pred > 40:
                    actions.append("1️⃣ **MEDIUM PRIORITY:** Send targeted email campaign")
                    actions.append("2️⃣ Offer bundling deals")
                    actions.append("3️⃣ Setup loyalty points reminder")
                    actions.append("4️⃣ Cross-sell relevant products")
                else:
                    actions.append("1️⃣ **ACTIVATION:** Send welcome discount (15-20%)")
                    actions.append("2️⃣ Newsletter subscription campaign")
                    actions.append("3️⃣ Free shipping promotion")
                    actions.append("4️⃣ Product recommendation email")
                
                for action in actions:
                    st.markdown(f"- {action}")
            
            # Estimated Metrics
            st.markdown("---")
            st.markdown("### 📈 Estimasi Metrik Bisnis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                clv = pred * (prev + 10) * 0.8
                st.metric("💰 Est. CLV", f"${clv:.2f}",
                         help="Customer Lifetime Value - Total nilai sepanjang hubungan dengan brand")
            
            with col2:
                repeat_prob = min((prev / 50) * 100, 95)
                st.metric("🔄 Repeat Purchase", f"{repeat_prob:.0f}%",
                         help="Probabilitas pelanggan akan kembali membeli")
            
            with col3:
                avg_order = pred * 0.95 if prev > 0 else pred
                st.metric("📊 Avg Order Value", f"${avg_order:.2f}",
                         help="Rata-rata nilai per transaksi")
            
            with col4:
                if prev > 15:
                    churn = "Low"
                elif prev > 5:
                    churn = "Medium"
                else:
                    churn = "High"
                st.metric("⚠️ Churn Risk", churn,
                         help="Risiko pelanggan berhenti bertransaksi")

with tab3:
    st.header("📖 Panduan Lengkap")
    
    st.markdown("""
    ## 🎯 Tentang XGBoost
    
    **XGBoost (Extreme Gradient Boosting)** adalah algoritma machine learning berbasis ensemble 
    yang menggabungkan prediksi dari banyak decision trees untuk menghasilkan prediksi yang sangat akurat.
    
    ### 🔬 Cara Kerja Sederhana
    
    Bayangkan Anda ingin tahu harga rumah. XGBoost seperti:
    
    1. **Decision Tree #1** melihat lokasi: "Lokasi bagus = +$50k"
    2. **Decision Tree #2** melihat luas: "Luas besar = +$30k"  
    3. **Decision Tree #3** melihat umur: "Baru = +$20k"
    4. ...dan seterusnya hingga 100 trees
    5. **Hasil Akhir:** Kombinasi semua pendapat = prediksi akurat!
    
    ### ✅ Keunggulan XGBoost
    
    - 🏆 **Akurasi Tinggi:** Sering menang di kompetisi Kaggle
    - ⚡ **Cepat & Efisien:** Optimized untuk performa tinggi
    - 🛡️ **Robust:** Handle missing values dan outliers
    - 🎯 **Flexible:** Banyak parameter untuk fine-tuning
    - 📊 **Interpretable:** Dapat melihat fitur mana yang penting
    
    ---
    
    ## 📊 Penjelasan Metrik
    
    ### MAE (Mean Absolute Error)
    ```
    MAE = Rata-rata |Prediksi - Aktual|
    ```
    **Contoh:** MAE = $5 artinya rata-rata prediksi meleset $5
    
    ### RMSE (Root Mean Squared Error)
    ```
    RMSE = √(Rata-rata (Prediksi - Aktual)²)
    ```
    **Kegunaan:** Memberikan penalti lebih besar untuk error yang besar
    
    ### R² Score
    ```
    R² = 1 - (Error Model / Total Variasi)
    ```
    **Interpretasi:**
    - R² = 0.9 → Model explain 90% variasi data (Excellent!)
    - R² = 0.7 → Model explain 70% variasi data (Good)
    - R² = 0.5 → Model explain 50% variasi data (Need improvement)
    
    ---
    
    ## 💡 Tips Penggunaan
    
    ### Untuk Mendapat Prediksi Akurat:
    
    1. ✅ **Isi Data Lengkap:** Semua field harus diisi
    2. ✅ **Data Konsisten:** Gunakan format yang sama dengan training
    3. ✅ **Update Model:** Retrain setiap 1-3 bulan dengan data baru
    4. ✅ **Validasi:** Cek apakah prediksi masuk akal
    
    ### Best Practices:
    
    - 📊 **Monitoring:** Track performa model di production
    - 🔄 **Retraining:** Update model dengan data terbaru
    - 📈 **A/B Testing:** Test performa sebelum full deployment
    - 📝 **Documentation:** Catat semua assumptions dan limitations
    
    ---
    
    ## ❓ FAQ
    
    **Q: Berapa akurasi model ini?**  
    A: Lihat R² Score di hasil training. >0.8 = sangat baik, 0.7-0.8 = baik.
    
    **Q: Apakah prediksi 100% akurat?**  
    A: Tidak ada model yang 100% akurat. Gunakan sebagai guidance, bukan keputusan final.
    
    **Q: Berapa data minimum untuk training?**  
    A: Minimal 500 rows, optimal 1000+ rows untuk hasil terbaik.
    
    **Q: Kapan harus retrain model?**  
    A: Setiap 1-3 bulan, atau jika performa menurun >5%.
    
    **Q: Bagaimana cara improve akurasi?**  
    A: 1) Tambah data, 2) Feature engineering, 3) Hyperparameter tuning, 4) Ensemble methods.
    
    **Q: Apakah bisa untuk industri lain?**  
    A: Ya! XGBoost universal, bisa untuk berbagai use case dengan data yang sesuai.
    
    ---
    
    ## 📞 Dukungan
    
    Butuh bantuan? Hubungi:
    - 📧 Email: support@yourcompany.com
    - 💬 Chat: Live chat di website
    - 📚 Docs: docs.yourcompany.com
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; color: white;'>
    <h3>🛒 Analisis XGBoost - Purchase Prediction System</h3>
    <p><b>Version 3.0</b> | Powered by XGBoost & Streamlit</p>
    <p>Dibuat dengan ❤️ untuk membantu bisnis membuat keputusan berbasis data</p>
    <p style='font-size: 12px; margin-top: 10px;'>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
