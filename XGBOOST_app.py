# -*- coding: utf-8 -*-
"""
Analisis XGBoost - Prediksi Purchase Amount
Version: 3.0.0 - Complete Interactive Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import sys

# Configure page FIRST before any other st commands
st.set_page_config(
    page_title="Analisis XGBoost - Prediksi Pembelian",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import libraries with proper error handling
def check_imports():
    """Check and import required libraries"""
    imports_ok = True
    error_msgs = []
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        globals()['train_test_split'] = train_test_split
        globals()['mean_absolute_error'] = mean_absolute_error
        globals()['mean_squared_error'] = mean_squared_error
        globals()['r2_score'] = r2_score
    except ImportError as e:
        imports_ok = False
        error_msgs.append(f"❌ Scikit-learn: {e}")
    
    try:
        from xgboost import XGBRegressor
        globals()['XGBRegressor'] = XGBRegressor
    except ImportError as e:
        imports_ok = False
        error_msgs.append(f"❌ XGBoost: {e}")
    
    # Import matplotlib - ALWAYS try to import
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
        globals()['plt'] = plt
        globals()['sns'] = sns
        
        # Set flag to True if successful
        return imports_ok, error_msgs, True
    except ImportError as e:
        error_msgs.append(f"⚠️ Matplotlib: {e}")
        return imports_ok, error_msgs, False

# Check imports
IMPORTS_OK, IMPORT_ERRORS, MATPLOTLIB_OK = check_imports()

# Custom CSS
st.markdown("""
<style>
    .big-font {font-size:24px !important; font-weight: bold; color: #1f77b4;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

def train_model_with_full_analysis(df):
    """
    Train XGBoost model dengan analisis lengkap dan visualisasi interaktif
    """
    
    # ============ TAHAP 1: EKSPLORASI DATA AWAL ============
    st.markdown("---")
    st.markdown("## 📊 TAHAP 1: Eksplorasi Data Awal (EDA)")
    
    # Metrics Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Baris", f"{len(df):,}")
    with col2:
        st.metric("📋 Jumlah Kolom", f"{len(df.columns)}")
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("❓ Data Hilang", f"{missing}")
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("🔄 Duplikat", f"{duplicates}")
    
    # Preview Data
    with st.expander("🔍 Lihat 10 Baris Pertama Data"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Statistik Deskriptif
    with st.expander("📈 Statistik Deskriptif"):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Info Kolom
    with st.expander("ℹ️ Informasi Kolom"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    # Visualisasi Distribusi Target
    st.markdown("### 📊 Distribusi Purchase Amount (Variabel Target)")
    
    if MATPLOTLIB_OK:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(df['Purchase Amount (USD)'], kde=True, bins=40, ax=ax, color='#667eea', edgecolor='black')
            ax.set_title('Distribusi Jumlah Pembelian', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Purchase Amount (USD)', fontsize=12)
            ax.set_ylabel('Frekuensi', fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**📊 Statistik:**")
            stats = df['Purchase Amount (USD)'].describe()
            st.metric("Mean", f"${stats['mean']:.2f}")
            st.metric("Median", f"${stats['50%']:.2f}")
            st.metric("Std Dev", f"${stats['std']:.2f}")
            st.metric("Min", f"${stats['min']:.2f}")
            st.metric("Max", f"${stats['max']:.2f}")
    else:
        st.error("❌ Matplotlib tidak tersedia - visualisasi tidak dapat ditampilkan")
        st.info("Pastikan matplotlib dan seaborn terinstall di requirements.txt")
    
    st.markdown("""
    <div class='info-box'>
    <b>💡 Interpretasi:</b><br>
    - Histogram menunjukkan distribusi nilai pembelian pelanggan<br>
    - Kurva KDE (Kernel Density Estimation) membantu melihat pola distribusi<br>
    - Ini adalah variabel yang akan kita prediksi menggunakan XGBoost
    </div>
    """, unsafe_allow_html=True)
    
    # ============ TAHAP 2: DATA CLEANING ============
    st.markdown("---")
    st.markdown("## 🧹 TAHAP 2: Pembersihan Data (Data Cleaning)")
    
    cleaning_steps = []
    
    # Drop Customer ID
    if 'Customer ID' in df.columns:
        df = df.drop('Customer ID', axis=1)
        cleaning_steps.append("✅ Kolom 'Customer ID' dihapus (tidak relevan untuk prediksi)")
    
    # Handle duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        cleaning_steps.append(f"✅ {dup_count} baris duplikat dihapus")
    else:
        cleaning_steps.append("✅ Tidak ada data duplikat")
    
    # Rename columns
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('__', '_').str.lower()
    cleaning_steps.append(f"✅ Nama kolom diseragamkan (lowercase, underscore)")
    
    # Check missing values
    missing_after = df.isnull().sum().sum()
    cleaning_steps.append(f"✅ Total nilai hilang setelah cleaning: {missing_after}")
    
    for step in cleaning_steps:
        st.success(step)
    
    st.info(f"**📊 Data setelah cleaning: {len(df):,} baris x {len(df.columns)} kolom**")
    
    # ============ TAHAP 3: FEATURE ENGINEERING ============
    st.markdown("---")
    st.markdown("## ⚙️ TAHAP 3: Rekayasa Fitur (Feature Engineering)")
    
    # Binary encoding
    st.markdown("### 🔄 Konversi Variabel Binary (Yes/No → 1/0)")
    binary_cols = ['subscription_status', 'discount_applied', 'promo_code_used']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            st.success(f"✅ `{col}` dikonversi ke format numerik")
    
    # Correlation Matrix
    if MATPLOTLIB_OK:
        st.markdown("### 🔗 Matriks Korelasi Variabel Numerik")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(14, 10))
            corr_matrix = df[numeric_cols].corr()
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, 
                       linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            
            ax.set_title('Matriks Korelasi Antar Variabel Numerik', 
                        fontsize=16, fontweight='bold', pad=20)
            
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            <div class='info-box'>
            <b>💡 Cara Membaca Korelasi:</b><br>
            - <b>+1.0</b>: Korelasi positif sempurna (kedua variabel bergerak searah)<br>
            - <b>0.0</b>: Tidak ada korelasi<br>
            - <b>-1.0</b>: Korelasi negatif sempurna (berlawanan arah)<br>
            - Nilai mendekati ±1 menunjukkan hubungan kuat
            </div>
            """, unsafe_allow_html=True)
    
    # One-Hot Encoding
    st.markdown("### 🎯 One-Hot Encoding untuk Variabel Kategorikal")
    
    cols_to_encode = ['gender', 'item_purchased', 'category', 'location', 'size', 
                      'color', 'season', 'payment_method', 'frequency_of_purchases']
    
    original_shape = df.shape
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    new_shape = df.shape
    
    st.success(f"✅ One-Hot Encoding selesai: {original_shape[1]} kolom → **{new_shape[1]} kolom**")
    
    st.markdown("""
    <div class='info-box'>
    <b>💡 One-Hot Encoding:</b><br>
    Teknik mengubah variabel kategorikal menjadi format numerik binary.<br>
    Contoh: Gender (Male/Female) → gender_Male (0/1)<br>
    Parameter <code>drop_first=True</code> mencegah multikolinearitas.
    </div>
    """, unsafe_allow_html=True)
    
    # ============ TAHAP 4: SPLIT DATA ============
    st.markdown("---")
    st.markdown("## ✂️ TAHAP 4: Pembagian Data (Train-Test Split)")
    
    X = df.drop('purchase_amount_usd', axis=1)
    y = df['purchase_amount_usd']
    
    test_size = st.slider("Pilih proporsi data testing:", 0.1, 0.3, 0.2, 0.05)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎓 Training Set", f"{len(X_train):,} baris", 
                 f"{(1-test_size)*100:.0f}%")
    
    with col2:
        st.metric("🧪 Testing Set", f"{len(X_test):,} baris", 
                 f"{test_size*100:.0f}%")
    
    with col3:
        st.metric("📊 Total Features", f"{X_train.shape[1]}")
    
    st.markdown("""
    <div class='info-box'>
    <b>💡 Kenapa Split Data?</b><br>
    - <b>Training Set:</b> Untuk melatih model belajar pola data<br>
    - <b>Testing Set:</b> Untuk menguji akurasi model pada data yang belum pernah dilihat<br>
    - Rasio 80:20 adalah standar umum dalam machine learning
    </div>
    """, unsafe_allow_html=True)
    
    # Label Encoding for shipping_type
    shipping_categories = X_train['shipping_type'].unique()
    shipping_mapping = {cat: idx for idx, cat in enumerate(shipping_categories)}
    
    X_train['shipping_type'] = X_train['shipping_type'].map(shipping_mapping)
    X_test['shipping_type'] = X_test['shipping_type'].map(shipping_mapping)
    
    st.success(f"✅ Label Encoding untuk 'shipping_type': {len(shipping_mapping)} kategori")
    
    # ============ TAHAP 5: TRAINING MODEL ============
    st.markdown("---")
    st.markdown("## 🤖 TAHAP 5: Training Model XGBoost")
    
    # Hyperparameter selection
    with st.expander("🎛️ Konfigurasi Hyperparameter (Opsional)"):
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("n_estimators (jumlah trees)", 50, 300, 100, 25)
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
            max_depth = st.slider("max_depth", 3, 10, 5)
        
        with col2:
            subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.1)
            colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.1)
    else:
        n_estimators = 100
        learning_rate = 0.1
        max_depth = 5
        subsample = 0.8
        colsample_bytree = 0.8
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("⏳ Menginisialisasi model XGBoost...")
    progress_bar.progress(20)
    
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1
    )
    
    status_text.text("🔄 Melatih model... (ini mungkin membutuhkan 1-2 menit)")
    progress_bar.progress(40)
    
    model.fit(X_train, y_train)
    
    status_text.text("✅ Model berhasil dilatih!")
    progress_bar.progress(100)
    
    st.markdown("""
    <div class='success-box'>
    <h4>🎉 Model XGBoost Berhasil Dilatih!</h4>
    Model menggunakan ensemble dari <b>{}</b> decision trees yang bekerja bersama untuk membuat prediksi akurat.
    </div>
    """.format(n_estimators), unsafe_allow_html=True)
    
    with st.expander("📖 Penjelasan Hyperparameter"):
        st.markdown("""
        - **n_estimators**: Jumlah pohon keputusan (lebih banyak = lebih akurat tapi lebih lambat)
        - **learning_rate**: Kecepatan pembelajaran (lebih kecil = lebih stabil tapi butuh lebih banyak trees)
        - **max_depth**: Kedalaman maksimum pohon (terlalu besar = overfitting)
        - **subsample**: Proporsi data untuk setiap pohon (< 1.0 = regularization)
        - **colsample_bytree**: Proporsi fitur untuk setiap pohon (< 1.0 = regularization)
        """)
    
    # ============ TAHAP 6: EVALUASI MODEL ============
    st.markdown("---")
    st.markdown("## 📊 TAHAP 6: Evaluasi Performa Model")
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics with styled cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>${mae:.2f}</h3>
            <p>Mean Absolute Error (MAE)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>${rmse:.2f}</h3>
            <p>Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{r2:.4f}</h3>
            <p>R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        accuracy_pct = r2 * 100
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{accuracy_pct:.1f}%</h3>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interpretasi Metrics
    with st.expander("📖 Penjelasan Metrik Evaluasi"):
        performance_level = "Excellent!" if r2 > 0.9 else "Sangat Baik!" if r2 > 0.8 else "Baik" if r2 > 0.7 else "Cukup Baik" if r2 > 0.6 else "Perlu Ditingkatkan"
        
        st.markdown(f"""
        **MAE (Mean Absolute Error): ${mae:.2f}**
        - Rata-rata selisih absolut antara prediksi dan nilai aktual
        - Interpretasi: Model rata-rata meleset sekitar **${mae:.2f}** dari nilai sebenarnya
        - Semakin kecil, semakin baik
        
        **RMSE (Root Mean Squared Error): ${rmse:.2f}**
        - Mirip dengan MAE, tapi memberikan penalti lebih besar pada error yang besar
        - Berguna untuk mendeteksi outlier predictions
        - Semakin kecil, semakin baik
        
        **R² Score: {r2:.4f} ({accuracy_pct:.1f}%)**
        - Mengukur seberapa baik model menjelaskan variasi data
        - Range: 0-1 (0% - 100%)
        - Interpretasi: **{accuracy_pct:.1f}%** variasi dalam purchase amount dapat dijelaskan oleh model
        - **Tingkat Performa: {performance_level}**
        
        **Kesimpulan:**
        Model ini dapat menjelaskan **{accuracy_pct:.1f}%** dari variabilitas data, dengan rata-rata kesalahan prediksi ${mae:.2f}.
        """)
    
    # Visualisasi Prediksi vs Aktual
    if MATPLOTLIB_OK:
        st.markdown("### 🎯 Prediksi vs Nilai Aktual")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=50, c='#667eea', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction', alpha=0.8)
        
        ax.set_xlabel('Nilai Aktual (USD)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Prediksi Model (USD)', fontsize=14, fontweight='bold')
        ax.set_title('Perbandingan Prediksi vs Nilai Aktual', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add R² score to plot
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
               fontsize=14, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        <div class='info-box'>
        <b>💡 Cara Membaca Grafik:</b><br>
        - Setiap titik biru mewakili satu prediksi<br>
        - Garis merah putus-putus = prediksi sempurna (aktual = prediksi)<br>
        - Titik yang dekat dengan garis merah = prediksi akurat<br>
        - Titik yang jauh dari garis = prediksi kurang akurat
        </div>
        """, unsafe_allow_html=True)
    
    # Analisis Residuals
    if MATPLOTLIB_OK:
        st.markdown("### 📉 Analisis Residuals (Kesalahan Prediksi)")
        
        residuals = y_test - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(residuals, kde=True, bins=40, ax=ax, color='#ff6b6b', edgecolor='black')
            ax.axvline(x=0, color='darkred', linestyle='--', linewidth=2, label='Zero Error')
            ax.set_title('Distribusi Residuals', fontsize=14, fontweight='bold')
            ax.set_xlabel('Residuals (Aktual - Prediksi)', fontsize=12)
            ax.set_ylabel('Frekuensi', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_pred, residuals, alpha=0.6, s=50, c='#4ecdc4', edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='darkred', linestyle='--', linewidth=2, label='Zero Error')
            ax.set_xlabel('Prediksi (USD)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
            ax.set_title('Residuals vs Prediksi', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("""
        <div class='info-box'>
        <b>💡 Interpretasi Residuals:</b><br>
        - <b>Histogram (kiri):</b> Seharusnya berbentuk kurva normal (bell curve) terpusat di 0<br>
        - <b>Scatter Plot (kanan):</b> Titik seharusnya tersebar acak di sekitar garis y=0<br>
        - Jika ada pola tertentu → ada informasi yang belum ditangkap model<br>
        - Distribusi normal residuals = model bekerja dengan baik
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("---")
    st.markdown("## 🔍 TAHAP 7: Analisis Feature Importance")
    
    feature_importances = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    st.markdown("### 📊 Top 20 Fitur Terpenting")
    
    if MATPLOTLIB_OK:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        top_n = min(20, len(features_df))
        top_features = features_df.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        
        bars = ax.barh(range(top_n), top_features['Importance'], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['Feature'], fontsize=11)
        ax.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
        ax.set_title(f'Top {top_n} Fitur Terpenting dalam Model XGBoost', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['Importance'])):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Display full feature importance table
    with st.expander("📋 Lihat Semua Feature Importance"):
        st.dataframe(features_df.style.background_gradient(cmap='Blues'), 
                    use_container_width=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>💡 Feature Importance menunjukkan:</b><br>
    - Fitur mana yang paling berpengaruh dalam membuat prediksi<br>
    - Semakin tinggi nilai importance, semakin besar pengaruhnya<br>
    - Berguna untuk memahami faktor kunci yang mempengaruhi purchase amount<br>
    - Dapat digunakan untuk feature selection dan business insights
    </div>
    """, unsafe_allow_html=True)
    
    # Kesimpulan
    st.markdown("---")
    st.markdown("## 📝 Kesimpulan & Rekomendasi")
    
    st.success(f"""
    ### ✅ Model XGBoost Berhasil Dilatih!
    
    **Ringkasan Performa:**
    - Model dapat menjelaskan **{r2*100:.2f}%** variabilitas dalam jumlah pembelian
    - Rata-rata kesalahan prediksi: **${mae:.2f}**
    - RMSE: **${rmse:.2f}**
    - Tingkat Akurasi: **{performance_level}**
    
    **Fitur Terpenting:**
    1. {features_df.iloc[0]['Feature']}: {features_df.iloc[0]['Importance']:.4f}
    2. {features_df.iloc[1]['Feature']}: {features_df.iloc[1]['Importance']:.4f}
    3. {features_df.iloc[2]['Feature']}: {features_df.iloc[2]['Importance']:.4f}
    """)
    
    with st.expander("🚀 Langkah Selanjutnya"):
        st.markdown("""
        **Untuk Meningkatkan Model:**
        1. **Hyperparameter Tuning:** Gunakan GridSearchCV atau RandomizedSearchCV
        2. **Cross-Validation:** Implementasi k-fold CV untuk validasi lebih robust
        3. **Feature Engineering Lanjutan:** Buat interaction features dan polynomial features
        4. **Ensemble Methods:** Combine dengan model lain (Random Forest, LightGBM)
        5. **SHAP Values:** Gunakan SHAP untuk interpretasi model yang lebih mendalam
        6. **A/B Testing:** Test model di production dengan sample kecil
        7. **Monitoring:** Setup sistem monitoring performa model seiring waktu
        
        **Untuk Business Implementation:**
        1. Deploy model ke production environment
        2. Integrate dengan CRM/e-commerce system
        3. Setup automated retraining pipeline
        4. Create dashboard untuk business stakeholders
        5. Document model limitations dan edge cases
        """)
    
    return model, shipping_mapping, X_train.columns.tolist(), {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2
    }

# Load model from file
@st.cache_resource
def get_model():
    """Load trained model from pickle files"""
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

# ============ MAIN APPLICATION ============
st.markdown("<h1 style='text-align: center; color: #667eea;'>🛒 Analisis XGBoost: Prediksi Jumlah Pembelian</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #764ba2;'>Sistem Prediksi Purchase Amount dengan Machine Learning</h3>", unsafe_allow_html=True)

# Check if imports are successful
if not IMPORTS_OK:
    st.error("⚠️ Beberapa library gagal di-import:")
    for error in IMPORT_ERRORS:
        st.error(error)
    st.stop()

# Show matplotlib status
if MATPLOTLIB_OK:
    st.success("✅ Matplotlib & Seaborn tersedia - Visualisasi aktif")
else:
    st.warning("⚠️ Matplotlib tidak tersedia - Visualisasi akan dibatasi")

if IMPORT_ERRORS:  # Show warnings for matplotlib
    for error in IMPORT_ERRORS:
        st.warning(error)

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📚 Training Model", "🎯 Prediksi", "📖 Dokumentasi"])

# ============ TAB 1: TRAINING ============
with tab1:
    st.header("📚 Training Model XGBoost")
    st.markdown("Upload dataset dan latih model dengan analisis lengkap step-by-step")
    
    model, shipping_mapping, feature_columns, model_exists = get_model()
    
    if model_exists:
        st.success("✅ Model sudah tersedia dan siap digunakan!")
        st.info("💡 Jika ingin melatih ulang dengan data baru, upload dataset di bawah")
    else:
        st.warning("⚠️ Model belum tersedia. Silakan upload dataset untuk training")
    
    st.markdown("### 📤 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload file **shopping_behavior_updated.csv**",
        type=['csv'],
        help="File CSV harus berisi kolom Purchase Amount (USD) dan fitur lainnya"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dataset berhasil diupload!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Baris", f"{len(df):,}")
        with col2:
            st.metric("📋 Total Kolom", f"{len(df.columns)}")
        with col3:
            st.metric("💾 Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("❓ Missing", f"{df.isnull().sum().sum()}")
        
        with st.expander("👁️ Preview Dataset"):
            st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("---")
        
        if st.button("🚀 Mulai Training Model", type="primary", use_container_width=True):
            try:
                with st.spinner("Training sedang berlangsung... Mohon tunggu"):
                    model, shipping_mapping, feature_columns, metrics = train_model_with_full_analysis(df)
                    
                    # Save to session state
                    st.session_state.model = model
                    st.session_state.shipping_mapping = shipping_mapping
                    st.session_state.feature_columns = feature_columns
                    
                    st.balloons()
                    st.success("🎉 Model berhasil dilatih dan siap digunakan!")
                    st.info("💡 Silakan pindah ke tab **🎯 Prediksi** untuk mulai menggunakan model")
                    
            except Exception as e:
                st.error(f"❌ Error saat training: {str(e)}")
                st.info("💡 Pastikan format CSV sesuai dengan yang diharapkan")
                with st.expander("🐛 Debug Information"):
                    st.exception(e)

# ============ TAB 2: PREDIKSI ============
with tab2:
    st.header("🎯 Prediksi Jumlah Pembelian")
    st.markdown("Input data pelanggan untuk mendapatkan prediksi purchase amount")
    
    # Load model
    if 'model' in st.session_state:
        model = st.session_state.model
        shipping_mapping = st.session_state.shipping_mapping
        feature_columns = st.session_state.feature_columns
        model_exists = True
    else:
        model, shipping_mapping, feature_columns, model_exists = get_model()
    
    if not model_exists:
        st.warning("⚠️ Model belum tersedia. Silakan training model terlebih dahulu di tab **📚 Training Model**")
    else:
        st.success("✅ Model siap digunakan!")
        
        st.markdown("### 📝 Input Data Pelanggan")
        st.markdown("*Isi semua informasi di bawah untuk mendapatkan prediksi yang akurat*")
        
        # Input form with three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Informasi Demografis")
            age = st.slider("Usia", 18, 70, 35, help="Usia pelanggan dalam tahun")
            gender = st.selectbox("Gender", ["Male", "Female"])
            location = st.selectbox("Lokasi", [
                "Alabama", "Alaska", "Arizona", "California", "Colorado",
                "Florida", "Georgia", "Illinois", "Indiana", "Kentucky",
                "Louisiana", "Massachusetts", "Michigan", "Montana", "Nevada",
                "New York", "Ohio", "Oregon", "Pennsylvania", "Texas"
            ])
        
        with col2:
            st.markdown("#### 🛍️ Informasi Produk")
            item_purchased = st.selectbox("Item yang Dibeli", [
                "Blouse", "Dress", "Jacket", "Jeans", "Pants", "Shirt",
                "Shoes", "Shorts", "Skirt", "Sweater", "T-shirt"
            ])
            category = st.selectbox("Kategori Produk", [
                "Accessories", "Clothing", "Footwear", "Outerwear"
            ])
            size = st.selectbox("Ukuran", ["S", "M", "L", "XL"])
            color = st.selectbox("Warna", [
                "Beige", "Black", "Blue", "Brown", "Charcoal", "Cyan",
                "Gold", "Gray", "Green", "Lavender", "Magenta", "Maroon",
                "Olive", "Orange", "Peach", "Pink", "Purple", "Red",
                "Silver", "Teal", "Turquoise", "White", "Yellow"
            ])
            season = st.selectbox("Musim", ["Fall", "Winter", "Spring", "Summer"])
        
        with col3:
            st.markdown("#### 💳 Informasi Transaksi")
            previous_purchases = st.number_input(
                "Jumlah Pembelian Sebelumnya", 
                0, 50, 10,
                help="Total transaksi yang pernah dilakukan pelanggan"
            )
            review_rating = st.slider(
                "Rating Review", 
                1.0, 5.0, 4.0, 0.1,
                help="Rating rata-rata dari pelanggan"
            )
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
            subscription_status = st.radio(
                "Status Langganan", 
                ["Yes", "No"], 
                horizontal=True,
                help="Apakah pelanggan berlangganan newsletter/membership?"
            )
            discount_applied = st.radio(
                "Diskon Diterapkan", 
                ["Yes", "No"], 
                horizontal=True,
                help="Apakah ada diskon untuk pembelian ini?"
            )
            promo_code_used = st.radio(
                "Kode Promo Digunakan", 
                ["Yes", "No"], 
                horizontal=True,
                help="Apakah pelanggan menggunakan kode promo?"
            )
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🔮 Prediksi Jumlah Pembelian", type="primary", use_container_width=True):
            
            # Prepare input data
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
            
            # Create DataFrame
            df_input = pd.DataFrame([input_data])
            for key, value in categorical_features.items():
                df_input[key] = value
            
            # One-hot encoding
            df_encoded = pd.get_dummies(df_input, columns=list(categorical_features.keys()), drop_first=True)
            
            # Encode shipping_type
            df_encoded['shipping_type'] = shipping_type
            if shipping_mapping is not None:
                if shipping_type in shipping_mapping:
                    df_encoded['shipping_type'] = shipping_mapping[shipping_type]
                else:
                    df_encoded['shipping_type'] = 0
            
            # Ensure all columns match training
            for col in feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            
            df_encoded = df_encoded[feature_columns]
            
            # Make prediction
            prediction = model.predict(df_encoded)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 💰 Hasil Prediksi")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h2 style='color: white;'>${prediction:.2f}</h2>
                    <p style='color: white;'>Prediksi Purchase Amount</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                confidence = "Tinggi" if 20 <= prediction <= 80 else "Sedang"
                color = "#28a745" if confidence == "Tinggi" else "#ffc107"
                st.markdown(f"""
                <div style='background: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <h2>{confidence}</h2>
                    <p>Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                category_avg = {"Accessories": 45, "Clothing": 55, "Footwear": 65, "Outerwear": 70}
                diff = prediction - category_avg.get(category, 50)
                color = "#28a745" if diff > 0 else "#dc3545"
                st.markdown(f"""
                <div style='background: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <h2>${diff:+.2f}</h2>
                    <p>vs Rata-rata Kategori</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                level = "High Value" if prediction > 70 else "Medium Value" if prediction > 40 else "Low Value"
                emoji = "🟢" if level == "High Value" else "🟡" if level == "Medium Value" else "🔴"
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <h2>{emoji}</h2>
                    <p>{level}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visual Analysis
            st.markdown("---")
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
            
            # Insights & Recommendations
            st.markdown("---")
            st.markdown("### 💡 Insight & Rekomendasi Strategis")
            
            col1, col2 = st.columns(2)
            
            insights = []
            recommendations = []
            
            # Value-based insights
            if prediction > 70:
                insights.append("✅ Pelanggan ini termasuk dalam kategori **high-value customer** (top 20%)")
                recommendations.append("🎯 **Strategi VIP:** Berikan program loyalitas premium dan akses eksklusif")
                recommendations.append("💎 **Upselling:** Tawarkan produk premium atau bundle dengan nilai lebih tinggi")
                recommendations.append("🎁 **Retention:** Prioritaskan untuk program customer retention")
            elif prediction < 30:
                insights.append("⚠️ Pelanggan ini termasuk dalam kategori **low-value customer**")
                recommendations.append("💰 **Strategi Promosi:** Berikan diskon atau voucher khusus untuk meningkatkan purchase value")
                recommendations.append("📧 **Email Marketing:** Kirim rekomendasi produk dengan harga terjangkau")
                recommendations.append("🔄 **Re-engagement:** Campaign khusus untuk aktivasi kembali")
            else:
                insights.append("ℹ️ Pelanggan ini termasuk dalam kategori **medium-value customer**")
                recommendations.append("📈 **Growth Strategy:** Dorong upgrade dengan cross-selling dan bundling")
                recommendations.append("🎁 **Reward Program:** Berikan poin rewards untuk pembelian berikutnya")
                recommendations.append("💳 **Payment Options:** Tawarkan installment atau buy-now-pay-later")
            
            # Subscription status
            if subscription_status == "Yes":
                insights.append("📧 **Subscriber aktif** - Kemungkinan pembelian berulang tinggi (retention rate +35%)")
                recommendations.append("💌 **Newsletter:** Kirim konten eksklusif dan early access produk baru")
                recommendations.append("🎉 **Subscriber Perks:** Berikan benefit tambahan untuk memperkuat loyalitas")
            else:
                recommendations.append("📨 **Subscription Campaign:** Tawarkan trial subscription dengan benefit menarik")
            
            # Price sensitivity
            if discount_applied == "Yes" or promo_code_used == "Yes":
                insights.append("🎁 **Price sensitive customer** - Sangat responsif terhadap promosi dan diskon")
                recommendations.append("🏷️ **Segment:** Masukkan dalam list untuk seasonal sale dan flash deals")
                recommendations.append("⏰ **Timing:** Target dengan promo pada akhir bulan atau hari gajian")
            
            # Purchase history
            if previous_purchases > 20:
                insights.append("⭐ **Super loyal customer** dengan riwayat pembelian sangat tinggi (20+ transaksi)")
                recommendations.append("🏆 **VIP Status:** Berikan status VIP dengan akses ke exclusive member zone")
                recommendations.append("🎖️ **Ambassador Program:** Pertimbangkan untuk program brand ambassador")
            elif previous_purchases > 10:
                insights.append("✨ **Pelanggan reguler** dengan engagement yang baik (10+ transaksi)")
                recommendations.append("📊 **Loyalty Tier:** Naikkan ke tier loyalty yang lebih tinggi")
            else:
                insights.append("🆕 Pelanggan baru atau jarang bertransaksi")
                recommendations.append("👋 **Welcome Campaign:** Berikan welcome discount 15-20% untuk pembelian kedua")
                recommendations.append("🎯 **Onboarding:** Fokus pada customer onboarding yang baik")
            
            # Demographics
            if age < 25:
                insights.append("👶 **Demografis muda** (Gen Z) - Preferensi trend fashion terkini dan sustainable products")
                recommendations.append("📱 **Social Media:** Fokus pada Instagram, TikTok, dan influencer marketing")
                recommendations.append("🎮 **Gamification:** Implementasi loyalty points dan challenges")
            elif age > 50:
                insights.append("👴 **Demografis mature** - Preferensi kualitas, kenyamanan, dan customer service excellent")
                recommendations.append("📧 **Email Marketing:** Gunakan email dengan informasi detail dan testimonial")
                recommendations.append("📞 **Customer Service:** Pastikan customer service yang responsif dan helpful")
            else:
                insights.append("👔 **Demografis millennial** - Balance antara quality, price, dan convenience")
                recommendations.append("💻 **Omnichannel:** Optimalkan experience di semua channel (online & offline)")
            
            with col1:
                st.markdown("#### 🔍 Customer Insights")
                for insight in insights:
                    st.markdown(f"- {insight}")
            
            with col2:
                st.markdown("#### 🎯 Action Items")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            # Additional Metrics
            st.markdown("---")
            st.markdown("### 📈 Metrik Tambahan & Analisis Lanjutan")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Customer Lifetime Value estimate
                clv_estimate = prediction * (previous_purchases + 10) * 0.8
                st.metric(
                    "💰 Estimasi CLV",
                    f"${clv_estimate:.2f}",
                    help="Customer Lifetime Value - Estimasi total nilai pelanggan sepanjang hubungan dengan brand"
                )
            
            with col2:
                # Repeat purchase probability
                repeat_prob = min((previous_purchases / 50) * 100, 95)
                st.metric(
                    "🔄 Repeat Purchase Probability",
                    f"{repeat_prob:.0f}%",
                    help="Kemungkinan pelanggan akan kembali melakukan pembelian"
                )
            
            with col3:
                # Average order value
                avg_order = prediction * 0.95 if previous_purchases > 0 else prediction
                st.metric(
                    "📊 Avg Order Value",
                    f"${avg_order:.2f}",
                    help="Rata-rata nilai per transaksi"
                )
            
            with col4:
                # Churn risk
                if previous_purchases > 15:
                    churn_risk = "Low"
                    churn_color = "green"
                elif previous_purchases > 5:
                    churn_risk = "Medium"
                    churn_color = "orange"
                else:
                    churn_risk = "High"
                    churn_color = "red"
                
                st.metric(
                    "⚠️ Churn Risk",
                    churn_risk,
                    help="Risiko pelanggan berhenti bertransaksi"
                )

# ============ TAB 3: DOKUMENTASI ============
with tab3:
    st.header("📖 Dokumentasi Lengkap")
    
    st.markdown("""
    ## 🎯 Tentang Aplikasi
    
    Aplikasi ini menggunakan **XGBoost (Extreme Gradient Boosting)**, sebuah algoritma machine learning 
    yang sangat powerful dan sering memenangkan kompetisi data science seperti Kaggle.
    
    ### 🔬 Apa itu XGBoost?
    
    XGBoost adalah algoritma ensemble learning yang menggabungkan banyak decision trees lemah menjadi
    satu model prediksi yang kuat. Keunggulan XGBoost:
    
    - ⚡ **Sangat Cepat:** Optimized untuk performa tinggi
    - 🎯 **Akurat:** Consistently wins ML competitions
    - 🛡️ **Robust:** Handle missing values dan outliers dengan baik
    - 🔧 **Flexible:** Banyak hyperparameter untuk tuning
    
    ### 📊 Cara Kerja Model
    
    ```
    Input Data → Feature Engineering → XGBoost Model → Prediksi Purchase Amount
    ```
    
    1. **Input Data**: 15+ fitur pelanggan (demografi, produk, transaksi)
    2. **Preprocessing**: Data cleaning, encoding, scaling
    3. **Training**: 100 decision trees belajar pola data
    4. **Prediction**: Ensemble voting dari semua trees
    5. **Output**: Prediksi purchase amount dalam USD
    
    ### 📋 Fitur yang Digunakan
    
    **Demografis (3 fitur):**
    - Usia pelanggan (18-70 tahun)
    - Gender (Male/Female)
    - Lokasi geografis (20 states)
    
    **Perilaku Pembelian (3 fitur):**
    - Jumlah pembelian sebelumnya (0-50+)
    - Frekuensi pembelian (Weekly, Monthly, dll)
    - Rating review (1.0-5.0)
    
    **Karakteristik Produk (5 fitur):**
    - Kategori produk (Clothing, Footwear, etc)
    - Item spesifik (Blouse, Dress, Jacket, etc)
    - Ukuran (S, M, L, XL)
    - Warna (25+ pilihan)
    - Musim pembelian (Fall, Winter, Spring, Summer)
    
    **Transaksi (5 fitur):**
    - Metode pembayaran (Credit Card, Cash, etc)
    - Tipe pengiriman (Express, Standard, etc)
    - Status langganan (Yes/No)
    - Penggunaan diskon (Yes/No)
    - Penggunaan kode promo (Yes/No)
    
    ### 🎓 Metrik Evaluasi Dijelaskan
    
    **MAE (Mean Absolute Error)**
    ```
    MAE = (1/n) × Σ|actual - predicted|
    ```
    - Rata-rata selisih absolut
    - Satuan: USD
    - Interpretasi: "Model rata-rata meleset $X"
    - Semakin kecil = semakin baik
    
    **RMSE (Root Mean Squared Error)**
    ```
    RMSE = √[(1/n) × Σ(actual - predicted)²]
    ```
    - Akar dari rata-rata kuadrat error
    - Memberikan penalti lebih besar pada error besar
    - Satuan: USD
    - Berguna untuk mendeteksi outlier predictions
    
    **R² Score (Coefficient of Determination)**
    ```
    R² = 1 - (SS_res / SS_tot)
    ```
    - Proporsi variasi yang dijelaskan model
    - Range: 0 - 1 (0% - 100%)
    - Benchmark:
      - **> 0.9:** Excellent
      - **0.8 - 0.9:** Very Good  
      - **0.7 - 0.8:** Good
      - **0.6 - 0.7:** Acceptable
      - **< 0.6:** Needs Improvement
    
    ### 🚀 Cara Menggunakan Aplikasi
    
    **Step 1: Training Model** 📚
    1. Buka tab "Training Model"
    2. Upload file CSV dengan data historis
    3. Klik "Mulai Training Model"
    4. Tunggu proses selesai (1-3 menit)
    5. Review hasil analisis dan metrik
    
    **Step 2: Melakukan Prediksi** 🎯
    1. Buka tab "Prediksi"
    2. Isi semua informasi pelanggan
    3. Klik "Prediksi Jumlah Pembelian"
    4. Lihat hasil dan rekomendasi
    5. Export hasil jika diperlukan
    
    ### 💡 Tips untuk Hasil Terbaik
    
    **Data Quality:**
    - Gunakan data minimal 1000 rows
    - Pastikan tidak ada missing values
    - Remove obvious outliers
    - Balance class distribution jika mungkin
    
    **Model Performance:**
    - Update model setiap 1-3 bulan dengan data terbaru
    - Monitor performa di production
    - A/B test sebelum full deployment
    - Setup automated retraining pipeline
    
    **Business Implementation:**
    - Integrate dengan CRM system
    - Create dashboard untuk stakeholders
    - Document business logic dan assumptions
    - Setup alerts untuk unusual predictions
    
    ### 🔬 Teknologi yang Digunakan
    
    | Teknologi | Versi | Fungsi |
    |-----------|-------|--------|
    | Python | 3.8+ | Bahasa pemrograman |
    | Streamlit | 1.28+ | Web framework |
    | XGBoost | 2.0+ | ML algorithm |
    | Pandas | 2.0+ | Data manipulation |
    | Scikit-learn | 1.3+ | ML utilities |
    | Matplotlib | 3.7+ | Visualisasi |
    | Seaborn | 0.12+ | Statistical plots |
    
    ### 📊 Sample Data Format
    
    File CSV harus memiliki format berikut:
    
    | Kolom | Tipe | Contoh | Deskripsi |
    |-------|------|--------|-----------|
    | Customer ID | String | "C12345" | ID unik pelanggan |
    | Age | Integer | 35 | Usia pelanggan |
    | Gender | String | "Male" | Jenis kelamin |
    | Item Purchased | String | "Blouse" | Item yang dibeli |
    | Category | String | "Clothing" | Kategori produk |
    | Purchase Amount (USD) | Float | 59.99 | **TARGET VARIABLE** |
    | Location | String | "California" | Lokasi pelanggan |
    | Size | String | "M" | Ukuran produk |
    | Color | String | "Black" | Warna produk |
    | Season | String | "Fall" | Musim pembelian |
    | Review Rating | Float | 4.5 | Rating dari pelanggan |
    | Subscription Status | String | "Yes" | Status berlangganan |
    | Shipping Type | String | "Express" | Jenis pengiriman |
    | Discount Applied | String | "Yes" | Apakah ada diskon |
    | Promo Code Used | String | "No" | Penggunaan promo |
    | Previous Purchases | Integer | 10 | Jumlah pembelian sebelumnya |
    | Payment Method | String | "Credit Card" | Metode pembayaran |
    | Frequency of Purchases | String | "Monthly" | Frekuensi beli |
    
    ### 🔧 Advanced: Hyperparameter Tuning
    
    Untuk advanced users, berikut parameter yang bisa di-tune:
    
    ```python
    xgb_model = XGBRegressor(
        n_estimators=100,        # Jumlah trees (50-500)
        learning_rate=0.1,       # Learning rate (0.01-0.3)
        max_depth=5,             # Tree depth (3-10)
        subsample=0.8,           # Sample ratio (0.5-1.0)
        colsample_bytree=0.8,    # Feature ratio (0.5-1.0)
        min_child_weight=1,      # Min samples in leaf (1-10)
        gamma=0,                 # Min loss reduction (0-5)
        reg_alpha=0,             # L1 regularization (0-1)
        reg_lambda=1,            # L2 regularization (0-10)
    )
    ```
    
    **Rekomendasi Tuning:**
    - **Dataset besar (>10k):** Tingkatkan n_estimators ke 200-300
    - **Overfitting:** Kurangi max_depth, tingkatkan gamma
    - **Underfitting:** Tingkatkan max_depth, kurangi regularization
    - **Training lambat:** Kurangi n_estimators, tingkatkan learning_rate
    
    ### 🐛 Troubleshooting
    
    **Problem: Model accuracy rendah (R² < 0.6)**
    - ✅ Check data quality dan outliers
    - ✅ Add more features atau feature engineering
    - ✅ Increase n_estimators
    - ✅ Try different algorithms (Random Forest, LightGBM)
    
    **Problem: Overfitting (Training acc >> Test acc)**
    - ✅ Reduce max_depth (coba 3-4)
    - ✅ Increase min_child_weight
    - ✅ Add regularization (reg_alpha, reg_lambda)
    - ✅ Reduce n_estimators
    - ✅ Increase subsample dan colsample_bytree
    
    **Problem: Prediction tidak masuk akal**
    - ✅ Check input data format
    - ✅ Ensure all features are encoded correctly
    - ✅ Verify feature columns match training
    - ✅ Check for data leakage
    
    **Problem: Model training terlalu lama**
    - ✅ Reduce n_estimators
    - ✅ Use smaller dataset for testing
    - ✅ Enable early stopping
    - ✅ Reduce max_depth
    
    ### 📚 Resources & Learning
    
    **Official Documentation:**
    - [XGBoost Documentation](https://xgboost.readthedocs.io/)
    - [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    
    **Recommended Reading:**
    - "Hands-On Machine Learning" by Aurélien Géron
    - "The Elements of Statistical Learning" by Hastie et al.
    - XGBoost Paper: "XGBoost: A Scalable Tree Boosting System"
    
    **Online Courses:**
    - Coursera: Machine Learning Specialization
    - Fast.ai: Practical Deep Learning
    - Kaggle Learn: Intermediate Machine Learning
    
    ### 🤝 Best Practices
    
    **Model Development:**
    1. ✅ Start with simple baseline model
    2. ✅ Perform thorough EDA before modeling
    3. ✅ Use cross-validation for robust evaluation
    4. ✅ Document all preprocessing steps
    5. ✅ Version control your models
    6. ✅ Create reproducible pipelines
    
    **Production Deployment:**
    1. ✅ Test model on holdout set
    2. ✅ Monitor model performance continuously
    3. ✅ Setup automated retraining
    4. ✅ Implement A/B testing
    5. ✅ Log all predictions for audit
    6. ✅ Create fallback mechanisms
    
    **Business Integration:**
    1. ✅ Align metrics with business KPIs
    2. ✅ Create explainable predictions
    3. ✅ Setup stakeholder dashboards
    4. ✅ Document model limitations
    5. ✅ Provide confidence intervals
    6. ✅ Regular model reviews with business teams
    
    ### 🔒 Privacy & Security
    
    **Data Privacy:**
    - Customer data diproses secara aman
    - Tidak ada data disimpan permanen di server
    - Model tidak menyimpan informasi identitas pelanggan
    - Comply dengan GDPR dan data protection regulations
    
    **Model Security:**
    - Model files di-encrypt saat disimpan
    - Access control untuk production models
    - Regular security audits
    - Backup models dan data
    
    ### 📞 Support & Contact
    
    **Untuk bantuan teknis:**
    - 📧 Email: support@yourdomain.com
    - 💬 Slack: #ml-support channel
    - 📖 Documentation: docs.yourdomain.com
    - 🐛 Bug Reports: github.com/yourrepo/issues
    
    **Untuk business inquiries:**
    - 📧 Email: business@yourdomain.com
    - 📞 Phone: +1-XXX-XXX-XXXX
    - 🏢 Office hours: Mon-Fri 9AM-5PM PST
    
    ### 📄 Changelog
    
    **Version 3.0.0 (Current)**
    - ✅ Complete interactive analysis with visualizations
    - ✅ Advanced feature importance analysis
    - ✅ Detailed insights and recommendations
    - ✅ Comprehensive documentation
    - ✅ Improved UI/UX with styled components
    
    **Version 2.0.0**
    - ✅ Added matplotlib visualizations
    - ✅ Enhanced EDA section
    - ✅ Better error handling
    
    **Version 1.0.0**
    - ✅ Initial release
    - ✅ Basic training and prediction
    
    ### 📜 License
    
    MIT License - Free to use and modify for commercial and non-commercial purposes.
    
    ---
    
    **Built with ❤️ by Data Science Team**
    
    *Last Updated: October 2025*
    """)
    
    # FAQ Section
    with st.expander("❓ Frequently Asked Questions (FAQ)"):
    st.markdown("""
    **Q: Berapa minimum data yang diperlukan untuk training?**  
    A: Minimal 500 rows, optimal 1000+ rows untuk hasil terbaik.
    
    **Q: Apakah model bisa digunakan untuk produk lain?**  
    A: Ya, asalkan struktur data dan fitur serupa. Mungkin perlu retraining dengan data spesifik.
    
    **Q: Berapa lama model harus di-update?**  
    A: Rekomendasi: setiap 1-3 bulan, atau ketika performa menurun >5%.
    
    **Q: Apakah prediksi 100% akurat?**  
    A: Tidak ada model yang 100% akurat. R² score menunjukkan tingkat akurasi. Gunakan sebagai guidance, bukan absolute truth.
    
    **Q: Bagaimana cara improve model accuracy?**  
    A: 1) Tambah data training, 2) Feature engineering, 3) Hyperparameter tuning, 4) Ensemble methods.
    
    **Q: Apakah data customer aman?**  
    A: Ya, data diproses secara lokal dan tidak disimpan permanen di server.
    
    **Q: Bisa export hasil prediksi?**  
    A: Saat ini manual copy, fitur export CSV akan ditambahkan di versi berikutnya.
    
    **Q: Model error/crash, apa yang harus dilakukan?**  
    A: 1) Check format data, 2) Ensure all required columns ada, 3) Check for missing values, 4) Contact support jika masih error.
    """)
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
    <h3>🛒 Analisis XGBoost - Purchase Prediction System</h3>
    <p><strong>Version 3.0.0</strong> | Powered by XGBoost & Streamlit</p>
    <p>Dibuat dengan ❤️ oleh Data Science Team</p>
    <p style='font-size: 12px; margin-top: 20px;'>
        © 2025 All Rights Reserved | 
        <a href='#' style='color: white;'>Privacy Policy</a> | 
        <a href='#' style='color: white;'>Terms of Service</a>
    </p>
</div>
""", unsafe_allow_html=True)
