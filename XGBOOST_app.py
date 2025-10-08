import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CRITICAL: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis XGBoost",
    page_icon="🛒",
    layout="wide"
)

# Fungsi untuk train model dengan visualisasi
def train_model_with_viz(df):
    """Train XGBoost model dengan analisis lengkap"""
    
    st.subheader("📊 Tahap 1: Eksplorasi Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Kolom", f"{len(df.columns)}")
    with col3:
        st.metric("Data Hilang", f"{df.isnull().sum().sum()}")
    
    # Visualisasi Distribusi Target
    st.markdown("### 📊 Distribusi Purchase Amount")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Purchase Amount (USD)'], kde=True, bins=30, ax=ax, color='skyblue')
    ax.set_title('Distribusi Jumlah Pembelian', fontsize=14, fontweight='bold')
    ax.set_xlabel('Purchase Amount (USD)')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)
    plt.close()
    
    # Data Cleaning
    st.markdown("---")
    st.subheader("🧹 Tahap 2: Pembersihan Data")
    
    if 'Customer ID' in df.columns:
        df = df.drop('Customer ID', axis=1)
        st.success("✅ Kolom 'Customer ID' dihapus")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        st.success(f"✅ {duplicates} baris duplikat dihapus")
    
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('__', '_').str.lower()
    
    # Feature Engineering
    st.markdown("---")
    st.subheader("⚙️ Tahap 3: Rekayasa Fitur")
    
    df['subscription_status'] = df['subscription_status'].map({'Yes': 1, 'No': 0})
    df['discount_applied'] = df['discount_applied'].map({'Yes': 1, 'No': 0})
    df['promo_code_used'] = df['promo_code_used'].map({'Yes': 1, 'No': 0})
    
    st.success("✅ Variabel Yes/No diubah ke 1/0")
    
    # Korelasi
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        st.markdown("### 🔗 Matriks Korelasi")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Korelasi Variabel Numerik', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    # One-Hot Encoding
    cols_to_onehot = ['gender', 'item_purchased', 'category', 'location', 'size', 'color', 'season', 'payment_method', 'frequency_of_purchases']
    df = pd.get_dummies(df, columns=cols_to_onehot, drop_first=True)
    
    st.success(f"✅ One-Hot Encoding selesai. Total kolom: {df.shape[1]}")
    
    # Split data
    st.markdown("---")
    st.subheader("✂️ Tahap 4: Pembagian Data")
    
    X = df.drop('purchase_amount_usd', axis=1)
    y = df['purchase_amount_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Set", f"{len(X_train):,} (80%)")
    with col2:
        st.metric("Testing Set", f"{len(X_test):,} (20%)")
    
    # Encoding shipping_type
    shipping_type_mapping = {cat: idx for idx, cat in enumerate(X_train['shipping_type'].unique())}
    X_train['shipping_type'] = X_train['shipping_type'].map(shipping_type_mapping)
    X_test['shipping_type'] = X_test['shipping_type'].map(shipping_type_mapping)
    
    # Train model
    st.markdown("---")
    st.subheader("🤖 Tahap 5: Training Model")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("⏳ Training model...")
    progress_bar.progress(30)
    
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
    
    xgb_model.fit(X_train, y_train)
    
    status_text.text("✅ Model selesai!")
    progress_bar.progress(100)
    
    # Evaluation
    st.markdown("---")
    st.subheader("📊 Tahap 6: Evaluasi Model")
    
    y_pred = xgb_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"${mae:.2f}")
    col2.metric("RMSE", f"${rmse:.2f}")
    col3.metric("R² Score", f"{r2:.4f}")
    col4.metric("Akurasi", f"{r2*100:.1f}%")
    
    # Visualisasi Prediksi vs Aktual
    st.markdown("### 🎯 Prediksi vs Nilai Aktual")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Nilai Aktual (USD)', fontsize=12)
    ax.set_ylabel('Prediksi (USD)', fontsize=12)
    ax.set_title('Prediksi vs Aktual', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Residuals
    st.markdown("### 📉 Analisis Residuals")
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(residuals, kde=True, bins=30, ax=ax, color='coral')
        ax.set_title('Distribusi Residuals', fontsize=12)
        ax.set_xlabel('Residuals')
        ax.axvline(x=0, color='red', linestyle='--', lw=2)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        ax.set_xlabel('Prediksi (USD)', fontsize=11)
        ax.set_ylabel('Residuals', fontsize=11)
        ax.set_title('Residuals vs Prediksi', fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Feature Importance
    st.markdown("---")
    st.subheader("🔍 Tahap 7: Feature Importance")
    
    feature_importances = xgb_model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=features_df.head(15), palette='viridis', ax=ax)
    ax.set_title('Top 15 Fitur Terpenting', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    st.pyplot(fig)
    plt.close()
    
    st.success(f"""
    ### ✅ Training Selesai!
    - Model berhasil dilatih dengan R² Score: **{r2:.4f}** ({r2*100:.1f}%)
    - Rata-rata error: **${mae:.2f}**
    - Fitur terpenting: **{features_df.iloc[0]['Feature']}**
    """)
    
    return xgb_model, shipping_type_mapping, X_train.columns.tolist(), {
        'MAE': mae, 'RMSE': rmse, 'R2': r2
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
st.title("🛒 Analisis XGBoost: Prediksi Pembelian")
st.markdown("---")

tab1, tab2 = st.tabs(["📚 Training Model", "🎯 Prediksi"])

with tab1:
    st.header("📚 Training Model XGBoost")
    
    model, shipping_mapping, feature_columns, model_exists = get_model()
    
    if model_exists:
        st.success("✅ Model sudah tersedia!")
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded: {len(df)} baris")
        
        with st.expander("Lihat Data"):
            st.dataframe(df.head())
        
        if st.button("🚀 Mulai Training", type="primary"):
            try:
                model, shipping_mapping, feature_columns, metrics = train_model_with_viz(df)
                
                st.session_state.model = model
                st.session_state.shipping_mapping = shipping_mapping
                st.session_state.feature_columns = feature_columns
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.header("🎯 Prediksi")
    
    if 'model' in st.session_state:
        model = st.session_state.model
        shipping_mapping = st.session_state.shipping_mapping
        feature_columns = st.session_state.feature_columns
        model_exists = True
    else:
        model, shipping_mapping, feature_columns, model_exists = get_model()
    
    if not model_exists:
        st.warning("⚠️ Train model dulu!")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Usia", 18, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            location = st.selectbox("Lokasi", ["California", "New York", "Texas"])
        
        with col2:
            item_purchased = st.selectbox("Item", ["Blouse", "Dress", "Jacket", "Jeans"])
            category = st.selectbox("Kategori", ["Clothing", "Footwear"])
            size = st.selectbox("Size", ["S", "M", "L", "XL"])
            color = st.selectbox("Warna", ["Black", "White", "Blue"])
            season = st.selectbox("Musim", ["Fall", "Winter", "Spring", "Summer"])
        
        with col3:
            previous_purchases = st.number_input("Pembelian Sebelumnya", 0, 50, 10)
            review_rating = st.slider("Rating", 1.0, 5.0, 4.0)
            payment_method = st.selectbox("Pembayaran", ["Credit Card", "Cash"])
            shipping_type = st.selectbox("Pengiriman", ["Express", "Standard"])
            frequency_of_purchases = st.selectbox("Frekuensi", ["Weekly", "Monthly"])
            subscription_status = st.radio("Langganan", ["Yes", "No"])
            discount_applied = st.radio("Diskon", ["Yes", "No"])
            promo_code_used = st.radio("Promo", ["Yes", "No"])
        
        if st.button("🔮 Prediksi", type="primary"):
            input_data = {
                'age': age,
                'previous_purchases': previous_purchases,
                'review_rating': review_rating,
                'subscription_status': 1 if subscription_status == "Yes" else 0,
                'discount_applied': 1 if discount_applied == "Yes" else 0,
                'promo_code_used': 1 if promo_code_used == "Yes" else 0,
            }
            
            categorical = {
                'gender': gender, 'item_purchased': item_purchased,
                'category': category, 'location': location,
                'size': size, 'color': color, 'season': season,
                'payment_method': payment_method,
                'frequency_of_purchases': frequency_of_purchases
            }
            
            df_input = pd.DataFrame([input_data])
            for k, v in categorical.items():
                df_input[k] = v
            
            df_encoded = pd.get_dummies(df_input, columns=list(categorical.keys()), drop_first=True)
            
            df_encoded['shipping_type'] = shipping_mapping.get(shipping_type, 0)
            
            for col in feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            
            df_encoded = df_encoded[feature_columns]
            prediction = model.predict(df_encoded)[0]
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Prediksi", f"${prediction:.2f}")
            col2.metric("📊 Level", "High" if prediction > 70 else "Medium" if prediction > 40 else "Low")
            col3.metric("📈 Progress", f"{min(prediction, 100):.0f}%")
            
            st.progress(min(prediction / 100, 1.0))

st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Streamlit & XGBoost")
