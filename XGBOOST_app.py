import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Import sklearn components
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    st.error(f"Sklearn import error: {e}")

# Import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    st.error(f"XGBoost import error: {e}")

# Import matplotlib with proper backend
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib tidak tersedia. Visualisasi akan dibatasi.")

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis XGBoost",
    page_icon="🛒",
    layout="wide"
)

# Fungsi train model
def train_model_with_viz(df):
    """Train XGBoost model dengan visualisasi"""
    
    if not SKLEARN_AVAILABLE or not XGBOOST_AVAILABLE:
        st.error("Library yang diperlukan tidak tersedia!")
        return None, None, None, None
    
    st.subheader("📊 Eksplorasi Data")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Kolom", f"{len(df.columns)}")
    col3.metric("Missing", f"{df.isnull().sum().sum()}")
    
    # Visualisasi distribusi
    if MATPLOTLIB_AVAILABLE:
        st.markdown("### 📊 Distribusi Purchase Amount")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Purchase Amount (USD)'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Purchase Amount (USD)')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Distribusi Pembelian')
        st.pyplot(fig)
        plt.close()
    
    # Data Cleaning
    st.markdown("---")
    st.subheader("🧹 Pembersihan Data")
    
    if 'Customer ID' in df.columns:
        df = df.drop('Customer ID', axis=1)
        st.success("✅ Customer ID dihapus")
    
    dup = df.duplicated().sum()
    if dup > 0:
        df.drop_duplicates(inplace=True)
        st.success(f"✅ {dup} duplikat dihapus")
    
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('__', '_').str.lower()
    
    # Feature Engineering
    st.markdown("---")
    st.subheader("⚙️ Feature Engineering")
    
    df['subscription_status'] = df['subscription_status'].map({'Yes': 1, 'No': 0})
    df['discount_applied'] = df['discount_applied'].map({'Yes': 1, 'No': 0})
    df['promo_code_used'] = df['promo_code_used'].map({'Yes': 1, 'No': 0})
    
    st.success("✅ Binary encoding selesai")
    
    # Korelasi
    if MATPLOTLIB_AVAILABLE:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            st.markdown("### 🔗 Matriks Korelasi")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            im = ax.imshow(corr, cmap='coolwarm', aspect='auto')
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax.set_yticklabels(numeric_cols)
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
    
    # One-Hot Encoding
    cols_to_onehot = ['gender', 'item_purchased', 'category', 'location', 'size', 'color', 'season', 'payment_method', 'frequency_of_purchases']
    df = pd.get_dummies(df, columns=cols_to_onehot, drop_first=True)
    
    st.success(f"✅ One-Hot Encoding: {df.shape[1]} kolom")
    
    # Split
    st.markdown("---")
    st.subheader("✂️ Split Data")
    
    X = df.drop('purchase_amount_usd', axis=1)
    y = df['purchase_amount_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    col1.metric("Training", f"{len(X_train):,}")
    col2.metric("Testing", f"{len(X_test):,}")
    
    # Encode shipping_type
    ship_map = {cat: idx for idx, cat in enumerate(X_train['shipping_type'].unique())}
    X_train['shipping_type'] = X_train['shipping_type'].map(ship_map)
    X_test['shipping_type'] = X_test['shipping_type'].map(ship_map)
    
    # Train
    st.markdown("---")
    st.subheader("🤖 Training Model")
    
    progress = st.progress(0)
    status = st.empty()
    
    status.text("Training...")
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
    
    model.fit(X_train, y_train)
    
    status.text("✅ Selesai!")
    progress.progress(100)
    
    # Evaluate
    st.markdown("---")
    st.subheader("📊 Evaluasi")
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mae:.2f}")
    col2.metric("RMSE", f"${rmse:.2f}")
    col3.metric("R²", f"{r2:.4f} ({r2*100:.1f}%)")
    
    # Visualisasi
    if MATPLOTLIB_AVAILABLE:
        st.markdown("### 🎯 Prediksi vs Aktual")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Aktual')
        ax.set_ylabel('Prediksi')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Residuals
        st.markdown("### 📉 Residuals")
        residuals = y_test - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(residuals, bins=30, color='coral', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', lw=2)
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_pred, residuals, alpha=0.5)
            ax.axhline(y=0, color='red', linestyle='--', lw=2)
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Residuals')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Feature Importance
        st.markdown("### 🔍 Feature Importance")
        
        feat_imp = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        top_n = min(15, len(feat_imp))
        ax.barh(range(top_n), feat_imp['Importance'].head(top_n))
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_imp['Feature'].head(top_n))
        ax.set_xlabel('Importance')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
    
    st.success(f"✅ Model siap! R² = {r2:.4f}")
    
    return model, ship_map, X_train.columns.tolist(), {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Load model
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
st.title("🛒 Analisis XGBoost")
st.markdown("---")

tab1, tab2 = st.tabs(["📚 Training", "🎯 Prediksi"])

with tab1:
    st.header("Training Model")
    
    model, ship_map, feat_cols, exists = get_model()
    
    if exists:
        st.success("✅ Model tersedia")
    
    file = st.file_uploader("Upload CSV", type=['csv'])
    
    if file:
        df = pd.read_csv(file)
        st.success(f"✅ {len(df)} baris")
        
        if st.button("🚀 Train", type="primary"):
            try:
                model, ship_map, feat_cols, metrics = train_model_with_viz(df)
                
                st.session_state.model = model
                st.session_state.ship_map = ship_map
                st.session_state.feat_cols = feat_cols
                
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("Prediksi")
    
    if 'model' in st.session_state:
        model = st.session_state.model
        ship_map = st.session_state.ship_map
        feat_cols = st.session_state.feat_cols
        exists = True
    else:
        model, ship_map, feat_cols, exists = get_model()
    
    if not exists:
        st.warning("⚠️ Train model dulu")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Usia", 18, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            location = st.selectbox("Lokasi", ["California", "New York", "Texas"])
        
        with col2:
            item = st.selectbox("Item", ["Blouse", "Dress", "Jacket", "Jeans"])
            category = st.selectbox("Kategori", ["Clothing", "Footwear"])
            size = st.selectbox("Size", ["S", "M", "L", "XL"])
            color = st.selectbox("Warna", ["Black", "White", "Blue"])
            season = st.selectbox("Musim", ["Fall", "Winter", "Spring", "Summer"])
        
        with col3:
            prev = st.number_input("Pembelian Sebelumnya", 0, 50, 10)
            rating = st.slider("Rating", 1.0, 5.0, 4.0)
            payment = st.selectbox("Pembayaran", ["Credit Card", "Cash"])
            shipping = st.selectbox("Pengiriman", ["Express", "Standard"])
            freq = st.selectbox("Frekuensi", ["Weekly", "Monthly"])
            subs = st.radio("Langganan", ["Yes", "No"])
            disc = st.radio("Diskon", ["Yes", "No"])
            promo = st.radio("Promo", ["Yes", "No"])
        
        if st.button("🔮 Prediksi", type="primary"):
            data = {
                'age': age, 'previous_purchases': prev, 'review_rating': rating,
                'subscription_status': 1 if subs == "Yes" else 0,
                'discount_applied': 1 if disc == "Yes" else 0,
                'promo_code_used': 1 if promo == "Yes" else 0,
            }
            
            cat_feat = {
                'gender': gender, 'item_purchased': item,
                'category': category, 'location': location,
                'size': size, 'color': color, 'season': season,
                'payment_method': payment, 'frequency_of_purchases': freq
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
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Prediksi", f"${pred:.2f}")
            col2.metric("Level", "High" if pred > 70 else "Medium" if pred > 40 else "Low")
            col3.metric("Score", f"{min(pred, 100):.0f}/100")
            
            st.progress(min(pred/100, 1.0))

st.markdown("---")
st.caption("Dibuat dengan ❤️")
