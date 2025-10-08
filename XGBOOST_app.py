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
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def train_model_with_viz(df):
    st.subheader("📊 Eksplorasi Data")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Kolom", len(df.columns))
    col3.metric("Missing", df.isnull().sum().sum())
    
    with st.expander("Preview Data"):
        st.dataframe(df.head())
    
    # Visualisasi distribusi
    if MATPLOTLIB_OK:
        st.markdown("### 📊 Distribusi Purchase Amount")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Purchase Amount (USD)'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Purchase Amount (USD)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Purchase Amount')
        st.pyplot(fig)
        plt.close()
    
    # Data Cleaning
    st.markdown("---")
    st.subheader("🧹 Data Cleaning")
    
    if 'Customer ID' in df.columns:
        df = df.drop('Customer ID', axis=1)
        st.success("✅ Customer ID removed")
    
    dup = df.duplicated().sum()
    if dup > 0:
        df.drop_duplicates(inplace=True)
        st.success(f"✅ {dup} duplicates removed")
    
    df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('__', '_').str.lower()
    
    # Feature Engineering
    st.markdown("---")
    st.subheader("⚙️ Feature Engineering")
    
    df['subscription_status'] = df['subscription_status'].map({'Yes': 1, 'No': 0})
    df['discount_applied'] = df['discount_applied'].map({'Yes': 1, 'No': 0})
    df['promo_code_used'] = df['promo_code_used'].map({'Yes': 1, 'No': 0})
    
    st.success("✅ Binary encoding done")
    
    # Correlation
    if MATPLOTLIB_OK:
        st.markdown("### 🔗 Correlation Matrix")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            plt.close()
    
    # One-Hot Encoding
    cols_to_encode = ['gender', 'item_purchased', 'category', 'location', 'size', 
                      'color', 'season', 'payment_method', 'frequency_of_purchases']
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    
    st.success(f"✅ One-Hot Encoding: {df.shape[1]} columns")
    
    # Split
    st.markdown("---")
    st.subheader("✂️ Train-Test Split")
    
    X = df.drop('purchase_amount_usd', axis=1)
    y = df['purchase_amount_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    col1.metric("Training", f"{len(X_train):,}")
    col2.metric("Testing", f"{len(X_test):,}")
    
    # Encode shipping
    ship_map = {cat: idx for idx, cat in enumerate(X_train['shipping_type'].unique())}
    X_train['shipping_type'] = X_train['shipping_type'].map(ship_map)
    X_test['shipping_type'] = X_test['shipping_type'].map(ship_map)
    
    # Train
    st.markdown("---")
    st.subheader("🤖 Training Model")
    
    progress = st.progress(0)
    status = st.empty()
    
    status.text("Training...")
    progress.progress(50)
    
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
    
    status.text("✅ Done!")
    progress.progress(100)
    
    # Evaluate
    st.markdown("---")
    st.subheader("📊 Evaluation")
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mae:.2f}")
    col2.metric("RMSE", f"${rmse:.2f}")
    col3.metric("R²", f"{r2:.4f}")
    
    # Visualizations
    if MATPLOTLIB_OK:
        st.markdown("### 🎯 Predictions vs Actual")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter
        ax1.scatter(y_test, y_pred, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Predictions vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        ax2.hist(residuals, bins=30, color='coral', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Feature Importance
        st.markdown("### 🔍 Feature Importance")
        
        feat_imp = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feat_imp)), feat_imp['Importance'])
        ax.set_yticks(range(len(feat_imp)))
        ax.set_yticklabels(feat_imp['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Important Features')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.success(f"✅ Model trained! R² = {r2:.4f}")
    
    return model, ship_map, X_train.columns.tolist(), {'MAE': mae, 'RMSE': rmse, 'R2': r2}

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
st.title("🛒 Analisis XGBoost - Prediksi Pembelian")

if not SKLEARN_OK:
    st.error("❌ Required libraries not available!")
    st.stop()

if MATPLOTLIB_OK:
    st.success("✅ Visualizations enabled")
else:
    st.warning("⚠️ Matplotlib unavailable - limited visualization")

st.markdown("---")

tab1, tab2 = st.tabs(["📚 Training", "🎯 Prediksi"])

with tab1:
    st.header("Training Model")
    
    model, ship_map, feat_cols, exists = get_model()
    
    if exists:
        st.success("✅ Model available!")
    
    file = st.file_uploader("Upload CSV", type=['csv'])
    
    if file:
        df = pd.read_csv(file)
        st.success(f"✅ {len(df)} rows loaded")
        
        if st.button("🚀 Train Model", type="primary"):
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
        st.warning("⚠️ Train model first")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Demographics")
            age = st.slider("Age", 18, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            location = st.selectbox("Location", ["California", "New York", "Texas"])
        
        with col2:
            st.subheader("🛍️ Product")
            item = st.selectbox("Item", ["Blouse", "Dress", "Jacket", "Jeans"])
            category = st.selectbox("Category", ["Clothing", "Footwear", "Accessories", "Outerwear"])
            size = st.selectbox("Size", ["S", "M", "L", "XL"])
            color = st.selectbox("Color", ["Black", "White", "Blue", "Red"])
            season = st.selectbox("Season", ["Fall", "Winter", "Spring", "Summer"])
        
        with col3:
            st.subheader("💳 Transaction")
            prev = st.number_input("Previous Purchases", 0, 50, 10)
            rating = st.slider("Rating", 1.0, 5.0, 4.0)
            payment = st.selectbox("Payment", ["Credit Card", "Cash", "Debit Card"])
            shipping = st.selectbox("Shipping", ["Express", "Standard", "Free Shipping"])
            freq = st.selectbox("Frequency", ["Weekly", "Monthly", "Quarterly"])
            subs = st.radio("Subscription", ["Yes", "No"])
            disc = st.radio("Discount", ["Yes", "No"])
            promo = st.radio("Promo", ["Yes", "No"])
        
        if st.button("🔮 Predict", type="primary"):
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
            
            st.markdown("---")
            st.markdown("## 💰 Result")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("💵 Prediction", f"${pred:.2f}")
            col2.metric("📊 Level", "High" if pred > 70 else "Medium" if pred > 40 else "Low")
            col3.metric("⭐ Score", f"{min(pred, 100):.0f}/100")
            
            # Fix progress bar - ensure value is between 0 and 1
            progress_value = max(0.0, min(pred / 100.0, 1.0))
            st.progress(progress_value)
            
            st.markdown("### 💡 Insights")
            if pred > 70:
                st.success("🟢 High-value customer - Premium offers recommended")
            elif pred > 40:
                st.info("🟡 Medium-value - Cross-sell opportunities")
            else:
                st.warning("🔴 Low-value - Promotional campaigns needed")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & XGBoost")
