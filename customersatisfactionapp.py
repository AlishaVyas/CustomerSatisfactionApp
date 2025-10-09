import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import gdown  # For downloading from Google Drive

# -----------------------------
# Load trained model and scaler from Google Drive
# -----------------------------
@st.cache_resource
def load_model_from_drive():
    model_url = "https://drive.google.com/uc?id=19M22GAduiTM898ofO38t5mFiUrzqzcRb"
    scaler_url = "https://drive.google.com/uc?id=1h07Hy0wt8ffaOthj_I2R2Ju2arghw-Lv"

    model_path = "rf_model.pkl"
    scaler_path = "scaler.pkl"

    # Download files if not already present
    gdown.download(model_url, model_path, quiet=False)
    gdown.download(scaler_url, scaler_path, quiet=False)

    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler

model, scaler = load_model_from_drive()
st.sidebar.info("✅ Model and Scaler loaded successfully from Google Drive.")

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(page_title="Customer Satisfaction Predictor", page_icon="😊", layout="wide")

st.title("🧩 Customer Satisfaction Prediction App")
st.markdown("Predict **Customer Satisfaction Score** based on demographic and feedback details.")

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio("Navigate", ["📊 Dataset Overview", "📈 Visualizations", "🤖 Prediction"])

# -----------------------------
# Upload Dataset Section
# -----------------------------
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"], help="Upload customer_feedback_satisfaction.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.warning("Using default dataset (please upload your own for exploration).")
    df = pd.read_csv("customer_feedback_satisfaction.csv")

# -----------------------------
# Page 1: Dataset Overview
# -----------------------------
if page == "📊 Dataset Overview":
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### 🧾 Basic Information")
    st.write(df.describe())

    st.markdown("### 🔢 Feedback Score Distribution")
    st.bar_chart(df['FeedbackScore'].value_counts())

# -----------------------------
# Page 2: Visualizations
# -----------------------------
elif page == "📈 Visualizations":
    st.subheader("📊 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribution of SatisfactionScore")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['SatisfactionScore'], kde=True, bins=20, color='skyblue', ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("#### SatisfactionScore vs LoyaltyLevel")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='LoyaltyLevel', y='SatisfactionScore', data=df, ax=ax)
        st.pyplot(fig)

    st.markdown("#### Correlation Heatmap")
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Country', 'FeedbackScore', 'LoyaltyLevel'], drop_first=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# -----------------------------
# Page 3: Prediction Interface
# -----------------------------
elif page == "🤖 Prediction":
    st.subheader("🎯 Predict Customer Satisfaction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Income ($)", min_value=10000, max_value=200000, value=50000)
            product_quality = st.slider("Product Quality (1–10)", 1, 10, 7)
            service_quality = st.slider("Service Quality (1–10)", 1, 10, 8)
            purchase_frequency = st.number_input("Purchase Frequency per Year", min_value=1, max_value=100, value=10)

        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            country = st.selectbox("Country", ["USA", "UK", "Canada", "Germany", "France"])
            feedback = st.selectbox("Feedback Score", ["Low", "Medium", "High"])
            loyalty = st.selectbox("Loyalty Level", ["Bronze", "Silver", "Gold"])

        submitted = st.form_submit_button("🔮 Predict Satisfaction Score")

    if submitted:
        # Prepare input
        new_customer = {
            'Age': age,
            'Income': income,
            'ProductQuality': product_quality,
            'ServiceQuality': service_quality,
            'PurchaseFrequency': purchase_frequency,
            'Gender': gender,
            'Country': country,
            'FeedbackScore': feedback,
            'LoyaltyLevel': loyalty
        }

        # Convert to DataFrame
        customer_df = pd.DataFrame([new_customer])

        # One-hot encode and align columns
        df_train = pd.read_csv("customer_feedback_satisfaction.csv")
        X_columns = pd.get_dummies(df_train, columns=['Gender', 'Country', 'FeedbackScore', 'LoyaltyLevel'], drop_first=True)
        X_columns = X_columns.drop(['CustomerID', 'SatisfactionScore'], axis=1)

        customer_encoded = pd.get_dummies(customer_df, columns=['Gender','Country','FeedbackScore','LoyaltyLevel'], drop_first=True)
        customer_encoded = customer_encoded.reindex(columns=X_columns.columns, fill_value=0)

        # Scale numerical columns
        num_cols = ['Age', 'Income', 'ProductQuality', 'ServiceQuality', 'PurchaseFrequency']
        customer_encoded[num_cols] = scaler.transform(customer_encoded[num_cols])

        # Predict
        predicted_score = model.predict(customer_encoded)[0]

        st.success(f"✅ **Predicted Satisfaction Score: {predicted_score:.2f}**")

        # Optional bar chart for feature importance
        st.markdown("#### Feature Importance")
        importances = pd.Series(model.feature_importances_, index=X_columns.columns)
        top_features = importances.sort_values(ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(6, 4))
        top_features.plot(kind='bar', ax=ax)
        plt.title("Top 5 Important Features")
        st.pyplot(fig)
