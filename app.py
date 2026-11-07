
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------
# 2. LOAD TRAINED MODEL
# ---------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

try:
    model, features = load_model()
except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()

# ---------------------------------------
# 3. DASHBOARD HEADER
# ---------------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "An intelligent machine learning system "
    "to identify potentially fraudulent credit card transactions."
)

st.divider()

# ---------------------------------------
# 4. SINGLE DATASET UPLOADER
# ---------------------------------------

st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file to analyse",
    type=["csv"],
    key="transaction_upload"
)

# ---------------------------------------
# 5. EMPTY DASHBOARD
# ---------------------------------------

if uploaded_file is None:

    st.info(
        "Please upload a transaction dataset to start fraud detection."
    )

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", "—")
    col2.metric("Normal Transactions", "—")
    col3.metric("Fraud Transactions", "—")
    col4.metric("Fraud Percentage", "—")

    st.divider()

    st.subheader("📈 Transaction Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Normal vs Fraud Transactions**")
        st.caption("Chart will appear after uploading a dataset.")

    with col2:
        st.markdown("**Transaction Amount Distribution**")
        st.caption("Chart will appear after uploading a dataset.")

    st.divider()

    st.subheader("🎯 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "—")
    col2.metric("Precision", "—")
    col3.metric("Recall", "—")
    col4.metric("F1 Score", "—")

    st.divider()

    st.subheader("🔍 Fraud Detection")
    st.write("Prediction results will appear after dataset upload.")

    st.stop()

# ---------------------------------------
# 6. READ UPLOADED DATASET
# ---------------------------------------

try:
    data = pd.read_csv(uploaded_file)

    if data.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    st.success(
        f"Dataset uploaded successfully! "
        f"{len(data):,} transactions found."
    )

except Exception as e:
    st.error(f"Unable to read CSV file: {e}")
    st.stop()

# ---------------------------------------
# 7. VALIDATE DATASET
# ---------------------------------------

missing_features = [
    col for col in features
    if col not in data.columns
]

if missing_features:
    st.error(
        "The uploaded dataset does not contain all required "
        "model features."
    )
    st.write("Missing columns:", missing_features)
    st.stop()

if "Amount" not in data.columns:
    st.error("Amount column is missing.")
    st.stop()

# Check actual class labels if provided
if "Class" in data.columns:

    if data["Class"].isnull().any() or not data["Class"].isin([0, 1]).all():
        st.error("Class column must contain only 0 and 1.")
        st.stop()

# ---------------------------------------
# 8. PREPARE INPUT DATA
# ---------------------------------------

input_data = data[features].apply(
    pd.to_numeric,
    errors="coerce"
)

if input_data.isnull().any().any():
    st.error(
        "Missing or non-numeric values found in required features."
    )
    st.stop()

# ---------------------------------------
# 9. PREDICT TRANSACTIONS
# ---------------------------------------

try:
    predictions = model.predict(input_data)

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

data["Prediction"] = [
    "Fraud" if p == 1 else "Normal"
    for p in predictions
]

predicted_normal = int((predictions == 0).sum())
predicted_fraud = int((predictions == 1).sum())

# ---------------------------------------
# 10. DATASET OVERVIEW
# ---------------------------------------

st.divider()
st.header("📊 Dataset Overview")

total = len(data)

if "Class" in data.columns:

    normal = int((data["Class"] == 0).sum())
    fraud = int((data["Class"] == 1).sum())

    fraud_percentage = (fraud / total * 100) if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Normal Transactions", f"{normal:,}")
    col3.metric("Fraud Transactions", f"{fraud:,}")
    col4.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")

    st.caption(
        "Dataset overview uses actual Class labels. "
        "Predictions are shown separately below."
    )

else:

    fraud_percentage = (
        predicted_fraud / total * 100 if total else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Predicted Normal", f"{predicted_normal:,}")
    col3.metric("Predicted Fraud", f"{predicted_fraud:,}")
    col4.metric("Predicted Fraud %", f"{fraud_percentage:.2f}%")

    st.info(
        "No Class column found. Counts represent model predictions."
    )

# ---------------------------------------
# 11. TWO COMPACT GRAPHS SIDE BY SIDE
# ---------------------------------------

st.divider()
st.header("📈 Transaction Analysis")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Normal vs Fraud")

    fig, ax = plt.subplots(figsize=(4, 2.7))

    if "Class" in data.columns:

        sns.countplot(
            data=data,
            x="Class",
            order=[0, 1],
            palette="Set2",
            ax=ax
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Fraud"])
        ax.set_title("Actual Transactions")

    else:

        sns.countplot(
            data=data,
            x="Prediction",
            order=["Normal", "Fraud"],
            palette="Set2",
            ax=ax
        )

        ax.set_title("Predicted Transactions")

    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Count")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col2:

    st.subheader("Transaction Amount")

    fig, ax = plt.subplots(figsize=(4, 2.7))

    sns.histplot(
        data=data,
        x="Amount",
        bins=40,
        ax=ax
    )

    ax.set_title("Amount Distribution")
    ax.set_xlabel("Transaction Amount")
    ax.set_ylabel("Frequency")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ---------------------------------------
# 12. MODEL PERFORMANCE
# ---------------------------------------

st.divider()
st.header("🎯 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "99.95%")
col2.metric("Precision", "97.10%")
col3.metric("Recall", "70.53%")
col4.metric("F1 Score", "81.71%")

st.caption(
    "Metrics obtained during the original Random Forest test evaluation."
)

# ---------------------------------------
# 13. FRAUD DETECTION RESULTS
# ---------------------------------------

st.divider()
st.header("🔍 Fraud Detection Results")

col1, col2 = st.columns(2)

col1.metric("Predicted Normal", f"{predicted_normal:,}")
col2.metric("Predicted Fraud", f"{predicted_fraud:,}")

if predicted_fraud > 0:
    st.warning(
        f"{predicted_fraud} transactions were identified "
        "as potentially fraudulent by the model."
    )
else:
    st.success("No transactions were predicted as fraudulent.")

# ---------------------------------------
# 14. DOWNLOAD PREDICTION REPORT
# ---------------------------------------

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Prediction Report",
    data=csv,
    file_name="fraud_prediction_results.csv",
    mime="text/csv"
)

st.success("Fraud detection analysis completed!")
