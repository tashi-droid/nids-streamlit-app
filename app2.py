import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model

st.set_page_config(layout="wide")
st.title("AI-Powered Network Intrusion Detection System (NIDS)")

# --- Load Pretrained Model ---
MODEL_PATH = "cnn_nids_model.h5"
try:
    model = load_model(MODEL_PATH)
    st.success(" Pretrained model loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- Upload File ---
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Clean Data ---
    df.columns = df.columns.str.strip()
    if 'Label' in df.columns:
        df = df[df['Label'].notna()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Encode non-numeric features automatically
    for col in df.select_dtypes(include='object').columns:
        if col != 'Label':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    st.subheader("Uploaded & Cleaned Data Preview")
    st.write(df.head())

    # --- Download Cleaned CSV ---
    cleaned_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("️ Download Cleaned CSV", data=cleaned_csv, file_name="cleaned_data.csv", mime="text/csv")

    # --- Column Selection ---
    st.subheader("️ Feature & Target Configuration")
    all_columns = df.columns.tolist()

    with st.expander("Adjust selected columns"):
        feature_cols = st.multiselect("Select feature columns", all_columns, default=all_columns[:-1])
        target_col = st.selectbox("Select target column (optional, for evaluation)", ["None"] + all_columns)

    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column.")
        st.stop()

    X = df[feature_cols]

    # --- Quick Prediction Summary ---
    st.subheader("Prediction Summary")
    try:
        y_pred = np.argmax(model.predict(X), axis=1)

        if target_col != "None":
            encoder = LabelEncoder()
            y = df[target_col]
            y_encoded = encoder.fit_transform(y)
            pred_labels = encoder.inverse_transform(y_pred)
            label_counts = pd.Series(pred_labels).value_counts()
        else:
            label_counts = pd.Series(y_pred).value_counts()

        benign_count = 0
        threat_count = 0

        for label, count in label_counts.items():
            label_str = str(label).lower()
            if "benign" in label_str or "normal" in label_str:
                benign_count += count
            else:
                threat_count += count

        st.success(f"Prediction Summary: Benign = {benign_count}, Threats = {threat_count}")

        # --- Optional Evaluation ---
        if target_col != "None":
            report = classification_report(y_encoded, y_pred, target_names=encoder.classes_, output_dict=True)
            matrix = confusion_matrix(y_encoded, y_pred)

            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(matrix, annot=True, fmt='d', xticklabels=encoder.classes_,
                        yticklabels=encoder.classes_, cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(fig)

            st.subheader("Target Label Distribution")
            st.bar_chart(y.value_counts())

        # --- Real-time Manual Prediction ---
        st.sidebar.subheader("Manual Intrusion Prediction")
        user_input = {}
        for col in feature_cols:
            dtype = X[col].dtype
            if np.issubdtype(dtype, np.number):
                mean_val = df[col].mean()
                if pd.isna(mean_val) or not np.isfinite(mean_val):
                    mean_val = 0.0
                user_input[col] = st.sidebar.number_input(col, value=float(mean_val))
            else:
                options = df[col].dropna().unique().tolist()
                if not options:
                    options = ["Unknown"]
                user_input[col] = st.sidebar.selectbox(col, options)

        if st.sidebar.button("🔍 Predict Intrusion"):
            input_df = pd.DataFrame([user_input])
            try:
                pred = np.argmax(model.predict(input_df), axis=1)[0]
                if target_col != "None":
                    pred_label = encoder.inverse_transform([pred])[0]
                    st.sidebar.success(f"Prediction: {pred_label}")
                else:
                    st.sidebar.success(f"Prediction class index: {pred}")
            except Exception as e:
                st.sidebar.error(f"Prediction error: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
