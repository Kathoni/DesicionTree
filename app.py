import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Title
st.title("📊 Decision Tree Prediction App")
st.write("Upload your dataset to get predictions using the trained Decision Tree model.")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("decision_tree_model.pkl")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # User selects features for prediction
    st.write("### 🛠 Select feature columns to feed into the model")
    feature_cols = st.multiselect("Choose Features:", df.columns.tolist())

    if st.button("Predict"):
        if len(feature_cols) == 0:
            st.warning("Please select at least one feature column.")
        else:
            X = df[feature_cols]

            # Make predictions
            predictions = model.predict(X)

            df["Prediction"] = predictions

            st.write("### ✅ Predictions")
            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
