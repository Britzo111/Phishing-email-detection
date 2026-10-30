import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI-Based Phishing Email Detection",
    page_icon="📧",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
**Project:**
AI-Based Phishing Email Detection Tool Using Machine Learning and NLP

**Model Used:**
- Random Forest

**Feature Extraction:**
- TF-IDF Vectorizer

**Developer:**
Britto Benny
""")

# -----------------------------
# Main Title
# -----------------------------
st.title("📧 AI-Based Phishing Email Detection")

st.markdown("""
Welcome to the **AI-Based Phishing Email Detection Tool**.

This application analyses the content of an email using **Natural Language Processing (NLP)** and a **Random Forest Machine Learning model** to determine whether an email is **Legitimate (Ham)** or **Phishing**.
""")

st.divider()

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_tuned.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
# -----------------------------
# SHAP Explainability
# -----------------------------

explainer = shap.TreeExplainer(model)

# -----------------------------
# Input
# -----------------------------
st.subheader("✉️ Email Details")

subject = st.text_input("Email Subject")

body = st.text_area(
    "Email Body",
    height=250,
    placeholder="Paste the email content here..."
)

if st.button("🔍 Analyse Email"):

    if subject.strip() == "" and body.strip() == "":
        st.warning("Please enter an email.")

    else:
        # Combine subject and body
        email = subject + " " + body

        # -----------------------------
        # Email Statistics
        # -----------------------------

        import re

        word_count = len(email.split())
        character_count = len(email)

        urls = re.findall(
            r'https?://\S+|www\.\S+',
            email
        )

        email_addresses = re.findall(
            r'[\w\.-]+@[\w\.-]+\.\w+',
            email
        )

        # -----------------------------
        # Suspicious Keywords
        # -----------------------------

        suspicious_keywords = [
            "urgent",
            "verify",
            "verification",
            "password",
            "account suspended",
            "click here",
            "confirm your account",
            "limited time",
            "winner",
            "prize",
            "bank",
            "payment",
            "login",
            "security alert",
            "act now"
        ]

        email_lower = email.lower()

        detected_keywords = [
            keyword
            for keyword in suspicious_keywords
            if keyword in email_lower
        ]

        # -----------------------------
        # Machine Learning Prediction
        # -----------------------------

        email_vector = vectorizer.transform([email])

        prediction = model.predict(email_vector)[0]
        probability = model.predict_proba(email_vector)[0]
        
        

        # -----------------------------
        # Results
        # -----------------------------

        st.divider()

        st.subheader("📊 Email Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Words", word_count)

        with col2:
            st.metric("Characters", character_count)

        with col3:
            st.metric("URLs", len(urls))

        with col4:
            st.metric("Email Addresses", len(email_addresses))

        st.divider()

        # -----------------------------
        # ML Prediction
        # -----------------------------

        st.subheader("🤖 Machine Learning Prediction")

        if prediction == 1:

            confidence = probability[1]

            st.error("🚨 PHISHING EMAIL DETECTED")

            st.metric(
                "Phishing Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(float(confidence))

        else:

            confidence = probability[0]

            st.success("✅ LEGITIMATE EMAIL")

            st.metric(
                "Legitimate Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(float(confidence))

        # -----------------------------
        # Suspicious Indicators
        # -----------------------------

        st.divider()

        st.subheader("⚠️ Suspicious Indicators")

        if len(urls) > 0:
            st.warning(
                f"🔗 {len(urls)} URL(s) detected in the email."
            )
        else:
            st.success("✅ No URLs detected.")

        if detected_keywords:

            st.warning(
                "⚠️ Suspicious keywords detected:"
            )

            st.write(
                ", ".join(detected_keywords)
            )

        else:

            st.success(
                "✅ No common suspicious keywords detected."
            )

        # -----------------------------
        # Risk Level
        # -----------------------------

        st.divider()

        st.subheader("🛡️ Risk Assessment")

        if prediction == 1 and confidence >= 0.80:

            st.error("🔴 HIGH RISK")

            st.write(
                "This email has been classified as highly likely "
                "to be phishing. Avoid clicking links or providing "
                "personal information."
            )

        elif prediction == 1:

            st.warning("🟠 MEDIUM RISK")

            st.write(
                "The email shows characteristics associated with "
                "phishing. Verify the sender before taking action."
            )

        else:

            st.success("🟢 LOW RISK")

            st.write(
                "The machine learning model classified this email "
                "as legitimate. However, always verify unexpected "
                "emails before taking action."
            )


# -----------------------------
# Model Explainability
# -----------------------------

st.divider()

st.subheader("🧠 Model Explainability")

st.write(
    "These are the highest-importance TF-IDF features "
    "present in the analysed email according to the "
    "Random Forest model."
)

# Get TF-IDF feature names
feature_names = vectorizer.get_feature_names_out()

# Convert email TF-IDF vector to dense format
email_dense = email_vector.toarray()[0]

# Get Random Forest feature importance
importances = model.feature_importances_

# Find features that are present in this email
active_features = []

for i, value in enumerate(email_dense):
    if value > 0:
        active_features.append(
            (
                feature_names[i],
                importances[i],
                value
            )
        )

# Sort by Random Forest importance
active_features.sort(
    key=lambda x: x[1],
    reverse=True
)

# Display top 10
if active_features:

    st.write("### Top TF-IDF Features")

    # Take top 10 features
    top_features = active_features[:10]

    # Create DataFrame
    feature_df = pd.DataFrame(
        top_features,
        columns=["Feature", "Importance", "TF-IDF"]
    )

    # Display chart
    st.bar_chart(
        feature_df.set_index("Feature")["Importance"],
        horizontal=True
    )

    # Display feature details
    st.write("### Feature Details")

    for _, row in feature_df.iterrows():

        st.write(
            f"**{row['Feature']}** — "
            f"importance: {row['Importance']:.6f}"
        )

else:

    st.info(
        "No significant TF-IDF features were found "
        "for this email."
    )