import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_text
import nltk
nltk.download("punkt")
nltk.download("stopwords")

# Load model and vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = load_model("model/cnn_email_classifier.h5")

label_map = {
    0: "Ham ✅",
    1: "Spam ⚠️",
    2: "Phishing 🚨"
}

explanation = {
    0: "This message appears to be a regular, non-malicious email...",
    1: "The email displays typical spam indicators...",
    2: "Phishing traits detected. This message attempts to impersonate a trusted service..."
}

st.set_page_config(page_title="AI-Powered Email Classifier", page_icon="📧")
st.title("📧 AI-Powered Email Classifier")
email_input = st.text_area("Paste your email content here:", height=250)

if st.button("Classify"):
    if not email_input.strip():
        st.warning("Please enter some email content.")
    else:
        clean_text = preprocess_text(email_input)
        vec = vectorizer.transform([clean_text])
        cnn_input = vec.toarray().reshape(1, vec.shape[1], 1)

        probs = model.predict(cnn_input)[0]
        predicted_label = np.argmax(probs)

        st.subheader("📊 Prediction Probabilities:")
        for i, p in enumerate(probs):
            st.write(f"{label_map[i]}: {p*100:.2f}%")

        st.success(f"✅ **Final Prediction:** {label_map[predicted_label]}")
        st.info(f"💡 **Why?** {explanation[predicted_label]}")
