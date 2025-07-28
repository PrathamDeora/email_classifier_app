# 📧 AI-Powered Email Classifier

A Streamlit web app that classifies emails as **Ham**, **Spam**, or **Phishing** using a Convolutional Neural Network (CNN) and TF-IDF vectorization.

## 🚀 Features
- NLP-based email text classification
- Explains why the email was labeled a certain way
- Streamlit interface with real-time prediction

## 📁 Project Structure
```
email_classifier_app/
├── app.py
├── model/
│   ├── cnn_email_classifier.h5
│   └── tfidf_vectorizer.pkl
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## 📦 Setup Instructions

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/email_classifier_app.git
   cd email_classifier_app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🤖 Model Details
- **Model:** CNN trained on labeled email dataset
- **Vectorizer:** TF-IDF
- **Classes:** Ham, Spam, Phishing

