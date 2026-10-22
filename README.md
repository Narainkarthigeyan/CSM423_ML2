# 🛡️ PhishGuard — Phishing URL Detection

PhishGuard is a machine learning-based application that analyzes URLs and classifies them as **potentially phishing** or **potentially legitimate**. It uses URL-based feature engineering and supervised learning to provide preliminary risk assessments through an interactive Streamlit dashboard.

## ✨ Features
- URL analysis and phishing risk assessment
- Extraction of 17 URL-based features
- Logistic Regression and Linear SVM model comparison
- Cross-validation and hyperparameter tuning
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Bulk URL scanning using CSV uploads
- Scan history and downloadable reports
- Phishing awareness and URL safety guidance

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib

## 🚀 Run Locally

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd PhishGuardAI
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
   ```

3. Ensure `app.py`, `phishguard_model.joblib`, and `phishguard_features.json` are in the project directory.

4. Launch the application:
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset
The project uses the [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) from the UCI Machine Learning Repository. The model uses handcrafted features extracted from URL text so the same features can be generated during prediction.

## ⚠️ Disclaimer
PhishGuard AI is an academic project and provides preliminary URL assessments only. It does not inspect live webpage content or verify domain reputation. Predictions may be incorrect, and a URL classified as legitimate is not guaranteed to be safe.

## 🎓 Project
Developed as an individual Machine Learning CA-1 project to demonstrate feature engineering, supervised learning, model evaluation, cross-validation, and hyperparameter tuning.

