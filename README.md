# 📊 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange)
![Joblib](https://img.shields.io/badge/Joblib-1.3.2-yellow)

A complete end-to-end Machine Learning pipeline that predicts whether a telecommunications customer is at a high risk of churning or discontinuing their service. Through an interactive web dashboard, users can input customer parameters in real-time and instantly derive an AI-driven churn probability.

🚀 **[Live Interactive App (Streamlit)](https://churn-prediction-01.streamlit.app/)**

---

## 📸 Dashboard Preview

Users input demographic profiles, subscription history, billing preferences, and tenure. Our custom pipeline rapidly parses the categorical features into machine-ready scaled matrices and classifies the risk levels as:
* 🔴 **HIGH RISK:** Probability >= 70%
* 🟡 **MEDIUM RISK:** Probability >= 40%
* 🟢 **LOW RISK:** Probability < 40%

---

## 🧠 Machine Learning Details

The predictive core has been trained on a cleaned subset of the Telecom Customer Churn dataset. During training, we explored several advanced algorithms:
1. **Logistic Regression (Main Deployment Model)** - Delivered the highest evaluation accuracy (~79%) with robust probability calibration.
2. RandomForestClassifier
3. DecisionTree
4. K-Nearest Neighbors

To guarantee deployment accuracy, categorical mapping (`pd.get_dummies()`) and numerical scaling (`StandardScaler`) processes were saved locally into serialized `.pkl` pipelines, ensuring the web inputs map perfectly into the exact multi-dimensional space the Machine Learning model was originally trained inside.

---

## 🛠️ How to run locally

### 1. Clone the repository
```bash
git clone https://github.com/Durjoy-06/end-to-end-customer-churn-prediction-system.git
cd end-to-end-customer-churn-prediction-system
```

### 2. Download Dependencies
Ensure you have Python installed, then install the required Python packages mathematically matching the model build:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
Deploy the interactive server securely on your local port:
```bash
streamlit run app.py
```
Open a browser and navigate to `http://localhost:8501`.

---

## 📁 Repository Structure
* **`app.py`** — The interactive Streamlit Web Interface logic handling UI layout and prediction generation.
* **`retrain.py`** — The comprehensive script to regenerate data preprocessing limits, extract insights, and fully train every ML Pipeline from scratch.
* **`models/`** — Serialized binaries storing ML `.pkl` classifiers and the `StandardScaler` ensuring immediate access to knowledge graphs without continuous retraining delay. 
* **`notebooks/`** — Detailed Jupyter environments tracing full Exploratory Data Analysis (EDA) statistics and model experimentation.
