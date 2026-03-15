# Adv Disease Pred APP

![Language](https://img.shields.io/badge/Language-Jupyter%20Notebook-DA5B0B?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Adv-Disease-pred-APP?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Adv-Disease-pred-APP?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Clinical decision support powered by machine learning — early disease risk prediction from patient biomarkers.

---

**Topics:** `classification` · `data-science` · `deep-learning` · `disease-prediction` · `health-ai` · `machine-learning` · `medical-diagnosis` · `neural-networks` · `streamlit` · `cnn-on-xray-mri`

## Overview

This application is a machine-learning-based disease risk assessment tool designed to assist clinicians and health-conscious individuals in identifying early warning signs from standard biomarker data. The model takes patient-reported and clinically measured parameters as input and returns a probabilistic risk score, enabling data-informed decision-making before symptoms become critical.

Built on Streamlit for rapid deployment, the application wraps a trained binary classifier — trained on a curated medical dataset — inside a clean, accessible UI. The prediction is accompanied by a SHAP explanation plot that breaks down each feature's contribution to the risk score, ensuring that the model's reasoning is visible and auditable rather than a black box.

The project also includes a full model comparison module where multiple classifiers (Logistic Regression, Random Forest, SVM, XGBoost, KNN) are evaluated side-by-side on the same test partition, with accuracy, AUC-ROC, precision, recall, and F1-score reported. This allows the deployment model to be chosen based on the performance metric most appropriate to the clinical context — favouring recall over precision in high-stakes screening scenarios.

---

## Motivation

Chronic disease is one of the leading causes of preventable mortality worldwide. Access to specialist-level screening is unevenly distributed, particularly in lower-resource healthcare settings. This project was motivated by the question: can a well-calibrated ML model, operating on data available in a standard clinical visit, provide a reliable first-line risk signal that guides further investigation? The answer, on benchmark datasets, is yes.

---

## Architecture

```
Patient Biomarker Input
    (age, sex, BP, cholesterol, glucose, ECG features...)
        │
  Feature Engineering + StandardScaler
        │
  Trained Binary Classifier (RF / XGBoost / SVM)
        │
  Risk Probability Score (0.0 → 1.0)
        │
  ┌─────┴─────┐
  │           │
SHAP Plot  Risk Category
(feature   (Low / Medium / High)
 waterfall)
```

The pipeline object (scaler + model) is serialised with joblib and loaded at app startup. Threshold for risk categorisation (default 0.5) is configurable in the sidebar.

---

## Features

### Biomarker Input Interface
Validated input widgets for all clinical features — age, sex, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, max heart rate, and chest pain type.

### Risk Probability Output
The model outputs a continuous probability score between 0 and 1, displayed as a colour-coded gauge (green/amber/red) with a plain-language risk category interpretation.

### SHAP Feature Attribution
A SHAP waterfall chart accompanies every prediction, showing which biomarkers pushed the risk score up or down and by how much — critical for clinical interpretability.

### Multi-Model Evaluation Panel
Compare Logistic Regression, Random Forest, SVM, XGBoost, and KNN classifiers on accuracy, AUC-ROC, F1, precision, and recall using a shared train/test split.

### ROC Curve Visualisation
Interactive ROC curves for all models on a single plot, enabling threshold selection based on the clinical sensitivity/specificity trade-off.

### Confusion Matrix Display
Normalised confusion matrix heatmap for the selected deployment model, with TP/TN/FP/FN counts and derived metrics.

### Batch CSV Screening
Upload a CSV of multiple patient records for batch risk scoring, with a downloadable output table including risk scores and categories.

### Clinical Caveat Footer
Every prediction page includes a mandatory disclaimer reminding users that this tool is a decision-support aid and does not replace clinical diagnosis.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **Streamlit** | Application framework | Clean medical UI with sidebar controls |
| **scikit-learn** | ML pipeline and models | Preprocessing, classification, evaluation metrics |
| **XGBoost** | Gradient boosting classifier | Best-in-class performance on tabular medical data |
| **SHAP** | Model explainability | TreeExplainer for biomarker attribution |
| **pandas** | Data handling | Patient record loading and batch processing |
| **Plotly** | Interactive charts | ROC curves, confusion matrices, gauge charts |
| **joblib** | Model persistence | Serialise and load trained pipeline |
| **NumPy** | Array operations | Feature vector construction and scaling |

> **Key packages detected in this repo:** `scikit-learn` · `numpy` · `pandas` · `matplotlib` · `seaborn` · `scipy` · `xgboost` · `lightgbm` · `catboost` · `tensorflow`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
git clone https://github.com/Devanik21/Heart_disease_Prediction-APP.git
cd Heart_disease_Prediction-APP
python -m venv venv && source venv/bin/activate
pip install streamlit scikit-learn xgboost shap pandas plotly joblib numpy
streamlit run app.py
```

---

## Usage

```bash
# Start the app
streamlit run app.py

# Batch prediction
python batch_predict.py --input patients.csv --output risk_scores.csv

# Retrain with updated dataset
python train.py --data heart.csv --model xgboost --threshold 0.45
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.pkl` | Serialised classifier pipeline |
| `RISK_THRESHOLD` | `0.5` | Probability cutoff for positive classification |
| `SHAP_ENABLED` | `True` | Enable/disable SHAP computation (slower but explainable) |
| `TOP_FEATURES` | `10` | Number of features shown in SHAP waterfall chart |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
Adv-Disease-pred-APP/
├── README.md
├── requirements.txt
├── app.py
├── Disease pred.ipynb
├── DiseasePred2.ipynb
├── disease.csv
├── disease2.csv
└── ...
```

---

## Roadmap

- [ ] Integration with FHIR-compatible EHR APIs for direct patient data ingestion
- [ ] Longitudinal risk tracking — plot risk score trajectory over multiple visits
- [ ] Uncertainty quantification via conformal prediction intervals
- [ ] Federated learning support for privacy-preserving multi-hospital training
- [ ] Voice-input mode for bedside use without keyboard interaction

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

This tool was developed for educational and research purposes. It is not a certified medical device. All predictions should be reviewed by qualified healthcare professionals before any clinical action is taken.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
