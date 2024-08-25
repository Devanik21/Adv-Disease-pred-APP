import streamlit as st
import pandas as pd
import joblib
import numpy as np
from visualization import plot_feature_importance, correlation_matrix
from model_interpretation import explain_model_prediction
from model_selection import load_model
from about import display_about
from data_preprocessing import preprocess_data

# Load the dataset
df = pd.read_csv("disease.csv")

# Sidebar for model selection
model_name = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "SVM"])
model = load_model(model_name)

# Preprocess user input data
input_df = preprocess_data(user_input_features())

# Prediction
with st.spinner('🔍 Making prediction...'):
    prediction = model.predict(input_df)

# Display result
disease = prediction[0]
color, severity = get_color_and_severity(disease)
st.markdown(f'<div class="result" style="color:{color};">🩺 The predicted disease is: <strong>{disease}</strong></div>', unsafe_allow_html=True)
st.markdown(f'<div class="result" style="color:{severity_colors.get(severity, "#000000")};">Severity: <strong>{severity}</strong></div>', unsafe_allow_html=True)

# Visualization
if st.sidebar.checkbox("Show Feature Importance"):
    plot_feature_importance(model, df.columns[:-1])

if st.sidebar.checkbox("Show Correlation Matrix"):
    correlation_matrix(df)

# Model Interpretation
if st.sidebar.checkbox("Explain Prediction"):
    explain_model_prediction(model, input_df)

# About section
if st.sidebar.checkbox("About this App"):
    display_about()
