import streamlit as st
import pandas as pd
import joblib
import numpy as np
from visualization import plot_feature_importance, correlation_matrix
from model_interpretation import explain_model_prediction
from about import display_about
from data_preprocessing import preprocess_data

# Load the dataset
df = pd.read_csv("disease.csv")

# Load the pre-trained model (Random Forest)
model = joblib.load("RF_Disease_pred.pkl")

# Set page configuration
st.set_page_config(
    page_title="Disease Prediction APP",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for user input
st.sidebar.header("🔍 Input Features")

def user_input_features():
    features = {}
    for col in df.columns[:-1]:  # Exclude the target column
        features[col] = st.sidebar.slider(f"{col}", 0, 1, 0)
    input_df = pd.DataFrame(features, index=[0])
    return input_df

# Get user input
input_df = preprocess_data(user_input_features())

# Display user input
st.subheader('📊 User Input Features')
st.write(input_df)

# Show loading spinner and make prediction
with st.spinner('🔍 Making prediction...'):
    prediction = model.predict(input_df)

# Display the prediction result
disease = prediction[0]
color, severity = get_color_and_severity(disease)
st.markdown(f'<div class="result" style="color:{color};">🩺 The predicted disease based on the input features is: <strong>{disease}</strong></div>', unsafe_allow_html=True)
st.markdown(f'<div class="result" style="color:{severity_colors.get(severity, "#000000")};">Severity: <strong>{severity}</strong></div>', unsafe_allow_html=True)

# Visualization options
if st.sidebar.checkbox("Show Feature Importance"):
    plot_feature_importance(model, df.columns[:-1])

if st.sidebar.checkbox("Show Correlation Matrix"):
    correlation_matrix(df)

# Model Interpretation option
if st.sidebar.checkbox("Explain Prediction"):
    explain_model_prediction(model, input_df)

# About section
if st.sidebar.checkbox("About this App"):
    display_about()

# Add an image or additional content
st.image("DNA.jpg", caption="Health and Wellness", use_column_width=True)
