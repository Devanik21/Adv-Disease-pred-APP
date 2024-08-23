import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the dataset
df = pd.read_csv("disease.csv")  # Update with the correct path if needed

# Load the trained model (adjust the path to where your model is saved)
model = joblib.load("RF_Disease_pred.pkl")  # Replace with your actual model path

# Set page configuration
st.set_page_config(
    page_title="Disease Prediction APP",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define color and severity mappings
disease_colors = {
    'Sudden Fever': '#FF4500',
    'Headache': '#FF6347',
    'Mouth Bleed': '#FFD700',
    'Nose Bleed': '#DAA520',
    'Muscle Pain': '#FF8C00',
    'Joint Pain': '#FF1493',
    'Vomiting': '#FF69B4',
    'Rash': '#FFB6C1',
    'Diarrhea': '#FF6347',
    'Hypotension': '#FF4500',
    # Add more diseases here
}

disease_severity = {
    'Sudden Fever': 'High',
    'Headache': 'Medium',
    'Mouth Bleed': 'Severe',
    'Nose Bleed': 'Moderate',
    'Muscle Pain': 'Low',
    'Joint Pain': 'Medium',
    'Vomiting': 'High',
    'Rash': 'Low',
    'Diarrhea': 'Moderate',
    'Hypotension': 'High',
    # Add more severity levels here
}

# Title of the web app with styling
st.markdown("""
    <style>
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #FF6347;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #4682B4;
        font-size: 24px;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
    }
    .note {
        font-size: 16px;
        color: #808080;
    }
    .sidebar {
        background-color: #f0f8ff;
    }
    .disease-high { color: #FF6347; }
    .disease-medium { color: #FFD700; }
    .disease-low { color: #32CD32; }
    .disease-severe { color: #FF4500; }
    </style>
    """, unsafe_allow_html=True)

# Title of the web app
st.markdown('<div class="title">Disease Prediction Web App</div>', unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    # Create a dictionary to hold feature inputs
    features = {}
    
    # Assuming the last column is 'prognosis' and the rest are features
    for col in df.columns[:-1]:  # Exclude the target column
        # All features are binary (0 or 1), so use a slider with values 0 and 1
        features[col] = st.sidebar.slider(f"{col}", 0, 1, 0)

    input_df = pd.DataFrame(features, index=[0])
    return input_df

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Show loading spinner
with st.spinner('Making prediction...'):
    # Make prediction
    prediction = model.predict(input_df)

# Display the prediction result
st.subheader('Prediction Result')

# Display prediction with dynamic color and severity
def get_color_and_severity(disease):
    color = disease_colors.get(disease, '#000000')  # Default to black if not found
    severity = disease_severity.get(disease, 'Unknown')
    return color, severity

disease = prediction[0]
color, severity = get_color_and_severity(disease)

st.markdown(f'<div class="result" style="color:{color};">🩺 The predicted disease based on the input features is: <strong>{disease}</strong></div>', unsafe_allow_html=True)
st.markdown(f'<div class="result" style="color:{color};">Severity: <strong>{severity}</strong></div>', unsafe_allow_html=True)

# Optionally, you can add more details or a description below the result
st.markdown("""
    <div class="note">
        <strong>Note:</strong> The prediction is based on the model's analysis of the provided symptoms. For accurate diagnosis, please consult a healthcare professional.
    </div>
    """, unsafe_allow_html=True)

# Add an image or additional content
st.image("DNA.jpg", caption="Health and Wellness", use_column_width=True)
