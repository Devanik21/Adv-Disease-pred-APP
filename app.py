import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the dataset
df = pd.read_csv("disease.csv")  # Update with the correct path if needed

# Load the trained model (adjust the path to where your model is saved)
model = joblib.load("RF_Disease_pred.pkl")  # Replace with your actual model path

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
        color: #2E8B57;
        font-weight: bold;
    }
    .note {
        font-size: 16px;
        color: #808080;
    }
    .sidebar {
        background-color: #f0f8ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the web app
st.markdown('<div class="title">Disease Prediction Web App</div>', unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("Input Features")
st.sidebar.markdown("<div class='sidebar'>Please enter the details below to get the disease prediction.</div>", unsafe_allow_html=True)

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

# Customize the prediction message
st.markdown(f'<div class="result">🩺 The predicted disease based on the input features is: **{prediction[0]}**</div>', unsafe_allow_html=True)

# Optionally, you can add more details or a description below the result
st.markdown("""
    <div class="note">
        <strong>Note:</strong> The prediction is based on the model's analysis of the provided symptoms. For accurate diagnosis, please consult a healthcare professional.
    </div>
    """, unsafe_allow_html=True)

# Add an image or additional content
st.image("DNA.jpg", caption="Health and Wellness", use_column_width=True)

