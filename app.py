import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the dataset
df = pd.read_csv("disease.csv")  # Update with the correct path if needed

# Load the trained model (adjust the path to where your model is saved)
model = joblib.load("RF_Disease_pred.pkl")  # Replace with your actual model path

import matplotlib.pyplot as plt

# Disease frequency
st.subheader('Disease Frequency')

disease_counts = df['prognosis'].value_counts()
fig, ax = plt.subplots()
disease_counts.plot(kind='bar', color=[disease_colors[disease] for disease in disease_counts.index], ax=ax)
ax.set_xlabel('Disease')
ax.set_ylabel('Count')
ax.set_title('Frequency of Diseases')
st.pyplot(fig)

# Severity distribution
st.subheader('Severity Distribution')

severity_counts = pd.Series(disease_severity).value_counts()
fig, ax = plt.subplots()
severity_counts.plot(kind='pie', autopct='%1.1f%%', colors=[severity_colors[severity] for severity in severity_counts.index], ax=ax)
ax.set_title('Distribution of Disease Severity')
st.pyplot(fig)

# Set page configuration
st.set_page_config(
    page_title="Disease Prediction APP",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

severity_colors = {
    'Low': '#90EE90',        # Light Green
    'Medium': '#FFFF00',     # Yellow
    'High': '#FF6347',       # Tomato
    'Moderate': '#FFA07A',   # Light Salmon
    'Severe': '#FF4500',     # Orange Red
    'Critical': '#FF0000'    # Red
}
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
    'Pleural Effusion': '#FF69B4',
    'Ascites': '#FF69B4',
    'Gastro Bleeding': '#FF69B4',
    'Swelling': '#DAA520',
    'Nausea': '#FF6347',
    'Chills': '#FFB6C1',
    'Myalgia': '#FF6347',
    'Digestion Trouble': '#DAA520',
    'Fatigue': '#FF6347',
    'Skin Lesions': '#DAA520',
    'Stomach Pain': '#DAA520',
    'Orbital Pain': '#FF6347',
    'Neck Pain': '#FFB6C1',
    'Weakness': '#FF4500',
    'Back Pain': '#FFB6C1',
    'Weight Loss': '#DAA520',
    'Gum Bleed': '#FF69B4',
    'Jaundice': '#FF69B4',
    'Coma': '#FF0000',
    'Dizziness': '#FF6347',
    'Inflammation': '#DAA520',
    'Red Eyes': '#FF6347',
    'Loss of Appetite': '#FF6347',
    'Urination Loss': '#FF69B4',
    'Slow Heart Rate': '#FF4500',
    'Abdominal Pain': '#DAA520',
    'Light Sensitivity': '#FFB6C1',
    'Yellow Skin': '#FF69B4',
    'Yellow Eyes': '#FF69B4',
    'Facial Distortion': '#FF69B4',
    'Microcephaly': '#FF0000',
    'Rigor': '#FF4500',
    'Bitter Tongue': '#FF6347',
    'Convulsion': '#FF0000',
    'Anemia': '#DAA520',
    'Cocacola Urine': '#FF69B4',
    'Hypoglycemia': '#FF4500',
    'Prostraction': '#FF0000',
    'Hyperpyrexia': '#FF0000',
    'Stiff Neck': '#FF4500',
    'Irritability': '#FF6347',
    'Confusion': '#FF69B4',
    'Tremor': '#FF6347',
    'Paralysis': '#FF0000',
    'Lymph Swells': '#DAA520',
    'Breathing Restriction': '#FF69B4',
    'Toe Inflammation': '#FFB6C1',
    'Finger Inflammation': '#FFB6C1',
    'Lips Irritation': '#FFB6C1',
    'Itchiness': '#FFB6C1',
    'Ulcers': '#DAA520',
    'Toenail Loss': '#FFB6C1',
    'Speech Problem': '#FF69B4',
    'Bullseye Rash': '#DAA520',
    'Dengue': '#FF69B4',
     'Chikungunya': '#DAA520'
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
    'Pleural Effusion': 'Severe',
    'Ascites': 'Severe',
    'Gastro Bleeding': 'Severe',
    'Swelling': 'Moderate',
    'Nausea': 'Medium',
    'Chills': 'Low',
    'Myalgia': 'Medium',
    'Digestion Trouble': 'Moderate',
    'Fatigue': 'Medium',
    'Skin Lesions': 'Moderate',
    'Stomach Pain': 'Moderate',
    'Orbital Pain': 'Medium',
    'Neck Pain': 'Low',
    'Weakness': 'High',
    'Back Pain': 'Low',
    'Weight Loss': 'Moderate',
    'Gum Bleed': 'Severe',
    'Jaundice': 'Severe',
    'Coma': 'Critical',
    'Dizziness': 'Medium',
    'Inflammation': 'Moderate',
    'Red Eyes': 'Medium',
    'Loss of Appetite': 'Medium',
    'Urination Loss': 'Severe',
    'Slow Heart Rate': 'High',
    'Abdominal Pain': 'Moderate',
    'Light Sensitivity': 'Low',
    'Yellow Skin': 'Severe',
    'Yellow Eyes': 'Severe',
    'Facial Distortion': 'Severe',
    'Microcephaly': 'Critical',
    'Rigor': 'High',
    'Bitter Tongue': 'Medium',
    'Convulsion': 'Critical',
    'Anemia': 'Moderate',
    'Cocacola Urine': 'Severe',
    'Hypoglycemia': 'High',
    'Prostraction': 'Critical',
    'Hyperpyrexia': 'Critical',
    'Stiff Neck': 'High',
    'Irritability': 'Medium',
    'Confusion': 'Severe',
    'Tremor': 'Medium',
    'Paralysis': 'Critical',
    'Lymph Swells': 'Moderate',
    'Breathing Restriction': 'Severe',
    'Toe Inflammation': 'Low',
    'Finger Inflammation': 'Low',
    'Lips Irritation': 'Low',
    'Itchiness': 'Low',
    'Ulcers': 'Moderate',
    'Toenail Loss': 'Low',
    'Speech Problem': 'Severe',
    'Bullseye Rash': 'Moderate',
    'Dengue': 'Severe',
     'Chikungunya':'Rare'
     
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
