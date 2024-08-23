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
    'Sudden Fever': '#FF4500',        # OrangeRed
    'Headache': '#FF6347',            # Tomato
    'Mouth Bleed': '#FFD700',         # Gold
    'Nose Bleed': '#DAA520',          # GoldenRod
    'Muscle Pain': '#FF8C00',         # DarkOrange
    'Joint Pain': '#FF1493',          # DeepPink
    'Vomiting': '#FF69B4',            # HotPink
    'Rash': '#FFB6C1',                # LightPink
    'Diarrhea': '#FF6347',            # Tomato (Same as Headache for consistency)
    'Hypotension': '#FF4500',         # OrangeRed (Same as Sudden Fever for consistency)
    'Pleural Effusion': '#FF8C00',    # DarkOrange (Same as Muscle Pain for consistency)
    'Ascites': '#FFD700',             # Gold (Same as Mouth Bleed for consistency)
    'Gastro Bleeding': '#DAA520',     # GoldenRod (Same as Nose Bleed for consistency)
    'Swelling': '#FF69B4',            # HotPink (Same as Vomiting for consistency)
    'Nausea': '#FFB6C1',              # LightPink (Same as Rash for consistency)
    'Chills': '#FF6347',              # Tomato (Same as Headache for consistency)
    'Myalgia': '#FF8C00',             # DarkOrange (Same as Muscle Pain for consistency)
    'Digestion Trouble': '#FF1493',   # DeepPink (Same as Joint Pain for consistency)
    'Fatigue': '#FF69B4',             # HotPink (Same as Vomiting for consistency)
    'Skin Lesions': '#FFB6C1',        # LightPink (Same as Rash for consistency)
    'Stomach Pain': '#FF6347',        # Tomato (Same as Headache for consistency)
    'Orbital Pain': '#FF4500',        # OrangeRed (Same as Sudden Fever for consistency)
    'Neck Pain': '#FF8C00',           # DarkOrange (Same as Muscle Pain for consistency)
    'Weakness': '#FFD700',            # Gold (Same as Mouth Bleed for consistency)
    'Back Pain': '#DAA520',           # GoldenRod (Same as Nose Bleed for consistency)
    'Weight Loss': '#FF69B4',         # HotPink (Same as Vomiting for consistency)
    'Gum Bleed': '#FFB6C1',           # LightPink (Same as Rash for consistency)
    'Jaundice': '#FF6347',            # Tomato (Same as Headache for consistency)
    'Coma': '#FF4500',                # OrangeRed (Same as Sudden Fever for consistency)
    'Dizziness': '#FF8C00',           # DarkOrange (Same as Muscle Pain for consistency)
    'Inflammation': '#FFD700',        # Gold (Same as Mouth Bleed for consistency)
    'Red Eyes': '#DAA520',            # GoldenRod (Same as Nose Bleed for consistency)
    'Loss of Appetite': '#FF69B4',    # HotPink (Same as Vomiting for consistency)
    'Urination Loss': '#FFB6C1',      # LightPink (Same as Rash for consistency)
    'Slow Heart Rate': '#FF6347',     # Tomato (Same as Headache for consistency)
    'Abdominal Pain': '#FF4500',      # OrangeRed (Same as Sudden Fever for consistency)
    'Light Sensitivity': '#FF8C00',   # DarkOrange (Same as Muscle Pain for consistency)
    'Yellow Skin': '#FFD700',         # Gold (Same as Mouth Bleed for consistency)
    'Yellow Eyes': '#DAA520',         # GoldenRod (Same as Nose Bleed for consistency)
    'Facial Distortion': '#FF69B4',   # HotPink (Same as Vomiting for consistency)
    'Microcephaly': '#FFB6C1',        # LightPink (Same as Rash for consistency)
    'Rigor': '#FF6347',               # Tomato (Same as Headache for consistency)
    'Bitter Tongue': '#FF4500',       # OrangeRed (Same as Sudden Fever for consistency)
    'Convulsion': '#FF8C00',          # DarkOrange (Same as Muscle Pain for consistency)
    'Anemia': '#FFD700',              # Gold (Same as Mouth Bleed for consistency)
    'Cocacola Urine': '#DAA520',      # GoldenRod (Same as Nose Bleed for consistency)
    'Hypoglycemia': '#FF69B4',        # HotPink (Same as Vomiting for consistency)
    'Prostraction': '#FFB6C1',        # LightPink (Same as Rash for consistency)
    'Hyperpyrexia': '#FF6347',        # Tomato (Same as Headache for consistency)
    'Stiff Neck': '#FF4500',          # OrangeRed (Same as Sudden Fever for consistency)
    'Irritability': '#FF8C00',        # DarkOrange (Same as Muscle Pain for consistency)
    'Confusion': '#FFD700',           # Gold (Same as Mouth Bleed for consistency)
    'Tremor': '#DAA520',              # GoldenRod (Same as Nose Bleed for consistency)
    'Paralysis': '#FF69B4',           # HotPink (Same as Vomiting for consistency)
    'Lymph Swells': '#FFB6C1',        # LightPink (Same as Rash for consistency)
    'Breathing Restriction': '#FF6347', # Tomato (Same as Headache for consistency)
    'Toe Inflammation': '#FF4500',    # OrangeRed (Same as Sudden Fever for consistency)
    'Finger Inflammation': '#FF8C00', # DarkOrange (Same as Muscle Pain for consistency)
    'Lips Irritation': '#FFD700',     # Gold (Same as Mouth Bleed for consistency)
    'Itchiness': '#DAA520',           # GoldenRod (Same as Nose Bleed for consistency)
    'Ulcers': '#FF69B4',              # HotPink (Same as Vomiting for consistency)
    'Toenail Loss': '#FFB6C1',        # LightPink (Same as Rash for consistency)
    'Speech Problem': '#FF6347',      # Tomato (Same as Headache for consistency)
    'Bullseye Rash': '#FF4500'        # OrangeRed (Same as Sudden Fever for consistency)
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
    'Bullseye Rash': 'Moderate'
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
