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
    page_title="Disease Prediction App",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define color and severity mappings
severity_colors = {
    'Low': '#90EE90',        # Light Green
    'Medium': '#FFFF00',     # Yellow
    'High': '#FF6347',       # Tomato
    'Moderate': '#FFA07A',   # Light Salmon
    'Severe': '#FF4500',     # Orange Red
    'Critical': '#FF0000'    # Red
}

disease_colors = {
    'Sudden Fever': severity_colors['High'],
    'Headache': severity_colors['Medium'],
    'Mouth Bleed': severity_colors['Severe'],
    'Nose Bleed': severity_colors['Moderate'],
    'Muscle Pain': severity_colors['Low'],
    'Joint Pain': severity_colors['Medium'],
    'Vomiting': severity_colors['High'],
    'Rash': severity_colors['Low'],
    'Diarrhea': severity_colors['Moderate'],
    'Hypotension': severity_colors['High'],
    'Pleural Effusion': severity_colors['Severe'],
    'Ascites': severity_colors['Severe'],
    'Gastro Bleeding': severity_colors['Severe'],
    'Swelling': severity_colors['Moderate'],
    'Nausea': severity_colors['Medium'],
    'Chills': severity_colors['Low'],
    'Myalgia': severity_colors['Medium'],
    'Digestion Trouble': severity_colors['Moderate'],
    'Fatigue': severity_colors['Medium'],
    'Skin Lesions': severity_colors['Moderate'],
    'Stomach Pain': severity_colors['Moderate'],
    'Orbital Pain': severity_colors['Medium'],
    'Neck Pain': severity_colors['Low'],
    'Weakness': severity_colors['High'],
    'Back Pain': severity_colors['Low'],
    'Weight Loss': severity_colors['Moderate'],
    'Gum Bleed': severity_colors['Severe'],
    'Jaundice': severity_colors['Severe'],
    'Coma': severity_colors['Critical'],
    'Dizziness': severity_colors['Medium'],
    'Inflammation': severity_colors['Moderate'],
    'Red Eyes': severity_colors['Medium'],
    'Loss of Appetite': severity_colors['Medium'],
    'Urination Loss': severity_colors['Severe'],
    'Slow Heart Rate': severity_colors['High'],
    'Abdominal Pain': severity_colors['Moderate'],
    'Light Sensitivity': severity_colors['Low'],
    'Yellow Skin': severity_colors['Severe'],
    'Yellow Eyes': severity_colors['Severe'],
    'Facial Distortion': severity_colors['Severe'],
    'Microcephaly': severity_colors['Critical'],
    'Rigor': severity_colors['High'],
    'Bitter Tongue': severity_colors['Medium'],
    'Convulsion': severity_colors['Critical'],
    'Anemia': severity_colors['Moderate'],
    'Cocacola Urine': severity_colors['Severe'],
    'Hypoglycemia': severity_colors['High'],
    'Prostraction': severity_colors['Critical'],
    'Hyperpyrexia': severity_colors['Critical'],
    'Stiff Neck': severity_colors['High'],
    'Irritability': severity_colors['Medium'],
    'Confusion': severity_colors['Severe'],
    'Tremor': severity_colors['Medium'],
    'Paralysis': severity_colors['Critical'],
    'Lymph Swells': severity_colors['Moderate'],
    'Breathing Restriction': severity_colors['Severe'],
    'Toe Inflammation': severity_colors['Low'],
    'Finger Inflammation': severity_colors['Low'],
    'Lips Irritation': severity_colors['Low'],
    'Itchiness': severity_colors['Low'],
    'Ulcers': severity_colors['Moderate'],
    'Toenail Loss': severity_colors['Low'],
    'Speech Problem': severity_colors['Severe'],
    'Bullseye Rash': severity_colors['Moderate'],
    'Dengue': severity_colors['Severe']
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
    'Dengue': 'Severe'
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
    .disease-critical { color: #FF0000; }
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .input-card {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .result-card {
        background-color: #fff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        width: 80%;
    }
    </style>
    <div class="title">Disease Prediction App</div>
    """, unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    st.markdown("- [Home](#)")
    st.markdown("- [Prediction](#)")
    st.markdown("- [Insights](#)")
    st.markdown("- [About](#)")

st.markdown("## Predict Your Disease")
st.markdown("Fill out the form below to get a prediction.")

# User input form
with st.form(key='prediction_form', clear_on_submit=True):
    st.subheader("Please select your symptoms:")
    symptoms = {
        'Sudden Fever': st.checkbox('Sudden Fever'),
        'Headache': st.checkbox('Headache'),
        'Mouth Bleed': st.checkbox('Mouth Bleed'),
        'Nose Bleed': st.checkbox('Nose Bleed'),
        'Muscle Pain': st.checkbox('Muscle Pain'),
        'Joint Pain': st.checkbox('Joint Pain'),
        'Vomiting': st.checkbox('Vomiting'),
        'Rash': st.checkbox('Rash'),
        'Diarrhea': st.checkbox('Diarrhea'),
        'Hypotension': st.checkbox('Hypotension'),
        'Pleural Effusion': st.checkbox('Pleural Effusion'),
        'Ascites': st.checkbox('Ascites'),
        'Gastro Bleeding': st.checkbox('Gastro Bleeding'),
        'Swelling': st.checkbox('Swelling'),
        'Nausea': st.checkbox('Nausea'),
        'Chills': st.checkbox('Chills'),
        'Myalgia': st.checkbox('Myalgia'),
        'Digestion Trouble': st.checkbox('Digestion Trouble'),
        'Fatigue': st.checkbox('Fatigue'),
        'Skin Lesions': st.checkbox('Skin Lesions'),
        'Stomach Pain': st.checkbox('Stomach Pain'),
        'Orbital Pain': st.checkbox('Orbital Pain'),
        'Neck Pain': st.checkbox('Neck Pain'),
        'Weakness': st.checkbox('Weakness'),
        'Back Pain': st.checkbox('Back Pain'),
        'Weight Loss': st.checkbox('Weight Loss'),
        'Gum Bleed': st.checkbox('Gum Bleed'),
        'Jaundice': st.checkbox('Jaundice'),
        'Coma': st.checkbox('Coma'),
        'Dizziness': st.checkbox('Dizziness'),
        'Inflammation': st.checkbox('Inflammation'),
        'Red Eyes': st.checkbox('Red Eyes'),
        'Loss of Appetite': st.checkbox('Loss of Appetite'),
        'Urination Loss': st.checkbox('Urination Loss'),
        'Slow Heart Rate': st.checkbox('Slow Heart Rate'),
        'Abdominal Pain': st.checkbox('Abdominal Pain'),
        'Light Sensitivity': st.checkbox('Light Sensitivity'),
        'Yellow Skin': st.checkbox('Yellow Skin'),
        'Yellow Eyes': st.checkbox('Yellow Eyes'),
        'Facial Distortion': st.checkbox('Facial Distortion'),
        'Microcephaly': st.checkbox('Microcephaly'),
        'Rigor': st.checkbox('Rigor'),
        'Bitter Tongue': st.checkbox('Bitter Tongue'),
        'Convulsion': st.checkbox('Convulsion'),
        'Anemia': st.checkbox('Anemia'),
        'Cocacola Urine': st.checkbox('Cocacola Urine'),
        'Hypoglycemia': st.checkbox('Hypoglycemia'),
        'Prostraction': st.checkbox('Prostraction'),
        'Hyperpyrexia': st.checkbox('Hyperpyrexia'),
        'Stiff Neck': st.checkbox('Stiff Neck'),
        'Irritability': st.checkbox('Irritability'),
        'Confusion': st.checkbox('Confusion'),
        'Tremor': st.checkbox('Tremor'),
        'Paralysis': st.checkbox('Paralysis'),
        'Lymph Swells': st.checkbox('Lymph Swells'),
        'Breathing Restriction': st.checkbox('Breathing Restriction'),
        'Toe Inflammation': st.checkbox('Toe Inflammation'),
        'Finger Inflammation': st.checkbox('Finger Inflammation'),
        'Lips Irritation': st.checkbox('Lips Irritation'),
        'Itchiness': st.checkbox('Itchiness'),
        'Ulcers': st.checkbox('Ulcers'),
        'Toenail Loss': st.checkbox('Toenail Loss'),
        'Speech Problem': st.checkbox('Speech Problem'),
        'Bullseye Rash': st.checkbox('Bullseye Rash'),
        'Dengue': st.checkbox('Dengue')
    }
    submit_button = st.form_submit_button(label='Predict Disease')

# Disease prediction logic
if submit_button:
    selected_symptoms = [symptom for symptom, selected in symptoms.items() if selected]
    input_data = np.array([selected_symptoms]).reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max() * 100

    # Display results
    st.markdown("## Prediction Result")
    st.markdown(f"<div class='result' style='background-color:{disease_colors.get(prediction, '#FFFFFF')}; padding: 10px;'>")
    st.markdown(f"**Predicted Disease:** {prediction}")
    st.markdown(f"**Probability:** {probability:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### Note:")
    st.markdown("This prediction is based on the input symptoms and the trained model. For accurate diagnosis, please consult a healthcare professional.")

# Add About section
st.markdown("## About")
st.markdown("""
    This application uses a Random Forest classifier to predict diseases based on symptoms. 
    The model is trained on historical disease data to provide accurate predictions.
    """)
