import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

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

# Title of the web app
st.markdown('<h1 style="color:#FF6347; text-align:center;">🧬 Disease Prediction Web App </h1>', unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("🔍 Input Features")

def user_input_features():
    features = {}
    for col in df.columns[:-1]:  # Exclude the target column
        features[col] = st.sidebar.selectbox(f"{col}", [0, 1], index=0, format_func=lambda x: 'No' if x==0 else 'Yes')
    input_df = pd.DataFrame(features, index=[0])
    return input_df

input_df = user_input_features()

# Display user input
st.subheader('📊 User Input Features')
st.write(input_df)

# Show loading spinner
with st.spinner('🔍 Making prediction...'):
    prediction = model.predict(input_df)

# Display the prediction result
st.subheader('🎯 Prediction Result')

def get_color_and_severity(disease):
    color = disease_colors.get(disease, '#000000')
    severity = disease_severity.get(disease, 'Unknown')
    return color, severity

disease = prediction[0]
color, severity = get_color_and_severity(disease)

# Add a pie chart for severity distribution
severity_counts = {k: list(disease_severity.values()).count(k) for k in set(disease_severity.values())}
fig, ax = plt.subplots()
ax.pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.1f%%', startangle=140)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

st.markdown(f'<h2 style="color:{color};">🩺 The predicted disease based on the input features is: <strong>{disease}</strong></h2>', unsafe_allow_html=True)
st.markdown(f'<h3 style="color:{severity_colors.get(severity, "#000000")};">Severity: <strong>{severity}</strong></h3>', unsafe_allow_html=True)

# User feedback
st.subheader("📝 Feedback")
feedback = st.text_area("Share your feedback or suggestions:", height=150)
if st.button('Submit Feedback'):
    st.success("Thank you for your feedback!")

# Add an image or additional content
st.image("DNA.jpg", caption="Health and Wellness", use_column_width=True)

# Optionally, add interactive elements or more dynamic content
st.markdown("""
    <h4 style="text-align:center; color:#808080;">
        <strong>Note:</strong> The prediction is based on the model's analysis of the provided symptoms. For accurate diagnosis, please consult a healthcare professional. 
    </h4>
    """, unsafe_allow_html=True)
