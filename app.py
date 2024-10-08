import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("disease.csv")  # Update with the correct path if needed

# Load the trained model (adjust the path to where your model is saved)
model = joblib.load("RF_Disease_pred.pkl")  # Replace with your actual model path

severity_colors = {
    'Low': '#90EE90',        # Light Green
    'Medium': '#FFFF00',     # Yellow
    'High': '#FF6347',       # Tomato
    'Moderate': '#FFA07A',   # Light Salmon
    'Severe': '#FF4500',     # Orange Red
    'Critical': '#FF0000'    # Red
}

# Set page configuration
st.set_page_config(
    page_title="Disease Prediction APP",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the web app
st.markdown('<h1 style="color:#FF6347; text-align:center;">🧬 Disease Prediction Web App </h1>', unsafe_allow_html=True)

# Add an image or additional content
st.image("DNA.jpg", caption="Health and Wellness", use_column_width=True)

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
    'Chikungunya':'Critical',
    'Tungiasis' : 'Moderate'
}
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
    'Chikungunya': '#DAA520',
    'Tungiasis' : '#FF69B4' 
}
# Define colors and severity mappings


def get_color_and_severity(disease):
    color = disease_colors.get(disease, '#000000')
    severity = disease_severity.get(disease, 'Unknown')
    return color, severity

disease = prediction[0]
color, severity = get_color_and_severity(disease)
st.markdown(f'<h2 style="color:{color};">🩺 The predicted disease based on the input features is: <strong>{disease}</strong></h2>', unsafe_allow_html=True)
st.markdown(f'<h3 style="color:{severity_colors.get(severity, "#000000")};">Severity: <strong>{severity}</strong></h3>', unsafe_allow_html=True)

st.subheader('🛩️Advanced Visualizations')

# Add a pie chart for severity distribution
severity_counts = {k: list(disease_severity.values()).count(k) for k in set(disease_severity.values())}
fig, ax = plt.subplots(figsize=(12, 10))
ax.pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.1f%%', startangle=140)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# Assuming df is your DataFrame and symptom_counts is already calculated
symptom_counts = df.iloc[:, :-1].sum()

# Increase figure size for better label visibility
fig, ax = plt.subplots(figsize=(16,12))  # Adjust the size as needed

# Plot bar chart
symptom_counts.plot(kind='bar', ax=ax, color='skyblue')

# Rotate x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Set titles and labels
ax.set_title('Frequency of Symptoms in Dataset')
ax.set_xlabel('Symptom')
ax.set_ylabel('Frequency')

# Display the plot in Streamlit
st.pyplot(fig)

# Histogram of Symptoms
fig, ax = plt.subplots(figsize=(14, 12))
df.iloc[:, :-1].sum(axis=0).plot(kind='hist', bins=30, ax=ax, color='lightcoral', edgecolor='black')
ax.set_title('Histogram of Symptom Frequencies')
ax.set_xlabel('Frequency')
st.pyplot(fig)


# Assuming you have already loaded your dataset as df

# Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns

# Increase the size of the figure to accommodate many columns
fig, ax = plt.subplots(figsize=(16, 12))  # Larger size for more columns

# Compute the correlation matrix
corr_matrix = numeric_df.corr()

# Mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Draw the heatmap
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    ax=ax,
    fmt='.2f',
    annot_kws={"size": 6},  # Smaller font size for annotations
    cbar_kws={"shrink": .8},
   
)

# Rotate the labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# Set title
ax.set_title('Correlation Heatmap of Numeric Features', fontsize=16)

# Display the heatmap in Streamlit
st.pyplot(fig)

# Line chart of average feature values
fig, ax = plt.subplots(figsize=(12,10))  # Increase the size for better label visibility
df.iloc[:, :-1].mean().plot(kind='line', ax=ax, marker='o', color='darkgreen')

# Rotate x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Set titles and labels
ax.set_title('Average Feature Values', fontsize=16)
ax.set_xlabel('Feature', fontsize=14)
ax.set_ylabel('Average Value', fontsize=14)

# Display the plot in Streamlit
st.pyplot(fig)

# Disease Distribution Pie Chart
disease_distribution = df['prognosis'].value_counts()
fig, ax = plt.subplots()
ax.pie(disease_distribution, labels=disease_distribution.index, autopct='%1.1f%%', startangle=140)
ax.axis('equal')
ax.set_title('Distribution of Diseases in Dataset')
st.pyplot(fig)

# User feedback
st.subheader("📝 Feedback")
feedback = st.text_area("Share your feedback or suggestions:", height=150)
if st.button('Submit Feedback'):
    st.success("Thank you for your feedback!")

# Optionally, add interactive elements or more dynamic content
st.markdown("""
    <h4 style="text-align:center; color:#808080;">
        <strong>Note:</strong> The prediction is based on the model's analysis of the provided symptoms. For accurate diagnosis, please consult a healthcare professional. 
    </h4>
    """, unsafe_allow_html=True)
