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
fig, ax = plt.subplots()
ax.pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.1f%%', startangle=140)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# Bar chart of symptom frequencies
symptom_counts = df.iloc[:, :-1].sum()
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the size as needed
symptom_counts.plot(kind='bar', ax=ax, color='skyblue')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Frequency of Symptoms in Dataset')
ax.set_xlabel('Symptom')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Histogram of Symptoms
fig, ax = plt.subplots()
df.iloc[:, :-1].sum(axis=0).plot(kind='hist', bins=30, ax=ax, color='lightcoral', edgecolor='black')
ax.set_title('Histogram of Symptom Frequencies')
ax.set_xlabel('Frequency')
st.pyplot(fig)

# Correlation Heatmap for Numeric Columns
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
fig, ax = plt.subplots(figsize=(14, 10))  # Adjust the size as needed
corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap='coolwarm',
    ax=ax,
    fmt='.2f',
    annot_kws={"size": 8},
    cbar_kws={"shrink": .8}
)
ax.set_title('Correlation Heatmap of Numeric Features', fontsize=16)
st.pyplot(fig)

# Average Feature Values Line Chart
fig, ax = plt.subplots(figsize=(12, 6))  # Increase the size for better label visibility
df.iloc[:, :-1].mean().plot(kind='line', ax=ax, marker='o', color='darkgreen')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Average Feature Values', fontsize=16)
ax.set_xlabel('Feature', fontsize=14)
ax.set_ylabel('Average Value', fontsize=14)
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

# Additional notes
st.markdown("""
    <h4 style="text-align:center; color:#808080;">
        <strong>Note:</strong> The prediction is based on the model's analysis of the provided symptoms. For accurate diagnosis, please consult a healthcare professional. 
    </h4>
    """, unsafe_allow_html=True)
