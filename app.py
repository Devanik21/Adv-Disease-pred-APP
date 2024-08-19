import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv("C:\\imp\\ml JUPYTER\\MY ML PROJECTS(BOOK)\\3.Deep learning\\ANN\\Recreation\\Disease\\Disease APP\\disease.csv")

# Load the trained model (adjust the path to where your model is saved)
model = joblib.load("C:\\imp\\ml JUPYTER\\MY ML PROJECTS(BOOK)\\3.Deep learning\\ANN\\Recreation\\Disease\\Disease APP\\RF_Disease_pred.pkl")  # Replace with your actual model path

# Title of the web app
st.title("Disease Prediction Web App")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    # Create a dictionary to hold feature inputs
    features = {}
    
    # Assuming the last column is 'prognosis' and the rest are features
    for col in df.columns[:-1]:  # Exclude the target column
        # Handling different types of features (numeric, categorical)
        if df[col].dtype == 'object':
            # For categorical features, use a select box
            unique_values = df[col].unique()
            features[col] = st.sidebar.selectbox(f"{col}", unique_values)
        else:
            # For numeric features, use a slider
            min_value = float(df[col].min())
            max_value = float(df[col].max())
            default_value = float(df[col].mean())
            features[col] = st.sidebar.slider(f"{col}", min_value, max_value, default_value)

    input_df = pd.DataFrame(features, index=[0])
    return input_df

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)

# Display the prediction result
st.subheader('Prediction Result')

# Customize the prediction message
st.success(f"🩺 The predicted disease based on the input features is: **{prediction[0]}**")

# Optionally, you can add more details or a description below the result
st.markdown("""
<div style="margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
    <strong>Note:</strong> The prediction is based on the model's analysis of the provided symptoms.
</div>
""", unsafe_allow_html=True)
