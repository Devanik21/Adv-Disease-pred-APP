# about.py

import streamlit as st

def display_about():
    st.markdown("""
    ## About this App
    This Disease Prediction Web App uses machine learning to predict diseases based on user-provided symptoms.

    ### Model Information
    The app supports multiple machine learning models:
    - Random Forest
    - Logistic Regression
    - Support Vector Machine (SVM)

    ### Dataset
    The dataset includes various symptoms and their associated diseases.

    ### Disclaimer
    This app is for educational purposes and is not a substitute for professional medical advice.
    """)
