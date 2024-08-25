# model_interpretation.py

import streamlit as st
import shap
import matplotlib.pyplot as plt

def explain_model_prediction(model, input_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    plt.title("SHAP values for the prediction")
    shap.force_plot(explainer.expected_value, shap_values[0], input_df, matplotlib=True)
    st.pyplot(plt)

    shap.summary_plot(shap_values, input_df, plot_type="bar")
    st.pyplot(plt)
