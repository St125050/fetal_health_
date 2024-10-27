import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')  # Ensure you've saved your model
scaler = joblib.load('scaler.pkl')  # Ensure you've saved your scaler

# Streamlit app title
st.title("Fetal Health Classification App")

# Instructions for feature values
st.header("Enter Fetal Health Features")

# Feature min and max values
feature_info = {
    "Baseline Value": (106, 160),
    "Accelerations": (0.0, 0.019),
    "Fetal Movement": (0.0, 0.481),
    "Uterine Contractions": (0.0, 0.015),
    "Light Decelerations": (0.0, 0.015),
    "Severe Decelerations": (0.0, 0.001),
    "Prolongued Decelerations": (0.0, 0.005),
    "Abnormal Short Term Variability": (12, 87),
    "Mean Value of Short Term Variability": (0.2, 7.0),
    "Percentage of Time with Abnormal Long Term Variability": (0.0, 91.0),
    "Histogram Min": (50, 159),
    "Histogram Max": (122, 238),
    "Histogram Number of Peaks": (0, 18),
    "Histogram Number of Zeroes": (0, 10),
    "Histogram Mode": (60, 187),
    "Histogram Mean": (73, 182),
    "Histogram Median": (77, 186),
    "Histogram Variance": (0, 269),
    "Histogram Tendency": (-1.0, 1.0),
}

for feature, (min_val, max_val) in feature_info.items():
    st.write(f"{feature}: Min = {min_val}, Max = {max_val}")

# User inputs for feature values
baseline_value = st.number_input("Baseline Value", min_value=0.0)
accelerations = st.number_input("Accelerations", min_value=0.0)
fetal_movement = st.number_input("Fetal Movement", min_value=0.0)
uterine_contractions = st.number_input("Uterine Contractions", min_value=0.0)
light_decelerations = st.number_input("Light Decelerations", min_value=0.0)
severe_decelerations = st.number_input("Severe Decelerations", min_value=0.0)
prolongued_decelerations = st.number_input("Prolongued Decelerations", min_value=0.0)
abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", min_value=0.0)
mean_value_of_short_term_variability = st.number_input("Mean Value of Short Term Variability", min_value=0.0)
percentage_of_time_with_abnormal_long_term_variability = st.number_input("Percentage of Time with Abnormal Long Term Variability", min_value=0.0)
histogram_min = st.number_input("Histogram Min", min_value=0.0)
histogram_max = st.number_input("Histogram Max", min_value=0.0)
histogram_number_of_peaks = st.number_input("Histogram Number of Peaks", min_value=0)
histogram_number_of_zeroes = st.number_input("Histogram Number of Zeroes", min_value=0)
histogram_mode = st.number_input("Histogram Mode", min_value=0.0)
histogram_mean = st.number_input("Histogram Mean", min_value=0.0)
histogram_median = st.number_input("Histogram Median", min_value=0.0)
histogram_variance = st.number_input("Histogram Variance", min_value=0.0)
histogram_tendency = st.number_input("Histogram Tendency", min_value=-1.0, max_value=1.0)

# Create a button to make predictions
if st.button("Predict Fetal Health"):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'baseline_value': [baseline_value],
        'accelerations': [accelerations],
        'fetal_movement': [fetal_movement],
        'uterine_contractions': [uterine_contractions],
        'light_decelerations': [light_decelerations],
        'severe_decelerations': [severe_decelerations],
        'prolongued_decelerations': [prolongued_decelerations],
        'abnormal_short_term_variability': [abnormal_short_term_variability],
        'mean_value_of_short_term_variability': [mean_value_of_short_term_variability],
        'percentage_of_time_with_abnormal_long_term_variability': [percentage_of_time_with_abnormal_long_term_variability],
        'histogram_min': [histogram_min],
        'histogram_max': [histogram_max],
        'histogram_number_of_peaks': [histogram_number_of_peaks],
        'histogram_number_of_zeroes': [histogram_number_of_zeroes],
        'histogram_mode': [histogram_mode],
        'histogram_mean': [histogram_mean],
        'histogram_median': [histogram_median],
        'histogram_variance': [histogram_variance],
        'histogram_tendency': [histogram_tendency]
    })

    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make predictions
    predictions = model.predict(scaled_data)

    # Show predictions
    st.write("Predicted Fetal Health:", predictions[0])
