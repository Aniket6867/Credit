# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 19:09:39 2025

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
model_path = r"C:\Users\HP\OneDrive\Desktop\ja\credit_card_model.sav"
loaded_model = pickle.load(open(model_path, 'rb'))

# Prediction function
def fraud_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)  # reshape into 2D
    prediction = loaded_model.predict(input_array)
    
    if prediction[0] == 0:
        return "✅ Normal Transaction"
    else:
        return "⚠️ Fraudulent Transaction"

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details:")

# Input fields for all features
time = st.number_input("Time", value=0.0)

# V1 - V28
features = []
for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0)
    features.append(value)

amount = st.number_input("Amount", value=0.0)

# Collect all inputs in correct order
input_data = [time] + features + [amount]

if st.button("Check Transaction"):
    result = fraud_prediction(input_data)
    st.success(result)

