# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 19:09:39 2025

@author: HP
"""

import numpy as np
import pickle
import streamlit as st
import os

# Path relative to this file
model_path = os.path.join(os.path.dirname(__file__), "credit_card_model.sav")

try:
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found! Make sure 'credit_card_model.sav' is uploaded to your GitHub repo.")
    st.stop()

# Prediction function
def fraud_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return "✅ Normal Transaction" if prediction[0] == 0 else "⚠️ Fraudulent Transaction"

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")

time = st.number_input("Time", value=0.0)
features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
amount = st.number_input("Amount", value=0.0)

input_data = [time] + features + [amount]

if st.button("Check Transaction"):
    result = fraud_prediction(input_data)
    st.success(result)
