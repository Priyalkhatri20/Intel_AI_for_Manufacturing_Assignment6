# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import os

# Load the trained model
model_path = "delivery_time_model (1).pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    st.success("✅ Model loaded from local file!")
else:
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()
# Title
st.title("🚚 Timelytics - Delivery Time Prediction")

# Input fields
product_category = st.selectbox("Product Category", ['Electronics', 'Clothing', 'Furniture', 'Toys', 'Books'])
customer_location = st.selectbox("Customer Location", ['New York', 'California', 'Texas', 'Florida', 'Washington'])
shipping_method = st.selectbox("Shipping Method", ['Standard', 'Expedited', 'Overnight'])

# Prediction
if st.button("Predict Delivery Time"):
    input_data = pd.DataFrame({
        'product_category': [product_category],
        'customer_location': [customer_location],
        'shipping_method': [shipping_method]
    })
    delivery_time = model.predict(input_data)[0]
    st.success(f"✅ Predicted Delivery Time: {delivery_time:.2f} days")
