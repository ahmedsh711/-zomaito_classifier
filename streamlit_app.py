import streamlit as st
import sys
import os
import pandas as pd

# Add src folder to sys.path (so Python can find your modules)
sys.path.append(os.path.abspath("../src"))

from inference import load_model, load_feature_names, predict

# Page config
st.set_page_config(page_title="Restaurant Success Predictor", layout="wide")

# Load model and features
@st.cache_resource
def load_resources():
    model = load_model('../models/restaurant_model.pkl')
    feature_names = load_feature_names('../data/preprocessed/feature_names.pkl')
    return model, feature_names

try:
    model, feature_names = load_resources()
except:
    st.error("Model not found! Please run train.py first.")
    st.stop()

# Title
st.title("Zomato Restaurant Success Predictor")
st.markdown("Predict whether a Bangalore restaurant will be **Successful** (rating ≥ 3.75)")

# Sidebar - Input form
st.sidebar.header("Restaurant Features")

# Basic features
online_order = st.sidebar.selectbox("Online Order Available?", ["Yes", "No"])
book_table = st.sidebar.selectbox("Table Booking Available?", ["Yes", "No"])
votes = st.sidebar.number_input("Number of Votes", min_value=0, max_value=10000, value=100)
cost_for_two = st.sidebar.number_input("Approx Cost for Two ", min_value=0, max_value=5000, value=500)

# Location features
location_freq = st.sidebar.slider("Location Popularity (restaurants in area)", 0, 3000, 1000)
city_freq = st.sidebar.slider("City Area Popularity", 0, 5000, 2000)

# Cuisine features
cuisine_count = st.sidebar.slider("Number of Cuisines", 1, 8, 3)

st.sidebar.subheader("Cuisine Types")
cuisine_north_indian = st.sidebar.checkbox("North Indian")
cuisine_chinese = st.sidebar.checkbox("Chinese")
cuisine_south_indian = st.sidebar.checkbox("South Indian")
cuisine_continental = st.sidebar.checkbox("Continental")

# Prepare input
input_data = {
    'online_order': 1 if online_order == "Yes" else 0,
    'book_table': 1 if book_table == "Yes" else 0,
    'votes': votes,
    'cost_for_two': cost_for_two,
    'location_freq': location_freq,
    'city_freq': city_freq,
    'cuisine_count': cuisine_count,
    'cuisine_north_indian': 1 if cuisine_north_indian else 0,
    'cuisine_chinese': 1 if cuisine_chinese else 0,
    'cuisine_south_indian': 1 if cuisine_south_indian else 0,
    'cuisine_continental': 1 if cuisine_continental else 0
}

# Predict button
if st.sidebar.button("Predict Success"):
    result = predict(model, input_data, feature_names)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", result['prediction'])
    with col2:
        st.metric("Success Probability", f"{result['success_probability']:.1%}")
    
    # Progress bar
    st.progress(result['success_probability'])
    
    # Interpretation
    if result['prediction'] == 'Successful':
        st.success("This restaurant is predicted to be **Successful** (rating ≥ 3.75)")
    else:
        st.warning("This restaurant is predicted to be **Unsuccessful** (rating < 3.75)")
    
    # Display input summary
    with st.expander("Input Summary"):
        st.json(input_data)

# Model info
st.markdown("---")
st.markdown("### About This Model")
st.info("""
This model predicts restaurant success in Bangalore based on:
- Service features (online ordering, table booking)
- Popularity metrics (votes, location frequency)
- Pricing strategy
- Cuisine diversity

**Success threshold:** Rating ≥ 3.75/5
""")
