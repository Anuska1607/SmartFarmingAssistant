# 🌾 Smart Farming Assistant
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import requests
from dotenv import load_dotenv
import os

# Load environment variables (for API key)
load_dotenv()
API_KEY = '0558c8724390fa208bc680de78725fef'

# ---------------------------
# Load models
# ---------------------------
@st.cache_resource
def load_models():
    rf_class = joblib.load('crop_classifier (2).joblib')   # trained with 7 features
    rf_reg = joblib.load('env_regressor (2).joblib')       # trained with 5 features
    le = joblib.load('label_encoder (3).joblib')
    return rf_class, rf_reg, le

rf_class, rf_reg, le = load_models()

# ---------------------------
# Load sample data for visualization
# ---------------------------
@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('crop_recommendation.csv')
    except:
        st.warning("Using dummy data for visualization as dataset file wasn't found.")
        np.random.seed(42)
        N = 2200
        return pd.DataFrame({
            'N': np.random.randint(0, 140, N),
            'P': np.random.randint(5, 145, N),
            'K': np.random.randint(5, 205, N),
            'ph': np.random.uniform(3.5, 10, N),
            'temperature': np.random.uniform(10, 40, N),
            'humidity': np.random.uniform(20, 90, N),
            'rainfall': np.random.uniform(20, 300, N),
            'moisture': np.random.uniform(10, 60, N),
            'label': np.random.choice(le.classes_, N)
        })

df = load_sample_data()

# ---------------------------
# Fetch live weather function
# ---------------------------
def get_weather(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    res = requests.get(url).json()
    if res.get("cod") != 200:
        return None
    return {
        "temperature": res['main']['temp'],
        "humidity": res['main']['humidity'],
        "rainfall": res.get('rain', {}).get('1h', 0.0)
    }

# ---------------------------
# Main app
# ---------------------------
def main():
    st.set_page_config(page_title="Smart Farming Assistant", layout="wide")
    rf_class, rf_reg, le = load_models()
    
    # Sidebar - Location + API
    st.sidebar.header("🌍 Farm Location")
    city_name = st.sidebar.text_input("Enter nearest city:", "Bangalore")
    api_key = API_KEY
   
    
    # Fetch live weather
    weather = None
    if api_key and city_name:
        weather = get_weather(api_key, city_name)
    
    # Main interface
    st.title("🌱 Smart Farming Decision Assistant")
    st.write("Optimize your farming decisions with AI-powered recommendations and **live weather data**")
    
    # Soil inputs
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider('Nitrogen (N) kg/ha', 0, 140, 70)
        P = st.slider('Phosphorous (P) kg/ha', 5, 145, 40)
        K = st.slider('Potassium (K) kg/ha', 5, 205, 50)
    with col2:
        ph = st.slider('pH Value', 3.5, 10.0, 6.5)
        rainfall = st.slider('Rainfall (mm)', 20.0, 300.0, weather['rainfall'] if weather else 100.0)
        moisture = st.slider('Soil Moisture (%)', 10, 60, 35)
    
    # Prediction button
    if st.button("Get Recommendations", type="primary"):
        if weather is None:
            st.error("⚠️ Could not fetch weather. Please check city name or API key.")
        else:
            st.success(f"Weather in {city_name}: 🌡 {weather['temperature']}°C, 💧 {weather['humidity']}%, 🌧 {weather['rainfall']}mm")
            
            # ✅ FIXED: Separate inputs for classifier (7 features) & regressor (5 features)
            input_class = [[N, P, K, ph, weather['rainfall'], weather['temperature'], weather['humidity']]]
            input_reg   = [[N, P, K, ph, weather['rainfall']]]
            
            # Predictions
            crop_num = rf_class.predict(input_class)[0]
            crop_name = le.inverse_transform([crop_num])[0]
            temp_pred, humid_pred = rf_reg.predict(input_reg)[0]
            
            # Results
            st.success(f"## Recommended Crop: {crop_name}")
            
            # Probability chart
            st.subheader("Top Crop Options")
            probs = rf_class.predict_proba(input_class)[0]
            top_n = 5
            top_indices = probs.argsort()[-top_n:][::-1]
            fig, ax = plt.subplots()
            sns.barplot(x=probs[top_indices], y=le.inverse_transform(top_indices), palette="viridis", ax=ax)
            st.pyplot(fig)
            
            # Weather advice
            st.subheader("🌤️ Weather Adaptation")
            cols = st.columns(2)
            cols[0].metric("Expected Temperature", f"{temp_pred:.1f}°C")
            cols[1].metric("Expected Humidity", f"{humid_pred:.1f}%")
            
            # Soil advice
            st.subheader("🌱 Soil Management")
            if ph < 5.5:
                st.warning("Soil is acidic. Consider adding lime.")
            elif ph > 7.5:
                st.warning("Soil is alkaline. Consider sulfur amendments.")
            
            # Market tips (mock)
            st.subheader("💰 Market Insights")
            st.write(f"Current price for {crop_name}: ${np.random.randint(20, 80)}/kg")

    # Data visualization
    st.header('Data Exploration')
    available_features = [col for col in df.columns if col != 'label']
    selected_feature = st.selectbox('Select feature to visualize', available_features)
    selected_crop = st.selectbox('Select crop to filter', ['All'] + list(le.classes_))

    if selected_crop == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['label'] == selected_crop]

    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df[selected_feature], kde=True, ax=ax2)
    ax2.set_title(f'Distribution of {selected_feature} for {selected_crop}')
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
