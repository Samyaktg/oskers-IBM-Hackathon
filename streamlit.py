import streamlit as st
import requests
import datetime
import pandas as pd
import numpy as np
import joblib

# Load the trained LSTM model
model_filename = 'lstm_weather_model.pkl'
model = joblib.load(model_filename)

# Function to get historical weather data from WeatherAPI
def get_historical_weather(api_key, location, date):
    url = f"http://api.weatherapi.com/v1/history.json"
    
    params = {
        'key': api_key,
        'q': location,
        'dt': date,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
        return None

# Function to get weather data for the past year
def get_weather_past_year(api_key, location):
    weather_data_list = []
    
    for i in range(365):
        date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
        weather_data = get_historical_weather(api_key, location, date)
        
        if weather_data:
            forecast_day = weather_data.get('forecast', {}).get('forecastday', [])[0].get('day', {})
            precipitation = forecast_day.get('totalprecip_mm', 0)
            weather_data_list.append({
                'date': date,
                'precipitation_mm': precipitation,
            })
    
    return weather_data_list

# Function to display weather data and make predictions
def display_weather_data(location):
    # Your WeatherAPI API Key
    api_key = 'f446c70ff0e146d9b6b160659241910'
    
    # Get weather data for the past year
    weather_data = get_weather_past_year(api_key, location)
    
    # Convert to DataFrame
    weather_df = pd.DataFrame(weather_data)
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    weather_df.set_index('date', inplace=True)
    
    # Prepare data for prediction
    features = ['avg_temp_c', 'humidity', 'wind_speed_kph', 'day_of_week', 'month', 'season']
    
    # Ensure the features are present in the DataFrame
    if all(feature in weather_df.columns for feature in features):
        X = weather_df[features].values
        X_scaled = scaler.transform(X)  # Use the same scaler used during training
        
        # Create sequences for LSTM
        sequence_length = 10
        X_sequences = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:i + sequence_length])
        
        X_sequences = np.array(X_sequences)
        
        # Make predictions
        predicted_diff = model.predict(X_sequences)
        
        # Display results
        st.write(f"### Weather Prediction for {location}")
        st.write(f"**Predicted Precipitation Difference:** {predicted_diff[-1][0]} mm")
    else:
        st.error("Required features are missing from the weather data.")

# Streamlit UI
st.title("Weather Prediction App")
location = st.text_input("Enter a location (e.g., Mumbai, India):", "Mumbai, India")

if st.button("Get Weather Prediction"):
    display_weather_data(location)