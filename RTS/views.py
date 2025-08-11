from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import os
import json
import pandas as pd
import numpy as np
import joblib
import requests
import threading
import logging
from datetime import datetime, timezone
from .utils.prediction import predict_day_generation, preprocess_new_csv

logger = logging.getLogger(__name__)
file_lock = threading.Lock()

def index(request):
    return render(request, 'weather_map/index.html')

def validate_lat_lon(lat, lon):
    try:
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None, None
        return lat, lon
    except (ValueError, TypeError):
        return None, None
def get_weather(request):
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')
    lat, lon = validate_lat_lon(lat, lon)
    if lat is None or lon is None:
        return JsonResponse({'error': 'Invalid latitude or longitude'}, status=400)

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation_instant,temperature_2m,surface_pressure,relative_humidity_2m,"
        f"wind_speed_80m,rain,snowfall,cloud_cover,is_day,apparent_temperature,"
        f"precipitation_probability,showers,precipitation,pressure_msl,visibility,"
        f"soil_temperature_0cm,soil_temperature_6cm,uv_index,uv_index_clear_sky,"
        f"shortwave_radiation,direct_radiation,direct_radiation_instant"
        f"&daily=sunrise,sunset,sunshine_duration,daylight_duration,weather_code,"
        f"temperature_2m_max,shortwave_radiation_sum,uv_index_max,temperature_2m_min"
        f"&current=temperature_2m,relative_humidity_2m,cloud_cover"
        f"&past_days=7&forecast_days=3&timezone=auto"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Weather API request failed: {e}")
        return JsonResponse({'error': 'Failed to fetch weather data'}, status=500)

    data = response.json()
    hourly, daily, current = data.get("hourly", {}), data.get("daily", {}), data.get("current", {})

    # Build hourly dataframe
    df_hourly = pd.DataFrame(hourly)
    df_hourly["Time"] = pd.to_datetime(df_hourly["time"])
    df_hourly["latitude"], df_hourly["longitude"] = lat, lon

    # Extract date from hourly timestamps
    df_hourly["date"] = df_hourly["Time"].dt.date

    # Build daily dataframe with dates
    daily_df = pd.DataFrame(daily)
    daily_df["date"] = pd.to_datetime(daily_df["time"]).dt.date

    # Merge hourly and daily on date
    df_merged = df_hourly.merge(daily_df, on="date", how="left")

    # Build current dataframe with date and merge with df_merged
    current_df = pd.DataFrame([current])
    current_df["date"] = datetime.now(timezone.utc).date()

    # Add current weather fields (same for every hourly row)
    for key, value in current.items():
        df_merged[f"current_{key}"] = value

    # Save the merged data
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'your_weather_data.csv')
    with file_lock:
        df_merged.to_csv(csv_path, index=False)

    # Summarize main weather metrics
    current_temp = current_df["temperature_2m"] if "temperature_2m" in current_df else None
    current_humidity = current_df["relative_humidity_2m"] if "relative_humidity_2m" in current_df else None
    avg_radiation = df_merged["shortwave_radiation_instant"].mean() if "shortwave_radiation_instant" in df_merged else None
    current_cloud_cover = current_df["cloud_cover"] if "cloud_cover" in current_df else None

    # get current hour from device
    current_hour = datetime.now(timezone.utc).hour

    # get isday value for current_hour
    isday = df_merged[df_merged['Time'].dt.hour == current_hour]['is_day'].values[0] if 'is_day' in df_merged.columns else None
    
    weather_summary = {
        'is_day': int(isday) if isday is not None else None,
        'temperature': round(current_temp.iloc[0], 1) if current_temp is not None else "--",
        'humidity': int(current_humidity.iloc[0]) if not current_humidity.empty else "--",
        'radiation': int(avg_radiation) if avg_radiation is not None else "--",
        'cloud_cover': int(current_cloud_cover.iloc[0]) if not current_cloud_cover.empty else "--",
    }

    return JsonResponse({
        'weather': weather_summary,
        'hourly': df_merged.to_dict(orient="list"), #ex- "time": ["2025-07-08T12:00", "2025-07-08T13:00"],"temperature_2m": [30.2, 31.5],
        'daily': daily,
        'current': current,
        'current_date': datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        'message': 'Weather data fetched successfully and saved.'
    })


def run_prediction_in_background(csv_path, model_path='RTS/utils/solar_generation_model.pkl'):
    try:
        with file_lock:
            df = pd.read_csv(csv_path, parse_dates=["Time"])
        logger.info(f"Loaded data for prediction: {df.shape}")

        _, X_df = preprocess_new_csv(df.copy())
        model = joblib.load(model_path)
        hourly_pred = np.clip(model.predict(X_df), 0, None)

        df["predicted_energy_kWh"] = hourly_pred
        with file_lock:
            df.to_csv(csv_path, index=False)
        logger.info("Predictions saved to CSV.")

        df['date'] = df['Time'].dt.date
        daily_energy = df.groupby('date')['predicted_energy_kWh'].sum().reset_index()
        daily_energy['date'] = daily_energy['date'].astype(str)
        today = datetime.now(timezone.utc).date()
        today_energy = daily_energy[daily_energy['date'] == str(today)]['predicted_energy_kWh'].sum()

        prediction = {
            "forecast_energy": {
                "hours": df['Time'].dt.strftime("%H:%M").tolist(),
                "predicted_energy": df['predicted_energy_kWh'].round(2).tolist(),
            },
            "daily_energy": {
                "dates": daily_energy['date'].tolist(),
                "predicted_energy_kWh": daily_energy['predicted_energy_kWh'].round(2).tolist(),
            },
            "total_energy": round(today_energy, 2),
        }

        os.makedirs('data', exist_ok=True)
        with file_lock:
            with open('data/prediction_results.json', 'w') as f:
                json.dump(prediction, f)
        logger.info("Prediction results saved to JSON.")

        history_csv = 'data/prediction_history.csv'
        daily_energy['latitude'], daily_energy['longitude'] = df['latitude'].iloc[0], df['longitude'].iloc[0]
        daily_energy['prediction_time'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        daily_energy['total_energy'] = today_energy
        with file_lock:
            if not os.path.exists(history_csv):
                daily_energy.to_csv(history_csv, index=False, mode='w')
            else:
                daily_energy.to_csv(history_csv, index=False, header=False, mode='a')
        logger.info(f"Prediction appended to history CSV: {history_csv}")
    except Exception as e:
        logger.error(f"[Prediction error]: {str(e)}", exc_info=True)

# def get_prediction_results(request):
#     result_path = 'data/prediction_results.json'
#     if not os.path.exists(result_path):
#         return JsonResponse({'status': 'pending'}, status=202)
#     try:
#         with file_lock:
#             with open(result_path, 'r') as f:
#                 prediction = json.load(f)
#         return JsonResponse({'status': 'ready', 'prediction': prediction})
#     except Exception as e:
#         logger.error(f"Failed to fetch results: {e}", exc_info=True)
#         return JsonResponse({'error': f"Failed to fetch results: {str(e)}"}, status=500)

def confirm_location_and_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)
    
    lat, lon, date_str, latest_weather_data = request.POST.get('lat'), request.POST.get('lon'), request.POST.get('date'), request.POST.get('latestWeatherData')
    latest_weather_data = json.loads(latest_weather_data) # Convert string to dict
    lat, lon = validate_lat_lon(lat, lon)
    if lat is None or lon is None or not date_str:
        return JsonResponse({'error': 'Invalid parameters'}, status=400)

    try:
        df = pd.DataFrame(latest_weather_data['hourly'])
        df["Time"] = pd.to_datetime(df["Time"])
        df["latitude"], df["longitude"] = lat, lon

        selected_date = pd.to_datetime(date_str).date()
        # df_selected = df[df['Time'].dt.date == selected_date].copy()
        # if df_selected.empty:
        #     return JsonResponse({'error': f"No data found for date {date_str}."}, status=400)

        model_path = os.path.join('RTS', 'utils', 'solar_generation_model.pkl')
        total_energy, X_df = predict_day_generation(df, model_path)
        total_energy = float(total_energy)

        current_df = pd.DataFrame([latest_weather_data['current']])
        current_temp = current_df["temperature_2m"] if "temperature_2m" in current_df else None
        current_humidity = current_df["relative_humidity_2m"] if "relative_humidity_2m" in current_df else None
        avg_radiation = df["shortwave_radiation_instant"].mean() if "shortwave_radiation_instant" in df else None
        current_cloud_cover = current_df["cloud_cover"] if "cloud_cover" in current_df else None

        # get current hour from device
        current_hour = datetime.now(timezone.utc).hour

        # get isday value for current_hour
        isday = df[df['Time'].dt.hour == current_hour]['is_day'].values[0] if 'is_day' in df.columns else None
    

        weather = {
            'is_day': int(isday) if isday is not None else None,
            'temperature': round(current_temp.iloc[0], 1) if current_temp is not None else "--",
            'humidity': int(current_humidity.iloc[0]) if not current_humidity.empty else "--",
            'radiation': int(avg_radiation) if avg_radiation is not None else "--",
            'cloud_cover': int(current_cloud_cover.iloc[0]) if not current_cloud_cover.empty else "--",
        }


        if 'predicted_energy_kWh' not in X_df.columns:
            _, X_df = preprocess_new_csv(X_df.copy())
            model = joblib.load(model_path)
            hourly_pred = np.clip(model.predict(X_df), 0, None)
            X_df["predicted_energy_kWh"] = hourly_pred

        hourly_energy = X_df[['Time', 'predicted_energy_kWh', 'GHI']].copy() # Add GHI column to hourly_energy
        
        # Check if Time is already datetime, if not convert it
        if not pd.api.types.is_datetime64_any_dtype(hourly_energy['Time']):
            hourly_energy['Time'] = pd.to_datetime(hourly_energy['Time'])
            
        # Now safely use dt accessor
        hourly_energy['Time'] = hourly_energy['Time'].dt.strftime("%H:%M")

        # Check if Time is already datetime, if not convert it
        if not pd.api.types.is_datetime64_any_dtype(X_df['Time']):
            X_df['Time'] = pd.to_datetime(X_df['Time'])

        X_df['date'] = X_df['Time'].dt.date
        if 'predicted_energy_kWh' in X_df.columns:
            daily_energy = X_df.groupby('date')['predicted_energy_kWh'].sum().reset_index()
            daily_energy['date'] = daily_energy['date'].astype(str)
        else:
            daily_energy = pd.DataFrame({'date': [selected_date], 'predicted_energy_kWh': [total_energy]})
            daily_energy['date'] = daily_energy['date'].astype(str)

        # Create prediction results JSON
        prediction = {
            "forecast_energy": {
                "hours": hourly_energy['Time'].tolist(),
                "predicted_energy": hourly_energy['predicted_energy_kWh'].round(2).tolist(),
                "GHI": hourly_energy['GHI'].round(2).tolist(), # Add GHI to predicted_energy list
            },
            "daily_energy": {
                "dates": daily_energy['date'].tolist(),
                "predicted_energy_kWh": daily_energy['predicted_energy_kWh'].round(2).tolist(),
            },
            "total_energy": float(round(total_energy, 2)),
            "weather": {
                'is_day': int(isday) if isday is not None else None,
                'temperature': float(round(current_temp.iloc[0], 1)) if current_temp is not None else "--",
                'humidity': int(current_humidity.iloc[0]) if not current_humidity.empty else "--",
                'radiation': int(avg_radiation) if avg_radiation is not None else "--",
                'cloud_cover': int(current_cloud_cover.iloc[0]) if not current_cloud_cover.empty else "--",
            },
        }

        os.makedirs('data', exist_ok=True)
        with file_lock:
            with open('data/prediction_results.json', 'w') as f:
                json.dump(prediction, f)
        logger.info("Prediction results saved to JSON.")

        # return JsonResponse({'status': 'success', 'message': 'Prediction started'})
        return JsonResponse({'status': 'success', 'prediction': prediction})

    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.error(f"Error in confirm_location_and_predict: {e}", exc_info=True)
        return JsonResponse({'error': f"Unexpected error: {str(e)}"}, status=500)
