import os
import time
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def preprocess_new_csv(df):
    # Rename time columns
    if 'Time' not in df.columns and 'time' in df.columns:
        df = df.rename(columns={"time": "Time"})
    if 'Time' in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
    else:
        df["Time"] = pd.to_datetime(df.index)

    # Rename weather columns
    df = df.rename(columns={
        "shortwave_radiation_instant": "GHI",
        "temperature_2m": "temp",
        "surface_pressure": "pressure",
        "relative_humidity_2m": "humidity",
        "wind_speed_80m": "wind_speed",
        "rain": "rain_1h",
        "snowfall": "snow_1h",
        "cloud_cover": "clouds_all",
        "is_day": "isSun",
    })

    # Derived features
    df["dayLength"] = df["daylight_duration"]
    df["sunlightTime"] = df["sunshine_duration"]
    df["SunlightTime/daylength"] = (df["sunlightTime"] / df["dayLength"]).clip(0, 1)
    df["hour"] = df["Time"].dt.hour
    df["month"] = df["Time"].dt.month
    df["weather_type"] = df["weather_code"]

    # Cyclical time features
    df["hour_norm"] = df["hour"] + df["Time"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_norm"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_norm"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Time blocks
    def categorize_time(h):
        if 20 <= h or h < 3: return 0
        elif 3 <= h < 5: return 1
        elif 5 <= h < 9: return 2
        elif 9 <= h < 11: return 3
        elif 11 <= h < 13: return 4
        elif 13 <= h < 15: return 5
        elif 15 <= h < 17: return 6
        elif 17 <= h < 20: return 7
        else: return 8
    df["time_block"] = df["hour_norm"].apply(categorize_time)

    df["low_ghi"] = (df["GHI"] < 15).astype(int)
    df["is_night"] = 1 - df["isSun"]

    # Drop unnecessary columns
    drop_cols = [
        "Time", "time", "time_y", "sunrise", "sunset", "apparent_temperature",
        "precipitation_probability", "showers", "precipitation", "pressure_msl", "visibility",
        "soil_temperature_0cm", "soil_temperature_6cm", "uv_index", "uv_index_clear_sky",
        "shortwave_radiation", "direct_radiation", "direct_radiation_instant",
        "sunshine_duration", "daylight_duration", "weather_code", "weather_type", "sunlightTime","latitude", "longitude"
    ]
    X_df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Expected features in model input
    expected_columns = [
        'GHI', 'temp', 'pressure', 'humidity', 'wind_speed', 'rain_1h', 'snow_1h',
        'clouds_all', 'isSun', 'dayLength', 'SunlightTime/daylength', 'hour', 'month',
        'hour_norm', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'time_block',
        'low_ghi', 'is_night'
    ]
    for col in expected_columns:
        if col not in X_df.columns:
            raise ValueError(f"Missing expected column: {col}")
    X_df = X_df[expected_columns]

    return df, X_df

def predict_day_generation(df, model_path='RTS/utils/solar_generation_model.pkl'):
    try:
        df_copy = df.copy()
        df_preprocessed, X = preprocess_new_csv(df_copy)
        os.makedirs('data', exist_ok=True)
        df_preprocessed['Time'] = df_preprocessed['Time'].dt.strftime("%Y-%m-%d %H:%M:%S") # Convert to string for CSV export
        model = joblib.load(model_path)
        y_pred = np.clip(model.predict(X), 0, None)
        df_preprocessed["predicted_energy_kWh"] = y_pred
        df_preprocessed.to_csv('data/new_prediction_data.csv', index=False)
        logger.info("Predictions saved to CSV.")

        X['predicted_energy_kWh'] = y_pred
        X['Time'] = df_preprocessed['Time'].values
        cols = ['Time'] + [col for col in X.columns if col != 'Time']
        X = X[cols]
        X.to_csv('data/old_prediction_data.csv', index=False)
        total_energy = float(df_preprocessed["predicted_energy_kWh"].sum())

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(
        #     x=df_preprocessed["Time"], y=df_preprocessed["predicted_energy_kWh"],
        #     mode='lines+markers', name='Predicted Energy (kWh)',
        #     line=dict(color='#28a745', width=2), marker=dict(size=6)
        # ))
        # fig.add_trace(go.Scatter(
        #     x=df_preprocessed["Time"], y=df_preprocessed["GHI"],
        #     mode='lines', name='Solar Radiation (W/m²)',
        #     line=dict(color='#ffc107', width=1.5, dash='dot'), yaxis='y2'
        # ))
        # fig.update_layout(
        #     title='Solar Energy Generation Prediction',
        #     xaxis_title='Time',
        #     yaxis_title='Energy Generated (kWh)',
        #     yaxis2=dict(title='Solar Radiation (W/m²)', overlaying='y', side='right'),
        #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        #     hovermode='x unified', template='plotly_white', margin=dict(l=50, r=50, t=80, b=50)
        # )

        # static_dir = os.path.join(settings.BASE_DIR, 'static', 'plots')
        # os.makedirs(static_dir, exist_ok=True)
        # plot_filename = f"prediction_plot_{int(time.time())}.html"
        # plot_path = os.path.join(static_dir, plot_filename)
        # fig.write_html(plot_path)

        return total_energy,X #, f"static/plots/{plot_filename}"

    except Exception as e:
        logger.error(f"Error in predict_day_generation: {e}", exc_info=True)
        raise ValueError(f"Prediction failed: {str(e)}")
