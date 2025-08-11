# Solar Generation Prediction System

A Django-based web application that predicts solar power generation based on weather data using machine learning models. The system provides interactive visualizations and real-time predictions for solar energy output.

## 🌟 Features

- **Weather-based Solar Prediction**: Uses machine learning models to predict solar generation based on weather conditions
- **Interactive Weather Map**: Visual representation of weather data and predictions
- **Real-time Data Processing**: Handles both historical and new prediction data
- **Comprehensive Data Management**: Tracks prediction history and results
- **Responsive Web Interface**: Clean, user-friendly dashboard for monitoring predictions
- **API Integration**: Fetches real-time weather data from Open-Meteo API
- **Multi-format Data Export**: CSV and JSON output for predictions

## 🛠️ Technology Stack

- **Backend**: Django 4.x
- **Machine Learning**: scikit-learn (pickle models)
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly for interactive charts
- **API**: Open-Meteo Weather API
- **Database**: SQLite (development)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap

## 📁 Project Structure

```
frontend/
├── frontend/              # Django project settings
├── RTS/                  # Main application
│   ├── models.py         # Database models
│   ├── views.py          # Request handlers
│   ├── urls.py           # URL routing
│   ├── utils/            # Utility modules
│   │   ├── prediction.py # ML prediction logic
│   │   └── solar_generation_model.pkl # Trained model
│   ├── templates/        # HTML templates
│   │   ├── base.html     # Base template
│   │   └── weather_map/  # Weather visualization templates
│   └── migrations/       # Database migrations
├── static/               # Static files
│   └── plots/           # Generated visualization files
├── data/                # Data storage
│   ├── new_prediction_data.csv
│   ├── old_prediction_data.csv
│   ├── prediction_history.csv
│   ├── prediction_results.json
│   └── your_weather_data.csv
└── requirements.txt     # Python dependencies
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd frontend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Start development server**
   ```bash
   python manage.py runserver
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:8000`

## 📊 Data Flow

1. **Weather Data Collection**: System fetches real-time weather data from Open-Meteo API
2. **Data Processing**: Raw weather data is processed and cleaned
3. **Feature Engineering**: Creates derived features for ML model
4. **Prediction**: Uses pre-trained model to predict solar generation
5. **Visualization**: Generates interactive plots and charts
6. **Storage**: Saves predictions to CSV/JSON files

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```
DEBUG=True
SECRET_KEY=your-secret-key
```

### Model Configuration
The ML model (`solar_generation_model.pkl`) is pre-trained and located in `RTS/utils/`. Features include:
- Global Horizontal Irradiance (GHI)
- Temperature, humidity, pressure
- Wind speed and cloud cover
- Time-based features (hour, month, day/night cycles)
- Solar position features

## 📈 API Endpoints

- `GET /` - Main dashboard with weather map
- `GET /weather/` - Fetch weather data for given coordinates
- `POST /predict/` - Generate solar generation predictions
- `GET /api/predictions/` - Retrieve prediction results

## 🧪 Usage Examples

### Fetch Weather Data
```javascript
// Example API call
fetch('/weather/?lat=40.7128&lon=-74.0060')
  .then(response => response.json())
  .then(data => console.log(data));
```

### Generate Predictions
```javascript
// Example prediction request
fetch('/confirm-location-and-predict/', {
  method: 'POST',
  body: new FormData(form)
});
```

## 📊 Output Data

The system generates several output files:

1. **your_weather_data.csv**: Raw weather data with coordinates
2. **prediction_results.json**: JSON format prediction results
3. **prediction_history.csv**: Historical predictions for analysis
4. **new_prediction_data.csv**: Latest predictions with features
5. **old_prediction_data.csv**: Previous prediction data

## 🧪 Testing

Run the test suite:
```bash
python manage.py test
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure `solar_generation_model.pkl` exists in `RTS/utils/`
   - Check file permissions

2. **API Connection Issues**
   - Verify internet connection for Open-Meteo API
   - Check rate limits (1000 requests/day for free tier)

3. **Data Processing Errors**
   - Ensure all required columns are present in weather data
   - Check date format consistency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for providing weather data API
- Django community for the excellent web framework
- scikit-learn team for machine learning capabilities

## 📞 Support

For support, email support@solar-prediction.com or create an issue in the GitHub repository.
