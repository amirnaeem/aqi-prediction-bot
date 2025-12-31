# City coordinates dictionary
CITIES = {
    "KHI": {
        "name": "Karachi",
        "lat": 24.8607,
        "lon": 67.0011,
        "code": "KHI"
    },
    "ISB": {
        "name": "Islamabad",
        "lat": 33.6844,
        "lon": 73.0479,
        "code": "ISB"
    },
    "LHR": {
        "name": "Lahore",
        "lat": 31.5497,
        "lon": 74.3436,
        "code": "LHR"
    }
}

# Default city (for backward compatibility)
DEFAULT_CITY = "KHI"
LAT = CITIES[DEFAULT_CITY]["lat"]
LON = CITIES[DEFAULT_CITY]["lon"]

# Function to generate URLs for a specific city
def get_air_quality_url(city_code: str = DEFAULT_CITY):
    """Generate air quality API URL for a specific city."""
    city = CITIES.get(city_code, CITIES[DEFAULT_CITY])
    return (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={city['lat']}&longitude={city['lon']}"
    f"&forecast_days=1"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
)

def get_weather_forecast_url(city_code: str = DEFAULT_CITY):
    """Generate weather forecast API URL for a specific city."""
    city = CITIES.get(city_code, CITIES[DEFAULT_CITY])
    return (
    "https://api.open-meteo.com/v1/forecast"
        f"?latitude={city['lat']}&longitude={city['lon']}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    "&forecast_days=1"
)

# Base URLs for latest fetch (backward compatibility - uses default city)
AIR_QUALITY_URL = get_air_quality_url()
WEATHER_FORECAST_URL = get_weather_forecast_url()

# base urls for historical data
from datetime import datetime, timedelta
start_date = "2024-01-01"
end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

def get_historic_air_quality_url(city_code: str = DEFAULT_CITY):
    """Generate historical air quality API URL for a specific city."""
    city = CITIES.get(city_code, CITIES[DEFAULT_CITY])
    return (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={city['lat']}&longitude={city['lon']}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    )

def get_historic_weather_url(city_code: str = DEFAULT_CITY):
    """Generate historical weather API URL for a specific city."""
    city = CITIES.get(city_code, CITIES[DEFAULT_CITY])
    return (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={city['lat']}&longitude={city['lon']}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )

# Backward compatibility
aq_historic_url = get_historic_air_quality_url()
weather_historic_url = get_historic_weather_url()

# Data path
RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed"
HIST_PATH = "data/historical"

import os
SAVE_LOCAL = os.getenv("SAVE_LOCAL", "false").lower() in ("1", "true", "yes")