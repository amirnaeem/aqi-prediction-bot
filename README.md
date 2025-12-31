# Multi-City AQI Prediction Dashboard

> **An end-to-end machine learning pipeline for real-time air quality forecasting across multiple cities using Open-Meteo APIs, Hopsworks Feature Store, and Streamlit.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Hopsworks](https://img.shields.io/badge/Feature%20Store-Hopsworks-green)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-yellow)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Overview

The **Multi-City AQI Prediction Dashboard** is a fully automated air quality prediction system that forecasts Air Quality Index (AQI) for the **current day and next three days** across multiple cities in Pakistan. The system integrates data from **Open-Meteo's Air Quality and Weather APIs**, computes pollutant-based AQI following **U.S. EPA standards**, and continuously updates the data pipeline through **GitHub Actions CI/CD**.

The system automates every stage — from **data ingestion and feature engineering** to **model training, prediction, and visualization** — while storing all processed data in the **Hopsworks Feature Store**.

### Supported Cities

- **Karachi (KHI)** - 24.8607°N, 67.0011°E
- **Islamabad (ISB)** - 33.6844°N, 73.0479°E
- **Lahore (LHR)** - 31.5497°N, 74.3436°E

---

## Technologies Used

- **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
- **Hopsworks Feature Store** - Cloud-based feature storage and serving
- **Open-Meteo APIs** - Air Quality, Weather Forecast, Historical Archive
- **GitHub Actions** - Automated CI/CD pipelines
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive data visualization
- **Lucide Icons** - Modern icon library

---

## Key Features

✅ **Multi-City Support** - Monitor and predict AQI for Karachi, Islamabad, and Lahore  
✅ **Real-time Data Fetching** - Automated data ingestion from Open-Meteo APIs  
✅ **Scientific AQI Computation** - U.S. EPA (2016) methodology for accurate calculations  
✅ **Automated Pipeline** - Daily CI/CD workflows for data ingestion and model retraining  
✅ **Interactive Dashboard** - Professional Streamlit interface with unified charts  
✅ **Dark Mode Support** - Automatic theme detection with responsive UI  
✅ **City-Specific Models** - Support for both unified and city-specific ML models  
✅ **Unified Visualizations** - Compare multiple cities in single interactive charts  
✅ **Feature Engineering** - Time-based, cyclic, and ratio-based features  
✅ **Production-Ready** - Modular, scalable architecture

---

## System Architecture

```
┌─────────────────────┐
│  Open-Meteo APIs    │
│ (Air + Weather Data)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Processing &   │
│ Feature Engineering │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Hopsworks Feature   │
│       Store         │
│  (Multi-City Data)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Model Training &   │
│   Evaluation        │
│ (City-Specific or   │
│    Unified)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Streamlit Dashboard │
│  (3-Day Forecast)   │
│  Multi-City View    │
└─────────────────────┘
```

---

## Folder Structure

```
aqi-prediction-bot/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Daily feature ingestion pipeline
│       └── training_pipeline.yml  # Daily model retraining pipeline
│
├── data/                           # Local data storage (optional)
│   ├── final/
│   ├── historical/
│   ├── processed/
│   └── predictions/
│
├── models/                         # Trained ML models
│   ├── best_model_random_forest.pkl
│   ├── best_model_xgboost.pkl
│   └── [city-specific models]
│
├── notebooks/                      # Jupyter notebooks for EDA
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_eda_feature_analysis.ipynb
│
├── reports/                        # Project documentation
│   └── Final Report.pdf
│
├── src/                            # Source code
│   ├── aqi_utils.py               # EPA-based AQI computation
│   ├── backfill_data.py           # Historical data fetching
│   ├── clean_data.py              # Data cleaning utilities
│   ├── config.py                  # Configuration (cities, API URLs)
│   ├── fetch_data.py              # API data fetching
│   ├── process_data.py            # JSON to DataFrame conversion
│   ├── process_features.py        # Feature engineering
│   ├── merge_features.py          # Data merging utilities
│   ├── upload_to_hopswork.py     # Hopsworks upload
│   ├── run_feature_pipeline.py   # End-to-end pipeline orchestration
│   ├── train_model.py            # Model training (city-specific or unified)
│   └── predict_evaluate.py       # Prediction and evaluation
│
├── streamlit_app/                  # Streamlit dashboard
│   └── app.py                     # Main dashboard application
│
├── .env                            # Environment variables (not in git)
├── requirements.txt               # Python dependencies
├── run_frontend.bat              # Windows launcher script
├── MULTI_CITY_GUIDE.md           # Multi-city usage guide
└── README.md                      # This file
```

---

## Data & Methodology

**Data Source:** [Open-Meteo Air Quality & Weather APIs](https://open-meteo.com/)  
**Reference Document:** *U.S. EPA Technical Assistance Document for Reporting AQI (May 2016)*  
**Pollutants Used:** PM₂.₅, PM₁₀, NO₂, CO, SO₂, O₃  
**Weather Parameters:** Temperature, Humidity, Wind Speed, Wind Direction

### AQI Computation

Each pollutant's AQI is computed using EPA's official breakpoint interpolation formula. The overall AQI = **max(sub-indexes)** per hour.

### AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0–50 | Good | Air quality is satisfactory |
| 51–100 | Moderate | Acceptable for most people |
| 101–150 | Unhealthy for Sensitive Groups | Sensitive groups may experience health effects |
| 151–200 | Unhealthy | Everyone may begin to experience health effects |
| 201–300 | Very Unhealthy | Health alert: everyone may experience serious effects |
| 301–500 | Hazardous | Health warning: emergency conditions |

---

## Setup Guide

### Prerequisites

- Python 3.10 or higher
- Hopsworks account and API key
- Git

### Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/amirnaeem/aqi-prediction-bot.git
   cd aqi-prediction-bot
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**

   Create a `.env` file in the root directory:
   ```env
   HOPSWORKS_API_KEY=your_api_key_here
   SAVE_LOCAL=false
   ```

### Running the Pipeline

#### Fetch Data for All Cities
```bash
python src/run_feature_pipeline.py
```

#### Fetch Data for a Specific City
```bash
python src/run_feature_pipeline.py KHI    # Karachi
python src/run_feature_pipeline.py ISB    # Islamabad
python src/run_feature_pipeline.py LHR    # Lahore
```

#### Train Models

**Option A: City-Specific Models (Recommended)**
```bash
python src/train_model.py KHI
python src/train_model.py ISB
python src/train_model.py LHR
```

**Option B: Unified Model (All Cities)**
```bash
python src/train_model.py
```

#### Launch Dashboard
```bash
cd streamlit_app
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

---

## Dashboard Features

### Current AQI Overview
- Real-time AQI cards for all supported cities
- Color-coded status indicators
- Quick health impact assessment

### Forecast Analysis
- **Unified Hourly Trend Chart** - Compare 72-hour predictions across cities
- **Daily Average Forecast** - Bar chart showing daily averages
- **Forecast Summary Table** - Detailed daily predictions
- **24-Hour Trend Analysis** - Expected changes in air quality

### Interactive Features
- Multi-city selection with checkboxes
- Dark mode support (automatic theme detection)
- Responsive design with modern UI
- Professional typography and spacing

---

## Model Performance

| Model | RMSE (Test) | R² | Use Case |
|-------|-------------|----|----------| 
| Ridge Regression | 10.4 | 0.93 | Linear baseline |
| XGBoost | 7.9 | 0.97 | High accuracy, risk of overfitting |
| **Random Forest** | **6.59** | **0.99** | ✅ **Best overall performance** |

### Model Selection Strategy

- **City-Specific Models**: Better accuracy for individual cities, captures local patterns
- **Unified Models**: Better generalization, useful when city-specific data is limited

---

## CI/CD Automation

| Pipeline | Schedule | Purpose |
|----------|----------|---------|
| `feature_pipeline.yml` | 8:10 AM PKT (03:10 UTC) | Fetches new data & uploads to Hopsworks |
| `training_pipeline.yml` | 8:30 AM PKT (03:30 UTC) | Retrains ML model on updated data |

Both workflows run automatically via **GitHub Actions**, ensuring daily data and model freshness.

---

## Feature Engineering

The system generates comprehensive features including:

- **Time-based Features**: Hour, day, month, day of week
- **Cyclic Features**: Sinusoidal transformations for temporal patterns
- **Ratio Features**: PM ratios, temperature-humidity ratios
- **Derived Metrics**: AQI change rate, wind effect
- **Weather Features**: Temperature, humidity, wind speed/direction

Total feature count: **23+ features** (after leakage removal)

---

## Multi-City Support

The system supports monitoring and prediction for multiple cities:

1. **Data Storage**: All city data stored in Hopsworks Feature Group `aqi_features` (version 3)
2. **Primary Key**: Composite key `(datetime_str, city)` for multi-city data
3. **City Identification**: Each row includes `city` (code) and `city_name` columns
4. **Model Flexibility**: Support for both city-specific and unified models

For detailed multi-city usage instructions, see [MULTI_CITY_GUIDE.md](MULTI_CITY_GUIDE.md).

---

## Troubleshooting

### "No data found for [City]"
- Run the feature pipeline for that city: `python src/run_feature_pipeline.py [CITY_CODE]`
- Verify data exists in Hopsworks Feature Store

### "Could not load model"
- Train a model: `python src/train_model.py [CITY_CODE]` (city-specific)
- Or use unified model: `python src/train_model.py`
- Check that model files exist in `models/` directory

### Dashboard shows errors
- Verify `HOPSWORKS_API_KEY` is set in `.env` file
- Ensure feature pipeline has been run at least once
- Check network connectivity to Hopsworks

### API connection issues
- Verify Open-Meteo APIs are accessible
- Check internet connection
- Review API rate limits

---

## Future Enhancements

- 🧠 **SHAP-based Interpretability** - Model explainability features
- 🌆 **Additional Cities** - Expand to more Pakistani cities
- ☁️ **Cloud Deployment** - Streamlit Cloud / HuggingFace Spaces
- 📱 **Alert System** - SMS/Email notifications for high pollution
- 📊 **Historical Analysis** - Long-term trends and patterns
- 🔄 **Real-time Updates** - WebSocket-based live updates

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Author

**Credit -> Mariam Khan**  
*B.S. Computer Science — Internship Project*  
📍 Karachi, Pakistan  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/mariam-khan0424)
🔗 [github Repo](https://github.com/mariamkhan04/AQI-Prediction-Bot)

---

## License

This project is licensed under the **MIT License** — feel free to use and modify with attribution.

---

## Acknowledgments

- **Open-Meteo** for providing free air quality and weather APIs
- **Hopsworks** for the feature store platform
- **U.S. EPA** for AQI computation standards
- **Streamlit** for the dashboard framework
