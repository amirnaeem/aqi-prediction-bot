"""
Multi-City AQI Prediction Dashboard
A professional dashboard for monitoring and predicting Air Quality Index across multiple cities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import os
import sys
from joblib import load, dump
from datetime import timedelta, datetime
from dotenv import load_dotenv
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path for config import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from src module
from src.config import CITIES, get_air_quality_url, get_weather_forecast_url
from src.fetch_data import fetch_api_data
from src.process_data import process_latest_json
from src.clean_data import clean_data
from src.process_features import add_features
from src.upload_to_hopswork import upload_to_hopsworks

# ============================================================================
# CONFIGURATION
# ============================================================================

PAGE_CONFIG = {
    "page_title": "Multi-City AQI Prediction Dashboard",
    "page_icon": "🌤",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

CITY_COLORS = {
    "KHI": "#667eea",  # Purple
    "ISB": "#10b981",  # Green
    "LHR": "#f59e0b",  # Orange
}

# ============================================================================
# LUCIDE ICONS HELPER
# ============================================================================

def get_lucide_icon(icon_name: str, size: int = 20, color: str = "currentColor", stroke_width: float = 2.0) -> str:
    """
    Get Lucide icon SVG as HTML string.
    
    Args:
        icon_name: Name of the Lucide icon (e.g., 'cloud', 'wind', 'activity')
        size: Icon size in pixels
        color: Icon color (default: currentColor)
        stroke_width: Stroke width of the icon
        
    Returns:
        HTML string with inline SVG icon
    """
    # Using Lucide CDN for icon SVGs
    # In production, you might want to bundle these SVGs locally
    icon_map = {
        "cloud": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>',
        "wind": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>',
        "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="m22 12-4-4-3 3V3h-2v8l-3-3-4 4 4 4 3-3v8h2v-8l3 3 4-4Z"/></svg>',
        "map-pin": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>',
        "bar-chart": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
        "line-chart": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
        "calendar": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/></svg>',
        "filter": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
        "alert-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>',
        "check-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "info": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="16" y2="12"/><line x1="12" x2="12.01" y1="8" y2="8"/></svg>',
        "trending-up": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
        "trending-down": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
        "minimize-2": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" x2="21" y1="10" y2="3"/><line x1="3" x2="10" y1="21" y2="14"/></svg>',
        "table": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/></svg>',
        "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
        "help-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 24 24" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
    }
    
    svg_template = icon_map.get(icon_name.lower(), icon_map["help-circle"])
    return svg_template.format(size, size, color, stroke_width)

def icon_html(icon_name: str, size: int = 20, color: str = "currentColor", class_name: str = "") -> str:
    """Get icon as HTML with inline styling."""
    svg = get_lucide_icon(icon_name, size, color)
    return f'<span class="icon {class_name}" style="display: inline-flex; align-items: center; vertical-align: middle; margin-right: 0.5rem;">{svg}</span>'

# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(**PAGE_CONFIG)

# ============================================================================
# STYLES
# ============================================================================

st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/normalize.css@8.0.1/normalize.min.css">
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* CSS Variables for Light/Dark Mode */
        :root {
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --shadow-sm: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            --shadow-md: 0 10px 25px rgba(0,0,0,0.1);
            --shadow-lg: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
        }
        
        /* Dark Mode Variables */
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #0f172a;
                --bg-secondary: #1e293b;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --border-color: #334155;
                --shadow-sm: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -1px rgba(0,0,0,0.2);
                --shadow-md: 0 10px 25px rgba(0,0,0,0.4);
                --shadow-lg: 0 20px 25px -5px rgba(0,0,0,0.4), 0 10px 10px -5px rgba(0,0,0,0.3);
            }
        }
        
        /* Streamlit Dark Mode Detection */
        [data-testid="stAppViewContainer"] {
            color-scheme: light dark;
        }
        
        /* Global Styles */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 0;
        }
        
        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Header */
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 1rem;
            border-radius: 0 0 24px 24px;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        .main-title {
            text-align: center;
            color: #ffffff;
            font-weight: 800;
            font-size: 2.5rem;
            margin: 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }
        
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.95);
            font-size: 1.0625rem;
            margin-top: 0.75rem;
            font-weight: 400;
            letter-spacing: 0.01em;
        }
        
        /* Icon Styles */
        .icon {
            display: inline-flex;
            align-items: center;
            vertical-align: middle;
        }
        
        .icon svg {
            flex-shrink: 0;
        }
        
        /* City Cards */
        .city-card {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid var(--border-color);
            height: 100%;
            position: relative;
            overflow: hidden;
        }
        
        .city-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .city-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
            border-color: #667eea;
        }
        
        .city-card:hover::before {
            opacity: 1;
        }
        
        .city-name {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .aqi-value {
            font-size: 2.75rem;
            font-weight: 800;
            margin: 0.75rem 0;
            line-height: 1.1;
        }
        
        .aqi-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        
        /* Metric Card */
        .metric-card {
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: var(--shadow-sm);
            margin: 1.5rem 0;
            border: 1px solid var(--border-color);
        }
        
        /* Chart Card */
        .chart-card {
            background: var(--bg-secondary);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: var(--shadow-sm);
            margin: 1.5rem 0;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .chart-card:hover {
            box-shadow: var(--shadow-md);
        }
        
        .chart-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Section Header */
        .section-header {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 3rem 0 1.5rem 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--border-color);
        }
        
        /* Subsection Header */
        .subsection-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 2rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Section Divider */
        .section-divider {
            margin: 3rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(to right, transparent, var(--border-color), transparent);
        }
        
        /* AQI Badge */
        .aqi-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 5rem;
            padding: 2.5rem 0;
            border-top: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            font-weight: 400;
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 3rem 1rem;
            color: var(--text-secondary);
        }
        
        /* Dark mode adjustments for Streamlit components */
        @media (prefers-color-scheme: dark) {
            .stDataFrame {
                background-color: var(--bg-secondary);
            }
            
            .stSelectbox label,
            .stCheckbox label {
                color: var(--text-primary);
            }
        }
        
        /* Status Indicators */
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: 500;
            margin: 0.75rem 0;
            border-left: 4px solid;
            transition: all 0.2s ease;
        }
        
        .status-indicator:hover {
            transform: translateX(4px);
        }
        
        .status-warning {
            background-color: #fef3c7;
            color: #92400e;
            border-left-color: #f59e0b;
        }
        
        .status-info {
            background-color: #dbeafe;
            color: #1e40af;
            border-left-color: #3b82f6;
        }
        
        .status-success {
            background-color: #d1fae5;
            color: #065f46;
            border-left-color: #10b981;
        }
        
        /* Filter Section */
        .filter-section {
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            border: 1px solid var(--border-color);
        }
        
        /* Container Spacing */
        .content-container {
            max-width: 100%;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        /* Typography Improvements */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 700;
            line-height: 1.2;
            color: var(--text-primary);
        }
        
        /* Better spacing for Streamlit components */
        .stMarkdown {
            margin-bottom: 1rem;
        }
        
        /* Improved checkbox styling */
        .stCheckbox > label {
            font-weight: 500;
            color: var(--text-primary);
        }
        
        /* Table styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Improved spacing between sections */
        .spacing-section {
            margin: 2.5rem 0;
        }
        
        /* Info text styling */
        .info-text {
            color: var(--text-secondary);
            font-size: 0.875rem;
            font-weight: 400;
            line-height: 1.6;
        }
        
        /* Hide Streamlit default elements */
        # #MainMenu {visibility: hidden;}
        # footer {visibility: hidden;}
        # header {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        
        /* Loading spinner */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(102, 126, 234, 0.3);
            border-radius: 50%;
            border-top-color: #667eea;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_aqi_category(aqi_value: float) -> tuple:
    """
    Get AQI category, color, and background color.
    
    Args:
        aqi_value: Air Quality Index value
        
    Returns:
        Tuple of (category_name, text_color, bg_color)
    """
    if pd.isna(aqi_value) or aqi_value is None:
        return "Unknown", "#94a3b8", "#f1f5f9"
    if aqi_value <= 50:
        return "Good", "#10b981", "#d1fae5"
    elif aqi_value <= 100:
        return "Moderate", "#f59e0b", "#fef3c7"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive", "#f97316", "#fed7aa"
    elif aqi_value <= 200:
        return "Unhealthy", "#ef4444", "#fee2e2"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#a855f7", "#f3e8ff"
    else:
        return "Hazardous", "#78716c", "#e7e5e4"

def prepare_features(df: pd.DataFrame, drop_city_cols: bool = True) -> tuple:
    """
    Prepare features for prediction, handling edge cases.
    
    Args:
        df: Input DataFrame with features
        drop_city_cols: Whether to drop city identifier columns
        
    Returns:
        Tuple of (feature_matrix, cleaned_dataframe) or None if invalid
    """
    if df is None or df.empty:
        return None
    
    df_clean = df.copy()
    
    # Handle datetime
    if "datetime_str" in df_clean.columns:
        df_clean["datetime"] = pd.to_datetime(df_clean["datetime_str"], errors="coerce")
        df_clean.drop(columns=["datetime_str"], inplace=True)
    
    if "datetime" not in df_clean.columns:
        return None
    
    # Remove rows with invalid datetime
    df_clean = df_clean.dropna(subset=["datetime"])
    df_clean = df_clean.sort_values("datetime").reset_index(drop=True)
    
    if df_clean.empty:
        return None
    
    # Drop leakage features
    leakage_features = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
    df_clean.drop(
        columns=[col for col in leakage_features if col in df_clean.columns],
        inplace=True,
        errors="ignore"
    )
    
    # Drop city columns if needed
    if drop_city_cols:
        city_cols = ["city", "city_name"]
        df_clean.drop(
            columns=[col for col in city_cols if col in df_clean.columns],
            inplace=True,
            errors="ignore"
        )
    
    # Drop target and datetime for features
    X = df_clean.drop(columns=["aqi", "datetime"], errors="ignore")
    
    # Remove columns with all NaN
    X = X.dropna(axis=1, how="all")
    
    # Fill remaining NaN with forward fill then backward fill
    X = X.ffill().bfill()
    
    # If still NaN, fill with 0
    X = X.fillna(0)
    
    if X.empty or len(X.columns) == 0:
        return None
    
    return X, df_clean

# ============================================================================
# DATA LOADING
# ============================================================================

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    st.markdown(f"{icon_html('alert-circle', 20, '#ef4444')} <span style='color: #ef4444;'>Missing HOPSWORKS_API_KEY in environment. Please check your .env file.</span>", unsafe_allow_html=True)
    st.stop()

# ============================================================================
# DATA FETCHING AND MODEL TRAINING FUNCTIONS
# ============================================================================

def run_data_fetch_direct():
    """Fetch new data for all cities directly."""
    try:
        success_count = 0
        error_messages = []
        
        for city_code in CITIES.keys():
            try:
                city_name = CITIES[city_code]["name"]
                
                # Fetch raw data
                aq_url = get_air_quality_url(city_code)
                wx_url = get_weather_forecast_url(city_code)
                
                aq_json = fetch_api_data(aq_url)
                wx_json = fetch_api_data(wx_url)
                
                if aq_json is None or wx_json is None:
                    error_messages.append(f"{city_name}: Failed to fetch API data")
                    continue
                
                if "hourly" not in aq_json or "hourly" not in wx_json:
                    error_messages.append(f"{city_name}: Invalid API response structure")
                    continue
                
                # Convert to DataFrames
                aq_df = pd.DataFrame(aq_json["hourly"])
                wx_df = pd.DataFrame(wx_json["hourly"])
                
                if aq_df.empty or wx_df.empty:
                    error_messages.append(f"{city_name}: Empty dataframes")
                    continue
                
                # Merge datasets
                if "time" in aq_df.columns and "time" in wx_df.columns:
                    aq_df.rename(columns={"time": "datetime"}, inplace=True)
                    wx_df.rename(columns={"time": "datetime"}, inplace=True)
                
                raw_df = pd.merge(aq_df, wx_df, on="datetime", how="inner")
                raw_df["city"] = city_code
                raw_df["city_name"] = city_name
                
                # Process data
                processed_df = process_latest_json(raw_df)
                cleaned_df = clean_data(processed_df)
                featured_df = add_features(cleaned_df)
                
                # Ensure city columns are preserved
                if "city" not in featured_df.columns:
                    featured_df["city"] = city_code
                if "city_name" not in featured_df.columns:
                    featured_df["city_name"] = city_name
                
                # Upload to Hopsworks
                upload_to_hopsworks(featured_df, city_code=city_code)
                success_count += 1
                
            except Exception as e:
                error_messages.append(f"{CITIES[city_code]['name']}: {str(e)}")
                continue
        
        if success_count > 0:
            return True, f"Data fetched for {success_count}/{len(CITIES)} cities"
        else:
            return False, f"Failed to fetch data: {'; '.join(error_messages)}"
            
    except Exception as e:
        return False, str(e)

def run_model_training_direct():
    """Train the model with latest data directly."""
    try:
        # Connect to Hopsworks
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        
        # Get feature group
        try:
            fg = fs.get_feature_group("aqi_features", version=3)
        except:
            fg = fs.get_feature_group("aqi_features", version=2)
        
        df = fg.read()
        
        if df is None or df.empty:
            return False, "No data available in feature store"
        
        # Drop city columns for unified model
        if "city" in df.columns:
            df.drop(columns=["city", "city_name"], inplace=True, errors="ignore")
        
        # Prepare datetime
        if "datetime_str" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime_str"])
            df.drop(columns=["datetime_str"], inplace=True)
        
        df = df.sort_values(by="datetime").reset_index(drop=True)
        
        # Drop leakage features
        leakage_features = [col for col in df.columns if "rolling" in col or "lag" in col]
        df.drop(columns=leakage_features, inplace=True, errors="ignore")
        
        # Add noise to pollutants
        np.random.seed(42)
        pollutant_cols = ["pm10", "pm2_5", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]
        for col in pollutant_cols:
            if col in df.columns:
                df[col] = df[col] * (1 + np.random.normal(0, 0.05, len(df)))
        
        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Time-based split
        split_index = int(len(df) * 0.8)
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
        
        X_train = train_df.drop(columns=["aqi", "datetime"], errors="ignore")
        y_train = train_df["aqi"]
        X_test = test_df.drop(columns=["aqi", "datetime"], errors="ignore")
        y_test = test_df["aqi"]
        
        if X_train.empty or X_test.empty:
            return False, "Insufficient data for training"
        
        # Train models
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBoost": XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method="hist"
            )
        }
        
        best_model = None
        best_name = None
        best_rmse = float('inf')
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name
        
        # Save best model
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_filename = f"best_model_{best_name.replace(' ', '_').lower()}.pkl"
        model_path = os.path.join(model_dir, model_filename)
        dump(best_model, model_path)
        
        return True, f"Model trained successfully (RMSE: {best_rmse:.3f})"
        
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=3600)
def fetch_all_cities_data():
    """Fetch data for all cities from Hopsworks."""
    try:
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        
        # Try version 3 first, then version 2
        fg = None
        for version in [3, 2]:
            try:
                fg = fs.get_feature_group("aqi_features", version=version)
                break
            except Exception:
                continue
        
        if fg is None:
            return None, "Could not find feature group"
        
        df = fg.read()
        
        if df is None or df.empty:
            return None, "No data in feature store"
        
        # Handle datetime
        if "datetime_str" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime_str"], errors="coerce")
            df.drop(columns=["datetime_str"], inplace=True)
        
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_prediction_model():
    """Load the ML model for predictions."""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "../models")
        
        if not os.path.exists(models_dir):
            return None, "Models directory not found"
        
        model_files = [f for f in os.listdir(models_dir) if f.startswith("best_model_") and f.endswith(".pkl")]
        
        if not model_files:
            return None, "No model files found"
        
        model_path = os.path.join(models_dir, model_files[0])
        
        if not os.path.exists(model_path):
            return None, "Model file not found"
        
        model = load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"""
    <div class="header-container">
        <h1 class="main-title">
            {get_lucide_icon("cloud", 40, "#ffffff", 2.0)}
            Multi-City AQI Prediction Dashboard
        </h1>
        <p class="subtitle">Real-time Air Quality monitoring across Pakistan's major cities</p>
    </div>
""", unsafe_allow_html=True)

# Add control buttons section
st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    fetch_button = st.button(
        "🔄 Fetch New Data",
        help="Fetch latest data from APIs for all cities and update Hopsworks",
        use_container_width=True,
        type="primary"
    )

with col2:
    train_button = st.button(
        "🤖 Train Model",
        help="Train ML model with latest data from Hopsworks",
        use_container_width=True,
        type="primary"
    )

# Handle fetch button click
if fetch_button:
    try:
        with st.spinner("🔄 Fetching new data from APIs for all cities... This may take a few minutes."):
            fetch_success, fetch_message = run_data_fetch_direct()
            st.session_state.fetch_status = (fetch_success, fetch_message)
            
            if fetch_success:
                # Clear cache to force reload of data
                fetch_all_cities_data.clear()
                st.success(f"✅ {fetch_message}")
                st.info("🔄 Refreshing data...")
                st.rerun()  # Reload the page to show updated data
            else:
                st.error(f"❌ {fetch_message}")
                st.info("💡 You can try again or continue with existing data from Hopsworks.")
    except Exception as e:
        error_msg = f"Error during data fetch: {str(e)}"
        st.session_state.fetch_status = (False, error_msg)
        st.error(f"❌ {error_msg}")
        st.info("💡 Please check your internet connection and API availability, then try again.")

# Handle train button click
if train_button:
    try:
        with st.spinner("🤖 Training model with latest data from Hopsworks... This may take a few minutes."):
            train_success, train_message = run_model_training_direct()
            st.session_state.train_status = (train_success, train_message)
            
            if train_success:
                # Clear cache to force reload of model
                load_prediction_model.clear()
                st.success(f"✅ {train_message}")
                st.info("🔄 Loading new model...")
                st.rerun()  # Reload the page to use the new model
            else:
                st.error(f"❌ {train_message}")
                st.info("💡 You can try again or continue with existing model.")
    except Exception as e:
        error_msg = f"Error during model training: {str(e)}"
        st.session_state.train_status = (False, error_msg)
        st.error(f"❌ {error_msg}")
        st.info("💡 Please ensure data is available in Hopsworks, then try again.")

# Show persistent status messages from previous operations (only if buttons weren't just clicked)
if not fetch_button and 'fetch_status' in st.session_state and st.session_state.fetch_status:
    fetch_success, fetch_message = st.session_state.fetch_status
    if fetch_success:
        st.info(f"ℹ️ Last fetch: {fetch_message}")
    else:
        st.warning(f"⚠️ Last fetch failed: {fetch_message}")

if not train_button and 'train_status' in st.session_state and st.session_state.train_status:
    train_success, train_message = st.session_state.train_status
    if train_success:
        st.info(f"ℹ️ Last training: {train_message}")
    else:
        st.warning(f"⚠️ Last training failed: {train_message}")

st.markdown('<hr style="margin: 1.5rem 0;">', unsafe_allow_html=True)

# ============================================================================
# DATA FETCH AND TRAIN BUTTONS
# ============================================================================

# Initialize session state for tracking operations
if 'fetch_status' not in st.session_state:
    st.session_state.fetch_status = None
if 'train_status' not in st.session_state:
    st.session_state.train_status = None

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

with st.spinner("Loading data from Hopsworks..."):
    all_cities_df, error_msg = fetch_all_cities_data()

if all_cities_df is None:
    st.markdown(f"{icon_html('alert-circle', 20, '#ef4444')} <span style='color: #ef4444;'>Could not load data from Hopsworks.</span>", unsafe_allow_html=True)
    if error_msg:
        st.error(f"Error: {error_msg}")
    st.markdown(f"{icon_html('info', 20, '#3b82f6')} <span style='color: #3b82f6;'>Please check your connection and ensure the feature pipeline has been run.</span>", unsafe_allow_html=True)
    st.stop()

if all_cities_df.empty:
    st.markdown(f"{icon_html('alert-circle', 20, '#f59e0b')} <span style='color: #f59e0b;'>No data available in the feature store.</span>", unsafe_allow_html=True)
    st.markdown(f"{icon_html('info', 20, '#3b82f6')} <span style='color: #3b82f6;'>Please run `python src/run_feature_pipeline.py` to fetch data.</span>", unsafe_allow_html=True)
    st.stop()

with st.spinner("Loading ML model..."):
    model, model_error = load_prediction_model()

if model is None:
    st.markdown(f"{icon_html('alert-circle', 20, '#ef4444')} <span style='color: #ef4444;'>Could not load prediction model.</span>", unsafe_allow_html=True)
    if model_error:
        st.error(f"Error: {model_error}")
    st.markdown(f"{icon_html('info', 20, '#3b82f6')} <span style='color: #3b82f6;'>Please run `python src/train_model.py` to train a model.</span>", unsafe_allow_html=True)
    st.stop()

# ============================================================================
# CURRENT AQI SECTION
# ============================================================================

st.markdown(f'<div class="section-header">{get_lucide_icon("map-pin", 28, "currentColor", 2.5)} Current Air Quality Index</div>', unsafe_allow_html=True)
st.markdown('<p class="info-text" style="margin-top: -1rem; margin-bottom: 2rem;">Real-time monitoring across all supported cities</p>', unsafe_allow_html=True)

# Prepare data for each city
city_data = {}
cities_with_errors = []

for city_code, city_info in CITIES.items():
    try:
        # Get city data
        if "city" in all_cities_df.columns:
            city_df = all_cities_df[all_cities_df["city"] == city_code].copy()
        else:
            city_df = all_cities_df.copy()
        
        if city_df is None or city_df.empty:
            cities_with_errors.append((city_info["name"], "No data available"))
            continue
        
        # Prepare features
        result = prepare_features(city_df, drop_city_cols=True)
        if result is None:
            cities_with_errors.append((city_info["name"], "Could not prepare features"))
            continue
        
        X_city, city_df_clean = result
        
        if X_city.empty or len(X_city.columns) == 0:
            cities_with_errors.append((city_info["name"], "No valid features"))
            continue
        
        # Get the latest row for prediction
        if len(X_city) == 0:
            cities_with_errors.append((city_info["name"], "No data rows"))
            continue
        
        today_data = X_city.iloc[-1:].values
        if today_data.size == 0:
            cities_with_errors.append((city_info["name"], "Empty feature vector"))
            continue
        
        try:
            current_aqi = model.predict(today_data)[0]
            if pd.isna(current_aqi) or not np.isfinite(current_aqi):
                cities_with_errors.append((city_info["name"], "Invalid prediction"))
                continue
        except Exception:
            cities_with_errors.append((city_info["name"], "Prediction failed"))
            continue
        
        category, color, bg_color = get_aqi_category(current_aqi)
        city_data[city_code] = {
            "name": city_info["name"],
            "aqi": current_aqi,
            "category": category,
            "color": color,
            "bg_color": bg_color
        }
    except Exception as e:
        cities_with_errors.append((city_info["name"], str(e)))
        continue

# Display city cards in a grid
if city_data:
    num_cities = len(city_data)
    cols_per_row = min(3, num_cities)
    cols = st.columns(cols_per_row)
    
    for idx, (city_code, data) in enumerate(city_data.items()):
        col_idx = idx % cols_per_row
        with cols[col_idx]:
            st.markdown(f"""
                <div class="city-card">
                    <div class="city-name">
                        {get_lucide_icon("map-pin", 18, "#64748b", 2.0)}
                        {data['name']}
                    </div>
                    <div class="aqi-label">Current AQI</div>
                    <div class="aqi-value" style="color: {data['color']}">{data['aqi']:.0f}</div>
                    <div class="aqi-badge" style="background-color: {data['bg_color']}; color: {data['color']}">
                        {get_lucide_icon("activity", 16, data['color'], 2.0)}
                        {data['category']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    if num_cities > 3:
        # Add remaining cities in new rows
        for idx in range(3, num_cities):
            if idx % 3 == 0:
                cols = st.columns(3)
            with cols[idx % 3]:
                city_code = list(city_data.keys())[idx]
                data = city_data[city_code]
                st.markdown(f"""
                    <div class="city-card">
                        <div class="city-name">
                            {get_lucide_icon("map-pin", 18, "#64748b", 2.0)}
                            {data['name']}
                        </div>
                        <div class="aqi-label">Current AQI</div>
                        <div class="aqi-value" style="color: {data['color']}">{data['aqi']:.0f}</div>
                        <div class="aqi-badge" style="background-color: {data['bg_color']}; color: {data['color']}">
                            {get_lucide_icon("activity", 16, data['color'], 2.0)}
                            {data['category']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="empty-state">
            <p>No current AQI data available for any city.</p>
            <p style="font-size: 0.875rem; margin-top: 0.5rem;">Please run the feature pipeline to fetch data.</p>
    </div>
    """, unsafe_allow_html=True)

# Show errors if any
if cities_with_errors:
    with st.expander(f"{icon_html('alert-circle', 18, '#f59e0b')} Cities with Data Issues", expanded=False):
        for city_name, error in cities_with_errors:
            st.markdown(f"**{city_name}**: <span style='color: var(--text-secondary);'>{error}</span>", unsafe_allow_html=True)

st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

# ============================================================================
# DETAILED FORECAST SECTION
# ============================================================================

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(f'<div class="section-header">{get_lucide_icon("bar-chart", 28, "currentColor", 2.5)} Forecast Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="info-text" style="margin-top: -1rem; margin-bottom: 2rem;">3-day predictions with hourly and daily trends</p>', unsafe_allow_html=True)

# City filter - Multi-select checkboxes
st.markdown('<div class="filter-section">', unsafe_allow_html=True)
st.markdown(f'<div class="subsection-header">{get_lucide_icon("filter", 20, "currentColor", 2.0)} Select Cities</div>', unsafe_allow_html=True)
city_options = {code: f"{info['name']} ({code})" for code, info in CITIES.items()}

# Create checkboxes in columns for better layout
filter_cols = st.columns(len(CITIES))
selected_cities = []
for idx, (city_code, city_label) in enumerate(city_options.items()):
    with filter_cols[idx]:
        if st.checkbox(city_label, value=True, key=f"city_check_{city_code}"):
            selected_cities.append(city_code)

st.markdown('</div>', unsafe_allow_html=True)

if not selected_cities:
    st.markdown(f"{icon_html('alert-circle', 20, '#f59e0b')} <span style='color: #f59e0b;'>Please select at least one city to view forecasts.</span>", unsafe_allow_html=True)
    st.stop()

cities_to_show = selected_cities

def prepare_city_forecast(city_code: str, city_df: pd.DataFrame) -> dict:
    """Prepare forecast data for a single city with error handling."""
    try:
        result = prepare_features(city_df, drop_city_cols=True)
        if result is None:
            return None
        
        X, city_df_clean = result
        
        if X.empty or len(X.columns) == 0:
            return None
        
        # Current AQI
        if len(X) == 0:
            return None
        
        today_data = X.iloc[-1:].values
        if today_data.size == 0:
            return None
        
        try:
            current_aqi = model.predict(today_data)[0]
            if pd.isna(current_aqi) or not np.isfinite(current_aqi):
                return None
        except Exception:
            return None
        
        # Future predictions
        last_date = city_df_clean["datetime"].max()
        if pd.isna(last_date):
            return None
        
        future_dates = [last_date + timedelta(hours=i) for i in range(1, 73)]
        base_features = X.iloc[-1].copy()
        
        # ±5% random variation
        vary_cols = [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "ozone", "sulphur_dioxide", "temperature_2m",
            "relative_humidity_2m", "wind_speed_10m"
        ]
        
        future_data = pd.DataFrame([base_features for _ in range(72)])
        for col in vary_cols:
            if col in future_data.columns:
                noise = np.random.normal(0, 0.05, size=72)
                future_data[col] = future_data[col] * (1 + noise)
        
        future_data["datetime"] = future_dates
        
        try:
            future_preds = model.predict(future_data.drop(columns=["datetime"], errors="ignore"))
            # Filter out invalid predictions
            future_preds = np.where(np.isfinite(future_preds), future_preds, np.nan)
        except Exception:
            return None
        
        future_results = pd.DataFrame({
            "datetime": future_dates,
            "predicted_AQI": future_preds
        })
        future_results = future_results.dropna()
        
        if future_results.empty:
            return None
        
        future_results["date"] = future_results["datetime"].dt.date
        daily_avg = future_results.groupby("date")["predicted_AQI"].mean().reset_index()
        
        if daily_avg.empty:
            return None
        
        return {
            "current_aqi": float(current_aqi),
            "future_results": future_results,
            "daily_avg": daily_avg,
            "avg_today": float(city_df_clean.tail(24)["aqi"].mean()) if "aqi" in city_df_clean.columns and not city_df_clean.tail(24)["aqi"].isna().all() else None,
            "avg_next_24h": float(future_results.head(24)["predicted_AQI"].mean()) if len(future_results) >= 24 else None
        }
    except Exception:
        return None

# Generate forecasts for all selected cities
all_forecasts = {}
forecast_count = 0

for city_code in cities_to_show:
    city_name = CITIES[city_code]["name"]
    
    # Get city data
    if "city" in all_cities_df.columns:
        city_df = all_cities_df[all_cities_df["city"] == city_code].copy()
    else:
        city_df = all_cities_df.copy()
    
    if city_df is None or city_df.empty:
        continue
    
    # Prepare forecast
    forecast_data = prepare_city_forecast(city_code, city_df)
    
    if forecast_data is None:
        continue
    
    forecast_count += 1
    all_forecasts[city_code] = {
        "name": city_name,
        "data": forecast_data,
        "color": CITY_COLORS.get(city_code, "#94a3b8")
    }

if forecast_count == 0:
    st.markdown("""
        <div class="empty-state">
            <p>No forecasts could be generated for the selected cities.</p>
            <p style="font-size: 0.875rem; margin-top: 0.5rem;">Please ensure data is available and run the feature pipeline if needed.</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================================
# UNIFIED CHARTS SECTION
# ============================================================================

st.markdown(f'<div class="subsection-header">{get_lucide_icon("calendar", 22, "currentColor", 2.0)} 3-Day Forecast Overview</div>', unsafe_allow_html=True)

# Prepare data for unified charts
hourly_data = {}
daily_data = {}

for city_code, city_info in all_forecasts.items():
    city_name = city_info["name"]
    forecast = city_info["data"]
    
    # Hourly data
    if not forecast["future_results"].empty:
        hourly_df = forecast["future_results"].copy()
        hourly_df["city"] = city_name
        hourly_data[city_code] = hourly_df
    
    # Daily data
    if not forecast["daily_avg"].empty:
        daily_df = forecast["daily_avg"].copy()
        daily_df["city"] = city_name
        daily_data[city_code] = daily_df

# Create unified hourly chart using Plotly
if hourly_data and len(hourly_data) > 0:
    # Detect dark mode from Streamlit theme
    try:
        theme_config = st.get_option("theme.base")
        is_dark = theme_config == "dark"
    except:
        is_dark = False
    
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    icon_color = "#f1f5f9" if is_dark else "#1e293b"
    st.markdown(f'<div class="chart-title">{get_lucide_icon("line-chart", 20, icon_color, 2.0)} Hourly AQI Trend</div>', unsafe_allow_html=True)
    st.markdown('<p class="info-text" style="margin-top: -1rem; margin-bottom: 1.5rem;">72-hour prediction across selected cities</p>', unsafe_allow_html=True)
    
    fig_hourly = go.Figure()
    
    for city_code, hourly_df in hourly_data.items():
        city_name = all_forecasts[city_code]["name"]
        color = all_forecasts[city_code]["color"]
        
        fig_hourly.add_trace(go.Scatter(
            x=hourly_df["datetime"],
            y=hourly_df["predicted_AQI"],
            mode='lines',
            name=city_name,
            line=dict(color=color, width=2.5),
            hovertemplate=f'<b>{city_name}</b><br>' +
                         'Time: %{x}<br>' +
                         'AQI: %{y:.1f}<extra></extra>'
        ))
    
    fig_hourly.update_layout(
        title="",
        xaxis_title="Date & Time",
        yaxis_title="Predicted AQI",
        hovermode='x unified',
        height=400,
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9" if is_dark else "#1e293b"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Create unified daily chart using Plotly
if daily_data and len(daily_data) > 0:
    # Detect dark mode from Streamlit theme
    try:
        theme_config = st.get_option("theme.base")
        is_dark = theme_config == "dark"
    except:
        is_dark = False
    
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    icon_color = "#f1f5f9" if is_dark else "#1e293b"
    st.markdown(f'<div class="chart-title">{get_lucide_icon("bar-chart", 20, icon_color, 2.0)} Daily Average Forecast</div>', unsafe_allow_html=True)
    st.markdown('<p class="info-text" style="margin-top: -1rem; margin-bottom: 1.5rem;">Average AQI predictions by day</p>', unsafe_allow_html=True)
    
    fig_daily = go.Figure()
    
    for city_code, daily_df in daily_data.items():
        city_name = all_forecasts[city_code]["name"]
        color = all_forecasts[city_code]["color"]
        
        fig_daily.add_trace(go.Bar(
            x=daily_df["date"],
            y=daily_df["predicted_AQI"],
            name=city_name,
            marker_color=color,
            opacity=0.8,
            hovertemplate=f'<b>{city_name}</b><br>' +
                          'Date: %{x}<br>' +
                          'Avg AQI: %{y:.1f}<extra></extra>'
        ))
    
    fig_daily.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="Average Predicted AQI",
        barmode='group',
        hovermode='x unified',
        height=400,
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9" if is_dark else "#1e293b"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Combined forecast summary table
if daily_data and len(daily_data) > 0:
    # Detect dark mode for icon color
    try:
        theme_config = st.get_option("theme.base")
        is_dark = theme_config == "dark"
    except:
        is_dark = False
    icon_color = "#f1f5f9" if is_dark else "#1e293b"
    st.markdown(f'<div class="subsection-header">{get_lucide_icon("table", 22, icon_color, 2.0)} Forecast Summary</div>', unsafe_allow_html=True)
    st.markdown('<p class="info-text" style="margin-top: -0.5rem; margin-bottom: 1rem;">Detailed daily averages for all selected cities</p>', unsafe_allow_html=True)
    # Create combined summary table
    summary_rows = []
    for city_code, daily_df in daily_data.items():
        city_name = all_forecasts[city_code]["name"]
        for _, row in daily_df.iterrows():
            summary_rows.append({
                "City": city_name,
                "Date": row["date"],
                "Average AQI": f"{row['predicted_AQI']:.1f}"
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(["City", "Date"])
        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.dataframe(
            summary_df, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "City": st.column_config.TextColumn("City", width="medium"),
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Average AQI": st.column_config.NumberColumn("Average AQI", format="%.1f")
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Current AQI comparison cards
st.markdown('<div class="spacing-section"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="subsection-header">{get_lucide_icon("activity", 22, "currentColor", 2.0)} Current Status</div>', unsafe_allow_html=True)
st.markdown('<p class="info-text" style="margin-top: -0.5rem; margin-bottom: 1.5rem;">Latest predicted AQI values for selected cities</p>', unsafe_allow_html=True)
current_aqi_cols = st.columns(len(all_forecasts))

for idx, (city_code, city_info) in enumerate(all_forecasts.items()):
    with current_aqi_cols[idx]:
        forecast = city_info["data"]
        city_name = city_info["name"]
        category, color, bg_color = get_aqi_category(forecast["current_aqi"])
        
        st.markdown(f"""
            <div class="city-card">
                <div class="city-name">
                    {get_lucide_icon("map-pin", 18, "#64748b", 2.0)}
                    {city_name}
                </div>
                <div class="aqi-label">Current AQI</div>
                <div class="aqi-value" style="color: {color}">{forecast['current_aqi']:.0f}</div>
                <div class="aqi-badge" style="background-color: {bg_color}; color: {color}">
                    {get_lucide_icon("activity", 16, color, 2.0)}
                    {category}
                </div>
            </div>
        """, unsafe_allow_html=True)

# Trend interpretation for each city
st.markdown('<div class="spacing-section"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="subsection-header">{get_lucide_icon("trending-up", 22, "currentColor", 2.0)} 24-Hour Trend Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="info-text" style="margin-top: -0.5rem; margin-bottom: 1rem;">Expected changes in air quality over the next 24 hours</p>', unsafe_allow_html=True)
for city_code, city_info in all_forecasts.items():
    city_name = city_info["name"]
    forecast = city_info["data"]
    
    if forecast["avg_today"] is not None and forecast["avg_next_24h"] is not None:
        diff = forecast["avg_next_24h"] - forecast["avg_today"]
        if diff > 5:
            st.markdown(f"""
                <div class="status-indicator status-warning">
                    {get_lucide_icon("trending-up", 18, "#92400e", 2.0)}
                    <strong>{city_name}</strong>: Air quality expected to worsen slightly in the next 24 hours. (Change: +{diff:.1f})
                </div>
            """, unsafe_allow_html=True)
        elif diff < -5:
            st.markdown(f"""
                <div class="status-indicator status-info">
                    {get_lucide_icon("trending-down", 18, "#1e40af", 2.0)}
                    <strong>{city_name}</strong>: Air quality expected to improve slightly in the next 24 hours. (Change: {diff:.1f})
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="status-indicator status-success">
                    {get_lucide_icon("minimize-2", 18, "#065f46", 2.0)}
                    <strong>{city_name}</strong>: Air quality expected to remain stable in the next 24 hours. (Change: {diff:+.1f})
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="status-indicator status-info">
                {get_lucide_icon("info", 18, "#1e40af", 2.0)}
                <strong>{city_name}</strong>: Insufficient data to determine 24-hour trend.
            </div>
        """, unsafe_allow_html=True)
