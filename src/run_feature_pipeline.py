# Purpose: End-to-end automation of the Feature Pipeline

import os
import sys
import pandas as pd
from datetime import datetime

# --- Import project modules safely ---
try:
    from src.config import SAVE_LOCAL, get_air_quality_url, get_weather_forecast_url, CITIES, DEFAULT_CITY
    from src.fetch_data import fetch_api_data
    from src.process_data import process_latest_json
    from src.clean_data import clean_data
    from src.process_features import add_features
    from src.upload_to_hopswork import upload_to_hopsworks
except ModuleNotFoundError:
    from config import SAVE_LOCAL, get_air_quality_url, get_weather_forecast_url, CITIES, DEFAULT_CITY
    from fetch_data import fetch_api_data
    from process_data import process_latest_json
    from clean_data import clean_data
    from process_features import add_features
    from upload_to_hopswork import upload_to_hopsworks


def run_pipeline_for_city(city_code: str = DEFAULT_CITY):
    """
    Run the complete feature pipeline for a specific city.
    
    Args:
        city_code: City code (KHI, ISB, LHR)
    """
    if city_code not in CITIES:
        raise ValueError(f"❌ Invalid city code: {city_code}. Available: {list(CITIES.keys())}")
    
    city_name = CITIES[city_code]["name"]
    print(f"\n🚀 Starting Daily Feature Pipeline for {city_name} AQI\n")

try:
    # 2. Step 1: Fetch Latest Raw Data
        print(f"\n🌤️ Fetching latest Air Quality + Weather data for {city_name}...")
        
        # Generate city-specific URLs
        aq_url = get_air_quality_url(city_code)
        wx_url = get_weather_forecast_url(city_code)
    
    # Fetch raw JSON dictionaries
        aq_json = fetch_api_data(aq_url)
        wx_json = fetch_api_data(wx_url)

        # Validate API responses
        if aq_json is None or wx_json is None:
            error_msg = f"❌ Failed to fetch data from APIs for {city_name}.\n"
            error_msg += f"   Air Quality API: {'✓ Success' if aq_json else '✗ Failed'}\n"
            error_msg += f"   Weather API: {'✓ Success' if wx_json else '✗ Failed'}\n"
            error_msg += "\n💡 Possible causes:\n"
            error_msg += "   - No internet connection\n"
            error_msg += "   - DNS resolution issues\n"
            error_msg += "   - API service temporarily unavailable\n"
            error_msg += "\n   Please check your internet connection and try again."
            raise ValueError(error_msg)
        
        if "hourly" not in aq_json or "hourly" not in wx_json:
            raise ValueError("❌ API response missing 'hourly' data. Check API response structure.")

    # Convert to DataFrames safely
    aq_df = pd.DataFrame(aq_json["hourly"])
    wx_df = pd.DataFrame(wx_json["hourly"])
        
        if aq_df.empty or wx_df.empty:
            raise ValueError(f"❌ Empty dataframes returned. Air Quality rows: {len(aq_df)}, Weather rows: {len(wx_df)}")

    # Add datetime column (for merging)
    if "time" in aq_df.columns and "time" in wx_df.columns:
        aq_df.rename(columns={"time": "datetime"}, inplace=True)
        wx_df.rename(columns={"time": "datetime"}, inplace=True)

    # Merge both datasets on time/datetime
    raw_df = pd.merge(aq_df, wx_df, on="datetime", how="inner")
        
        # Add city information
        raw_df["city"] = city_code
        raw_df["city_name"] = city_name
        
    print(f"✅ Combined raw data fetched with shape: {raw_df.shape}")

    # 3. Step 2: Process Raw Data
    print("\n⚙️ Processing raw JSON data into structured DataFrame...")
    processed_df = process_latest_json(raw_df)
    print(f"✅ Processed data shape: {processed_df.shape}")

    # 4. Step 3: Clean Data (EDA-1 logic) 
    print("\n🧹 Cleaning processed data...")
    cleaned_df = clean_data(processed_df)
    print(f"✅ Cleaned data shape: {cleaned_df.shape}")

    # 5. Step 4: Feature Engineering (EDA-2 logic) 
    print("\n🧠 Generating engineered features...")
    featured_df = add_features(cleaned_df)
    print(f"✅ Feature engineering complete — shape: {featured_df.shape}")

        # Ensure city columns are preserved
        if "city" not in featured_df.columns:
            featured_df["city"] = city_code
        if "city_name" not in featured_df.columns:
            featured_df["city_name"] = city_name

    # 6. Step 5: Upload to Hopsworks
        print(f"\n📦 Uploading final dataset to Hopsworks Feature Store for {city_name}...")
        upload_to_hopsworks(featured_df, city_code=city_code)

    # 7. Verification Step: Read data back from Feature Store
    print("\n🔍 Verifying uploaded data from Feature Store...")
    try:
        import hopsworks
            from dotenv import load_dotenv
            load_dotenv()
            import os
            api_key = os.getenv("HOPSWORKS_API_KEY")

            project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
            # Try version 3 (multi-city) first, fallback to version 2 (legacy)
            try:
                fg = fs.get_feature_group("aqi_features", version=3)
            except:
        fg = fs.get_feature_group("aqi_features", version=2)

        df_check = fg.read()  # Read full feature group
        df_check["datetime_str"] = pd.to_datetime(df_check["datetime_str"])

            # Filter by city if city column exists
            if "city" in df_check.columns:
                df_check = df_check[df_check["city"] == city_code]
                print(f"\n📊 Data for {city_name} ({city_code}):")

        # Sort chronologically to check range
        df_check.sort_values("datetime_str", inplace=True)
        print("\n🧭 Feature Store Data Time Range:")
        print(f"Start → {df_check['datetime_str'].min()}")
        print(f"End   → {df_check['datetime_str'].max()}")

        # Display small samples
        print("\n📊 Head of Feature Store:")
        print(df_check.head(3))
        print("\n📊 Tail of Feature Store:")
        print(df_check.tail(3))
    except Exception as e:
        print("⚠️ Could not verify data from Feature Store:")
        print(str(e))

        print(f"\n🎉 Feature pipeline executed successfully for {city_name}!")
        return featured_df

except Exception as e:
        print(f"\n❌ Pipeline failed for {city_name} due to error:")
    print(str(e))
        raise

finally:
        print(f"\n🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==============================================")


# Main execution - supports command line argument or runs for all cities
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run for specific city from command line
        city_arg = sys.argv[1].upper()
        run_pipeline_for_city(city_arg)
    else:
        # Run for all cities
        print("\n🌍 Running pipeline for all cities...\n")
        for city_code in CITIES.keys():
            try:
                run_pipeline_for_city(city_code)
                print(f"\n{'='*50}\n")
            except Exception as e:
                print(f"❌ Failed to process {CITIES[city_code]['name']}: {e}\n")
                continue
