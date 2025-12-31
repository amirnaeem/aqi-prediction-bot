import hopsworks
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime

try:
    from src.config import SAVE_LOCAL
except Exception:
    from config import SAVE_LOCAL


def upload_to_hopsworks(df: pd.DataFrame = None, city_code: str = None):
    """
    Upload final processed feature DataFrame to Hopsworks Feature Store.
    If df is not provided, it loads the latest 'final_selected_features.csv'.
    
    Args:
        df: DataFrame to upload (must include 'city' column)
        city_code: City code (KHI, ISB, LHR) - used if city column missing
    """

    print("🔗 Connecting to Hopsworks Feature Store...")

    # 1. Load environment variables (API key) 
    load_dotenv()
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise ValueError("❌ Missing HOPSWORKS_API_KEY in .env file")

    # 2. Authenticate & connect to project
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    print("✅ Connected to Hopsworks Feature Store")

    # 3. Load DataFrame (if not passed) 
    if df is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(BASE_DIR, "data", "final", "final_selected_features.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        df = pd.read_csv(file_path)
        print(f"📂 Loaded dataset → {file_path}")

    print(f"📊 Dataset shape before upload: {df.shape}")

    # 4. Ensure city column exists
    if "city" not in df.columns:
        if city_code:
            df["city"] = city_code
            try:
                from src.config import CITIES
            except:
                from config import CITIES
            df["city_name"] = CITIES.get(city_code, {}).get("name", city_code)
        else:
            # Default to Karachi for backward compatibility
            df["city"] = "KHI"
            df["city_name"] = "Karachi"
    
    # Ensure city_name exists
    if "city_name" not in df.columns:
        try:
            from src.config import CITIES
        except:
            from config import CITIES
        df["city_name"] = df["city"].map(lambda x: CITIES.get(x, {}).get("name", x))

    # 5. Datetime handling 
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["datetime_str"] = df["datetime"].astype(str)
        df.drop(columns=["datetime"], inplace=True)
    elif "datetime_str" not in df.columns:
        raise ValueError("❌ Missing datetime column in DataFrame")

    # 6. Drop extra columns not in FG schema
    drop_extras = ["year", "month_num", "day_num"]
    df = df.drop(columns=[c for c in drop_extras if c in df.columns], errors="ignore")

    # 7. Enforce correct dtypes (align with FG schema)
    int_cols = ["month", "hour", "day", "weekday", "high_pollution_flag"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.int64)

    # Ensure city is string type
    if "city" in df.columns:
        df["city"] = df["city"].astype(str)
    if "city_name" in df.columns:
        df["city_name"] = df["city_name"].astype(str)

    # 8. Define Feature Group metadata
    FEATURE_GROUP_NAME = "aqi_features"
    FEATURE_GROUP_VERSION = 3  # New version with city columns

    # 9. Get or create Feature Group (primary key includes city for multi-city support)
    # This will create version 3 if it doesn't exist, or use existing if it does
    print(f"📋 Getting or creating Feature Group: {FEATURE_GROUP_NAME}_v{FEATURE_GROUP_VERSION}")
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["datetime_str", "city"],  # Composite key: datetime + city
        description="Multi-city AQI selected features (daily ingestion) - Karachi, Islamabad, Lahore",
        online_enabled=True
    )

    # 10. Insert into Feature Store 
    city_list = df["city"].unique().tolist() if "city" in df.columns else ["unknown"]
    print(f"🚀 Uploading to Hopsworks Feature Store for city(ies): {', '.join(city_list)}...")
    
    try:
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully uploaded {len(df)} rows to Feature Group → '{FEATURE_GROUP_NAME}_v{FEATURE_GROUP_VERSION}'")
    except Exception as e:
        error_msg = str(e)
        if "not compatible with Feature Group schema" in error_msg:
            print(f"\n⚠️ Schema mismatch detected. Attempting to update Feature Group schema...")
            print(f"   Error: {error_msg}\n")
            print(f"💡 Solution: The Feature Group needs to be recreated with the new schema.")
            print(f"   You may need to delete version {FEATURE_GROUP_VERSION} and recreate it, or use a different version number.")
            raise ValueError(f"Schema incompatibility. Please recreate Feature Group {FEATURE_GROUP_NAME}_v{FEATURE_GROUP_VERSION} with city columns included.")
        else:
            raise

    # 11. local snapshot
    if SAVE_LOCAL:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(BASE_DIR, "data", "final", "uploaded_snapshot.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"💾 Snapshot saved locally → {out_path}")
    else:
        print("⚙️ Skipping local snapshot save (cloud mode).")

    print("🎉 Upload complete.")
    return df


# --- Run standalone test safely ---
if __name__ == "__main__":
    upload_to_hopsworks()