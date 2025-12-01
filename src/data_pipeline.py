import pandas as pd
import numpy as np
import requests
import json
import time
import os
from typing import List, Dict, Any

# --- CONFIGURATION ---
KAGGLE_FILE = 'Code/data/googleplaystore_apps.csv'
REVIEWS_FILE = 'Code/data/googleplaystore_apps_user_reviews.csv'
GOOGLE_PLAY_CLEANED_FILE = 'Code/data/processed/googleplaystore_final_data.csv'
COMBINED_FILE = 'Code/data/processed/googleplaystore_and_appstore_combined_data.csv'
API_CACHE_FILE = 'Code/data/processed/appstore_api_response.json' # Cache file for App Store lookups

# MOCK API CONFIGURATION
# NOTE: The script defaults to MOCK API calls inside fetch_appstore_data()
RAPIDAPI_KEY = '8065b3eb6fmsh9ff6a831c5406d4p1ff81ejsncec291d7a03d' 
RAPIDAPI_HOST = 'appstore-scrapper-api.p.rapidapi.com'
RAPIDAPI_URL = f'https://{RAPIDAPI_HOST}/v1/app-store-api/search'

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}
# --- END CONFIGURATION ---

def load_and_clean_kaggle_data() -> pd.DataFrame:
    """
    Loads, cleans, and standardizes the Kaggle Google Play dataset.
    This function sources data ONLY from local CSV files.
    """
    print("-> Starting Kaggle data ingestion and cleaning (from local CSVs)...")

    try:
        # Load the data
        df = pd.read_csv(KAGGLE_FILE)
    except FileNotFoundError:
        print(f"Error: Required file not found. Check if '{KAGGLE_FILE}' is in the 'data/' folder.")
        raise

    # Drop rows where 'Installs' is 'Free' (a known data anomaly)
    df = df[df['Installs'] != 'Free'].copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Clean 'Installs' and convert to Int64 (nullable integer)
    df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '', regex=False).astype('Int64')

    # Clean 'Price' and convert to float
    df['Price'] = df['Price'].str.replace('$', '', regex=False).astype(float)

    # Convert 'Reviews' to integer
    df['Reviews'] = df['Reviews'].astype(int)

    # Helper function to convert app size string to bytes
    def size_to_bytes(size):
        if pd.isna(size):
            return np.nan
        size = str(size).upper()
        if 'M' in size:
            return float(size.replace('M', '')) * (1024 ** 2)
        elif 'K' in size:
            return float(size.replace('K', '')) * 1024
        return np.nan

    # Apply size conversion and drop original column
    df['Size_Bytes'] = df['Size'].apply(size_to_bytes)
    df.drop('Size', axis=1, inplace=True)

    # Standardize metadata columns
    df.rename(columns={'App': 'Name'}, inplace=True)
    df['Source'] = 'Google Play Store' # Explicitly label the source

    print("-> Kaggle data cleaning complete.")
    # Return a clean, standardized subset of columns
    return df[['Name', 'Category', 'Rating', 'Reviews', 'Installs', 'Type', 'Price',
               'Content Rating', 'Last Updated', 'Android Ver', 'Source', 'Size_Bytes']].copy()


def process_and_merge_reviews(df_google: pd.DataFrame) -> pd.DataFrame:
    """Loads and cleans reviews data from local CSV, then merges calculated sentiment polarity with the main dataset."""
    print("-> Processing user reviews (from local CSVs)...")
    try:
        df_reviews = pd.read_csv(REVIEWS_FILE)
    except FileNotFoundError:
        print(f"Error: Required file not found. Check if '{REVIEWS_FILE}' is in the 'data/' folder. Skipping sentiment calculation.")
        # Add a placeholder column if reviews file is missing
        df_google['Avg_Sentiment_Polarity'] = 0.0
        return df_google

    # Calculate average sentiment polarity per app
    df_reviews_avg = df_reviews.groupby('App')['Sentiment_Polarity'].mean().reset_index()
    df_reviews_avg.rename(columns={'App': 'Name', 'Sentiment_Polarity': 'Avg_Sentiment_Polarity'}, inplace=True)

    # Merge sentiment data back into the main Google Play DataFrame
    df_merged = pd.merge(df_google, df_reviews_avg, on='Name', how='left')

    # Fill missing sentiment scores (where no reviews were available) with 0 (neutral)
    df_merged['Avg_Sentiment_Polarity'] = df_merged['Avg_Sentiment_Polarity'].fillna(0)

    print("-> User review processing and merge complete.")
    return df_merged


def fetch_appstore_data(app_names: List[str], use_mock: bool = True) -> pd.DataFrame:
    """
    Fetches data from the App Store API or mock data, using a local cache.
    This function is responsible for all external data fetching and caching.
    """
    print(f"\n-> Fetching App Store data for {len(app_names)} unique apps...")
    api_data_list = []

    # --- Caching Implementation: Load Existing Cache ---
    app_cache = {}
    if os.path.exists(API_CACHE_FILE):
        try:
            with open(API_CACHE_FILE, 'r') as f:
                app_cache = json.load(f)
            print(f"  -> Loaded {len(app_cache)} apps from local cache.")
        except json.JSONDecodeError:
            print("  -> WARNING: Cache file corrupted. Starting with an empty cache.")

    newly_fetched_count = 0

    for app_name in app_names:

        # 1. Check Cache first (Avoid redundant API/Mock calls)
        if app_name in app_cache:
            # The cached result is already standardized
            api_data_list.append(app_cache[app_name])
            continue

        # 2. Make API Call (or Mock Call)
        app_details = None

        if use_mock:
            # MOCK API RESPONSE (for a single best result)
            app_details = {
                "results": [
                    {
                        "Name": app_name,
                        "Category": np.random.choice(['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY', 'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE', 'FAMILY', 'FOOD_AND_DRINK', 'GAME', 'HEALTH_AND_FITNESS', 'LIFESTYLE', 'MAPS_AND_NAVIGATION', 'MEDICAL', 'NEWS_AND_MAGAZINES', 'PHOTOGRAPHY', 'SHOPPING', 'SOCIAL', 'SPORTS', 'WEATHER']),
                        "Rating": round(np.random.uniform(4.0, 5.0), 1),
                        "Reviews": np.random.randint(5000, 500000),
                        "Installs": np.random.randint(100, 50000000),
                        "Type": np.random.choice(['Free', 'Paid'], p=[0.85, 0.15]),
                        "Price": np.random.choice([0.0, round(np.random.uniform(0.99, 9.99), 2)], p=[0.85, 0.15]),
                        "Content Rating": np.random.choice(['Everyone', '9+', '12+']),
                        "Size_Bytes": np.random.randint(15 * (1024**2), 150 * (1024**2)),
                        "Required Android Ver": "iOS 14.0+", # Mock data field name (unstandardized key)
                        "Last Updated": time.strftime("%B %d, %Y"), # Mock data field name (unstandardized key)
                        "Avg_Sentiment_Polarity": round(np.random.uniform(-0.5, 0.9), 2),
                        "Source": "App Store (Mock)"
                    }
                ]
            }
        else:
            # REAL API call (Requires a valid RAPIDAPI_KEY)
            try:
                # In a real scenario, we'd add proper error handling and exponential backoff here.
                response = requests.get(RAPIDAPI_URL, headers=HEADERS, params={'query': app_name})
                response.raise_for_status()
                app_details = response.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                print(f"  -> Failed to fetch/decode real API response for {app_name}. Error: {e}")
                continue

        # 3. Process Result, Select Best Match, Add to List, and Update Cache
        if 'results' in app_details and len(app_details['results']) > 0:

            # Optimization: Only select the first result (best match)
            best_match_result = app_details['results'][0]

            # Standardize column names for merging
            # NOTE: We use .pop() to grab the value and simultaneously remove the old key
            best_match_result['Review_Count'] = best_match_result.pop('Reviews')
            best_match_result['Required_Android_Version'] = best_match_result.pop('Required Android Ver')
            best_match_result['Last_Updated_Date'] = best_match_result.pop('Last Updated')

            api_data_list.append(best_match_result)

            # --- Caching Implementation: Update Cache ---
            app_cache[app_name] = best_match_result
            newly_fetched_count += 1
            # -------------------------------------------

        else:
            print(f"  -> No results found for {app_name}.")

    # --- Caching Implementation: Save Cache ---
    os.makedirs(os.path.dirname(API_CACHE_FILE), exist_ok=True)
    with open(API_CACHE_FILE, 'w') as f:
        json.dump(app_cache, f, indent=4)
    print(f"  -> Saved {len(app_cache)} app records (including {newly_fetched_count} new) to cache.")
    # ------------------------------------------

    df_appstore = pd.DataFrame(api_data_list)

    # Rename the remaining complex column name for final consistency
    df_appstore.rename(columns={'Content Rating': 'Content_Rating'}, inplace=True)

    print(f"-> App Store data processed. Shape: {df_appstore.shape}")
    return df_appstore


def build_unified_dataset(use_mock_api=True) -> pd.DataFrame:
    """Loads, cleans, fetches, and merges all data sources into one unified DataFrame."""

    # 1. Load and process Google Play data (from local CSVs)
    df_google = load_and_clean_kaggle_data()
    df_google_full = process_and_merge_reviews(df_google)

    # Define the consistent column structure for the final DataFrame
    columns_to_keep = [
        'Name', 'Category', 'Rating', 'Review_Count', 'Installs',
        'Type', 'Price', 'Content_Rating', 'Size_Bytes', 'Required_Android_Version',
        'Last_Updated_Date', 'Avg_Sentiment_Polarity', 'Source'
    ]

    # 4. Rename and select columns in Google Play data to match the final structure
    df_google_full.rename(columns={
        'Reviews': 'Review_Count',
        'Content Rating': 'Content_Rating',
        'Last Updated': 'Last_Updated_Date',
        'Android Ver': 'Required_Android_Version'
    }, inplace=True)
    
    # Select all columns for Google Play
    df_google_all_data = df_google_full[columns_to_keep]

    # --- MODIFICATION 1: SAVE CLEAN GOOGLE PLAY DATA ---
    os.makedirs(os.path.dirname(GOOGLE_PLAY_CLEANED_FILE), exist_ok=True)
    df_google_all_data.to_csv(GOOGLE_PLAY_CLEANED_FILE, index=False)
    print(f"\nDELIVERABLE 1: Saved clean Google Play dataset to {GOOGLE_PLAY_CLEANED_FILE}. Total rows: {len(df_google_all_data)}")
    # ----------------------------------------------------


    # --- MODIFICATION 2: STRATIFIED SAMPLING FOR BALANCED OUTPUT ---
    
    # Target: 400 total apps per category in the COMBINED file, balanced (200 Play + 200 App Store)
    # We first sample 200 from Google Play, and use those same 200 to fetch App Store data.
    TARGET_APPS_PER_CATEGORY_FOR_SAMPLE = 200
    
    print(f"\n-> Implementing stratified sampling for balanced output (Target: {TARGET_APPS_PER_CATEGORY_FOR_SAMPLE} apps per category for cross-platform comparison)...")

    # Sample only from the full Google Play data to get the subset for cross-platform comparison
    df_google_sampled = df_google_full.groupby('Category', dropna=True).apply(
        lambda x: x.sample(
            n=min(TARGET_APPS_PER_CATEGORY_FOR_SAMPLE, len(x)), 
            replace=False, 
            random_state=42 # Use a seed for reproducibility
        )
    ).reset_index(drop=True)

    # Create the list of app names to send to the API (these are the same apps as the Google Play sample)
    sample_apps = df_google_sampled['Name'].unique().tolist()
    
    print(f"-> Stratified sampling complete. Total unique apps to fetch for App Store: {len(sample_apps)}")
    
    # --- MODIFIED STRATIFIED SAMPLING END ---

    # 3. Fetch App Store data (via API/Mock API)
    df_appstore = fetch_appstore_data(sample_apps, use_mock=use_mock_api)

    # 5. Reindex App Store data to ensure all columns are present (filling missing with NaN)
    df_appstore_final = df_appstore.reindex(columns=columns_to_keep, fill_value=np.nan)

    # 6. Combine the datasets (only the balanced sample of Google Play + the fetched App Store data)
    # NOTE: We only combine the SAMPLED Google Play data to achieve the balanced per-category structure.
    df_combined = pd.concat([df_google_sampled[columns_to_keep], df_appstore_final], ignore_index=True)

    # 7. Final type consistency and save
    df_combined['Installs'] = df_combined['Installs'].astype('Int64')

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)
    df_combined.to_csv(COMBINED_FILE, index=False)

    print(f"\nDELIVERABLE 2: Saved BALANCED combined dataset to {COMBINED_FILE}. Total rows: {len(df_combined)}")
    print(f"NOTE: The combined file contains a balanced sample of {TARGET_APPS_PER_CATEGORY_FOR_SAMPLE} Google Play apps and {len(df_appstore_final)} App Store apps.")
    return df_combined

if __name__ == '__main__':
    # Run the full pipeline. Set use_mock_api=False to use the real (but rate-limited) API.
    build_unified_dataset(use_mock_api=True)