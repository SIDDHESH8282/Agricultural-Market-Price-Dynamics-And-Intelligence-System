import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = 'preprocessed_final_2014_2025_FULL_FEATURES_corrected_SORTED_NO_NULLS.csv'
ARTIFACTS_DIR = 'model_artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def create_features(df, target_col, lags=[1, 7, 30], windows=[7, 30]):
    """
    Generates lag and rolling mean features for a specific target column.
    """
    df = df.copy()
    
    # Lags
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('market_id')[target_col].shift(lag)
        
    # Rolling Means
    for window in windows:
        df[f'roll_mean_{window}'] = df.groupby('market_id')[target_col].transform(lambda x: x.rolling(window).mean())
        
    return df

def train_and_save_model(df, target_col, model_name, feature_prefix):
    print(f"\n--- Training {model_name} Model ---")
    
    # Target Encoding
    encodings = {}
    for col in ['state', 'district', 'market_id']:
        encoding_map = df.groupby(col)[target_col].mean().to_dict()
        encodings[col] = encoding_map
        df[f'{col}_encoded'] = df[col].map(encoding_map)
        
    # Feature Engineering
    df_processed = create_features(df, target_col)
    df_processed = df_processed.dropna()
    
    features = [
        'state_encoded', 'district_encoded', 'market_id_encoded',
        'month', 'year', 'day_of_week',
        'lag_1', 'lag_7', 'lag_30',
        'roll_mean_7', 'roll_mean_30'
    ]
    
    # Special case: Price model can use arrivals as a feature
    if target_col == 'modal_price_rs_qtl':
        features.append('arrivals_tonnes')
        
    X = df_processed[features]
    y = df_processed[target_col]
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Save Artifacts
    print(f"Saving {model_name} artifacts...")
    with open(os.path.join(ARTIFACTS_DIR, f'xgb_{feature_prefix}_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(ARTIFACTS_DIR, f'{feature_prefix}_encodings.pkl'), 'wb') as f:
        pickle.dump(encodings, f)
    with open(os.path.join(ARTIFACTS_DIR, f'{feature_prefix}_features.pkl'), 'wb') as f:
        pickle.dump(features, f)

def train_model():
    print("Loading data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Date conversion
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)
    else:
        raise ValueError("Date column not found")

    # Create market_id
    if 'market_id' not in df.columns:
        df['market_id'] = df['state'] + '_' + df['district'] + '_' + df['market']

    print("Preprocessing...")
    
    # Common Features
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['day_of_week'] = df[date_col].dt.dayofweek
    
    # 1. Train Price Model
    train_and_save_model(df.copy(), 'modal_price_rs_qtl', 'Price', 'price')
    
    # 2. Train Arrivals Model
    train_and_save_model(df.copy(), 'arrivals_tonnes', 'Arrivals', 'arrival')
    
    # 3. Train Temperature Model
    # Check if temp column exists, handle missing
    if 'temp_avg' in df.columns:
        # Fill missing temps with mean or ffill
        df['temp_avg'] = df['temp_avg'].fillna(method='ffill').fillna(25.0)
        train_and_save_model(df.copy(), 'temp_avg', 'Temperature', 'temp')
    else:
        print("Warning: 'temp_avg' column not found. Skipping Temperature model.")

    # 4. Train Rainfall Model
    if 'rain' in df.columns:
        df['rain'] = df['rain'].fillna(0.0)
        train_and_save_model(df.copy(), 'rain', 'Rainfall', 'rain')
    else:
        print("Warning: 'rain' column not found. Skipping Rainfall model.")
        
    print("✅ All models trained and saved!")

if __name__ == "__main__":
    train_model()
