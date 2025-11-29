import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta

# Load artifacts
ARTIFACTS_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'xgb_model.pkl')
ENCODINGS_PATH = os.path.join(ARTIFACTS_DIR, 'encodings.pkl')
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, 'feature_names.pkl')

model = None
encodings = None
feature_names = None

def load_model_artifacts():
    global model, encodings, feature_names
    if model is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(ENCODINGS_PATH, 'rb') as f:
                encodings = pickle.load(f)
            with open(FEATURES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            return False
    return True

def prepare_features(state, district, market, date, history_df):
    """
    Prepare a single row of features for prediction.
    Requires historical data to calculate lags/rolling means.
    """
    if not load_model_artifacts():
        return None
        
    market_id = f"{state}_{district}_{market}"
    
    # Get recent history for this market
    market_df = history_df[history_df['market'] == market].sort_values('Date')
    
    if market_df.empty:
        # Fallback: use district average or global average?
        # For now, return None if no history for this specific market
        return None
        
    # Calculate features
    # 1. Encodings (Handle unknown categories with global mean if possible, or 0)
    state_enc = encodings['state'].get(state, np.mean(list(encodings['state'].values())))
    dist_enc = encodings['district'].get(district, np.mean(list(encodings['district'].values())))
    market_enc = encodings['market_id'].get(market_id, np.mean(list(encodings['market_id'].values())))
    
    # 2. Date features
    month = date.month
    year = date.year
    day_of_week = date.dayofweek
    
    # 3. Lags & Rolling (Need to look back from the target date)
    # If target date is far in future, we might need to use recursive forecasting.
    # For simplicity in "Custom Date", if date > last_date, we use the latest available data as proxy for lags
    # (Naive approach for single point, but better is recursive)
    
    last_row = market_df.iloc[-1]
    
    # Simple assumption: If predicting for tomorrow, use today's data. 
    # If predicting for next month, we should ideally simulate forward.
    # Here we will use the LATEST available data for lags.
    
    lag_1 = last_row['modal_price_rs_qtl']
    # Try to find exactly 7 days ago, else use 7th last record
    lag_7 = market_df.iloc[-7]['modal_price_rs_qtl'] if len(market_df) >= 7 else lag_1
    lag_30 = market_df.iloc[-30]['modal_price_rs_qtl'] if len(market_df) >= 30 else lag_1
    
    roll_mean_7 = market_df.tail(7)['modal_price_rs_qtl'].mean()
    roll_mean_30 = market_df.tail(30)['modal_price_rs_qtl'].mean()
    
    arrivals = last_row['arrivals_tonnes'] # Assume constant arrivals or use seasonal avg
    
    # Weather (Placeholder if not available)
    temp_avg = last_row.get('temp_avg', 25.0)
    rain = last_row.get('rain', 0.0)
    
    # Construct feature vector in correct order
    feature_dict = {
        'state_encoded': state_enc,
        'district_encoded': dist_enc,
        'market_id_encoded': market_enc,
        'month': month,
        'year': year,
        'day_of_week': day_of_week,
        'lag_1': lag_1,
        'lag_7': lag_7,
        'lag_30': lag_30,
        'roll_mean_7': roll_mean_7,
        'roll_mean_30': roll_mean_30,
        'arrivals_tonnes': arrivals,
        'temp_avg': temp_avg,
        'rain': rain
    }
    
    # Ensure all features from training are present
    input_vector = []
    for feat in feature_names:
        input_vector.append(feature_dict.get(feat, 0))
        
    return pd.DataFrame([input_vector], columns=feature_names)

def run_forecast(history_df, horizon_days=30, supply_shock=0, transport_cost=1.0, seasonality_strength=1.0):
    """
    Recursive forecasting using the trained XGBoost model.
    """
    if not load_model_artifacts() or history_df.empty:
        return pd.DataFrame()
        
    # We need to identify the market we are forecasting for.
    # Assuming history_df contains data for ONE market (filtered in UI)
    # If not, we take the most frequent one.
    if 'market' in history_df.columns:
        market = history_df['market'].mode()[0]
        state = history_df['state'].mode()[0]
        district = history_df['district'].mode()[0]
    else:
        # Fallback if columns missing (shouldn't happen with data_loader)
        return pd.DataFrame()

    last_date = history_df['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
    
    predictions = []
    
    # Initial state for recursion
    current_history = history_df.copy()
    
    for date in future_dates:
        # Prepare features for this date
        X = prepare_features(state, district, market, date, current_history)
        
        if X is not None:
            # Predict
            pred_price = model.predict(X)[0]
            
            # Apply scenarios
            shock_effect = pred_price * (supply_shock / 100.0)
            transport_effect = pred_price * (transport_cost - 1.0)
            # Seasonality is implicitly in the model (month feature), but we can amplify it
            # Simple amplification: (pred - mean) * strength + mean
            
            final_price = pred_price + shock_effect + transport_effect
            final_price = max(0, final_price)
            
            predictions.append(final_price)
            
            # Update history for next step (Recursive)
            new_row = {
                'Date': date,
                'modal_price_rs_qtl': final_price,
                'market': market,
                'state': state,
                'district': district,
                'arrivals_tonnes': current_history['arrivals_tonnes'].iloc[-1], # Assume constant
                'temp_avg': current_history['temp_avg'].iloc[-1] if 'temp_avg' in current_history else 25,
                'rain': 0 # Assume no rain in future? Or seasonal avg
            }
            current_history = pd.concat([current_history, pd.DataFrame([new_row])], ignore_index=True)
        else:
            predictions.append(0)

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions,
        'Lower_CI': [p * 0.9 for p in predictions], # Simple CI for now
        'Upper_CI': [p * 1.1 for p in predictions]
    })
    
    return forecast_df

def predict_custom_date(state, district, market, date, history_df):
    """
    Predict price for a specific custom date.
    """
    if not load_model_artifacts():
        return None
        
    X = prepare_features(state, district, market, date, history_df)
    
    if X is not None:
        price = model.predict(X)[0]
        return max(0, price)
    return None
