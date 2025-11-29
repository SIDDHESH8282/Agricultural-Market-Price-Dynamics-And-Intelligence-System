from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from typing import List, Optional

app = FastAPI(title="Onion Price Forecast API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model Artifacts ---
ARTIFACTS_DIR = '../model_artifacts'  # Assuming running from backend/ dir
if not os.path.exists(ARTIFACTS_DIR):
    ARTIFACTS_DIR = 'model_artifacts' # Fallback if running from root

try:
    # Load Price Model
    with open(os.path.join(ARTIFACTS_DIR, 'xgb_price_model.pkl'), 'rb') as f:
        price_model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'price_encodings.pkl'), 'rb') as f:
        price_encodings = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'price_features.pkl'), 'rb') as f:
        price_features = pickle.load(f)
        
    # Load Arrivals Model
    with open(os.path.join(ARTIFACTS_DIR, 'xgb_arrivals_model.pkl'), 'rb') as f:
        arrival_model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'arrival_encodings.pkl'), 'rb') as f:
        arrival_encodings = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'arrival_features.pkl'), 'rb') as f:
        arrival_features = pickle.load(f)
        
    print("SUCCESS: Dual AI Models (Price & Supply) loaded successfully.")
    
    # Load Temp Model
    with open(os.path.join(ARTIFACTS_DIR, 'xgb_temp_model.pkl'), 'rb') as f:
        temp_model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'temp_encodings.pkl'), 'rb') as f:
        temp_encodings = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'temp_features.pkl'), 'rb') as f:
        temp_features = pickle.load(f)

    # Load Rain Model
    with open(os.path.join(ARTIFACTS_DIR, 'xgb_rain_model.pkl'), 'rb') as f:
        rain_model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'rain_encodings.pkl'), 'rb') as f:
        rain_encodings = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'rain_features.pkl'), 'rb') as f:
        rain_features = pickle.load(f)
        
    print("SUCCESS: Weather AI Models (Temp & Rain) loaded successfully.")

except Exception as e:
    print(f"ERROR loading artifacts: {e}")
    price_model = None
    arrival_model = None
    temp_model = None
    rain_model = None

# --- Load Data (for history) ---
DATA_FILE = '../preprocessed_final_2014_2025_FULL_FEATURES_corrected_SORTED_NO_NULLS.csv'
if not os.path.exists(DATA_FILE):
    DATA_FILE = 'preprocessed_final_2014_2025_FULL_FEATURES_corrected_SORTED_NO_NULLS.csv'

try:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    # Basic preprocessing for lookup
    date_col = next((c for c in df.columns if 'date' in c.lower()), 'date')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.rename(columns={date_col: 'Date', 'modal_price_rs_qtl': 'Price', 'arrivals_tonnes': 'Arrivals'})
    df['market_id'] = df['state'] + '_' + df['district'] + '_' + df['market']
    print("SUCCESS: Data loaded successfully.")
except Exception as e:
    print(f"ERROR loading data: {e}")
    df = pd.DataFrame()

# --- Pydantic Models ---
class ForecastRequest(BaseModel):
    state: str
    district: str
    market: str
    horizon: int = 30
    supply_shock: float = 0.0
    transport_cost: float = 1.0
    seasonality: float = 1.0

class CustomDateRequest(BaseModel):
    state: str
    district: str
    market: str
    date: str # YYYY-MM-DD

class MarketSummaryRequest(BaseModel):
    state: str
    district: str
    market: str

class MarketListRequest(BaseModel):
    state: str
    district: str

class ArbitrageRequest(BaseModel):
    source_state: str
    source_district: str
    source_market: str
    dest_state: str
    dest_district: str
    dest_market: str

class MarketPredictionRequest(BaseModel):
    state: str
    district: str
    market: str
    horizon: int = 30  # 30, 90, or 180 days

# --- Helper Functions ---
def prepare_features(state, district, market, date_obj, history_df, model_type='price'):
    market_id = f"{state}_{district}_{market}"
    market_df = history_df[history_df['market'] == market].sort_values('Date')
    
    if market_df.empty:
        return None
        
    # Select Encodings & Features based on model type
    if model_type == 'price':
        curr_encodings = price_encodings
        curr_features = price_features
        target_col = 'Price'
    else:
        curr_encodings = arrival_encodings
        curr_features = arrival_features
        target_col = 'Arrivals'

    # Encodings
    state_enc = curr_encodings['state'].get(state, np.mean(list(curr_encodings['state'].values())))
    dist_enc = curr_encodings['district'].get(district, np.mean(list(curr_encodings['district'].values())))
    market_enc = curr_encodings['market_id'].get(market_id, np.mean(list(curr_encodings['market_id'].values())))
    
    # Lags (Naive: use latest available)
    last_row = market_df.iloc[-1]
    lag_1 = last_row[target_col]
    lag_7 = market_df.iloc[-7][target_col] if len(market_df) >= 7 else lag_1
    lag_30 = market_df.iloc[-30][target_col] if len(market_df) >= 30 else lag_1
    
    roll_mean_7 = market_df.tail(7)[target_col].mean()
    roll_mean_30 = market_df.tail(30)[target_col].mean()
    
    feature_dict = {
        'state_encoded': state_enc,
        'district_encoded': dist_enc,
        'market_id_encoded': market_enc,
        'month': date_obj.month,
        'year': date_obj.year,
        'day_of_week': date_obj.dayofweek,
        'lag_1': lag_1,
        'lag_7': lag_7,
        'lag_30': lag_30,
        'roll_mean_7': roll_mean_7,
        'roll_mean_30': roll_mean_30,
        'arrivals_tonnes': last_row['Arrivals'], # Only used for price model
        'temp_avg': last_row.get('temp_avg', 25.0),
        'rain': 0.0
    }
    
    return pd.DataFrame([[feature_dict.get(f, 0) for f in curr_features]], columns=curr_features)

def generate_ai_insight(predictions, dates, params: ForecastRequest, current_price):
    if not predictions:
        return {}
    
    start_price = predictions[0]
    end_price = predictions[-1]
    max_price = max(predictions)
    min_price = min(predictions)
    
    pct_change = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0
    
    # 1. Trend Analysis
    trend = "stable"
    if pct_change > 15:
        trend = "surging significantly"
    elif pct_change > 5:
        trend = "rising moderately"
    elif pct_change < -15:
        trend = "crashing"
    elif pct_change < -5:
        trend = "falling"
        
    summary = f"Over the next {params.horizon} days, onion prices in {params.market} are expected to be **{trend}**. The price is projected to move from ₹{start_price:.0f} to ₹{end_price:.0f} ({pct_change:+.1f}%)."

    # 2. Driver Analysis
    drivers = []
    if params.supply_shock > 5:
        drivers.append(f"**Supply Shortage**: The {params.supply_shock}% supply shock is the primary driver pushing prices up.")
    elif params.supply_shock < -5:
        drivers.append(f"**Supply Surplus**: A {abs(params.supply_shock)}% excess in supply is keeping prices suppressed.")
        
    if params.transport_cost > 1.2:
        drivers.append(f"**High Transport Costs**: Elevated fuel/transport costs (x{params.transport_cost}) are adding a premium to the final price.")
        
    # Seasonality check (simple heuristic based on month)
    start_date = datetime.strptime(dates[0], '%Y-%m-%d')
    if start_date.month in [9, 10, 11]: # Late monsoon/winter
        drivers.append("**Seasonal Effect**: Historical patterns suggest volatility during this late monsoon period.")
    elif start_date.month in [3, 4, 5]: # Summer
        drivers.append("**Seasonal Effect**: Summer harvest arrivals typically stabilize prices.")

    if not drivers:
        drivers.append("**Market Stability**: No major external shocks detected; price movement is driven by standard demand-supply cycles.")

    # 3. Strategic Advice
    advice = ""
    action = "HOLD"
    if pct_change > 10:
        action = "BUY NOW"
        advice = "Prices are likely to spike. **Buyers** should stock up immediately to avoid higher costs. **Farmers** might benefit from holding stock slightly longer if storage permits."
    elif pct_change < -10:
        action = "WAIT"
        advice = "Prices are expected to drop. **Buyers** should wait for the market to cool down. **Farmers** should consider selling early to avoid value loss."
    else:
        advice = "The market is relatively stable. Standard trading practices recommended."

    return {
        "summary": summary,
        "drivers": drivers,
        "advice": advice,
        "action": action
    }

def generate_weekly_summary(dates, prices, market_name):
    if not prices:
        return {}
    
    start_date = datetime.strptime(dates[0], '%Y-%m-%d')
    end_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    date_range = f"{start_date.strftime('%d %b %y')} - {end_date.strftime('%d %b %y')}"
    
    min_p, max_p = min(prices), max(prices)
    start_p, end_p = prices[0], prices[-1]
    
    # Trend Logic
    trend_desc = ""
    if end_p > start_p * 1.05:
        trend_desc = "Prices are rising compared to the start of the week, driven by strong buyer demand amid limited arrivals."
    elif end_p < start_p * 0.95:
        trend_desc = "Prices are showing a downward trend due to increased market arrivals and reduced buying pressure."
    else:
        trend_desc = "Prices are remaining relatively stable with minor fluctuations, indicating a balanced demand-supply scenario."

    # Drivers Logic (Seasonality)
    month = start_date.month
    drivers = "Steady demand from local retail markets."
    if month in [10, 11, 12]:
        drivers = "Increased demand from upcoming festivals and export orders supporting higher prices."
    elif month in [4, 5]:
        drivers = "Peak harvest arrivals are putting downward pressure on prices."
    elif month in [7, 8]:
        drivers = "Monsoon constraints on transport are slightly impacting supply."

    # Tip
    tip = ""
    if "rising" in trend_desc:
        tip = "Farmers and traders should focus on quality sorting and timely selling to capitalize on current demand. Proper storage is essential."
    elif "downward" in trend_desc:
        tip = "Buyers might find good opportunities to stock up as prices soften. Farmers should monitor daily rates closely."
    else:
        tip = "Traders should maintain regular inventory levels. Farmers can sell in batches to average out price fluctuations."

    return {
        "title": f"Weekly Outlook for {market_name}",
        "date_range": date_range,
        "sections": [
            {"title": "Market Trend", "content": trend_desc},
            {"title": "Key Drivers", "content": drivers},
            {"title": "Actionable Tip", "content": tip}
        ]
    }

# --- Endpoints ---

@app.get("/meta/locations")
def get_locations():
    if df.empty:
        return {"states": [], "districts": {}, "markets": {}}
    
    states = sorted(df['state'].unique().tolist())
    districts = df.groupby('state')['district'].unique().apply(sorted).to_dict()
    
    markets = df.groupby('district')['market'].unique().apply(sorted).to_dict()
    
    return {"states": states, "districts": districts, "markets": markets}

@app.post("/predict/horizon")
def predict_horizon(req: ForecastRequest):
    if price_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Filter history
    market_df = df[(df['state'] == req.state) & (df['district'] == req.district) & (df['market'] == req.market)]
    if market_df.empty:
        raise HTTPException(status_code=404, detail="Market data not found")
    
    last_date = market_df['Date'].max()
    current_price = market_df.iloc[-1]['Price']
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, req.horizon + 1)]
    
    price_preds = []
    arrival_preds = []
    dates = []
    
    current_history = market_df.copy()
    
    for date in future_dates:
        # Predict Price
        X_price = prepare_features(req.state, req.district, req.market, date, current_history, 'price')
        if X_price is not None:
            pred_p = float(price_model.predict(X_price)[0])
            
            # Scenarios
            shock = pred_p * (req.supply_shock / 100.0)
            transport = pred_p * (req.transport_cost - 1.0)
            final_price = max(0, pred_p + shock + transport)
            price_preds.append(final_price)
        else:
            price_preds.append(0.0)
            final_price = 0.0
            
        # Predict Arrivals
        X_arrival = prepare_features(req.state, req.district, req.market, date, current_history, 'arrival')
        if X_arrival is not None:
            pred_a = float(arrival_model.predict(X_arrival)[0])
            arrival_preds.append(max(0, pred_a))
        else:
            arrival_preds.append(0.0)
            pred_a = 0.0

        dates.append(date.strftime('%Y-%m-%d'))
        
        # Update history for recursion (simplified)
        new_row = {
            'Date': date,
            'Price': final_price,
            'market': req.market,
            'state': req.state,
            'district': req.district,
            'Arrivals': pred_a,
            'temp_avg': current_history['temp_avg'].iloc[-1] if 'temp_avg' in current_history else 25,
            'rain': 0
        }
        current_history = pd.concat([current_history, pd.DataFrame([new_row])], ignore_index=True)
            
    # Generate Insights
    insight = generate_ai_insight(price_preds, dates, req, current_price)
    
    # Generate Weekly Summary (using first 7 days of forecast)
    weekly_summary = {}
    if len(price_preds) >= 7:
        weekly_summary = generate_weekly_summary(dates[:7], price_preds[:7], req.market)
            
    return {
        "dates": dates,
        "prices": price_preds,
        "arrivals": arrival_preds,
        "lower_ci": [p * 0.9 for p in price_preds],
        "upper_ci": [p * 1.1 for p in price_preds],
        "ai_insight": insight,
        "weekly_summary": weekly_summary
    }

@app.post("/predict/custom")
def predict_custom(req: CustomDateRequest):
    if price_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    target_date = pd.to_datetime(req.date)
    market_df = df[(df['state'] == req.state) & (df['district'] == req.district) & (df['market'] == req.market)]
    
    if market_df.empty:
        raise HTTPException(status_code=404, detail="Market data not found")
        
    last_known_date = market_df['Date'].max()
    
    days_to_predict = (target_date - last_known_date).days + 7
    
    if days_to_predict <= 0:
         # Handle past dates or immediate dates roughly
         days_to_predict = 7 

    # Run recursive forecast
    current_history = market_df.copy()
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    
    price_preds = []
    arrival_preds = []
    dates = []
    
    for date in future_dates:
        # Predict Price
        X_price = prepare_features(req.state, req.district, req.market, date, current_history, 'price')
        if X_price is not None:
            pred_p = float(price_model.predict(X_price)[0])
            price_preds.append(pred_p)
        else:
            price_preds.append(0.0)
            pred_p = 0.0
            
        # Predict Arrivals
        X_arrival = prepare_features(req.state, req.district, req.market, date, current_history, 'arrival')
        if X_arrival is not None:
            pred_a = float(arrival_model.predict(X_arrival)[0])
            arrival_preds.append(max(0, pred_a))
        else:
            arrival_preds.append(0.0)
            pred_a = 0.0

        dates.append(date.strftime('%Y-%m-%d'))
            
        new_row = {
            'Date': date,
            'Price': pred_p,
            'market': req.market,
            'state': req.state,
            'district': req.district,
            'Arrivals': pred_a,
            'temp_avg': current_history['temp_avg'].iloc[-1] if 'temp_avg' in current_history else 25,
            'rain': 0
        }
        current_history = pd.concat([current_history, pd.DataFrame([new_row])], ignore_index=True)

    target_date_str = req.date
    try:
        idx = dates.index(target_date_str)
        target_price = price_preds[idx]
        target_arrival = arrival_preds[idx]
        
        next_7_dates = dates[idx:idx+7]
        next_7_prices = price_preds[idx:idx+7]
        next_7_arrivals = arrival_preds[idx:idx+7]
        
        weekly_summary = generate_weekly_summary(next_7_dates, next_7_prices, req.market)
        
        return {
            "date": req.date, 
            "price": max(0, target_price),
            "arrival": max(0, target_arrival),
            "forecast_7_days": {
                "dates": next_7_dates,
                "prices": next_7_prices,
                "arrivals": next_7_arrivals
            },
            "weekly_summary": weekly_summary
        }
        
    except ValueError:
        # Fallback if date not found in generated range
        return {
             "date": req.date,
             "price": 0,
             "arrival": 0,
             "forecast_7_days": None,
             "weekly_summary": None
        }

@app.get("/trading/insights")
def get_trading_insights():
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date].copy()
    
    if latest_df.empty:
        latest_date = df.iloc[-1]['Date']
        latest_df = df[df['Date'] == latest_date].copy()

    sorted_df = latest_df.sort_values('Price')
    
    buy_markets = sorted_df.head(5)[['market', 'district', 'state', 'Price']].to_dict('records')
    sell_markets = sorted_df.tail(5).sort_values('Price', ascending=False)[['market', 'district', 'state', 'Price']].to_dict('records')
    
    start_date = latest_date - timedelta(days=30)
    trend_df = df[df['Date'] >= start_date]
    
    buy_trends = []
    for m in buy_markets:
        m_data = trend_df[(trend_df['market'] == m['market']) & (trend_df['district'] == m['district'])]
        buy_trends.append({
            "market": m['market'],
            "data": m_data[['Date', 'Price']].sort_values('Date').to_dict('records')
        })
        
    sell_trends = []
    for m in sell_markets:
        m_data = trend_df[(trend_df['market'] == m['market']) & (trend_df['district'] == m['district'])]
        sell_trends.append({
            "market": m['market'],
            "data": m_data[['Date', 'Price']].sort_values('Date').to_dict('records')
        })

    best_buy = buy_markets[0]
    best_sell = sell_markets[0]
    spread = best_sell['Price'] - best_buy['Price']
    
    ai_summary = f"""
    <p><b>Market Arbitrage Opportunity:</b> The market currently shows a significant price disparity of <b>₹{spread:.0f}/qtl</b>. 
    The most attractive opportunity is to source high-quality onions from <b>{best_buy['market']}</b> ({best_buy['state']}) at <b>₹{best_buy['Price']:.0f}</b> 
    and target the <b>{best_sell['market']}</b> ({best_sell['state']}) market where prices are peaking at <b>₹{best_sell['Price']:.0f}</b>.</p>
    
    <p><b>Strategic Action:</b> Traders should execute immediate buy orders in the <b>{best_buy['district']}</b> region. 
    Given the 30-day trend, prices in {best_sell['market']} are {'rising' if sell_trends[0]['data'][-1]['Price'] > sell_trends[0]['data'][0]['Price'] else 'stabilizing'}, 
    making it a prime destination for offloading stock.</p>
    
    <p><b>Risk Assessment:</b> Ensure transport logistics are secured as the distance may impact net margins. 
    Verify quality parameters (size, moisture) in {best_buy['market']} to avoid rejection at the destination mandi.</p>
    """

    return {
        "latest_date": latest_date.strftime('%Y-%m-%d'),
        "buy_markets": buy_markets,
        "sell_markets": sell_markets,
        "buy_trends": buy_trends,
        "sell_trends": sell_trends,
        "ai_summary": ai_summary
    }

@app.post("/meta/markets")
def get_markets_with_prices(req: MarketListRequest):
    if df.empty:
        return []
        
    latest_date = df['Date'].max()
    district_df = df[
        (df['state'] == req.state) & 
        (df['district'] == req.district) & 
        (df['Date'] == latest_date)
    ].copy()
    
    if district_df.empty:
        district_df = df[
            (df['state'] == req.state) & 
            (df['district'] == req.district)
        ].sort_values('Date').groupby('market').last().reset_index()

    district_df = district_df.sort_values('Price')
    return district_df[['market', 'Price']].to_dict(orient='records')

@app.post("/market-summary")
def get_market_summary(req: MarketSummaryRequest):
    market_df = df[(df['state'] == req.state) & (df['district'] == req.district) & (df['market'] == req.market)].sort_values('Date')
    
    if market_df.empty:
        raise HTTPException(status_code=404, detail="Market data not found")
    
    last_30 = market_df.tail(30)
    if last_30.empty:
        return {}
        
    avg_price = last_30['Price'].mean()
    min_price = last_30['Price'].min()
    max_price = last_30['Price'].max()
    latest_price = market_df.iloc[-1]['Price']
    latest_arrivals = market_df.iloc[-1]['Arrivals']
    
    # Weather Stats
    avg_temp = last_30['temp_avg'].mean() if 'temp_avg' in last_30 else 0
    total_rain = last_30['rain'].sum() if 'rain' in last_30 else 0
    
    prev_30 = market_df.iloc[-60:-30]
    if not prev_30.empty:
        prev_avg = prev_30['Price'].mean()
        prev_arrivals_avg = prev_30['Arrivals'].mean()
        change_pct = ((avg_price - prev_avg) / prev_avg) * 100
        arrivals_change_pct = ((last_30['Arrivals'].mean() - prev_arrivals_avg) / prev_arrivals_avg) * 100 if prev_arrivals_avg > 0 else 0
    else:
        change_pct = 0.0
        arrivals_change_pct = 0.0
    
    # DETAILED PRICE FLUCTUATION ANALYSIS
    price_factors = []
    
    # 1. Supply-Demand Analysis
    if arrivals_change_pct < -20:
        impact_pct = 40
        price_factors.append({
            "factor": "Supply Shortage",
            "impact_pct": impact_pct,
            "explanation": f"Arrivals dropped by {abs(arrivals_change_pct):.1f}%, creating supply constraints",
            "evidence": f"Average arrivals: {last_30['Arrivals'].mean():.0f} tonnes vs previous {prev_arrivals_avg:.0f} tonnes"
        })
    elif arrivals_change_pct > 20:
        impact_pct = 35
        price_factors.append({
            "factor": "Supply Surplus",
            "impact_pct": impact_pct,
            "explanation": f"Arrivals surged by {arrivals_change_pct:.1f}%, flooding the market",
            "evidence": f"Average arrivals: {last_30['Arrivals'].mean():.0f} tonnes vs previous {prev_arrivals_avg:.0f} tonnes"
        })
    
    # 2. Weather Impact
    weather_impact = 0
    if total_rain > 50:
        weather_impact = 25
        price_factors.append({
            "factor": "Weather Disruption",
            "impact_pct": weather_impact,
            "explanation": "Heavy rainfall disrupting harvest and transport logistics",
            "evidence": f"Total rainfall: {total_rain:.1f}mm in last 30 days"
        })
    elif avg_temp > 35:
        weather_impact = 15
        price_factors.append({
            "factor": "Heat Stress",
            "impact_pct": weather_impact,
            "explanation": "High temperatures affecting storage quality and shelf life",
            "evidence": f"Average temperature: {avg_temp:.1f}°C"
        })
    
    # 3. Seasonal Effects
    current_month = market_df.iloc[-1]['Date'].month
    if current_month in [10, 11, 12]:  # Festival season
        seasonal_impact = 20
        price_factors.append({
            "factor": "Seasonal Demand",
            "impact_pct": seasonal_impact,
            "explanation": "Festival season increasing consumer demand",
            "evidence": "October-December period with traditional high consumption"
        })
    elif current_month in [3, 4, 5]:  # Harvest season
        seasonal_impact = 15
        price_factors.append({
            "factor": "Harvest Season",
            "impact_pct": seasonal_impact,
            "explanation": "Peak harvest arrivals moderating prices",
            "evidence": "March-May harvest period increasing supply"
        })
    
    # 4. Price Volatility
    price_std = last_30['Price'].std()
    volatility_ratio = (price_std / avg_price * 100) if avg_price > 0 else 0
    if volatility_ratio > 15:
        price_factors.append({
            "factor": "High Volatility",
            "impact_pct": 10,
            "explanation": "Significant price swings indicating market uncertainty",
            "evidence": f"Price volatility: {volatility_ratio:.1f}%"
        })
    
    # Normalize impact percentages
    total_impact = sum(f['impact_pct'] for f in price_factors)
    if total_impact > 0:
        for factor in price_factors:
            factor['impact_pct'] = round((factor['impact_pct'] / total_impact) * 100, 1)
    
    # AGRO-CLIMATIC INSIGHTS
    agro_climatic = {
        "growing_conditions": "Favorable" if 20 < avg_temp < 32 and 10 < total_rain < 100 else "Challenging",
        "soil_moisture": "Adequate" if total_rain > 20 else "Low",
        "pest_risk": "High" if total_rain > 80 and avg_temp > 28 else "Low",
        "harvest_outlook": "Good" if arrivals_change_pct > 10 else "Moderate",
        "quality_index": "Premium" if volatility_ratio < 10 else "Standard"
    }
    
    # Detailed agro commentary
    agro_commentary = []
    if agro_climatic["growing_conditions"] == "Favorable":
        agro_commentary.append("Current weather conditions are optimal for onion cultivation with balanced temperature and rainfall.")
    else:
        if avg_temp > 35:
            agro_commentary.append("High temperatures may stress crops and reduce bulb quality.")
        if total_rain > 100:
            agro_commentary.append("Excessive rainfall increases disease pressure and may delay harvest.")
    
    if agro_climatic["pest_risk"] == "High":
        agro_commentary.append("Humid conditions elevate risk of fungal diseases like purple blotch and downy mildew.")
    
    # SUPPLY CHAIN ANALYSIS
    # Find major source districts
    state_df = df[(df['state'] == req.state) & (df['Date'] >= market_df['Date'].max() - timedelta(days=7))]
    district_arrivals = state_df.groupby('district')['Arrivals'].sum().sort_values(ascending=False)
    
    major_sources = [
        {"district": dist, "contribution_pct": round((arr / district_arrivals.sum()) * 100, 1)}
        for dist, arr in district_arrivals.head(5).items()
    ]
    
    supply_chain = {
        "major_sources": major_sources,
        "transport_status": "Normal" if weather_impact < 20 else "Disrupted",
        "storage_utilization": "High" if current_month in [1, 2, 3, 4] else "Moderate",
        "market_efficiency": "Efficient" if volatility_ratio < 12 else "Volatile"
    }
    
    # PRICE FORECAST (Next 7 Days)
    try:
        # Simple trend-based forecast
        recent_trend = latest_price - last_30.iloc[0]['Price']
        trend_direction = "rising" if recent_trend > 0 else "falling"
        forecast_7d = latest_price + (recent_trend / 30 * 7)  # Extrapolate
        
        forecast_risks = []
        if weather_impact > 20:
            forecast_risks.append("Weather disruption may cause price spikes")
        if volatility_ratio > 15:
            forecast_risks.append("High volatility increases forecast uncertainty")
        if arrivals_change_pct < -15:
            forecast_risks.append("Supply shortage likely to push prices higher")
        
        price_forecast = {
            "next_7_days": round(forecast_7d, 2),
            "trend": trend_direction,
            "confidence": "High" if volatility_ratio < 10 else "Moderate" if volatility_ratio < 15 else "Low",
            "risks": forecast_risks if forecast_risks else ["Market conditions appear stable"]
        }
    except:
        price_forecast = None
    
    # Market comparison
    latest_date = df['Date'].max()
    recent_state_df = df[
        (df['state'] == req.state) & 
        (df['Date'] >= latest_date - timedelta(days=7))
    ]
    
    state_markets = recent_state_df.sort_values('Date').groupby(['district', 'market']).last().reset_index()
    top_markets = state_markets.sort_values('Price', ascending=False).head(5)
    
    comparison_list = []
    for _, row in top_markets.iterrows():
        comparison_list.append({
            "market": row['market'],
            "district": row['district'],
            "price": row['Price'],
            "is_selected": row['market'] == req.market
        })
    
    return {
        "stats": {
            "avg_price": avg_price,
            "min_price": min_price,
            "max_price": max_price,
            "latest_price": latest_price,
            "change_pct": change_pct,
            "total_arrivals": last_30['Arrivals'].sum(),
            "avg_temp": avg_temp,
            "total_rain": total_rain,
            "volatility": round(volatility_ratio, 2)
        },
        "comparison": comparison_list,
        "price_analysis": {
            "factors": price_factors,
            "trend": "rising" if change_pct > 5 else "falling" if change_pct < -5 else "stable",
            "magnitude": abs(change_pct)
        },
        "agro_climatic": {
            **agro_climatic,
            "commentary": agro_commentary
        },
        "supply_chain": supply_chain,
        "forecast": price_forecast
    }

@app.post("/trading/analyze")
def analyze_arbitrage(req: ArbitrageRequest):
    source_df = df[
        (df['state'] == req.source_state) & 
        (df['district'] == req.source_district) & 
        (df['market'] == req.source_market)
    ].sort_values('Date').tail(30)
    
    dest_df = df[
        (df['state'] == req.dest_state) & 
        (df['district'] == req.dest_district) & 
        (df['market'] == req.dest_market)
    ].sort_values('Date').tail(30)
    
    if source_df.empty or dest_df.empty:
        raise HTTPException(status_code=404, detail="Market data not found")
        
    current_source = source_df.iloc[-1]['Price']
    current_dest = dest_df.iloc[-1]['Price']
    spread = current_dest - current_source
    
    recommendation = "HOLD"
    risk = "LOW"
    reason = ""
    
    if spread > 500:
        recommendation = "STRONG BUY"
        risk = "MEDIUM"
        reason = f"Massive spread of ₹{spread:.0f}/qtl detected. This is a prime arbitrage window."
    elif spread > 200:
        recommendation = "BUY"
        risk = "LOW"
        reason = f"Healthy spread of ₹{spread:.0f}/qtl. Good for standard trade volumes."
    elif spread < 0:
        recommendation = "AVOID"
        risk = "HIGH"
        reason = "Negative spread. You will lose money on this trade."
    else:
        recommendation = "WAIT"
        risk = "LOW"
        reason = "Spread is too thin to cover transport and handling costs."

    return {
        "source_history": source_df[['Date', 'Price']].to_dict(orient='records'),
        "dest_history": dest_df[['Date', 'Price']].to_dict(orient='records'),
        "analysis": {
            "spread": spread,
            "recommendation": recommendation,
            "risk": risk,
            "reason": reason
        }
    }

@app.get("/history")
def get_history(state: str, district: str, market: str):
    market_df = df[(df['state'] == state) & (df['district'] == district) & (df['market'] == market)]
    if market_df.empty:
        return []
    
    # Return last 365 days
    recent = market_df.sort_values('Date').tail(365)
    # Ensure columns exist
    cols = ['Date', 'Price', 'Arrivals']
    if 'temp_avg' in recent.columns:
        cols.append('temp_avg')
    if 'rain' in recent.columns:
        cols.append('rain')
        
    return recent[cols].to_dict(orient='records')

def prepare_weather_features(state, district, market, date_obj, history_df, model_type='temp'):
    market_id = f"{state}_{district}_{market}"
    market_df = history_df[history_df['market'] == market].sort_values('Date')
    
    if market_df.empty:
        return None
        
    if model_type == 'temp':
        curr_encodings = temp_encodings
        curr_features = temp_features
        target_col = 'temp_avg'
    else:
        curr_encodings = rain_encodings
        curr_features = rain_features
        target_col = 'rain'
        
    # Encodings
    state_enc = curr_encodings['state'].get(state, np.mean(list(curr_encodings['state'].values())))
    dist_enc = curr_encodings['district'].get(district, np.mean(list(curr_encodings['district'].values())))
    market_enc = curr_encodings['market_id'].get(market_id, np.mean(list(curr_encodings['market_id'].values())))
    
    # Lags
    last_row = market_df.iloc[-1]
    # Handle missing target in history if any
    val = last_row.get(target_col, 0)
    
    lag_1 = val
    lag_7 = market_df.iloc[-7][target_col] if len(market_df) >= 7 and target_col in market_df else lag_1
    lag_30 = market_df.iloc[-30][target_col] if len(market_df) >= 30 and target_col in market_df else lag_1
    
    roll_mean_7 = market_df.tail(7)[target_col].mean() if target_col in market_df else lag_1
    roll_mean_30 = market_df.tail(30)[target_col].mean() if target_col in market_df else lag_1
    
    feature_dict = {
        'state_encoded': state_enc,
        'district_encoded': dist_enc,
        'market_id_encoded': market_enc,
        'month': date_obj.month,
        'year': date_obj.year,
        'day_of_week': date_obj.dayofweek,
        'lag_1': lag_1,
        'lag_7': lag_7,
        'lag_30': lag_30,
        'roll_mean_7': roll_mean_7,
        'roll_mean_30': roll_mean_30
    }
    
    return pd.DataFrame([[feature_dict.get(f, 0) for f in curr_features]], columns=curr_features)

def generate_weather_forecast_insight(dates, temps, rains):
    avg_temp = sum(temps) / len(temps)
    total_rain = sum(rains)
    max_temp = max(temps)
    min_temp = min(temps)
    days_with_rain = sum(1 for r in rains if r > 2.0)
    
    # --- 1. Detailed Analysis ---
    analysis = []
    analysis.append(f"Over the next {len(dates)} days, the average temperature is projected to be **{avg_temp:.1f}°C**, ranging from a low of {min_temp:.1f}°C to a high of {max_temp:.1f}°C.")
    
    if total_rain > 50:
        analysis.append(f"Significant rainfall of **{total_rain:.1f}mm** is expected, distributed over approximately {days_with_rain} days.")
    elif total_rain > 10:
        analysis.append(f"Moderate rainfall ({total_rain:.1f}mm) is expected, which may affect soil moisture levels.")
    else:
        analysis.append("Conditions are expected to remain mostly dry with minimal rainfall.")
        
    if max_temp > 35:
        analysis.append("Several days of high heat (>35°C) are anticipated, which could stress crops.")
    
    # --- 2. Yield Forecast & Onion Growth Impact ---
    yield_impact = ""
    onion_status = ""
    
    # Onion Optimal: 15-25°C. 
    # Stress: >30°C or <10°C.
    # Harvest: Needs dry weather.
    
    if 15 <= avg_temp <= 25 and total_rain < 50:
        onion_status = "🟢 **Excellent Growth Conditions**"
        yield_impact = "The forecasted temperature range (15-25°C) is **optimal for bulb development**. With controlled rainfall, yield quality is expected to be high. Bulbs should develop good size and firmness."
    elif avg_temp > 30:
        onion_status = "🔴 **Heat Stress Risk**"
        yield_impact = "High temperatures may accelerate maturity prematurely, potentially leading to **smaller bulb size** and lower overall yields. Heat stress can also increase susceptibility to thrips."
    elif total_rain > 100:
        onion_status = "🟠 **Rotting Risk**"
        yield_impact = "Excessive moisture poses a severe risk of **fungal diseases (Purple Blotch)** and bulb rotting. Yield quantity may be maintained, but **storage quality could be severely compromised**."
    elif avg_temp < 12:
        onion_status = "🟡 **Slow Growth / Bolting**"
        yield_impact = "Lower temperatures may slow down vegetative growth. If the crop is in the late stage, prolonged cold can induce **bolting** (flowering), which renders the bulb unmarketable."
    else:
        onion_status = "🔵 **Moderate Conditions**"
        yield_impact = "Weather conditions are within acceptable limits. Standard yield expectations apply, provided standard irrigation and pest management practices are followed."

    # --- 3. Actionable Advice ---
    advice = ""
    if total_rain > 80:
        advice = "⚠️ **Drainage Priority**: Ensure fields have adequate drainage to prevent waterlogging. Delay harvesting until a dry spell is predicted."
    elif max_temp > 38:
        advice = "💧 **Irrigation Alert**: Increase irrigation frequency to combat heat stress. Apply mulch to retain soil moisture."
    elif days_with_rain > 10 and avg_temp > 25:
        advice = "fungicide **Disease Watch**: Warm and wet conditions favor fungal growth. Prophylactic fungicide application is recommended."
    elif min_temp < 10:
        advice = "❄️ **Frost Protection**: Monitor for frost warnings. Avoid late-evening irrigation."
    else:
        advice = "✅ **Standard Operations**: Conditions are favorable for routine field operations, fertilizer application, or harvesting."

    # --- 4. Notes ---
    notes = "Forecast reliability decreases beyond 14 days. Monitor local daily updates for short-term planning."

    return {
        "summary": f"{onion_status}: {analysis[0]}",
        "detailed_analysis": " ".join(analysis),
        "yield_forecast": yield_impact,
        "notes": notes,
        "advice": advice,
        "metrics": {
            "avg_temp": f"{avg_temp:.1f}°C",
            "total_rain": f"{total_rain:.1f}mm",
            "rain_days": str(days_with_rain)
        }
    }

@app.post("/predict/weather")
def predict_weather(req: ForecastRequest):
    if temp_model is None or rain_model is None:
        raise HTTPException(status_code=500, detail="Weather models not loaded")
        
    market_df = df[(df['state'] == req.state) & (df['district'] == req.district) & (df['market'] == req.market)]
    if market_df.empty:
        raise HTTPException(status_code=404, detail="Market data not found")
        
    last_date = market_df['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, req.horizon + 1)]
    
    temp_preds = []
    rain_preds = []
    dates = []
    
    current_history = market_df.copy()
    # Ensure temp/rain cols exist
    if 'temp_avg' not in current_history.columns:
        current_history['temp_avg'] = 25.0
    if 'rain' not in current_history.columns:
        current_history['rain'] = 0.0
        
    for date in future_dates:
        # Predict Temp
        X_temp = prepare_weather_features(req.state, req.district, req.market, date, current_history, 'temp')
        if X_temp is not None:
            pred_t = float(temp_model.predict(X_temp)[0])
            temp_preds.append(pred_t)
        else:
            temp_preds.append(25.0)
            pred_t = 25.0
            
        # Predict Rain
        X_rain = prepare_weather_features(req.state, req.district, req.market, date, current_history, 'rain')
        if X_rain is not None:
            pred_r = float(rain_model.predict(X_rain)[0])
            pred_r = max(0, pred_r) # Rain can't be negative
            rain_preds.append(pred_r)
        else:
            rain_preds.append(0.0)
            pred_r = 0.0
            
        dates.append(date.strftime('%Y-%m-%d'))
        
        # Update history
        new_row = {
            'Date': date,
            'market': req.market,
            'state': req.state,
            'district': req.district,
            'temp_avg': pred_t,
            'rain': pred_r,
            'Price': 0, # Dummy
            'Arrivals': 0 # Dummy
        }
        current_history = pd.concat([current_history, pd.DataFrame([new_row])], ignore_index=True)
        
    insight = generate_weather_forecast_insight(dates, temp_preds, rain_preds)
    
    return {
        "dates": dates,
        "temps": temp_preds,
        "rains": rain_preds,
        "insight": insight
    }

class ArbitrageForecastRequest(BaseModel):
    source_state: str
    source_district: str
    source_market: str
    dest_state: str
    dest_district: str
    dest_market: str
    horizon: int = 30

@app.post("/trading/forecast-arbitrage")
def forecast_arbitrage(req: ArbitrageForecastRequest):
    if df.empty or price_model is None:
        raise HTTPException(status_code=500, detail="Data or Model not loaded")
        
    latest_date = df['Date'].max()
    future_dates = [latest_date + timedelta(days=i) for i in range(1, req.horizon + 1)]
    
    # Helper to predict for a market (Reusable logic)
    def get_market_preds(state, district, market):
        market_history = df[(df['state'] == state) & (df['district'] == district) & (df['market'] == market)].sort_values('Date')
        if market_history.empty:
            return []
            
        preds = []
        current_hist = market_history.copy()
        
        # Ensure temp/rain cols
        if 'temp_avg' not in current_hist.columns: current_hist['temp_avg'] = 25.0
        if 'rain' not in current_hist.columns: current_hist['rain'] = 0.0
        
        market_id = f"{state}_{district}_{market}"
        s_enc = price_encodings['state'].get(state, 0)
        d_enc = price_encodings['district'].get(district, 0)
        m_enc = price_encodings['market_id'].get(market_id, 0)
            
        for date in future_dates:
            # Lags
            lag_1 = current_hist.iloc[-1]['Price']
            lag_7 = current_hist.iloc[-7]['Price'] if len(current_hist) >= 7 else lag_1
            lag_30 = current_hist.iloc[-30]['Price'] if len(current_hist) >= 30 else lag_1
            roll_7 = current_hist.tail(7)['Price'].mean()
            roll_30 = current_hist.tail(30)['Price'].mean()
            
            # Arrivals (Assume last known for simplicity in this context)
            arrivals = current_hist.iloc[-1]['Arrivals']
            
            features = pd.DataFrame([[
                s_enc, d_enc, m_enc,
                date.month, date.year, date.dayofweek,
                lag_1, lag_7, lag_30, roll_7, roll_30, arrivals
            ]], columns=price_features)
            
            pred_price = float(price_model.predict(features)[0])
            preds.append({"Date": date.strftime('%Y-%m-%d'), "Price": pred_price})
            
            # Update history
            new_row = current_hist.iloc[-1].copy()
            new_row['Date'] = date
            new_row['Price'] = pred_price
            current_hist = pd.concat([current_hist, pd.DataFrame([new_row])], ignore_index=True)
            
        return preds

    source_preds = get_market_preds(req.source_state, req.source_district, req.source_market)
    dest_preds = get_market_preds(req.dest_state, req.dest_district, req.dest_market)
    
    if not source_preds or not dest_preds:
        raise HTTPException(status_code=404, detail="Insufficient data for prediction")
        
    # Analyze Opportunity
    analysis_data = []
    best_opp = None
    max_roi = -float('inf')
    
    for i in range(len(source_preds)):
        s_price = source_preds[i]['Price']
        d_price = dest_preds[i]['Price']
        spread = d_price - s_price
        roi = (spread / s_price) * 100 if s_price > 0 else 0
        
        item = {
            "date": source_preds[i]['Date'],
            "source_price": s_price,
            "dest_price": d_price,
            "spread": spread,
            "roi": roi
        }
        analysis_data.append(item)
        
        if roi > max_roi:
            max_roi = roi
            best_opp = item
            
    # AI Insight Generation
    insight = ""
    if best_opp and best_opp['roi'] > 15:
        insight = f"""
        <p><b>🚀 High ROI Opportunity Detected:</b> The model predicts a peak trading window around <b>{best_opp['date']}</b>.</p>
        <p><b>Financial Projection:</b>
        <ul>
            <li><b>Buy at:</b> ₹{best_opp['source_price']:.0f} ({req.source_market})</li>
            <li><b>Sell at:</b> ₹{best_opp['dest_price']:.0f} ({req.dest_market})</li>
            <li><b>Gross Margin:</b> ₹{best_opp['spread']:.0f}/qtl</li>
            <li><b>Expected ROI:</b> <span class="text-green-600 font-bold">{best_opp['roi']:.1f}%</span></li>
        </ul>
        </p>
        <p><b>Strategy:</b> Accumulate stock in {req.source_market} 3-5 days prior to {best_opp['date']} to capitalize on the widening spread. The destination market ({req.dest_market}) is showing a strong upward trend.</p>
        """
    elif best_opp and best_opp['roi'] > 5:
        insight = f"""
        <p><b>⚠️ Moderate Opportunity:</b> A positive spread exists, peaking on <b>{best_opp['date']}</b> with an ROI of <b>{best_opp['roi']:.1f}%</b>.</p>
        <p><b>Advisory:</b> Margins are tight. Ensure transport costs are below ₹{best_opp['spread'] * 0.6:.0f}/qtl to maintain profitability. Strictly monitor quality to avoid deductions.</p>
        """
    else:
        insight = f"""
        <p><b>🛑 Market Warning:</b> No significant arbitrage opportunities detected in the next {req.horizon} days.</p>
        <p><b>Analysis:</b> Price convergence between {req.source_market} and {req.dest_market} erodes potential margins. Recommend exploring alternative destination markets.</p>
        """

    return {
        "forecast": analysis_data,
        "best_opportunity": best_opp,
        "ai_analysis": insight
    }

@app.post("/trading/predict-market")
def predict_market_opportunity(req: MarketPredictionRequest):
    """
    Advanced market prediction with multi-horizon forecasting and ROI analysis
    """
    if price_model is None or arrival_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Get market data
    market_df = df[(df['state'] == req.state) & (df['district'] == req.district) & (df['market'] == req.market)].copy()
    
    if market_df.empty:
        raise HTTPException(status_code=404, detail="Market not found")
    
    last_date = market_df['Date'].max()
    current_price = market_df[market_df['Date'] == last_date]['Price'].iloc[0]
    current_arrivals = market_df[market_df['Date'] == last_date]['Arrivals'].iloc[0]
    
    # Generate predictions for the requested horizon
    horizon_days = req.horizon
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
    
    current_history = market_df.copy()
    price_preds = []
    arrival_preds = []
    dates_list = []
    
    for date in future_dates:
        # Predict Price
        X_price = prepare_features(req.state, req.district, req.market, date, current_history, 'price')
        if X_price is not None:
            pred_p = float(price_model.predict(X_price)[0])
            price_preds.append(max(0, pred_p))
        else:
            price_preds.append(current_price)
            pred_p = current_price
        
        # Predict Arrivals
        X_arrival = prepare_features(req.state, req.district, req.market, date, current_history, 'arrival')
        if X_arrival is not None:
            pred_a = float(arrival_model.predict(X_arrival)[0])
            arrival_preds.append(max(0, pred_a))
        else:
            arrival_preds.append(current_arrivals)
            pred_a = current_arrivals
        
        dates_list.append(date.strftime('%Y-%m-%d'))
        
        # Update history for next iteration
        new_row = {
            'Date': date,
            'Price': pred_p,
            'market': req.market,
            'state': req.state,
            'district': req.district,
            'Arrivals': pred_a,
            'temp_avg': current_history['temp_avg'].iloc[-1] if 'temp_avg' in current_history else 25,
            'rain': 0
        }
        current_history = pd.concat([current_history, pd.DataFrame([new_row])], ignore_index=True)
    
    # Calculate confidence intervals based on historical volatility
    hist_30d = market_df.tail(30)['Price']
    volatility = hist_30d.std() if len(hist_30d) > 1 else 0
    if pd.isna(volatility):
        volatility = 0
    
    lower_ci = []
    upper_ci = []
    for i, price in enumerate(price_preds):
        days_out = i + 1
        uncertainty = 1.96 * volatility * np.sqrt(days_out / 30)
        lower_ci.append(max(0, price - uncertainty))
        upper_ci.append(price + uncertainty)
    
    # ROI Analysis
    peak_price = max(price_preds)
    low_price = min(price_preds)
    peak_idx = price_preds.index(peak_price)
    low_idx = price_preds.index(low_price)
    
    potential_gain_pct = ((peak_price - current_price) / current_price) * 100 if current_price > 0 else 0
    best_buy_date = dates_list[low_idx] if low_idx < peak_idx else dates_list[0]
    best_sell_date = dates_list[peak_idx]
    
    roi_from_low = ((peak_price - low_price) / low_price) * 100 if low_price > 0 else 0
    
    # AI Insights Generation
    avg_price = np.mean(price_preds)
    price_trend = "rising" if price_preds[-1] > price_preds[0] else "falling" if price_preds[-1] < price_preds[0] else "stable"
    trend_magnitude = abs(((price_preds[-1] - price_preds[0]) / price_preds[0]) * 100) if price_preds[0] > 0 else 0
    
    # Determine recommendation
    if potential_gain_pct > 15:
        recommendation = "BUY_NOW"
        rec_text = "Strong Buy"
    elif potential_gain_pct > 8:
        recommendation = "BUY"
        rec_text = "Buy"
    elif potential_gain_pct < -10:
        recommendation = "WAIT"
        rec_text = "Wait"
    else:
        recommendation = "HOLD"
        rec_text = "Hold"
    
    # Generate detailed insights
    summary = f"Over the next {horizon_days} days, {req.market} prices are expected to be **{price_trend}** with a {trend_magnitude:.1f}% change. "
    summary += f"Current price of ₹{current_price:.0f} is projected to reach a peak of ₹{peak_price:.0f} around {best_sell_date}."
    
    # Price drivers
    price_drivers = []
    avg_arrivals = np.mean(arrival_preds)
    arrivals_change_pct = ((arrival_preds[-1] - arrival_preds[0]) / arrival_preds[0]) * 100 if arrival_preds[0] > 0 else 0
    
    if arrivals_change_pct < -15:
        price_drivers.append("**Supply Constraint**: Arrivals are expected to decline by {:.1f}%, putting upward pressure on prices.".format(abs(arrivals_change_pct)))
    elif arrivals_change_pct > 15:
        price_drivers.append("**Supply Surge**: Arrivals projected to increase by {:.1f}%, which may suppress price growth.".format(arrivals_change_pct))
    
    # Seasonal factors
    start_month = future_dates[0].month
    if start_month in [10, 11, 12]:
        price_drivers.append("**Seasonal Demand**: Festival season typically boosts demand and supports higher prices.")
    elif start_month in [4, 5]:
        price_drivers.append("**Harvest Season**: Peak arrivals during this period usually moderate price increases.")
    
    if volatility > current_price * 0.1:
        price_drivers.append("**High Volatility**: Market shows significant price swings, creating both risk and opportunity.")
    
    if not price_drivers:
        price_drivers.append("**Stable Market**: No major disruptions expected; prices likely to follow seasonal patterns.")
    
    # Opportunities
    opportunities = []
    if roi_from_low > 12:
        opportunities.append(f"**Timing Opportunity**: Buying around {best_buy_date} at ₹{low_price:.0f} and selling on {best_sell_date} could yield {roi_from_low:.1f}% returns.")
    
    if trend_magnitude > 10:
        opportunities.append(f"**Trend Play**: Strong {price_trend} trend offers clear directional opportunity for {trend_magnitude:.1f}% potential gain.")
    
    # Risks
    risks = []
    if volatility > current_price * 0.15:
        risks.append("**Price Volatility**: High price swings increase uncertainty and potential for unexpected losses.")
    
    if abs(arrivals_change_pct) > 25:
        risks.append("**Supply Uncertainty**: Large fluctuations in arrivals could disrupt price predictions.")
    
    if horizon_days > 90:
        risks.append("**Forecast Horizon**: Longer-term predictions have higher uncertainty due to unforeseen market events.")
    
    if not risks:
        risks.append("**Low Risk**: Market conditions appear relatively stable with moderate uncertainty.")
    
    # Detailed recommendation
    if recommendation == "BUY_NOW":
        advice = f"**Action**: Purchase inventory immediately. Price is projected to rise {potential_gain_pct:.1f}% to ₹{peak_price:.0f}. Expected peak around {best_sell_date}. **Target buy price**: ₹{current_price:.0f} or below. **Target sell price**: ₹{peak_price:.0f}."
    elif recommendation == "BUY":
        advice = f"**Action**: Consider gradual accumulation. Moderate upside of {potential_gain_pct:.1f}% expected. **Optimal entry**: {best_buy_date} around ₹{low_price:.0f}. **Exit target**: {best_sell_date} at ₹{peak_price:.0f}."
    elif recommendation == "WAIT":
        advice = f"**Action**: Hold off on purchases. Prices projected to decline {abs(potential_gain_pct):.1f}%. **Wait for**: Price stabilization around {best_buy_date} at ₹{low_price:.0f} before entering."
    else:
        advice = f"**Action**: Maintain current positions. Market showing sideways movement. **Strategy**: Wait for clearer directional signals before making significant moves."
    
    return {
        "market_info": {
            "state": req.state,
            "district": req.district,
            "market": req.market,
            "current_price": round(current_price, 2),
            "current_arrivals": round(current_arrivals, 2),
            "last_updated": last_date.strftime('%Y-%m-%d')
        },
        "prediction": {
            "horizon_days": horizon_days,
            "dates": dates_list,
            "prices": [round(p, 2) for p in price_preds],
            "arrivals": [round(a, 2) for a in arrival_preds],
            "lower_ci": [round(l, 2) for l in lower_ci],
            "upper_ci": [round(u, 2) for u in upper_ci]
        },
        "roi_analysis": {
            "current_price": round(current_price, 2),
            "predicted_peak": round(peak_price, 2),
            "predicted_low": round(low_price, 2),
            "potential_gain_pct": round(potential_gain_pct, 2),
            "roi_from_optimal": round(roi_from_low, 2),
            "best_buy_date": best_buy_date,
            "best_sell_date": best_sell_date,
            "average_predicted_price": round(avg_price, 2)
        },
        "ai_insights": {
            "summary": summary,
            "price_drivers": price_drivers,
            "opportunities": opportunities,
            "risks": risks,
            "recommendation": recommendation,
            "recommendation_text": rec_text,
            "advice": advice,
            "trend": price_trend,
            "trend_magnitude_pct": round(trend_magnitude, 2),
            "volatility": round(volatility, 2)
        }
    }

@app.get("/dashboard/global-stats")
def get_global_dashboard_stats():
    """
    Get aggregated national statistics for dashboard
    """
    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")
    
    # Get latest date
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    
    # Calculate national average price
    national_avg_price = latest_data['Price'].mean()
    
    # Get previous month for comparison
    prev_month_date = latest_date - timedelta(days=30)
    prev_month_data = df[(df['Date'] >= prev_month_date) & (df['Date'] < latest_date)]
    prev_avg_price = prev_month_data['Price'].mean() if not prev_month_data.empty else national_avg_price
    
    price_change_pct = ((national_avg_price - prev_avg_price) / prev_avg_price * 100) if prev_avg_price > 0 else 0
    
    # Market sentiment
    if price_change_pct > 10:
        sentiment = "Bullish"
        sentiment_color = "green"
    elif price_change_pct < -10:
        sentiment = "Bearish"
        sentiment_color = "red"
    else:
        sentiment = "Neutral"
        sentiment_color = "gray"
    
    # Calculate volatility index (std dev of prices in last 30 days)
    recent_prices = df[df['Date'] >= prev_month_date]['Price']
    volatility_index = (recent_prices.std() / recent_prices.mean() * 100) if len(recent_prices) > 0 else 0
    
    # Top 5 states by average price
    state_prices = latest_data.groupby('state')['Price'].mean().sort_values(ascending=False)
    regional_stats = [
        {"state": state, "avg_price": round(price, 2)}
        for state, price in state_prices.head(5).items()
    ]
    
    # Top movers (biggest price changes in last 7 days)
    week_ago = latest_date - timedelta(days=7)
    week_data = df[df['Date'] == week_ago]
    
    top_movers = []
    for market_id in latest_data['market_id'].unique()[:20]:  # Limit to 20 markets
        current = latest_data[latest_data['market_id'] == market_id]
        previous = week_data[week_data['market_id'] == market_id]
        
        if not current.empty and not previous.empty:
            curr_price = current['Price'].iloc[0]
            prev_price = previous['Price'].iloc[0]
            change_pct = ((curr_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
            
            if abs(change_pct) > 5:  # Only include significant movers
                top_movers.append({
                    "market": current['market'].iloc[0],
                    "district": current['district'].iloc[0],
                    "state": current['state'].iloc[0],
                    "current_price": round(curr_price, 2),
                    "change_pct": round(change_pct, 2),
                    "direction": "up" if change_pct > 0 else "down"
                })
    
    # Sort by absolute change and take top 10
    top_movers = sorted(top_movers, key=lambda x: abs(x['change_pct']), reverse=True)[:10]
    
    # Total arrivals today
    total_arrivals = latest_data['Arrivals'].sum()
    
    return {
        "national_stats": {
            "avg_price": round(national_avg_price, 2),
            "price_change_pct": round(price_change_pct, 2),
            "total_arrivals": round(total_arrivals, 2),
            "total_markets": len(latest_data),
            "last_updated": latest_date.strftime('%Y-%m-%d')
        },
        "market_sentiment": {
            "sentiment": sentiment,
            "color": sentiment_color,
            "description": f"Market is {sentiment.lower()} with {abs(price_change_pct):.1f}% price change"
        },
        "volatility_index": {
            "value": round(volatility_index, 2),
            "level": "High" if volatility_index > 15 else "Moderate" if volatility_index > 8 else "Low"
        },
        "regional_stats": regional_stats,
        "top_movers": top_movers
    }

@app.get("/dashboard/ticker-data")
def get_ticker_data():
    """
    Get latest prices from top markets for live ticker
    """
    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")
    
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    
    # Get previous day for comparison
    prev_date = latest_date - timedelta(days=1)
    prev_data = df[df['Date'] == prev_date]
    
    # Get top 15 markets by trading volume (arrivals)
    top_markets = latest_data.nlargest(15, 'Arrivals')
    
    ticker_items = []
    for _, row in top_markets.iterrows():
        market_id = row['market_id']
        current_price = row['Price']
        
        # Find previous price
        prev_row = prev_data[prev_data['market_id'] == market_id]
        prev_price = prev_row['Price'].iloc[0] if not prev_row.empty else current_price
        
        change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        ticker_items.append({
            "market": row['market'],
            "state": row['state'],
            "price": round(current_price, 2),
            "change_pct": round(change_pct, 2),
            "direction": "up" if change_pct > 0 else "down" if change_pct < 0 else "neutral"
        })
    
    return {"ticker_items": ticker_items}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
