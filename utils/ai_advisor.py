import pandas as pd

def generate_summary(forecast_df, history_df):
    if forecast_df is None or forecast_df.empty:
        return "No forecast generated."
    
    start_price = history_df['modal_price_rs_qtl'].iloc[-1]
    end_price = forecast_df['Predicted_Price'].iloc[-1]
    max_price = forecast_df['Predicted_Price'].max()
    min_price = forecast_df['Predicted_Price'].min()
    
    pct_change = ((end_price - start_price) / start_price) * 100
    
    trend = "stable"
    if pct_change > 5:
        trend = "rising significantly"
    elif pct_change > 1:
        trend = "rising slightly"
    elif pct_change < -5:
        trend = "falling significantly"
    elif pct_change < -1:
        trend = "falling slightly"
        
    summary = f"""
    **Forecast Summary:**
    
    Over the next {len(forecast_df)} days, onion prices are expected to be **{trend}**.
    
    - **Current Price**: ₹{start_price:.2f}/qtl
    - **Predicted Price (End)**: ₹{end_price:.2f}/qtl ({pct_change:+.2f}%)
    - **Peak Price**: ₹{max_price:.2f}/qtl
    - **Lowest Price**: ₹{min_price:.2f}/qtl
    """
    
    return summary

def get_buying_advice(forecast_df, history_df):
    if forecast_df is None or forecast_df.empty:
        return "Insufficient data for advice."
        
    start_price = history_df['modal_price_rs_qtl'].iloc[-1]
    end_price = forecast_df['Predicted_Price'].iloc[-1]
    
    pct_change = ((end_price - start_price) / start_price) * 100
    
    if pct_change > 10:
        return "🔴 **SELL NOW / DO NOT BUY**: Prices are expected to spike. If you are a farmer, hold for a few days if possible. If you are a buyer, stock up immediately."
    elif pct_change < -10:
        return "🟢 **BUY LATER**: Prices are expected to drop significantly. Wait for the market to cool down."
    else:
        return "🟡 **HOLD / NEUTRAL**: Market is relatively stable. Standard trading recommended."

def explain_drivers(forecast_df, supply_shock, transport_cost, seasonality):
    drivers = []
    
    if supply_shock > 0:
        drivers.append(f"⚠️ **Supply Shock**: A {supply_shock}% disruption in supply is pushing prices up.")
    elif supply_shock < 0:
        drivers.append(f"📉 **Supply Surplus**: A {abs(supply_shock)}% increase in supply is dampening prices.")
        
    if transport_cost > 1.0:
        drivers.append(f"🚚 **Transport Costs**: High fuel/transport costs are adding a premium to the final price.")
        
    if seasonality > 1.0:
        drivers.append(f"📅 **Seasonality**: Historical patterns suggest a seasonal price increase during this period.")
        
    if not drivers:
        drivers.append("✅ **Normal Market Conditions**: No major external shocks detected.")
        
    return "\n\n".join(drivers)
