import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date as dt_date
from utils.data_loader import load_data, get_unique_locations, filter_data
from utils.forecasting import run_forecast, predict_custom_date
from utils.ai_advisor import generate_summary, get_buying_advice, explain_drivers

st.set_page_config(page_title="Forecast Studio", page_icon="🔮", layout="wide")

st.title("🔮 Forecast Studio & AI Advisor")

df = load_data()

if df is not None:
    states, districts, markets = get_unique_locations(df)
    
    # Layout: Controls on left (sidebar), Results on right
    with st.sidebar:
        st.header("1. Select Market")
        selected_state = st.selectbox("State", ['All'] + states, index=0)
        
        if selected_state != 'All':
            filtered_districts = sorted(df[df['state'] == selected_state]['district'].unique().tolist())
        else:
            filtered_districts = districts
        selected_district = st.selectbox("District", ['All'] + filtered_districts)
        
        if selected_district != 'All':
            filtered_markets = sorted(df[df['district'] == selected_district]['market'].unique().tolist())
        else:
            filtered_markets = markets
        selected_market = st.selectbox("Market", ['All'] + filtered_markets)
        
        st.divider()
        
        st.header("2. Forecast Mode")
        mode = st.radio("Choose Mode", ["Horizon Forecast", "Custom Date Prediction"])
        
        if mode == "Horizon Forecast":
            horizon = st.slider("Forecast Horizon (Days)", 7, 90, 30)
            
            st.subheader("Scenario Analysis")
            supply_shock = st.slider("Supply Shock (%)", -50, 50, 0, help="Positive = Shortage (Price Up), Negative = Surplus (Price Down)")
            transport_cost = st.slider("Transport Cost Multiplier", 1.0, 2.0, 1.0, 0.1, help="Multiplier for fuel/transport costs")
            seasonality = st.slider("Seasonality Strength", 0.0, 2.0, 1.0, help="Amplify or dampen seasonal effects")
            
            run_btn = st.button("Run Forecast", type="primary")
            
        else: # Custom Date
            custom_date = st.date_input("Select Date", min_value=dt_date.today())
            predict_btn = st.button("Predict Price", type="primary")

    # --- Main Content ---
    
    # Validate Selection
    if selected_state == 'All' or selected_district == 'All' or selected_market == 'All':
        st.info("👈 Please select a specific State, District, and Market to enable forecasting.")
    else:
        # Filter data for the specific market
        target_df = filter_data(df, selected_state, selected_district, selected_market)
        
        if target_df is None or target_df.empty:
            st.error("No historical data available for this market.")
        else:
            # Aggregate to daily level
            history_df = target_df.groupby('Date')[['modal_price_rs_qtl', 'arrivals_tonnes', 'state', 'district', 'market']].first().reset_index().sort_values('Date')
            
            # Add weather columns if they exist in original df but got lost in groupby (using first)
            # (Already handled by including them in groupby or merging back if needed. 
            # For now, assuming they are in target_df)
            cols_to_keep = ['temp_avg', 'rain']
            for col in cols_to_keep:
                if col in target_df.columns:
                    # simplistic join
                    pass 

            if mode == "Horizon Forecast":
                if run_btn:
                    with st.spinner("Running XGBoost Model..."):
                        forecast_df = run_forecast(history_df, horizon, supply_shock, transport_cost, seasonality)
                    
                    if not forecast_df.empty:
                        # 1. Chart
                        st.subheader(f"Price Forecast: {selected_market}, {selected_district}")
                        
                        fig = go.Figure()
                        
                        # Historical Data (Last 90 days)
                        recent_history = history_df.tail(90)
                        fig.add_trace(go.Scatter(x=recent_history['Date'], y=recent_history['modal_price_rs_qtl'], mode='lines', name='Historical', line=dict(color='gray')))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted_Price'], mode='lines', name='Forecast', line=dict(color='#2E7D32', width=3)))
                        
                        # Confidence Interval
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_df['Date'], forecast_df['Date'][::-1]]),
                            y=pd.concat([forecast_df['Upper_CI'], forecast_df['Lower_CI'][::-1]]),
                            fill='toself',
                            fillcolor='rgba(46, 125, 50, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False,
                            name='Confidence Interval'
                        ))
                        
                        fig.update_layout(title="Price Prediction with Confidence Intervals", yaxis_title="Price (₹/qtl)", xaxis_title="Date")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 2. AI Advisor
                        st.divider()
                        st.header("🤖 AI Market Advisor")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(generate_summary(forecast_df, history_df))
                            st.markdown("### Key Drivers")
                            st.markdown(explain_drivers(forecast_df, supply_shock, transport_cost, seasonality))
                            
                        with col2:
                            st.markdown("### Strategic Advice")
                            advice = get_buying_advice(forecast_df, history_df)
                            st.success(advice) if "BUY" in advice else st.warning(advice) if "HOLD" in advice else st.error(advice)
                            
                            st.markdown("### Detailed Metrics")
                            st.dataframe(forecast_df[['Date', 'Predicted_Price', 'Lower_CI', 'Upper_CI']].head(10))
            
            else: # Custom Date Mode
                if predict_btn:
                    with st.spinner("Calculating Prediction..."):
                        # Convert date to datetime
                        target_date = pd.to_datetime(custom_date)
                        price = predict_custom_date(selected_state, selected_district, selected_market, target_date, history_df)
                    
                    if price is not None:
                        st.success(f"### Predicted Price for {custom_date}:")
                        st.metric(label="Modal Price", value=f"₹{price:.2f}/qtl")
                        
                        # Context
                        last_price = history_df.iloc[-1]['modal_price_rs_qtl']
                        diff = price - last_price
                        pct = (diff / last_price) * 100
                        
                        st.info(f"This is a **{pct:+.2f}%** change from the last recorded price (₹{last_price:.2f}).")
                    else:
                        st.error("Could not generate prediction. Please check data availability.")

else:
    st.error("Data could not be loaded.")
