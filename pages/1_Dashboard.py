import streamlit as st
import plotly.express as px
import pandas as pd
from utils.data_loader import load_data

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Market Dashboard")

df = load_data()

if df is not None:
    # Get latest date
    latest_date = df['Date'].max()
    latest_data = df[df['Date'] == latest_date]
    
    # KPIs
    avg_price = latest_data['modal_price_rs_qtl'].mean()
    total_arrivals = latest_data['arrivals_tonnes'].sum()
    
    # Calculate 30-day change (approximate)
    past_date = latest_date - pd.Timedelta(days=30)
    past_data = df[df['Date'] >= past_date]
    past_avg_price = df[df['Date'] == past_date]['modal_price_rs_qtl'].mean() if not df[df['Date'] == past_date].empty else avg_price
    
    price_change = ((avg_price - past_avg_price) / past_avg_price) * 100 if past_avg_price else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Modal Price (Latest)", f"₹{avg_price:.2f}/qtl", f"{price_change:.2f}% (30d)")
        
    with col2:
        st.metric("Total Arrivals (Latest)", f"{total_arrivals:.1f} Tonnes")
        
    with col3:
        st.metric("Data Up To", latest_date.strftime('%Y-%m-%d'))
        
    st.divider()
    
    st.subheader("Recent Price Trend (All Markets)")
    # Aggregate by date for the chart
    daily_avg = df.groupby('Date')['modal_price_rs_qtl'].mean().reset_index()
    # Filter last 1 year for performance
    last_year = latest_date - pd.Timedelta(days=365)
    chart_data = daily_avg[daily_avg['Date'] > last_year]
    
    fig = px.line(chart_data, x='Date', y='modal_price_rs_qtl', title='Average Onion Price (Last 1 Year)')
    fig.update_layout(yaxis_title="Price (₹/qtl)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.warning("Please upload data or ensure the CSV is present.")
