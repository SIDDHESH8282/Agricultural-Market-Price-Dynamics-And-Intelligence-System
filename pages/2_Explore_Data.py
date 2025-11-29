import streamlit as st
import plotly.express as px
from utils.data_loader import load_data, get_unique_locations, filter_data

st.set_page_config(page_title="Explore Data", page_icon="🔍", layout="wide")

st.title("🔍 Explore Historical Data")

df = load_data()

if df is not None:
    states, districts, markets = get_unique_locations(df)
    
    with st.sidebar:
        st.header("Filters")
        selected_state = st.selectbox("Select State", ['All'] + states)
        
        # Filter districts based on state
        if selected_state != 'All':
            filtered_districts = sorted(df[df['state'] == selected_state]['district'].unique().tolist())
        else:
            filtered_districts = districts
            
        selected_district = st.selectbox("Select District", ['All'] + filtered_districts)
        
        # Filter markets based on district
        if selected_district != 'All':
            filtered_markets = sorted(df[df['district'] == selected_district]['market'].unique().tolist())
        else:
            filtered_markets = markets
            
        selected_market = st.selectbox("Select Market", ['All'] + filtered_markets)
        
    filtered_df = filter_data(df, selected_state, selected_district, selected_market)
    
    if not filtered_df.empty:
        st.subheader(f"Price History: {selected_state} > {selected_district} > {selected_market}")
        
        # Aggregate if multiple markets selected
        chart_df = filtered_df.groupby('Date')[['modal_price_rs_qtl', 'arrivals_tonnes']].mean().reset_index()
        
        fig = px.line(chart_df, x='Date', y='modal_price_rs_qtl', title='Modal Price Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Data Table"):
            st.dataframe(filtered_df.sort_values('Date', ascending=False).head(1000))
    else:
        st.info("No data found for the selected filters.")
else:
    st.error("Data could not be loaded.")
