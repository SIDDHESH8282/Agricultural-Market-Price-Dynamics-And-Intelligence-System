import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data():
    file_path = 'preprocessed_no_leakage_sorted_2014_2024_CLEANED (1).csv'
    if not os.path.exists(file_path):
        st.error(f"Data file not found at: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Date conversion
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: 'Date'})
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_unique_locations(df):
    if df is None:
        return [], [], []
    
    states = sorted(df['state'].unique().tolist())
    districts = sorted(df['district'].unique().tolist())
    markets = sorted(df['market'].unique().tolist())
    
    return states, districts, markets

def filter_data(df, state=None, district=None, market=None):
    if df is None:
        return None
    
    filtered_df = df.copy()
    
    if state and state != 'All':
        filtered_df = filtered_df[filtered_df['state'] == state]
    
    if district and district != 'All':
        filtered_df = filtered_df[filtered_df['district'] == district]
        
    if market and market != 'All':
        filtered_df = filtered_df[filtered_df['market'] == market]
        
    return filtered_df
