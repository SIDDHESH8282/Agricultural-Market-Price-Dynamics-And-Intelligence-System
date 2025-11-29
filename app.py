import streamlit as st

st.set_page_config(
    page_title="Onion Price Forecast",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧅 Onion Price Forecast & Market Studio")

st.markdown("""
### Welcome to the Advanced Agricultural Market Analysis Tool

This application provides deep insights into onion market dynamics, offering:

- **📊 Market Dashboard**: Real-time KPIs and trends.
- **🔍 Data Explorer**: Deep dive into historical price and arrival data.
- **🔮 Forecast Studio**: AI-powered price predictions with scenario analysis.
- **🤖 AI Advisor**: Smart summaries and actionable buy/sell recommendations.

**👈 Select a module from the sidebar to get started.**
""")

st.sidebar.success("Select a page above.")

st.sidebar.info("""
**About**
This tool uses advanced machine learning to forecast onion prices based on historical data, weather patterns, and market arrivals.
""")
