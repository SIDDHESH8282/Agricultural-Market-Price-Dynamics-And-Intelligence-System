# 🧅 Agricultural Market Intelligence Platform

> **AI-Powered Market Insights for Smarter Agricultural Trading**

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React-61dafb)
![ML](https://img.shields.io/badge/AI-XGBoost-9c27b0)

## 📖 Overview

The **Agricultural Market Intelligence Platform** is a comprehensive solution designed to empower traders, farmers, and analysts with real-time data, predictive analytics, and deep market insights. By leveraging 11 years of historical data and advanced machine learning models, the platform provides actionable intelligence to maximize profitability and minimize risk in agricultural trading.

### 🌟 Key Features

*   **📊 Live Market Dashboard**: Real-time ticker, national statistics, and top market movers.
*   **🔮 AI-Powered Predictions**: Multi-horizon price forecasting (30/90/180 days) with confidence intervals.
*   **💼 Traders Hub**: Arbitrage opportunity analyzer to find profitable buy/sell routes.
*   **🧠 Deep Analytics**: Detailed price factor breakdown (supply, weather, seasonality) and agro-climatic insights.
*   **🌍 Supply Chain Visibility**: Source district tracking and transport status monitoring.

---

## 🏗️ Architecture

The platform is built on a modern, scalable tech stack:

*   **Frontend**: React 18, Vite, Tailwind CSS, Recharts
*   **Backend**: FastAPI, Uvicorn, Pydantic
*   **Machine Learning**: XGBoost, Pandas, NumPy, Scikit-learn
*   **Data**: Historical market data (2014-2025)

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Node.js 16+

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/agri-market-intelligence.git
    cd agri-market-intelligence
    ```

2.  **Backend Setup**
    ```bash
    cd backend
    pip install -r ../requirements.txt
    python -m uvicorn main:app --reload --port 8000
    ```

3.  **Frontend Setup**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

4.  **Access the Application**
    *   Frontend: `http://localhost:5173`
    *   Backend API: `http://localhost:8000/docs`

---

## 💡 How It Works

### 1. Data Processing
The system ingests over 600,000 rows of historical market data, cleaning and preprocessing it to handle missing values and outliers.

### 2. Machine Learning Models
Four specialized XGBoost models are trained to predict:
*   **Price**: Future market rates
*   **Arrivals**: Expected supply volume
*   **Temperature**: Weather conditions
*   **Rainfall**: Supply chain disruption risks

### 3. AI Insights Engine
The backend analyzes model outputs to generate human-readable insights, such as:
*   "Prices rising due to 20% supply drop and festival demand."
*   "Arbitrage opportunity: Buy in Nashik, Sell in Delhi (Spread: ₹450)."

---

## 📸 Screenshots

*(Add your screenshots here)*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

*   Data sources provided by [Agmarknet](https://agmarknet.gov.in/)
*   Icons by [Lucide React](https://lucide.dev/)
*   Charts by [Recharts](https://recharts.org/)
