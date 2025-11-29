import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_URL,
});

export const getLocations = async () => {
    const response = await api.get('/meta/locations');
    return response.data;
};

export const getHistory = async (state, district, market) => {
    const response = await api.get('/history', {
        params: { state, district, market }
    });
    return response.data;
};

export const predictHorizon = async (params) => {
    const response = await api.post('/predict/horizon', params);
    return response.data;
};

export const predictCustom = async (params) => {
    const response = await api.post('/predict/custom', params);
    return response.data;
};

export const getTradingInsights = async () => {
    const response = await api.get('/trading/insights');
    return response.data;
};

export const fetchMarkets = async (state, district) => {
    const response = await api.post('/meta/markets', { state, district });
    return response.data;
};

export const analyzeArbitrage = async (params) => {
    const response = await api.post('/trading/analyze', params);
    return response.data;
};

export const getMarketSummary = async (state, district, market) => {
    const response = await api.post('/market-summary', { state, district, market });
    return response.data;
};

export const predictWeather = async (params) => {
    const response = await api.post('/predict/weather', params);
    return response.data;
};

export const forecastArbitrage = async (params) => {
    const response = await api.post('/trading/forecast-arbitrage', params);
    return response.data;
};

export const predictMarket = async (params) => {
    const response = await api.post('/trading/predict-market', params);
    return response.data;
};

export const getDashboardStats = async () => {
    const response = await api.get('/dashboard/global-stats');
    return response.data;
};

export const getTickerData = async () => {
    const response = await api.get('/dashboard/ticker-data');
    return response.data;
};
