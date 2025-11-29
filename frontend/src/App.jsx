import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import ExploreData from './pages/ExploreData';
import ForecastStudio from './pages/ForecastStudio';
import TradersHub from './pages/TradersHub';
import WeatherStation from './pages/WeatherStation';

function App() {
    return (
        <Router>
            <Routes>
                {/* Public Route */}
                <Route path="/login" element={<Login />} />

                {/* Redirect root to login */}
                <Route path="/" element={<Navigate to="/login" replace />} />

                {/* Protected Routes wrapped in AppLayout */}
                <Route element={<AppLayout />}>
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/explore" element={<ExploreData />} />
                    <Route path="/forecast" element={<ForecastStudio />} />
                    <Route path="/traders" element={<TradersHub />} />
                    <Route path="/weather" element={<WeatherStation />} />
                </Route>
            </Routes>
        </Router>
    );
}

export default App;
