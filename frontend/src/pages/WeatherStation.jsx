import React, { useEffect, useState } from 'react';
import { getLocations, getHistory, predictWeather } from '../services/api';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ComposedChart } from 'recharts';
import { CloudRain, ThermometerSun, Calendar, Info, AlertTriangle, CheckCircle } from 'lucide-react';

const WeatherStation = () => {
    const [locations, setLocations] = useState({ states: [], districts: {}, markets: {} });
    const [selectedState, setSelectedState] = useState('');
    const [selectedDistrict, setSelectedDistrict] = useState('');
    const [selectedMarket, setSelectedMarket] = useState('');

    const [history, setHistory] = useState([]);
    const [forecast, setForecast] = useState(null);
    const [timeRange, setTimeRange] = useState('1Y');
    const [forecastHorizon, setForecastHorizon] = useState(30);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        getLocations().then(data => {
            setLocations(data);
            if (data.states.length > 0) {
                const s = data.states[0];
                setSelectedState(s);
                const d = data.districts[s]?.[0];
                setSelectedDistrict(d);
                const m = data.markets[d]?.[0];
                setSelectedMarket(m);
            }
        });
    }, []);

    useEffect(() => {
        if (selectedState && selectedDistrict && selectedMarket) {
            setLoading(true);
            Promise.all([
                getHistory(selectedState, selectedDistrict, selectedMarket),
                predictWeather({ state: selectedState, district: selectedDistrict, market: selectedMarket, horizon: forecastHorizon })
            ]).then(([histData, forecastData]) => {
                setHistory(histData);

                // Format forecast data for chart
                const formattedForecast = forecastData.dates.map((date, i) => ({
                    Date: date,
                    temp: forecastData.temps[i],
                    rain: forecastData.rains[i]
                }));

                setForecast({
                    data: formattedForecast,
                    insight: forecastData.insight
                });
                setLoading(false);
            }).catch(err => {
                console.error(err);
                setLoading(false);
            });
        }
    }, [selectedState, selectedDistrict, selectedMarket, forecastHorizon]);

    const getFilteredHistory = () => {
        if (!history.length) return [];
        if (timeRange === 'ALL') return history;

        const now = new Date(history[history.length - 1].Date);
        const cutoff = new Date(now);

        switch (timeRange) {
            case '1W': cutoff.setDate(now.getDate() - 7); break;
            case '1M': cutoff.setMonth(now.getMonth() - 1); break;
            case '3M': cutoff.setMonth(now.getMonth() - 3); break;
            case '6M': cutoff.setMonth(now.getMonth() - 6); break;
            case '1Y': cutoff.setFullYear(now.getFullYear() - 1); break;
            default: return history;
        }

        return history.filter(item => new Date(item.Date) >= cutoff);
    };

    const filteredHistory = getFilteredHistory();

    return (
        <div className="space-y-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                        <ThermometerSun className="text-orange-500" /> Weather Station
                    </h1>
                    <p className="text-sm text-gray-500 mt-1">Advanced Agro-Climatic Intelligence & Forecasting</p>
                </div>

                {/* Filters */}
                <div className="flex flex-wrap gap-2">
                    <select
                        className="rounded-lg border-gray-300 shadow-sm focus:border-primary focus:ring-primary text-sm min-w-[150px]"
                        value={selectedState}
                        onChange={(e) => {
                            setSelectedState(e.target.value);
                            const d = locations.districts[e.target.value]?.[0];
                            setSelectedDistrict(d);
                            setSelectedMarket(locations.markets[d]?.[0]);
                        }}
                    >
                        {locations.states.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                    <select
                        className="rounded-lg border-gray-300 shadow-sm focus:border-primary focus:ring-primary text-sm min-w-[150px]"
                        value={selectedDistrict}
                        onChange={(e) => {
                            setSelectedDistrict(e.target.value);
                            setSelectedMarket(locations.markets[e.target.value]?.[0]);
                        }}
                    >
                        {locations.districts[selectedState]?.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                    <select
                        className="rounded-lg border-gray-300 shadow-sm focus:border-primary focus:ring-primary text-sm min-w-[150px]"
                        value={selectedMarket}
                        onChange={(e) => setSelectedMarket(e.target.value)}
                    >
                        {locations.markets[selectedDistrict]?.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                </div>
            </div>

            {/* AI Insight Banner */}
            {forecast && (
                <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
                    <div className="flex items-start gap-4 mb-6">
                        <div className="p-3 bg-blue-50 rounded-lg text-blue-600">
                            <Info size={24} />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold text-gray-900">AI Weather Advisor</h3>
                            <p className="text-gray-500 text-sm">Real-time agricultural intelligence based on {forecastHorizon}-day forecast</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Left Column: Analysis & Yield */}
                        <div className="space-y-6">
                            <div>
                                <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                                    <ThermometerSun size={18} className="text-orange-500" /> Detailed Analysis
                                </h4>
                                <p className="text-gray-600 text-sm leading-relaxed">
                                    {forecast.insight.detailed_analysis}
                                </p>
                            </div>

                            <div className="bg-green-50 p-4 rounded-lg border border-green-100">
                                <h4 className="font-semibold text-green-800 mb-2 flex items-center gap-2">
                                    <CheckCircle size={18} /> Yield Forecast
                                </h4>
                                <p className="text-green-700 text-sm">
                                    {forecast.insight.yield_forecast}
                                </p>
                            </div>
                        </div>

                        {/* Right Column: Advice & Metrics */}
                        <div className="space-y-6">
                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                                <h4 className="font-semibold text-blue-800 mb-2 flex items-center gap-2">
                                    <AlertTriangle size={18} /> Actionable Advice
                                </h4>
                                <p className="text-blue-700 text-sm font-medium">
                                    {forecast.insight.advice}
                                </p>
                            </div>

                            <div>
                                <h4 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                                    <Calendar size={18} className="text-purple-500" /> Forecast Notes
                                </h4>
                                <p className="text-gray-500 text-xs italic">
                                    {forecast.insight.notes}
                                </p>
                            </div>

                            <div className="flex gap-4 pt-2 border-t border-gray-100">
                                <div className="text-center px-4 py-2 bg-gray-50 rounded-lg">
                                    <span className="block text-xs text-gray-500">Avg Temp</span>
                                    <span className="block font-bold text-gray-900">{forecast.insight.metrics.avg_temp}</span>
                                </div>
                                <div className="text-center px-4 py-2 bg-gray-50 rounded-lg">
                                    <span className="block text-xs text-gray-500">Total Rain</span>
                                    <span className="block font-bold text-gray-900">{forecast.insight.metrics.total_rain}</span>
                                </div>
                                <div className="text-center px-4 py-2 bg-gray-50 rounded-lg">
                                    <span className="block text-xs text-gray-500">Rain Days</span>
                                    <span className="block font-bold text-gray-900">{forecast.insight.metrics.rain_days}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Historical Analysis */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="font-bold text-gray-800 flex items-center gap-2">
                            <Calendar size={18} /> Historical Weather Patterns
                        </h3>
                        <div className="flex bg-gray-100 rounded-lg p-1">
                            {['1W', '1M', '3M', '6M', '1Y', 'ALL'].map(range => (
                                <button
                                    key={range}
                                    onClick={() => setTimeRange(range)}
                                    className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${timeRange === range
                                        ? 'bg-white text-primary shadow-sm'
                                        : 'text-gray-500 hover:text-gray-900'
                                        }`}
                                >
                                    {range}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="h-[350px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={filteredHistory}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                <XAxis dataKey="Date" tickFormatter={(str) => new Date(str).toLocaleDateString()} stroke="#9CA3AF" minTickGap={30} />
                                <YAxis yAxisId="left" stroke="#F97316" label={{ value: 'Temp (°C)', angle: -90, position: 'insideLeft' }} />
                                <YAxis yAxisId="right" orientation="right" stroke="#60A5FA" label={{ value: 'Rain (mm)', angle: 90, position: 'insideRight' }} />
                                <Tooltip labelFormatter={(label) => new Date(label).toLocaleDateString()} />
                                <Legend />
                                <Line yAxisId="left" type="monotone" dataKey="temp_avg" stroke="#F97316" strokeWidth={2} dot={false} name="Temperature" />
                                <Bar yAxisId="right" dataKey="rain" fill="#60A5FA" name="Rainfall" barSize={20} />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Forecast Analysis */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="font-bold text-gray-800 flex items-center gap-2">
                            <AlertTriangle size={18} className="text-yellow-500" /> Weather Forecast
                        </h3>
                        <div className="flex items-center gap-2">
                            <div className="flex bg-gray-100 rounded-lg p-1">
                                {[
                                    { label: '1M', value: 30 },
                                    { label: '3M', value: 90 },
                                    { label: '6M', value: 180 },
                                    { label: '1Y', value: 365 }
                                ].map(item => (
                                    <button
                                        key={item.label}
                                        onClick={() => setForecastHorizon(item.value)}
                                        className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${forecastHorizon === item.value
                                            ? 'bg-white text-primary shadow-sm'
                                            : 'text-gray-500 hover:text-gray-900'
                                            }`}
                                    >
                                        {item.label}
                                    </button>
                                ))}
                            </div>
                            <span className="text-xs font-medium px-2 py-1 bg-green-100 text-green-700 rounded-full flex items-center gap-1">
                                <CheckCircle size={12} /> AI Ready
                            </span>
                        </div>
                    </div>
                    <div className="h-[350px]">
                        {forecast ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={forecast.data}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                    <XAxis dataKey="Date" tickFormatter={(str) => new Date(str).toLocaleDateString()} stroke="#9CA3AF" minTickGap={30} />
                                    <YAxis yAxisId="left" stroke="#F97316" label={{ value: 'Temp (°C)', angle: -90, position: 'insideLeft' }} />
                                    <YAxis yAxisId="right" orientation="right" stroke="#60A5FA" label={{ value: 'Rain (mm)', angle: 90, position: 'insideRight' }} />
                                    <Tooltip labelFormatter={(label) => new Date(label).toLocaleDateString()} />
                                    <Legend />
                                    <Line yAxisId="left" type="monotone" dataKey="temp" stroke="#F97316" strokeWidth={2} dot={false} name="Forecast Temp" />
                                    <Bar yAxisId="right" dataKey="rain" fill="#60A5FA" name="Forecast Rain" barSize={20} />
                                </ComposedChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex items-center justify-center text-gray-400">
                                Loading Forecast...
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WeatherStation;
