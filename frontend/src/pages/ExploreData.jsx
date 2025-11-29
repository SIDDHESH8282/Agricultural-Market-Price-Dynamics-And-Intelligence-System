import React, { useEffect, useState } from 'react';
import { getLocations, getHistory, getMarketSummary } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Bar, ComposedChart } from 'recharts';
import { ArrowUp, ArrowDown, TrendingUp, TrendingDown, Activity, MapPin, CloudRain, ThermometerSun, Info } from 'lucide-react';

const MarketStatsCards = ({ stats }) => {
    if (!stats) return null;

    const isPositive = stats.change_pct >= 0;

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {/* Average Price */}
            <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-100">
                <div className="flex justify-between items-start mb-2">
                    <div>
                        <p className="text-sm text-gray-500 font-medium">Average Price</p>
                        <h3 className="text-2xl font-bold text-gray-900 mt-1">₹{stats.avg_price.toFixed(0)}</h3>
                    </div>
                    <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
                        <Activity size={20} />
                    </div>
                </div>
                <p className="text-xs text-gray-400">Last 30 days</p>
            </div>

            {/* Price Variation */}
            <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-100">
                <div className="flex justify-between items-start mb-2">
                    <div>
                        <p className="text-sm text-gray-500 font-medium">Price Variation</p>
                        <div className="flex items-baseline gap-2 mt-1">
                            <h3 className="text-2xl font-bold text-gray-900">
                                {Math.abs(stats.change_pct).toFixed(1)}%
                            </h3>
                            <span className={`flex items-center text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                                {isPositive ? <ArrowUp size={16} /> : <ArrowDown size={16} />}
                            </span>
                        </div>
                    </div>
                    <div className={`p-2 rounded-lg ${isPositive ? 'bg-green-50 text-green-600' : 'bg-red-50 text-red-600'}`}>
                        {isPositive ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
                    </div>
                </div>
                <p className="text-xs text-gray-400">vs Previous 30 days</p>
            </div>

            {/* Weather Stats: Temp */}
            <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-100">
                <div className="flex justify-between items-start mb-2">
                    <div>
                        <p className="text-sm text-gray-500 font-medium">Avg Temperature</p>
                        <h3 className="text-2xl font-bold text-gray-900 mt-1">{stats.avg_temp?.toFixed(1)}°C</h3>
                    </div>
                    <div className="p-2 bg-orange-50 rounded-lg text-orange-600">
                        <ThermometerSun size={20} />
                    </div>
                </div>
                <p className="text-xs text-gray-400">Last 30 days avg</p>
            </div>

            {/* Weather Stats: Rain */}
            <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-100">
                <div className="flex justify-between items-start mb-2">
                    <div>
                        <p className="text-sm text-gray-500 font-medium">Total Rainfall</p>
                        <h3 className="text-2xl font-bold text-gray-900 mt-1">{stats.total_rain?.toFixed(1)} mm</h3>
                    </div>
                    <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
                        <CloudRain size={20} />
                    </div>
                </div>
                <p className="text-xs text-gray-400">Last 30 days total</p>
            </div>
        </div>
    );
};

const MarketComparison = ({ comparison }) => {
    if (!comparison || comparison.length === 0) return null;

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-full">
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                <MapPin size={18} /> State Leaderboard
            </h3>
            <div className="space-y-4">
                {comparison.map((item, idx) => (
                    <div
                        key={idx}
                        className={`flex items-center justify-between p-3 rounded-lg transition-colors ${item.is_selected ? 'bg-green-50 border border-green-100' : 'hover:bg-gray-50'}`}
                    >
                        <div className="flex items-center gap-3">
                            <span className={`w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold ${idx < 3 ? 'bg-yellow-100 text-yellow-700' : 'bg-gray-100 text-gray-500'}`}>
                                {idx + 1}
                            </span>
                            <div>
                                <p className={`text-sm font-medium ${item.is_selected ? 'text-green-900' : 'text-gray-900'}`}>
                                    {item.market}
                                </p>
                                <p className="text-xs text-gray-500">{item.district}</p>
                            </div>
                        </div>
                        <div className="text-right">
                            <p className={`text-sm font-bold ${item.is_selected ? 'text-green-700' : 'text-gray-900'}`}>
                                ₹{item.price.toFixed(0)}
                            </p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const WeatherInsight = ({ insight }) => {
    if (!insight) return null;
    return (
        <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 mb-6 flex items-start gap-3">
            <div className="mt-1 text-blue-600">
                <Info size={20} />
            </div>
            <div>
                <h4 className="font-bold text-blue-900 text-sm mb-1">Agro-Climatic Insight</h4>
                <p className="text-sm text-blue-800 leading-relaxed">{insight}</p>
            </div>
        </div>
    );
};

const ExploreData = () => {
    const [locations, setLocations] = useState({ states: [], districts: {}, markets: {} });
    const [selectedState, setSelectedState] = useState('');
    const [selectedDistrict, setSelectedDistrict] = useState('');
    const [selectedMarket, setSelectedMarket] = useState('');
    const [history, setHistory] = useState([]);
    const [marketSummary, setMarketSummary] = useState(null);
    const [timeRange, setTimeRange] = useState('1Y'); // 1W, 1M, 3M, 6M, 1Y, ALL
    const [viewMode, setViewMode] = useState('supply'); // supply, rain, temp

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
            getHistory(selectedState, selectedDistrict, selectedMarket).then(setHistory);
            getMarketSummary(selectedState, selectedDistrict, selectedMarket).then(setMarketSummary).catch(console.error);
        }
    }, [selectedState, selectedDistrict, selectedMarket]);

    // Filter history based on time range
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
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <h1 className="text-2xl font-bold text-gray-900">Explore Data</h1>

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

            {/* Market Stats Cards (Updated with Weather) */}
            {marketSummary && <MarketStatsCards stats={marketSummary.stats} />}

            {/* Weather Insight */}
            {marketSummary?.weather_insight && <WeatherInsight insight={marketSummary.weather_insight} />}

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Main Chart Section */}
                <div className="lg:col-span-3 space-y-6">
                    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
                            <h3 className="font-bold text-gray-800">Market Trends</h3>

                            <div className="flex items-center gap-4">
                                {/* View Mode Toggle */}
                                <div className="flex bg-gray-100 rounded-lg p-1">
                                    <button
                                        onClick={() => setViewMode('supply')}
                                        className={`px-3 py-1 text-xs font-medium rounded-md transition-all flex items-center gap-1 ${viewMode === 'supply' ? 'bg-white text-primary shadow-sm' : 'text-gray-500'}`}
                                    >
                                        Price vs Supply
                                    </button>
                                    <button
                                        onClick={() => setViewMode('rain')}
                                        className={`px-3 py-1 text-xs font-medium rounded-md transition-all flex items-center gap-1 ${viewMode === 'rain' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500'}`}
                                    >
                                        <CloudRain size={12} /> Rain
                                    </button>
                                    <button
                                        onClick={() => setViewMode('temp')}
                                        className={`px-3 py-1 text-xs font-medium rounded-md transition-all flex items-center gap-1 ${viewMode === 'temp' ? 'bg-white text-orange-600 shadow-sm' : 'text-gray-500'}`}
                                    >
                                        <ThermometerSun size={12} /> Temp
                                    </button>
                                </div>

                                {/* Time Range Selector */}
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
                        </div>

                        <div className="h-[500px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={filteredHistory}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                    <XAxis
                                        dataKey="Date"
                                        tickFormatter={(str) => new Date(str).toLocaleDateString()}
                                        stroke="#9CA3AF"
                                        minTickGap={30}
                                    />
                                    <YAxis yAxisId="left" stroke="#2E7D32" label={{ value: 'Price (₹/qtl)', angle: -90, position: 'insideLeft' }} />
                                    <YAxis
                                        yAxisId="right"
                                        orientation="right"
                                        stroke={viewMode === 'supply' ? '#FFD54F' : viewMode === 'rain' ? '#60A5FA' : '#F97316'}
                                        label={{
                                            value: viewMode === 'supply' ? 'Arrivals (T)' : viewMode === 'rain' ? 'Rain (mm)' : 'Temp (°C)',
                                            angle: 90,
                                            position: 'insideRight'
                                        }}
                                    />
                                    <Tooltip labelFormatter={(label) => new Date(label).toLocaleDateString()} />
                                    <Legend />
                                    <Line yAxisId="left" type="monotone" dataKey="Price" stroke="#2E7D32" strokeWidth={2} dot={false} name="Price" />

                                    {viewMode === 'supply' && (
                                        <Line yAxisId="right" type="monotone" dataKey="Arrivals" stroke="#FFD54F" strokeWidth={2} dot={false} name="Arrivals" />
                                    )}
                                    {viewMode === 'rain' && (
                                        <Bar yAxisId="right" dataKey="rain" fill="#60A5FA" name="Rainfall" barSize={20} />
                                    )}
                                    {viewMode === 'temp' && (
                                        <Line yAxisId="right" type="monotone" dataKey="temp_avg" stroke="#F97316" strokeWidth={2} dot={false} name="Temperature" />
                                    )}
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Table (Preserved) */}
                    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price (₹/qtl)</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Arrivals (Tonnes)</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {filteredHistory.slice().reverse().slice(0, 50).map((row, i) => (
                                        <tr key={i} className="hover:bg-gray-50">
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                                {new Date(row.Date).toLocaleDateString()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-primary">
                                                ₹{row.Price.toFixed(2)}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {row.Arrivals.toFixed(1)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Sidebar: Market Comparison */}
                <div className="lg:col-span-1">
                    {marketSummary && <MarketComparison comparison={marketSummary.comparison} />}
                </div>
            </div>
        </div>
    );
};

export default ExploreData;
