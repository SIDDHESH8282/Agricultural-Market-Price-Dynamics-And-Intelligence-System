import React, { useEffect, useState } from 'react';
import { getLocations, predictHorizon, predictCustom } from '../services/api';
import { ComposedChart, LineChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, AreaChart } from 'recharts';
import { Sliders, Calendar, AlertCircle, Sparkles, TrendingUp } from 'lucide-react';

const WeeklySummary = ({ data }) => {
    if (!data) return null;

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden mt-6">
            <div className="bg-indigo-50 px-6 py-4 border-b border-indigo-100 flex items-center gap-3">
                <div className="bg-indigo-100 p-2 rounded-lg">
                    <Sparkles className="text-indigo-600" size={20} />
                </div>
                <div>
                    <h3 className="font-bold text-indigo-900">{data.title}</h3>
                    <p className="text-xs text-indigo-600 font-medium">{data.date_range}</p>
                </div>
            </div>

            <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                {data.sections.map((section, idx) => (
                    <div key={idx} className="space-y-2">
                        <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider">{section.title}</h4>
                        <p className="text-sm text-gray-700 leading-relaxed font-medium">
                            {section.content}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
};

const ForecastStudio = () => {
    const [locations, setLocations] = useState({ states: [], districts: {}, markets: {} });
    const [selectedState, setSelectedState] = useState('');
    const [selectedDistrict, setSelectedDistrict] = useState('');
    const [selectedMarket, setSelectedMarket] = useState('');

    const [mode, setMode] = useState('horizon'); // 'horizon' or 'custom'
    const [horizon, setHorizon] = useState(30);
    const [supplyShock, setSupplyShock] = useState(0);
    const [transportCost, setTransportCost] = useState(1.0);
    const [customDate, setCustomDate] = useState('');

    const [forecastData, setForecastData] = useState(null);
    const [customPrediction, setCustomPrediction] = useState(null);
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

    const handleRunForecast = async () => {
        setLoading(true);
        try {
            if (mode === 'horizon') {
                const data = await predictHorizon({
                    state: selectedState,
                    district: selectedDistrict,
                    market: selectedMarket,
                    horizon,
                    supply_shock: supplyShock,
                    transport_cost: transportCost
                });

                // Transform for chart
                const chartData = data.dates.map((date, i) => ({
                    date,
                    price: data.prices[i],
                    arrivals: data.arrivals[i],
                    lower_ci: data.lower_ci[i],
                    upper_ci: data.upper_ci[i]
                }));
                // Attach insight to the array
                chartData.insight = data.ai_insight;
                chartData.weekly_summary = data.weekly_summary;
                setForecastData(chartData);
                setCustomPrediction(null);
            } else {
                const data = await predictCustom({
                    state: selectedState,
                    district: selectedDistrict,
                    market: selectedMarket,
                    date: customDate
                });
                setCustomPrediction(data);
                setForecastData(null);
            }
        } catch (err) {
            console.error(err);
            alert("Failed to run forecast. Ensure date is in the future.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-[calc(100vh-140px)]">
            {/* Controls Sidebar */}
            <div className="lg:col-span-1 bg-white p-6 rounded-xl shadow-sm border border-gray-100 overflow-y-auto">
                <h2 className="text-lg font-bold text-gray-900 mb-6 flex items-center gap-2">
                    <Sliders size={20} /> Configuration
                </h2>

                <div className="space-y-6">
                    {/* Location */}
                    <div className="space-y-3">
                        <label className="block text-sm font-medium text-gray-700">Market Selection</label>
                        <select
                            className="w-full rounded-lg border-gray-300 shadow-sm"
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
                            className="w-full rounded-lg border-gray-300 shadow-sm"
                            value={selectedDistrict}
                            onChange={(e) => {
                                setSelectedDistrict(e.target.value);
                                setSelectedMarket(locations.markets[e.target.value]?.[0]);
                            }}
                        >
                            {locations.districts[selectedState]?.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                        <select
                            className="w-full rounded-lg border-gray-300 shadow-sm"
                            value={selectedMarket}
                            onChange={(e) => setSelectedMarket(e.target.value)}
                        >
                            {locations.markets[selectedDistrict]?.map(m => <option key={m} value={m}>{m}</option>)}
                        </select>
                    </div>

                    <div className="h-px bg-gray-100" />

                    {/* Mode */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-3">Forecast Mode</label>
                        <div className="flex gap-2">
                            <button
                                onClick={() => setMode('horizon')}
                                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors ${mode === 'horizon' ? 'bg-primary text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                            >
                                Horizon
                            </button>
                            <button
                                onClick={() => setMode('custom')}
                                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors ${mode === 'custom' ? 'bg-primary text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }`}
                            >
                                Custom Date
                            </button>
                        </div>
                    </div>

                    {mode === 'horizon' ? (
                        <>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Horizon: {horizon} Days
                                </label>
                                <input
                                    type="range" min="7" max="90" value={horizon}
                                    onChange={(e) => setHorizon(parseInt(e.target.value))}
                                    className="w-full accent-primary"
                                />
                            </div>

                            <div className="space-y-4">
                                <h3 className="text-sm font-semibold text-gray-900">Scenarios</h3>
                                <div>
                                    <label className="block text-xs text-gray-500 mb-1">
                                        Supply Shock ({supplyShock > 0 ? '+' : ''}{supplyShock}%)
                                    </label>
                                    <input
                                        type="range" min="-50" max="50" value={supplyShock}
                                        onChange={(e) => setSupplyShock(parseInt(e.target.value))}
                                        className="w-full accent-accent"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs text-gray-500 mb-1">
                                        Transport Cost (x{transportCost})
                                    </label>
                                    <input
                                        type="range" min="1" max="2" step="0.1" value={transportCost}
                                        onChange={(e) => setTransportCost(parseFloat(e.target.value))}
                                        className="w-full accent-secondary"
                                    />
                                </div>
                            </div>
                        </>
                    ) : (
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Target Date</label>
                            <input
                                type="date"
                                className="w-full rounded-lg border-gray-300 shadow-sm mb-4"
                                value={customDate}
                                onChange={(e) => setCustomDate(e.target.value)}
                            />
                        </div>
                    )}

                    <button
                        onClick={handleRunForecast}
                        disabled={loading}
                        className="w-full py-3 bg-primary hover:bg-primary-dark text-white rounded-lg font-semibold shadow-md transition-all active:scale-95 disabled:opacity-50"
                    >
                        {loading ? 'Running Model...' : 'Run Forecast'}
                    </button>
                </div>
            </div>

            {/* Results Area */}
            <div className="lg:col-span-2 space-y-6 overflow-y-auto pr-2">
                {mode === 'horizon' && forecastData && (
                    <div className="flex flex-col gap-6">
                        {/* Main Forecast Chart */}
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                            <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center gap-2">
                                <TrendingUp size={20} /> Price & Supply Forecast
                            </h3>
                            <div className="h-[400px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={forecastData}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                                        <XAxis dataKey="date" tick={{ fontSize: 12 }} tickFormatter={(str) => new Date(str).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} />
                                        <YAxis yAxisId="left" label={{ value: 'Price (₹/qtl)', angle: -90, position: 'insideLeft' }} />
                                        <YAxis yAxisId="right" orientation="right" label={{ value: 'Arrivals (Tonnes)', angle: 90, position: 'insideRight' }} />
                                        <Tooltip />
                                        <Legend />
                                        <Bar yAxisId="right" dataKey="arrivals" name="Supply (Tonnes)" fill="#93c5fd" opacity={0.6} barSize={20} />
                                        <Line yAxisId="left" type="monotone" dataKey="price" name="Price (₹)" stroke="#16a34a" strokeWidth={3} dot={false} />
                                        <Line yAxisId="left" type="monotone" dataKey="lower_ci" name="Low Est." stroke="#16a34a" strokeDasharray="3 3" strokeWidth={1} dot={false} opacity={0.5} />
                                        <Line yAxisId="left" type="monotone" dataKey="upper_ci" name="High Est." stroke="#16a34a" strokeDasharray="3 3" strokeWidth={1} dot={false} opacity={0.5} />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* New Weekly Summary for Horizon */}
                        {forecastData.weekly_summary && <WeeklySummary data={forecastData.weekly_summary} />}

                        {/* AI Insights Section */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Summary & Drivers */}
                            <div className="p-5 bg-blue-50 rounded-xl border border-blue-100">
                                <h4 className="font-bold text-blue-900 mb-3 flex items-center gap-2">
                                    <AlertCircle size={18} /> AI Market Analysis
                                </h4>
                                <p className="text-sm text-blue-800 mb-4 leading-relaxed">
                                    <span dangerouslySetInnerHTML={{ __html: forecastData.insight?.summary?.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                                </p>

                                <h5 className="text-xs font-bold text-blue-900 uppercase tracking-wider mb-2">Key Drivers</h5>
                                <ul className="space-y-2">
                                    {forecastData.insight?.drivers?.map((driver, i) => (
                                        <li key={i} className="text-sm text-blue-800 flex items-start gap-2">
                                            <span className="mt-1">•</span>
                                            <span dangerouslySetInnerHTML={{ __html: driver.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            {/* Strategic Advice */}
                            <div className={`p-5 rounded-xl border ${forecastData.insight?.action === 'BUY NOW' ? 'bg-green-50 border-green-100' :
                                forecastData.insight?.action === 'WAIT' ? 'bg-red-50 border-red-100' :
                                    'bg-yellow-50 border-yellow-100'
                                }`}>
                                <h4 className={`font-bold mb-3 flex items-center gap-2 ${forecastData.insight?.action === 'BUY NOW' ? 'text-green-900' :
                                    forecastData.insight?.action === 'WAIT' ? 'text-red-900' :
                                        'text-yellow-900'
                                    }`}>
                                    <Calendar size={18} /> Strategic Advice
                                </h4>

                                <div className="flex items-center gap-3 mb-3">
                                    <span className={`px-3 py-1 rounded-full text-xs font-bold ${forecastData.insight?.action === 'BUY NOW' ? 'bg-green-200 text-green-800' :
                                        forecastData.insight?.action === 'WAIT' ? 'bg-red-200 text-red-800' :
                                            'bg-yellow-200 text-yellow-800'
                                        }`}>
                                        {forecastData.insight?.action}
                                    </span>
                                </div>

                                <p className={`text-sm leading-relaxed ${forecastData.insight?.action === 'BUY NOW' ? 'text-green-800' :
                                    forecastData.insight?.action === 'WAIT' ? 'text-red-800' :
                                        'text-yellow-800'
                                    }`}>
                                    <span dangerouslySetInnerHTML={{ __html: forecastData.insight?.advice?.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {mode === 'custom' && customPrediction && (
                    <div className="flex flex-col gap-6">
                        {/* Single Point Prediction */}
                        <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100 flex items-center justify-center">
                            <div className="text-center">
                                <p className="text-gray-500 font-medium mb-2">Forecast for {new Date(customPrediction.date).toLocaleDateString()}</p>
                                <div className="flex items-baseline justify-center gap-4 mb-4">
                                    <div className="text-5xl font-bold text-green-700">
                                        ₹{customPrediction.price.toFixed(0)}<span className="text-xl text-gray-400 font-normal">/qtl</span>
                                    </div>
                                    <div className="text-3xl font-bold text-blue-600">
                                        {customPrediction.arrival?.toFixed(0)}<span className="text-lg text-gray-400 font-normal"> Tonnes</span>
                                    </div>
                                </div>
                                <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 text-green-700 rounded-full text-sm font-medium">
                                    <Calendar size={16} />
                                    Confidence: High
                                </div>
                            </div>
                        </div>

                        {/* 7-Day Trend Chart */}
                        {customPrediction.forecast_7_days && (
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                                    <TrendingUp size={20} /> Next 7 Days Trend
                                </h3>
                                <div className="h-[300px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={customPrediction.forecast_7_days.dates.map((d, i) => ({
                                            date: d,
                                            price: customPrediction.forecast_7_days.prices[i],
                                            arrival: customPrediction.forecast_7_days.arrivals[i]
                                        }))}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                            <XAxis dataKey="date" stroke="#9CA3AF" tickFormatter={(str) => new Date(str).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} />
                                            <YAxis yAxisId="left" stroke="#16a34a" />
                                            <YAxis yAxisId="right" orientation="right" stroke="#93c5fd" />
                                            <Tooltip />
                                            <Legend />
                                            <Bar yAxisId="right" dataKey="arrival" name="Supply" fill="#93c5fd" opacity={0.5} />
                                            <Line yAxisId="left" type="monotone" dataKey="price" name="Price" stroke="#16a34a" strokeWidth={3} dot={true} />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}

                        {/* Weekly Summary */}
                        {customPrediction.weekly_summary && <WeeklySummary data={customPrediction.weekly_summary} />}
                    </div>
                )}

                {!forecastData && !customPrediction && !loading && (
                    <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100 flex items-center justify-center h-full text-gray-400">
                        Select parameters and run forecast to see results
                    </div>
                )}
            </div>
        </div>
    );
};

export default ForecastStudio;
