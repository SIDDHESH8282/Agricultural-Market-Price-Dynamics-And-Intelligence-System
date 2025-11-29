import React, { useEffect, useState } from 'react';
import { getTradingInsights, analyzeArbitrage, forecastArbitrage } from '../services/api';
import LocationSelector from '../components/LocationSelector';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, Legend } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, ArrowRight, AlertTriangle, CheckCircle, Sparkles, Calendar } from 'lucide-react';

const MarketCard = ({ market, trendData, type }) => {
    const isBuy = type === 'buy';
    const color = isBuy ? '#16a34a' : '#dc2626'; // Green or Red

    return (
        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start mb-2">
                <div>
                    <h4 className="font-bold text-gray-800">{market.market}</h4>
                    <p className="text-xs text-gray-500">{market.district}, {market.state}</p>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-bold ${isBuy ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                    ₹{market.Price}
                </span>
            </div>

            <div className="h-16 w-full mt-2">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trendData}>
                        <defs>
                            <linearGradient id={`grad${market.market}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={color} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <Area
                            type="monotone"
                            dataKey="Price"
                            stroke={color}
                            fill={`url(#grad${market.market})`}
                            strokeWidth={2}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const TradersHub = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    // Arbitrage State
    const [source, setSource] = useState(null);
    const [dest, setDest] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);

    // Forecast State
    const [forecastData, setForecastData] = useState(null);
    const [forecastHorizon, setForecastHorizon] = useState(30);
    const [forecasting, setForecasting] = useState(false);

    useEffect(() => {
        getTradingInsights().then(res => {
            setData(res);
            setLoading(false);
        });
    }, []);

    const handleAnalyze = async () => {
        if (!source || !dest) return;
        setAnalyzing(true);
        setAnalysis(null);
        setForecastData(null);

        try {
            // 1. Current Analysis
            const res = await analyzeArbitrage({
                source_state: source.state,
                source_district: source.district,
                source_market: source.market,
                dest_state: dest.state,
                dest_district: dest.district,
                dest_market: dest.market
            });
            setAnalysis(res);

            // 2. Future Forecast
            fetchForecast(30); // Default to 1M

        } catch (err) {
            console.error(err);
        }
        setAnalyzing(false);
    };

    const fetchForecast = async (horizon) => {
        setForecasting(true);
        setForecastHorizon(horizon);
        try {
            const res = await forecastArbitrage({
                source_state: source.state,
                source_district: source.district,
                source_market: source.market,
                dest_state: dest.state,
                dest_district: dest.district,
                dest_market: dest.market,
                horizon: horizon
            });
            setForecastData(res);
        } catch (err) {
            console.error(err);
        }
        setForecasting(false);
    };

    if (loading) return <div className="p-8 text-center text-gray-500">Loading Market Intelligence...</div>;

    return (
        <div className="space-y-8 pb-20">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Traders Hub</h1>
                    <p className="text-gray-500">Advanced Market Analysis & Arbitrage</p>
                </div>
                <div className="bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-xs font-bold">
                    Live Data
                </div>
            </div>

            {/* Top Markets Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Buy Zones */}
                <div>
                    <h3 className="text-lg font-bold text-green-800 mb-4 flex items-center gap-2">
                        <TrendingDown size={20} /> Best Buy Zones (Low Prices)
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {data.buy_markets.map((m, i) => (
                            <MarketCard
                                key={m.market}
                                market={m}
                                trendData={data.buy_trends[i].data}
                                type="buy"
                            />
                        ))}
                    </div>
                </div>

                {/* Sell Zones */}
                <div>
                    <h3 className="text-lg font-bold text-red-800 mb-4 flex items-center gap-2">
                        <TrendingUp size={20} /> Premium Sell Markets (High Prices)
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {data.sell_markets.map((m, i) => (
                            <MarketCard
                                key={m.market}
                                market={m}
                                trendData={data.sell_trends[i].data}
                                type="sell"
                            />
                        ))}
                    </div>
                </div>
            </div>

            {/* Arbitrage Analyzer */}
            <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="bg-gray-900 p-6 text-white">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <DollarSign className="text-yellow-400" /> Arbitrage Calculator
                    </h3>
                    <p className="text-gray-400 text-sm mt-1">Select any two markets to analyze profit potential.</p>
                </div>

                <div className="p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 relative">
                        <LocationSelector label="Source Market (Buy)" onSelect={setSource} type="source" />

                        {/* Arrow Icon */}
                        <div className="hidden md:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white p-2 rounded-full shadow-md z-10 border border-gray-100">
                            <ArrowRight className="text-gray-400" />
                        </div>

                        <LocationSelector label="Destination Market (Sell)" onSelect={setDest} type="dest" />
                    </div>

                    <div className="mt-6 flex justify-center">
                        <button
                            onClick={handleAnalyze}
                            disabled={!source || !dest || analyzing}
                            className="bg-gray-900 text-white px-8 py-3 rounded-lg font-bold hover:bg-gray-800 disabled:opacity-50 transition-colors flex items-center gap-2"
                        >
                            {analyzing ? 'Analyzing...' : 'Analyze Opportunity'}
                        </button>
                    </div>
                </div>

                {/* Analysis Result */}
                {analysis && (
                    <div className="border-t border-gray-100 bg-gray-50 p-8 animate-in fade-in slide-in-from-bottom-4">
                        {/* Current Status */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                            {/* Chart */}
                            <div className="lg:col-span-2 bg-white p-4 rounded-xl shadow-sm">
                                <h4 className="font-bold text-gray-700 mb-4">Historical Price Comparison</h4>
                                <div className="h-[300px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                                            <XAxis dataKey="Date" tick={false} />
                                            <YAxis />
                                            <Tooltip />
                                            <Legend />
                                            <Line data={analysis.source_history} dataKey="Price" name={source.market} stroke="#16a34a" strokeWidth={2} dot={false} />
                                            <Line data={analysis.dest_history} dataKey="Price" name={dest.market} stroke="#dc2626" strokeWidth={2} dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Insights */}
                            <div className="space-y-4">
                                <div className="bg-white p-6 rounded-xl shadow-sm border-l-4 border-blue-500">
                                    <span className="text-xs font-bold text-gray-400 uppercase">Current Spread</span>
                                    <div className="text-3xl font-bold text-gray-900 mt-1">
                                        ₹{analysis.analysis.spread.toFixed(0)}
                                        <span className="text-sm font-normal text-gray-500 ml-1">/ qtl</span>
                                    </div>
                                </div>

                                <div className={`p-6 rounded-xl shadow-sm border-l-4 ${analysis.analysis.recommendation === 'AVOID' ? 'bg-red-50 border-red-500' : 'bg-green-50 border-green-500'}`}>
                                    <span className="text-xs font-bold text-gray-400 uppercase">Recommendation</span>
                                    <div className={`text-2xl font-bold mt-1 ${analysis.analysis.recommendation === 'AVOID' ? 'text-red-700' : 'text-green-700'}`}>
                                        {analysis.analysis.recommendation}
                                    </div>
                                    <p className="text-sm text-gray-600 mt-2 leading-relaxed">
                                        {analysis.analysis.reason}
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Future Forecast Section */}
                        <div className="bg-indigo-50 rounded-xl p-6 border border-indigo-100">
                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
                                <div>
                                    <h3 className="text-xl font-bold text-indigo-900 flex items-center gap-2">
                                        <Sparkles className="text-yellow-500" /> Future Opportunity Analyzer
                                    </h3>
                                    <p className="text-indigo-600 text-sm">AI-driven ROI projections for the selected route.</p>
                                </div>

                                <div className="flex bg-white rounded-lg p-1 shadow-sm">
                                    {[
                                        { label: '1 Month', value: 30 },
                                        { label: '3 Months', value: 90 },
                                        { label: '6 Months', value: 180 }
                                    ].map(item => (
                                        <button
                                            key={item.label}
                                            onClick={() => fetchForecast(item.value)}
                                            className={`px-4 py-2 text-xs font-bold rounded-md transition-all ${forecastHorizon === item.value
                                                ? 'bg-indigo-600 text-white shadow-md'
                                                : 'text-gray-500 hover:text-gray-900'
                                                }`}
                                        >
                                            {item.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {forecasting ? (
                                <div className="h-[300px] flex items-center justify-center text-indigo-400">
                                    <Sparkles className="animate-spin mr-2" /> Generating Forecast...
                                </div>
                            ) : forecastData ? (
                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                                    <div className="lg:col-span-2 h-[300px] bg-white rounded-xl p-4 shadow-sm">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={forecastData.forecast}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                                                <XAxis dataKey="date" tickFormatter={(str) => new Date(str).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} />
                                                <YAxis label={{ value: 'Price (₹)', angle: -90, position: 'insideLeft' }} />
                                                <Tooltip labelFormatter={(label) => new Date(label).toLocaleDateString()} />
                                                <Legend />
                                                <Line type="monotone" dataKey="source_price" name={`Buy: ${source.market}`} stroke="#16a34a" strokeWidth={2} dot={false} />
                                                <Line type="monotone" dataKey="dest_price" name={`Sell: ${dest.market}`} stroke="#dc2626" strokeWidth={2} dot={false} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>

                                    <div className="bg-white p-6 rounded-xl shadow-sm border border-indigo-100">
                                        <h4 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
                                            <Calendar size={18} className="text-indigo-500" /> AI Strategy
                                        </h4>
                                        <div className="prose prose-sm text-gray-600" dangerouslySetInnerHTML={{ __html: forecastData.ai_analysis }} />
                                    </div>
                                </div>
                            ) : null}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TradersHub;
