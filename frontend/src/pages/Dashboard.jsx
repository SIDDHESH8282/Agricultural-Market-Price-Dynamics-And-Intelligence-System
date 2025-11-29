import React, { useEffect, useState } from 'react';
import { ArrowUpRight, ArrowDownRight, TrendingUp, Activity, Calendar, Globe, Zap, BarChart2, TrendingDown } from 'lucide-react';
import { getLocations, getHistory, getDashboardStats, getTickerData } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const KPICard = ({ title, value, change, trend, icon: Icon, color }) => (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <div className="flex justify-between items-start mb-4">
            <div className={`p-3 rounded-lg ${color}`}>
                <Icon size={24} className="text-white" />
            </div>
            {change && (
                <div className={`flex items-center gap-1 text-sm font-medium ${trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                    {trend === 'up' ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
                    {change}
                </div>
            )}
        </div>
        <h3 className="text-gray-500 text-sm font-medium mb-1">{title}</h3>
        <p className="text-2xl font-bold text-gray-900">{value}</p>
    </div>
);

const LiveTicker = ({ tickerData }) => {
    if (!tickerData || tickerData.length === 0) return null;

    // Duplicate items for seamless loop
    const items = [...tickerData, ...tickerData];

    return (
        <div className="bg-gradient-to-r from-gray-900 to-gray-800 py-3 overflow-hidden relative">
            <div className="absolute left-0 top-0 bottom-0 w-20 bg-gradient-to-r from-gray-900 to-transparent z-10"></div>
            <div className="absolute right-0 top-0 bottom-0 w-20 bg-gradient-to-l from-gray-900 to-transparent z-10"></div>

            <div className="flex animate-scroll">
                {items.map((item, index) => (
                    <div key={index} className="flex items-center space-x-2 mx-6 whitespace-nowrap">
                        <span className="text-white font-semibold">{item.market}</span>
                        <span className="text-gray-400 text-sm">({item.state})</span>
                        <span className="text-yellow-400 font-bold">₹{item.price}</span>
                        <span className={`text-sm font-medium ${item.direction === 'up' ? 'text-green-400' :
                                item.direction === 'down' ? 'text-red-400' : 'text-gray-400'
                            }`}>
                            {item.direction === 'up' ? '↑' : item.direction === 'down' ? '↓' : '→'}
                            {Math.abs(item.change_pct).toFixed(1)}%
                        </span>
                        <span className="text-gray-600">|</span>
                    </div>
                ))}
            </div>

            <style jsx>{`
                @keyframes scroll {
                    0% { transform: translateX(0); }
                    100% { transform: translateX(-50%); }
                }
                .animate-scroll {
                    animation: scroll 30s linear infinite;
                }
                .animate-scroll:hover {
                    animation-play-state: paused;
                }
            `}</style>
        </div>
    );
};

const Dashboard = () => {
    const [locations, setLocations] = useState({ states: [], districts: {}, markets: {} });
    const [selectedState, setSelectedState] = useState('');
    const [selectedDistrict, setSelectedDistrict] = useState('');
    const [selectedMarket, setSelectedMarket] = useState('');
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(false);

    // Dashboard stats
    const [dashboardStats, setDashboardStats] = useState(null);
    const [tickerData, setTickerData] = useState([]);
    const [statsLoading, setStatsLoading] = useState(true);

    useEffect(() => {
        getLocations().then(data => {
            setLocations(data);
            // Default selection
            if (data.states.length > 0) {
                const s = data.states[0];
                setSelectedState(s);
                const d = data.districts[s]?.[0];
                setSelectedDistrict(d);
                const m = data.markets[d]?.[0];
                setSelectedMarket(m);
            }
        });

        // Load dashboard stats
        loadDashboardData();

        // Refresh ticker every 30 seconds
        const interval = setInterval(loadDashboardData, 30000);
        return () => clearInterval(interval);
    }, []);

    const loadDashboardData = async () => {
        try {
            const [stats, ticker] = await Promise.all([
                getDashboardStats(),
                getTickerData()
            ]);
            setDashboardStats(stats);
            setTickerData(ticker.ticker_items);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        } finally {
            setStatsLoading(false);
        }
    };

    useEffect(() => {
        if (selectedState && selectedDistrict && selectedMarket) {
            setLoading(true);
            getHistory(selectedState, selectedDistrict, selectedMarket)
                .then(data => setHistory(data))
                .finally(() => setLoading(false));
        }
    }, [selectedState, selectedDistrict, selectedMarket]);

    // Calculate KPIs
    const latest = history.length > 0 ? history[history.length - 1] : null;
    const prev = history.length > 1 ? history[history.length - 2] : null;

    const price = latest?.Price || 0;
    const priceChange = prev ? ((price - prev.Price) / prev.Price) * 100 : 0;

    const arrivals = latest?.Arrivals || 0;

    return (
        <div className="space-y-6">
            {/* Live Ticker */}
            <LiveTicker tickerData={tickerData} />

            {/* Header & Filters */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                        <Globe className="text-primary" size={28} />
                        Market Overview
                    </h1>
                    <p className="text-gray-500">Real-time insights across India</p>
                </div>

                <div className="flex gap-3 bg-white p-2 rounded-lg shadow-sm border border-gray-200">
                    <select
                        className="bg-transparent border-none text-sm font-medium focus:ring-0 cursor-pointer"
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
                    <div className="w-px bg-gray-200 my-1" />
                    <select
                        className="bg-transparent border-none text-sm font-medium focus:ring-0 cursor-pointer"
                        value={selectedDistrict}
                        onChange={(e) => {
                            setSelectedDistrict(e.target.value);
                            setSelectedMarket(locations.markets[e.target.value]?.[0]);
                        }}
                    >
                        {locations.districts[selectedState]?.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                    <div className="w-px bg-gray-200 my-1" />
                    <select
                        className="bg-transparent border-none text-sm font-medium focus:ring-0 cursor-pointer"
                        value={selectedMarket}
                        onChange={(e) => setSelectedMarket(e.target.value)}
                    >
                        {locations.markets[selectedDistrict]?.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                </div>
            </div>

            {/* Global Stats KPIs */}
            {!statsLoading && dashboardStats && (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                        <KPICard
                            title="National Avg Price"
                            value={`₹${dashboardStats.national_stats.avg_price}/qtl`}
                            change={`${Math.abs(dashboardStats.national_stats.price_change_pct).toFixed(1)}%`}
                            trend={dashboardStats.national_stats.price_change_pct >= 0 ? 'up' : 'down'}
                            icon={TrendingUp}
                            color="bg-primary"
                        />
                        <KPICard
                            title="Total Arrivals"
                            value={`${(dashboardStats.national_stats.total_arrivals / 1000).toFixed(1)}K Tonnes`}
                            icon={Activity}
                            color="bg-accent"
                        />
                        <KPICard
                            title="Active Markets"
                            value={dashboardStats.national_stats.total_markets}
                            icon={BarChart2}
                            color="bg-secondary"
                        />
                        <div className={`p-6 rounded-xl shadow-sm border-2 ${dashboardStats.market_sentiment.color === 'green' ? 'bg-green-50 border-green-300' :
                                dashboardStats.market_sentiment.color === 'red' ? 'bg-red-50 border-red-300' :
                                    'bg-gray-50 border-gray-300'
                            }`}>
                            <div className="flex justify-between items-start mb-4">
                                <div className={`p-3 rounded-lg ${dashboardStats.market_sentiment.color === 'green' ? 'bg-green-600' :
                                        dashboardStats.market_sentiment.color === 'red' ? 'bg-red-600' :
                                            'bg-gray-600'
                                    }`}>
                                    <Zap size={24} className="text-white" />
                                </div>
                            </div>
                            <h3 className="text-gray-500 text-sm font-medium mb-1">Market Sentiment</h3>
                            <p className={`text-2xl font-bold ${dashboardStats.market_sentiment.color === 'green' ? 'text-green-700' :
                                    dashboardStats.market_sentiment.color === 'red' ? 'text-red-700' :
                                        'text-gray-700'
                                }`}>
                                {dashboardStats.market_sentiment.sentiment}
                            </p>
                            <p className="text-xs text-gray-600 mt-1">{dashboardStats.volatility_index.level} Volatility</p>
                        </div>
                    </div>

                    {/* Top Movers & Regional Stats */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Top Movers */}
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <Zap size={20} className="text-orange-500" />
                                Top Market Movers (7 Days)
                            </h3>
                            <div className="space-y-3">
                                {dashboardStats.top_movers.slice(0, 5).map((mover, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                                        <div className="flex-1">
                                            <p className="font-semibold text-gray-900 text-sm">{mover.market}</p>
                                            <p className="text-xs text-gray-500">{mover.district}, {mover.state}</p>
                                        </div>
                                        <div className="text-right">
                                            <p className="font-bold text-gray-900">₹{mover.current_price}</p>
                                            <p className={`text-sm font-medium flex items-center gap-1 ${mover.direction === 'up' ? 'text-green-600' : 'text-red-600'
                                                }`}>
                                                {mover.direction === 'up' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                                {Math.abs(mover.change_pct)}%
                                            </p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Regional Stats */}
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                                <Globe size={20} className="text-blue-500" />
                                Top States by Price
                            </h3>
                            <div className="space-y-3">
                                {dashboardStats.regional_stats.map((region, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                        <div className="flex items-center gap-3">
                                            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${idx === 0 ? 'bg-yellow-100 text-yellow-700' :
                                                    idx === 1 ? 'bg-gray-200 text-gray-700' :
                                                        idx === 2 ? 'bg-orange-100 text-orange-700' :
                                                            'bg-gray-100 text-gray-600'
                                                }`}>
                                                {idx + 1}
                                            </div>
                                            <p className="font-semibold text-gray-900">{region.state}</p>
                                        </div>
                                        <p className="font-bold text-gray-900">₹{region.avg_price}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </>
            )}

            {/* Selected Market KPIs */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <KPICard
                    title="Current Modal Price"
                    value={`₹${price.toFixed(2)}/qtl`}
                    change={`${Math.abs(priceChange).toFixed(2)}%`}
                    trend={priceChange >= 0 ? 'up' : 'down'}
                    icon={TrendingUp}
                    color="bg-primary"
                />
                <KPICard
                    title="Daily Arrivals"
                    value={`${arrivals.toFixed(1)} Tonnes`}
                    icon={Activity}
                    color="bg-accent"
                />
                <KPICard
                    title="Latest Data Point"
                    value={latest ? new Date(latest.Date).toLocaleDateString() : '-'}
                    icon={Calendar}
                    color="bg-secondary"
                />
            </div>

            {/* Main Chart */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="text-lg font-semibold text-gray-900 mb-6">Price Trend (Last 1 Year)</h3>
                <div className="h-[400px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={history}>
                            <defs>
                                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#2E7D32" stopOpacity={0.2} />
                                    <stop offset="95%" stopColor="#2E7D32" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                            <XAxis
                                dataKey="Date"
                                tickFormatter={(str) => new Date(str).toLocaleDateString(undefined, { month: 'short' })}
                                stroke="#9CA3AF"
                                tick={{ fontSize: 12 }}
                            />
                            <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                            <Tooltip
                                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                labelFormatter={(label) => new Date(label).toLocaleDateString()}
                            />
                            <Area
                                type="monotone"
                                dataKey="Price"
                                stroke="#2E7D32"
                                strokeWidth={2}
                                fillOpacity={1}
                                fill="url(#colorPrice)"
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
