import React, { useState, useEffect } from 'react';
import { getLocations, fetchMarkets } from '../services/api';
import { MapPin } from 'lucide-react';

const LocationSelector = ({ label, onSelect, type = 'source' }) => {
    const [locations, setLocations] = useState({ states: [], districts: {} });
    const [markets, setMarkets] = useState([]);

    const [selectedState, setSelectedState] = useState('');
    const [selectedDistrict, setSelectedDistrict] = useState('');
    const [selectedMarket, setSelectedMarket] = useState('');

    useEffect(() => {
        getLocations().then(setLocations);
    }, []);

    useEffect(() => {
        if (selectedState && selectedDistrict) {
            fetchMarkets(selectedState, selectedDistrict).then(setMarkets);
        } else {
            setMarkets([]);
        }
    }, [selectedState, selectedDistrict]);

    const handleMarketChange = (e) => {
        const marketName = e.target.value;
        setSelectedMarket(marketName);

        if (marketName) {
            const marketData = markets.find(m => m.market === marketName);
            onSelect({
                state: selectedState,
                district: selectedDistrict,
                market: marketName,
                price: marketData?.Price || 0
            });
        } else {
            onSelect(null);
        }
    };

    return (
        <div className={`p-4 rounded-xl border ${type === 'source' ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'}`}>
            <h4 className={`font-bold mb-3 flex items-center gap-2 ${type === 'source' ? 'text-green-800' : 'text-red-800'}`}>
                <MapPin size={18} /> {label}
            </h4>

            <div className="space-y-3">
                {/* State */}
                <div>
                    <label className="text-xs font-medium text-gray-500 ml-1">State</label>
                    <select
                        className="w-full p-2 rounded-lg border border-gray-200 text-sm focus:ring-2 focus:ring-primary/20 outline-none"
                        value={selectedState}
                        onChange={(e) => {
                            setSelectedState(e.target.value);
                            setSelectedDistrict('');
                            setSelectedMarket('');
                        }}
                    >
                        <option value="">Select State</option>
                        {locations.states.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                </div>

                {/* District */}
                <div>
                    <label className="text-xs font-medium text-gray-500 ml-1">District</label>
                    <select
                        className="w-full p-2 rounded-lg border border-gray-200 text-sm focus:ring-2 focus:ring-primary/20 outline-none"
                        value={selectedDistrict}
                        disabled={!selectedState}
                        onChange={(e) => {
                            setSelectedDistrict(e.target.value);
                            setSelectedMarket('');
                        }}
                    >
                        <option value="">Select District</option>
                        {selectedState && locations.districts[selectedState]?.map(d => (
                            <option key={d} value={d}>{d}</option>
                        ))}
                    </select>
                </div>

                {/* Market */}
                <div>
                    <label className="text-xs font-medium text-gray-500 ml-1">Market (Sorted by Price)</label>
                    <select
                        className="w-full p-2 rounded-lg border border-gray-200 text-sm focus:ring-2 focus:ring-primary/20 outline-none font-mono"
                        value={selectedMarket}
                        disabled={!selectedDistrict}
                        onChange={handleMarketChange}
                    >
                        <option value="">Select Market</option>
                        {markets.map(m => (
                            <option key={m.market} value={m.market}>
                                {m.market} (₹{m.Price})
                            </option>
                        ))}
                    </select>
                </div>
            </div>
        </div>
    );
};

export default LocationSelector;
