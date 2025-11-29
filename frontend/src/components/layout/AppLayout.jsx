import React from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { LayoutDashboard, LineChart, TrendingUp, Sprout, DollarSign, ThermometerSun } from 'lucide-react';
import { cn } from '../../lib/utils';

const SidebarItem = ({ to, icon: Icon, label }) => (
    <NavLink
        to={to}
        className={({ isActive }) =>
            cn(
                "flex items-center gap-3 px-4 py-3 rounded-lg transition-colors mb-1",
                isActive
                    ? "bg-primary text-white shadow-md"
                    : "text-gray-600 hover:bg-primary/10 hover:text-primary"
            )
        }
    >
        <Icon size={20} />
        <span className="font-medium">{label}</span>
    </NavLink>
);

const AppLayout = () => {
    return (
        <div className="flex h-screen bg-background">
            {/* Sidebar */}
            <aside className="w-64 bg-white border-r border-gray-200 flex flex-col shadow-sm z-10">
                <div className="p-6 border-b border-gray-100">
                    <div className="flex items-center gap-2 text-primary-dark">
                        <Sprout size={28} />
                        <h1 className="text-xl font-bold tracking-tight">OnionForecast</h1>
                    </div>
                </div>

                <nav className="flex-1 p-4 overflow-y-auto">
                    <div className="mb-6">
                        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 px-2">
                            Menu
                        </p>
                        <SidebarItem to="/dashboard" icon={LayoutDashboard} label="Dashboard" />
                        <SidebarItem to="/explore" icon={LineChart} label="Explore Data" />
                        <SidebarItem to="/forecast" icon={TrendingUp} label="Forecast Studio" />
                        <SidebarItem to="/weather" icon={ThermometerSun} label="Weather Station" />
                        <SidebarItem to="/traders" icon={DollarSign} label="Traders Hub" />
                    </div>
                </nav>

                <div className="p-4 border-t border-gray-100">
                    <div className="bg-primary/5 rounded-lg p-4">
                        <p className="text-xs text-gray-500 mb-1">Model Status</p>
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                            <span className="text-sm font-medium text-primary-dark">Online</span>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-auto">
                <header className="bg-white border-b border-gray-200 px-8 py-4 sticky top-0 z-0">
                    <h2 className="text-lg font-semibold text-gray-800">
                        Agricultural Market Intelligence
                    </h2>
                </header>
                <div className="p-8 max-w-7xl mx-auto">
                    <Outlet />
                </div>
            </main>
        </div>
    );
};

export default AppLayout;
