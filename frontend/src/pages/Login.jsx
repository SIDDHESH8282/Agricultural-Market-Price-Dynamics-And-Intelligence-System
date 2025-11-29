import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sprout, Lock, Mail, ArrowRight, Loader2 } from 'lucide-react';

const Login = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleLogin = (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        // Mock Authentication
        setTimeout(() => {
            if (email && password) {
                // In a real app, you'd validate against backend
                localStorage.setItem('isAuthenticated', 'true');
                navigate('/dashboard');
            } else {
                setError('Please enter valid credentials.');
                setLoading(false);
            }
        }, 1500);
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-50 to-emerald-100 p-4">
            <div className="bg-white w-full max-w-4xl rounded-2xl shadow-2xl overflow-hidden flex flex-col md:flex-row">

                {/* Left Side - Hero/Branding */}
                <div className="md:w-1/2 bg-primary p-12 text-white flex flex-col justify-between relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-full h-full opacity-10 pointer-events-none">
                        <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                            <path fill="#FFFFFF" d="M44.7,-76.4C58.9,-69.2,71.8,-59.1,81.6,-46.6C91.4,-34.1,98.1,-19.2,95.8,-5.3C93.5,8.6,82.2,21.5,70.6,32.3C59,43.1,47.1,51.8,34.8,58.8C22.5,65.8,9.8,71.1,-1.8,74.2C-13.4,77.3,-24,78.2,-34.3,73.8C-44.6,69.4,-54.6,59.7,-63.3,48.7C-72,37.7,-79.4,25.4,-82.1,11.9C-84.8,-1.6,-82.8,-16.3,-75.4,-28.5C-68,-40.7,-55.2,-50.4,-42.2,-58.1C-29.2,-65.8,-16,-71.5,-1.2,-69.4C13.6,-67.3,27.2,-57.4,44.7,-76.4Z" transform="translate(100 100)" />
                        </svg>
                    </div>

                    <div className="z-10">
                        <div className="flex items-center gap-3 mb-8">
                            <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
                                <Sprout size={32} className="text-white" />
                            </div>
                            <h1 className="text-2xl font-bold tracking-tight">OnionForecast</h1>
                        </div>

                        <div className="space-y-6">
                            <h2 className="text-4xl font-bold leading-tight">
                                Agricultural Intelligence at Your Fingertips
                            </h2>
                            <p className="text-green-100 text-lg">
                                Advanced price prediction, weather forecasting, and market insights for smarter trading decisions.
                            </p>
                        </div>
                    </div>

                    <div className="mt-12 z-10">
                        <div className="flex items-center gap-4 text-sm text-green-100">
                            <div className="flex -space-x-2">
                                <div className="w-8 h-8 rounded-full bg-green-400 border-2 border-primary flex items-center justify-center text-xs font-bold">AI</div>
                                <div className="w-8 h-8 rounded-full bg-green-300 border-2 border-primary flex items-center justify-center text-xs font-bold">ML</div>
                                <div className="w-8 h-8 rounded-full bg-green-200 border-2 border-primary flex items-center justify-center text-xs font-bold">DS</div>
                            </div>
                            <p>Powered by Advanced Analytics</p>
                        </div>
                    </div>
                </div>

                {/* Right Side - Login Form */}
                <div className="md:w-1/2 p-12 flex flex-col justify-center">
                    <div className="mb-8">
                        <h2 className="text-3xl font-bold text-gray-900 mb-2">Welcome Back</h2>
                        <p className="text-gray-500">Please sign in to access your dashboard.</p>
                    </div>

                    <form onSubmit={handleLogin} className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                            <div className="relative">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                    <Mail size={18} className="text-gray-400" />
                                </div>
                                <input
                                    type="email"
                                    required
                                    className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-primary focus:border-primary transition-colors"
                                    placeholder="trader@example.com"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                            <div className="relative">
                                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                    <Lock size={18} className="text-gray-400" />
                                </div>
                                <input
                                    type="password"
                                    required
                                    className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-primary focus:border-primary transition-colors"
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                />
                            </div>
                        </div>

                        {error && (
                            <div className="p-3 bg-red-50 text-red-600 text-sm rounded-lg flex items-center gap-2">
                                <span className="w-1.5 h-1.5 bg-red-600 rounded-full" />
                                {error}
                            </div>
                        )}

                        <div className="flex items-center justify-between text-sm">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input type="checkbox" className="rounded border-gray-300 text-primary focus:ring-primary" />
                                <span className="text-gray-600">Remember me</span>
                            </label>
                            <a href="#" className="text-primary hover:text-primary-dark font-medium">Forgot password?</a>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-primary hover:bg-primary-dark text-white font-bold py-3 px-4 rounded-lg transition-all transform hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-2 shadow-lg shadow-primary/30"
                        >
                            {loading ? (
                                <>
                                    <Loader2 size={20} className="animate-spin" />
                                    Signing in...
                                </>
                            ) : (
                                <>
                                    Sign In <ArrowRight size={20} />
                                </>
                            )}
                        </button>
                    </form>

                    <div className="mt-8 text-center text-sm text-gray-500">
                        Don't have an account? <a href="#" className="text-primary font-bold hover:underline">Request Access</a>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;
