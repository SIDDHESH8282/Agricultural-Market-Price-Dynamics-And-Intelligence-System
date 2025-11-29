@echo off
echo ==========================================
echo Setting up Onion Forecast UI
echo ==========================================

echo 1. Installing dependencies...
cd frontend
call npm install

echo 2. Starting Frontend...
echo The app will be available at http://localhost:5173
call npm run dev
pause
