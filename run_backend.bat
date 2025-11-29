@echo off
echo Starting FastAPI Backend...
cd backend
uvicorn main:app --reload --port 8000
pause
