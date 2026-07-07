@echo off
echo ============================================
echo  Hydrogen Bubble Analysis System — Startup
echo ============================================

:: Start backend
echo.
echo [1/2] Starting FastAPI backend...
cd backend
start cmd /k "pip install -r requirements.txt && python main.py"
cd ..

:: Wait 3 seconds for backend to boot
timeout /t 3 /nobreak >nul

:: Start frontend
echo [2/2] Starting React frontend...
cd frontend
start cmd /k "npm install && npm run dev"
cd ..

echo.
echo Both services starting...
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API docs: http://localhost:8000/docs
echo.
pause
