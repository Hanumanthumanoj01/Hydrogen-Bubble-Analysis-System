#!/bin/bash
echo "============================================"
echo " Hydrogen Bubble Analysis System — Startup"
echo "============================================"

# Backend
echo ""
echo "[1/2] Starting FastAPI backend..."
cd backend
pip install -r requirements.txt -q
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

sleep 2

# Frontend
echo "[2/2] Starting React frontend..."
cd frontend
npm install -q
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "Both services running:"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both."

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
