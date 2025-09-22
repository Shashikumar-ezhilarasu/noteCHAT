#!/bin/bash

echo "🚀 Starting AI Notebook Assistant..."

# Check if backend is setup
if [ ! -d "backend/venv" ]; then
    echo "❌ Backend not setup. Please run ./setup.sh first"
    exit 1
fi

# Start backend in background
echo "📡 Starting backend server..."
cd backend
source venv/bin/activate
python3 main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Start frontend
echo "🌐 Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Both servers started!"
echo "🔗 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to stop
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
