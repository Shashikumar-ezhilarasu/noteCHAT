#!/bin/bash

echo "🔥 Upload Documents to Firebase and Start Servers"
echo "=" * 50

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Backend not setup. Please run ./setup_with_firebase.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check Firebase configuration
if grep -q "your-private-key-id" firebase_admin_config.json; then
    echo "❌ Firebase not configured! Please update firebase_admin_config.json first"
    exit 1
fi

# Upload documents to Firebase
echo "📤 Uploading documents to Firebase Storage..."
python3 upload_documents.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Documents uploaded successfully!"
    echo ""
    echo "🚀 Starting servers..."
    
    # Start backend in background
    echo "📡 Starting backend server..."
    python3 main.py &
    BACKEND_PID=$!
    
    # Wait for backend to start
    echo "⏳ Waiting for backend to initialize..."
    sleep 8
    
    # Start frontend
    echo "🌐 Starting frontend..."
    cd ../frontend
    npm run dev &
    FRONTEND_PID=$!
    
    echo ""
    echo "🎉 All systems running!"
    echo "🔗 Frontend: http://localhost:3000"
    echo "🔗 Backend API: http://localhost:8000"
    echo "📚 Your documents are now searchable!"
    echo ""
    echo "Press Ctrl+C to stop both servers"
    
    # Wait for user to stop
    trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
    wait
else
    echo "❌ Document upload failed. Please check Firebase configuration."
    exit 1
fi
