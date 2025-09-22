#!/bin/bash

echo "🚀 Setting up AI Notebook Assistant..."

# Setup backend
echo "📦 Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

echo "✅ Backend setup complete!"

# Setup frontend
echo "📦 Setting up frontend..."
cd ../frontend

# Install npm dependencies
npm install

echo "✅ Frontend setup complete!"

echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Update backend/firebase_admin_config.json with your Firebase credentials"
echo "2. Upload your documents to Firebase Storage under 'notebooks/' folder"
echo "3. Start the backend: cd backend && source venv/bin/activate && python3 main.py"
echo "4. Start the frontend: cd frontend && npm run dev"
echo ""
echo "Then visit http://localhost:3000 to use your AI assistant!"
