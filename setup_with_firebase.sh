#!/bin/bash

echo "🔥 AI Notebook Assistant - Complete Setup with Firebase"
echo "=" * 60

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

# Check Firebase configuration
echo ""
echo "🔥 Firebase Configuration Check"
echo "-" * 30

cd ../backend

if [ -f "firebase_admin_config.json" ]; then
    # Check if it's still the template
    if grep -q "your-private-key-id" firebase_admin_config.json; then
        echo "⚠️  Firebase service account not configured yet!"
        echo ""
        echo "📋 To complete setup:"
        echo "1. Go to https://console.firebase.google.com"
        echo "2. Select your project: notechat-26c38"
        echo "3. Go to Project Settings > Service Accounts"
        echo "4. Click 'Generate new private key'"
        echo "5. Replace backend/firebase_admin_config.json with the downloaded file"
        echo ""
        echo "🚀 Once configured, run: ./upload_and_start.sh"
    else
        echo "✅ Firebase configuration found!"
        echo ""
        echo "🚀 Ready to upload documents and start!"
        echo "Run: ./upload_and_start.sh"
    fi
else
    echo "❌ firebase_admin_config.json not found!"
    echo "Please add your Firebase service account credentials."
fi

echo ""
echo "📁 Current documents in NOTES folder:"
if [ -d "../NOTES" ]; then
    ls -la "../NOTES"
else
    echo "❌ NOTES folder not found!"
fi

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Configure Firebase credentials (if not done)"
echo "2. Run: ./upload_and_start.sh to upload docs and start servers"
echo "3. Or manually: python3 upload_documents.py then ./start.sh"
