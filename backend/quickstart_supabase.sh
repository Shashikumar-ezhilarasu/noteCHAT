#!/bin/bash

# Quick Start Script for Supabase Vector Store Setup
# This script guides you through the entire setup process

echo "🚀 noteCHAT - Supabase Vector Store Quick Start"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -d "../NOTES" ]; then
    echo "❌ Error: NOTES folder not found!"
    echo "Please run this script from the backend directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Step 1: Check environment variables
echo "📋 Step 1: Checking Supabase credentials..."
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_KEY" ]; then
    echo ""
    echo "❌ Supabase credentials not found!"
    echo ""
    echo "Please set your Supabase credentials:"
    echo ""
    read -p "Enter your Supabase URL (https://xxx.supabase.co): " SUPABASE_URL
    read -p "Enter your Supabase anon key: " SUPABASE_KEY
    
    export SUPABASE_URL
    export SUPABASE_KEY
    
    echo ""
    echo "✅ Credentials set for this session"
    echo ""
    echo "💡 To make this permanent, add to ~/.zshrc:"
    echo "   export SUPABASE_URL='$SUPABASE_URL'"
    echo "   export SUPABASE_KEY='$SUPABASE_KEY'"
else
    echo "✅ Supabase credentials found"
fi

echo ""

# Step 2: Install dependencies
echo "📦 Step 2: Installing dependencies..."
pip install -q supabase sentence-transformers
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""

# Step 3: Show SQL schema instructions
echo "📋 Step 3: Database Schema Setup"
echo "================================"
echo ""
echo "⚠️  IMPORTANT: You need to run the SQL schema in Supabase first!"
echo ""
echo "Follow these steps:"
echo "  1. Open https://supabase.com and go to your project"
echo "  2. Click 'SQL Editor' in the left sidebar"
echo "  3. Click 'New Query'"
echo "  4. Copy the contents of 'supabase_schema.sql'"
echo "  5. Paste into the SQL Editor and click 'Run'"
echo ""
read -p "Press Enter after you've run the SQL schema in Supabase..."

echo ""

# Step 4: Process documents
echo "🔄 Step 4: Processing documents..."
echo "================================"
echo ""

python setup_supabase_vectorstore.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ SUCCESS! Your vector database is ready!"
    echo "================================================"
    echo ""
    echo "🎉 Next steps:"
    echo "  1. Check your Supabase dashboard > Table Editor > document_chunks"
    echo "  2. Try semantic search with your documents"
    echo "  3. Integrate with your noteCHAT application"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi
