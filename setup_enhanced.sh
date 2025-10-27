#!/bin/bash

# Enhanced noteCHAT Setup with Hugging Face Models
echo "🚀 Setting up Enhanced noteCHAT with Hugging Face Models..."

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Install enhanced requirements
echo "📦 Installing enhanced dependencies..."
pip install -r requirements_enhanced.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Pre-download Hugging Face models (optional but recommended)
echo "🤗 Pre-downloading Hugging Face models..."
python -c "
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

print('📥 Downloading SentenceTransformer...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ SentenceTransformer ready!')

print('📥 Downloading QA model...')
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
print('✅ QA model ready!')

print('📥 Downloading summarization model...')
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
print('✅ Summarization model ready!')

print('🎉 All Hugging Face models downloaded successfully!')
"

echo "✅ Enhanced setup complete!"
echo ""
echo "🚀 To start the enhanced server:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python main_enhanced.py"
echo ""
echo "🌐 The server will run on http://localhost:8000"
echo "📱 Frontend will be available on http://localhost:3000"
