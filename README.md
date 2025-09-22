# AI Notebook Assistant (Fixed Version)

A full-stack AI assistant that uses Firebase Storage, FastAPI, and Next.js to create a RAG (Retrieval-Augmented Generation) system for your documents.

## 🏗️ Architecture

```
📂 AI Notebook Assistant
├── 🔧 Backend (FastAPI + Firebase + Simple RAG)
│   ├── Firebase Storage (documents) - Optional
│   ├── Simple RAG Pipeline (keyword-based search)
│   └── Fallback to local files
├── 🌐 Frontend (Next.js + Tailwind CSS)
│   └── Query Interface
└── 📚 Documents (Your ML Notes)
```

## 🚀 Quick Setup

### Option 1: Automated Setup

```bash
# Run the setup script
./setup.sh

# Start both servers
./start.sh
```

### Option 2: Manual Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install minimal dependencies
pip install -r requirements.txt

# Start the server
python3 main.py
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 📋 Features

✅ **Firebase Storage Integration**: Automatically syncs documents from cloud storage  
✅ **Fallback Support**: Works with local files if Firebase is not configured  
✅ **Simple RAG**: Keyword-based search and context generation  
✅ **Modern UI**: Clean Next.js interface with Tailwind CSS  
✅ **Fast Setup**: Minimal dependencies for quick start

## 📚 Supported Documents

- PDF files (`.pdf`) - requires PyPDF2
- Word documents (`.docx`) - requires python-docx
- Text files (`.txt`)
- Markdown files (`.md`)

## 🔧 Configuration Options

### Option 1: Firebase Storage (Recommended for Production)

1. Create a Firebase project at https://console.firebase.google.com
2. Enable Firebase Storage
3. Create a service account and download credentials
4. Replace `backend/firebase_admin_config.json` with your credentials
5. Upload documents to `notebooks/` folder in Firebase Storage

### Option 2: Local Files (Quick Testing)

- The system automatically falls back to local files from `../NOTES/` folder
- No Firebase configuration needed
- Perfect for development and testing

## 🚀 Usage

1. **Start the system**: Run `./start.sh` or start backend/frontend manually
2. **Upload documents**: Either to Firebase Storage or place in local NOTES folder
3. **Visit http://localhost:3000**
4. **Ask questions** about your documents!

Example queries:

- "What is K-means clustering?"
- "Explain the difference between supervised and unsupervised learning"
- "How does hierarchical clustering work?"

## 🛠️ Troubleshooting

### Common Issues

**Backend won't start:**

- Use `python3` instead of `python`
- Check virtual environment is activated: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

**No documents found:**

- Check if files are in Firebase Storage under `notebooks/` folder
- Or place files in local `../NOTES/` folder as fallback
- Supported formats: PDF, DOCX, TXT, MD

**Frontend issues:**

- Ensure backend is running on port 8000
- Check browser console for errors
- Run `npm install` to install dependencies

### File Structure Check

```bash
# Your project should look like this:
project/
├── NOTES/                    # Your documents (fallback)
│   ├── *.pdf
│   ├── *.docx
│   └── *.txt
├── backend/
│   ├── main.py
│   ├── rag_pipeline_fixed.py
│   ├── requirements.txt
│   └── firebase_admin_config.json
├── frontend/
│   ├── pages/
│   ├── components/
│   └── package.json
├── setup.sh
└── start.sh
```

## 📖 API Endpoints

- `GET /`: Health check
- `GET /health`: Detailed system status
- `POST /query`: Submit a question (expects `{"question": "your question"}`)
- `GET /sources`: List available document sources

## 🔄 Upgrade Path

To add advanced features later:

1. Install full LangChain: `pip install langchain langchain-community`
2. Add vector embeddings: `pip install sentence-transformers chromadb`
3. Add local LLM: `pip install transformers torch`
4. Switch to `rag_pipeline.py` in `main.py`

## 📊 RAG Approaches Comparison

Below is a comparison of different RAG (Retrieval-Augmented Generation) approaches implemented in this project:

![RAG Approaches Comparison](./assets/rag_comparison_table.png)

### Confidence Score Evolution

- **Initial TF-IDF**: Simple percentage based on keyword matches (sometimes as low as 27.3%)
- **Sentence Transformers**: Normalized cosine similarity between embeddings
- **Cross-Encoder**: Raw scores (could be negative, e.g., -102.4%) now normalized using sigmoid function:
  ```
  confidence = 100 * (1 / (1 + exp(-raw_score)))
  ```

## 📄 License

This project is for educational purposes. Modify as needed for your use case.