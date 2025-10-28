# Supabase Vector Store Setup for noteCHAT

This guide will help you set up a cloud-based vector database using Supabase PostgreSQL with pgvector extension to store your document embeddings.

## Overview

- **Database**: Supabase PostgreSQL with pgvector
- **Storage**: Document chunks + 384-dimensional vector embeddings
- **Embedding Model**: all-MiniLM-L6-v2
- **Documents**: PDFs and DOCX files from your NOTES folder

## Prerequisites

1. A Supabase account (free tier works fine)
2. Python 3.8+ with pip
3. Your NOTES folder with PDF/DOCX files

## Step-by-Step Setup

### Step 1: Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up or log in
3. Click "New Project"
4. Fill in:
   - **Name**: `notechat-vectors` (or any name you like)
   - **Database Password**: Create a strong password (save this!)
   - **Region**: Choose closest to you
   - **Pricing Plan**: Free tier is sufficient
5. Click "Create new project"
6. Wait for project to be ready (~2 minutes)

### Step 2: Get Supabase Credentials

1. In your Supabase dashboard, go to **Settings** > **API**
2. Copy these two values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon public key**: Long string starting with `eyJ...`

### Step 3: Set Environment Variables

Open your terminal and run:

```bash
# Set Supabase credentials
export SUPABASE_URL='https://your-project.supabase.co'
export SUPABASE_KEY='your-anon-public-key'
```

**For permanent setup**, add to your `~/.zshrc` or `~/.bashrc`:

```bash
echo 'export SUPABASE_URL="https://your-project.supabase.co"' >> ~/.zshrc
echo 'export SUPABASE_KEY="your-anon-public-key"' >> ~/.zshrc
source ~/.zshrc
```

### Step 4: Install Dependencies

```bash
cd /Users/shashikumarezhil/Documents/SECTOR-17\ /noteCHAT/backend
source venv/bin/activate  # Activate virtual environment
pip install supabase sentence-transformers
```

### Step 5: Setup Database Schema

1. Go to your Supabase project
2. Click **SQL Editor** in the left sidebar
3. Click **New Query**
4. Open the file `supabase_schema.sql` and copy all its contents
5. Paste into the SQL Editor
6. Click **Run** to execute

This will:

- Enable pgvector extension
- Create `document_chunks` table
- Create search indexes
- Add helper functions for similarity search

### Step 6: Process and Upload Documents

Run the setup script:

```bash
python setup_supabase_vectorstore.py
```

The script will:

1. ✅ Check Supabase connection
2. 📄 Find all PDFs and DOCX in NOTES folder
3. 📝 Extract text from documents
4. ✂️ Split into semantic chunks (~200 words each)
5. 🧠 Generate vector embeddings (384 dimensions)
6. 📤 Upload to Supabase

**Expected Output:**

```
🚀 noteCHAT - Supabase Vector Store Setup
============================================================

✅ Supabase URL: https://xxxxx.supabase.co
✅ API Key: eyJhbGciOiJIUzI1NiIs...

🔧 Initializing Supabase Vector Store...
✅ Supabase Vector Store initialized

📁 Processing documents from: /Users/.../NOTES

📊 Found:
   - 35 PDF files
   - 12 DOCX files
   - 47 total files to process

🔄 Processing documents...
📄 Processing: ML Unit 1 part 1.pdf
✅ Created 45 chunks from ML Unit 1 part 1.pdf
📄 Processing: S1.pptx
✅ Created 23 chunks from S1.pptx
...
📊 Total: 605 chunks from 47 files
🧠 Generating embeddings for 605 chunks...
✅ Embeddings generated successfully
📤 Uploading 605 chunks to Supabase...
✅ Uploaded batch 1/7
✅ Uploaded batch 2/7
...
✅ Successfully uploaded 605 chunks to Supabase

============================================================
✅ SUCCESS!
============================================================

🎉 All documents have been processed and uploaded to Supabase!
```

### Step 7: Verify in Supabase

1. Go to **Table Editor** in Supabase
2. Select `document_chunks` table
3. You should see all your chunks with:
   - `content`: Text from documents
   - `source`: Filename
   - `page_number`: Page number (for PDFs)
   - `embedding`: Vector array [384 dimensions]

## Database Schema

### Table: `document_chunks`

| Column      | Type        | Description            |
| ----------- | ----------- | ---------------------- |
| id          | BIGSERIAL   | Primary key            |
| chunk_id    | TEXT        | Unique identifier      |
| content     | TEXT        | Document chunk text    |
| source      | TEXT        | Source filename        |
| page_number | INTEGER     | Page number (nullable) |
| embedding   | VECTOR(384) | Vector embedding       |
| created_at  | TIMESTAMP   | Creation time          |
| updated_at  | TIMESTAMP   | Update time            |

### Key Functions

- `match_documents(query_embedding, match_count, filter_source)`: Search similar documents
- `get_document_stats()`: Get statistics about stored documents

## Using the Vector Store

### Search for Similar Documents

```python
from supabase_vector_store import SupabaseVectorStore

# Initialize
vector_store = SupabaseVectorStore(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_KEY")
)

# Search
results = vector_store.search_similar("What is machine learning?", top_k=5)

for result in results:
    print(f"Source: {result['source']}")
    print(f"Content: {result['content']}")
    print(f"Similarity: {result['similarity']}")
    print("-" * 50)
```

## Troubleshooting

### Error: "Module not found: supabase"

```bash
pip install supabase sentence-transformers
```

### Error: "SUPABASE_URL not found"

```bash
# Set environment variables
export SUPABASE_URL='your-url'
export SUPABASE_KEY='your-key'
```

### Error: "relation 'document_chunks' does not exist"

- Make sure you ran the SQL schema in Supabase SQL Editor
- Check that the table was created in Table Editor

### Error: "extension 'vector' does not exist"

- pgvector should be enabled by default in Supabase
- If not, contact Supabase support

## Cost Estimation

**Supabase Free Tier includes:**

- 500 MB database storage
- 1 GB bandwidth
- 2 GB file storage

**Estimated usage:**

- ~605 chunks × ~500 bytes each = ~300 KB text
- ~605 vectors × 384 dimensions × 4 bytes = ~930 KB vectors
- **Total**: ~1.2 MB (well within free tier!)

## Next Steps

1. ✅ Vector database is ready
2. 🔄 Integrate with your noteCHAT app for semantic search
3. 🎯 Build RAG (Retrieval-Augmented Generation) pipeline
4. 📊 Add query logging and analytics
5. 🚀 Deploy your application

## Files Created

- `supabase_vector_store.py`: Main vector store class
- `setup_supabase_vectorstore.py`: Setup and processing script
- `supabase_schema.sql`: Database schema
- `SUPABASE_SETUP.md`: This guide

## Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [pgvector Guide](https://github.com/pgvector/pgvector)
- [Sentence Transformers](https://www.sbert.net/)

---

**Questions?** Check the Supabase dashboard or logs for any errors during upload.
