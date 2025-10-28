# 🚀 QUICK START: Supabase Vector Store Setup

## Current Status: ⏸️ WAITING FOR SQL SCHEMA

The script is currently waiting for you to run the SQL schema in your Supabase dashboard.

## ✅ STEP-BY-STEP INSTRUCTIONS

### Step 1: Open Supabase Dashboard

1. Go to: **https://supabase.com/dashboard**
2. Sign in with your account
3. Select your project: `cnyajpokraljdcrufetk`

### Step 2: Run SQL Schema

1. Click **"SQL Editor"** in the left sidebar
2. Click **"New query"** button
3. Copy the ENTIRE SQL code from `supabase_schema.sql` file
4. Paste it into the SQL Editor
5. Click **"Run"** (or press Cmd/Ctrl + Enter)

The SQL will:

- ✅ Enable pgvector extension
- ✅ Create `document_chunks` table
- ✅ Create search indexes
- ✅ Add similarity search function
- ✅ Add statistics function

### Step 3: Continue the Script

1. Go back to your terminal where the script is waiting
2. Press **Enter** to continue
3. The script will then:
   - 📄 Process all PDF and DOCX files from NOTES folder
   - ✂️ Split them into ~200-word chunks
   - 🧠 Generate vector embeddings (384 dimensions)
   - 📤 Upload to Supabase

## 📊 Expected Results

After completion, you should have:

- **Documents Processed**: ~47 files (PDFs and DOCX)
- **Total Chunks**: ~600-800 chunks
- **Vector Embeddings**: 384-dimensional vectors for each chunk
- **Database Size**: ~2-3 MB (well within free tier!)

## 🔍 Verify in Supabase

After the script completes:

1. Go to **Table Editor** in Supabase
2. Select **`document_chunks`** table
3. You should see columns:
   - `id`, `chunk_id`, `content`, `source`, `page_number`, `embedding`
4. Try searching by clicking on any row to see the data

## 📝 SQL Schema to Run

The complete SQL is in `supabase_schema.sql` file, or copy from the terminal output above.

## ⚡ Quick Copy (SQL Schema)

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create document_chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    page_number INTEGER,
    embedding vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS document_chunks_source_idx
ON document_chunks(source);

CREATE INDEX IF NOT EXISTS document_chunks_chunk_id_idx
ON document_chunks(chunk_id);

-- Create search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(384),
    match_count int DEFAULT 5,
    filter_source text DEFAULT NULL
)
RETURNS TABLE (
    id bigint,
    chunk_id text,
    content text,
    source text,
    page_number int,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        document_chunks.id,
        document_chunks.chunk_id,
        document_chunks.content,
        document_chunks.source,
        document_chunks.page_number,
        1 - (document_chunks.embedding <=> query_embedding) as similarity
    FROM document_chunks
    WHERE
        CASE
            WHEN filter_source IS NOT NULL THEN document_chunks.source = filter_source
            ELSE TRUE
        END
    ORDER BY document_chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

## 🎯 After Setup Complete

Once the vector database is populated, you can:

1. **Semantic Search**: Find similar documents using vector similarity
2. **Question Answering**: Build RAG applications
3. **Knowledge Base**: Query your ML notes intelligently
4. **API Integration**: Connect to your noteCHAT application

## 📞 Need Help?

- Check Supabase dashboard for any error messages
- Verify that pgvector extension is enabled
- Make sure you have the correct API keys set

---

**Current Session Info:**

- Supabase URL: https://cnyajpokraljdcrufetk.supabase.co
- Script: Waiting for SQL schema confirmation
- Location: `/Users/shashikumarezhil/Documents/SECTOR-17 /noteCHAT/backend`
