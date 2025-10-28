# Setup Supabase Search Function

## ✅ Current Status

- ✅ 442 document chunks uploaded successfully
- ✅ 30 unique documents stored
- ✅ Vector embeddings (384 dimensions) stored
- ⚠️ Search function needs to be created

## 🔧 Quick Fix - Create Search Function

### Option 1: Using Supabase Dashboard (Recommended)

1. Go to https://supabase.com/dashboard
2. Select your project
3. Click on "SQL Editor" in the left sidebar
4. Click "New Query"
5. Copy and paste this SQL:

```sql
-- Function to search for similar documents using cosine similarity
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

6. Click "Run" button
7. You should see "Success. No rows returned" message

### Option 2: Run Full Schema (If you haven't already)

If you haven't set up the complete schema yet, run the full SQL from `supabase_schema.sql`:

1. Open `backend/supabase_schema.sql` file
2. Copy all the contents
3. Paste in Supabase SQL Editor
4. Click "Run"

## 🧪 Test Your Setup

After creating the function, run this command to test:

```bash
cd backend
source venv/bin/activate
export SUPABASE_URL='https://cnyajpokraljdcrufetk.supabase.co'
export SUPABASE_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNueWFqcG9rcmFsamRjcnVmZXRrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE2MzE3NzcsImV4cCI6MjA3NzIwNzc3N30.QidVCb2zw6GdB1fbedIV6tdGEJYvJnOxJcUzdU5QN6U'
python3 check_database.py
```

You should now see "✅ Search function works!" at the end.

## 📊 What You Have Now

### Database Stats

- **Total Chunks**: 442
- **Unique Documents**: 30
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Chunking Strategy**: ~200 words per chunk with 1-sentence overlap

### Sample Documents

- HMM.docx
- ML Unit 1 part 1.pdf
- ML Unit 1 part 2.pdf
- UNIT 1 Machine Learning.pdf
- S1-S8 Notes (various PDFs)
- Unit-2.OLD pdf.pdf
- Probability Theory notes
- K-means clustering documents
- And 20 more...

## 🎯 Next Steps

1. **Create the search function** (see above)
2. **Test semantic search** by running `check_database.py`
3. **Integrate with noteCHAT** to use the Supabase vector store instead of local cache

## 🔍 Useful Links

- **Supabase Dashboard**: https://supabase.com/dashboard
- **Your Project URL**: https://cnyajpokraljdcrufetk.supabase.co
- **Table Editor**: https://supabase.com/dashboard/project/_/editor
