"""
Quick script to check what's in the Supabase database
"""
import os
from supabase import create_client

def main():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Please set SUPABASE_URL and SUPABASE_KEY environment variables")
        return
    
    print("🔍 Checking Supabase Database...")
    print("=" * 60)
    
    supabase = create_client(supabase_url, supabase_key)
    
    # Get total count
    response = supabase.table("document_chunks").select("id", count="exact").limit(1).execute()
    total_count = response.count
    print(f"📊 Total chunks in database: {total_count}")
    
    # Get sample data
    response = supabase.table("document_chunks").select("chunk_id, source, page_number, content").limit(5).execute()
    
    print("\n📝 Sample chunks:")
    print("-" * 60)
    for i, chunk in enumerate(response.data, 1):
        print(f"\n{i}. Chunk ID: {chunk['chunk_id']}")
        print(f"   Source: {chunk['source']}")
        print(f"   Page: {chunk.get('page_number', 'N/A')}")
        print(f"   Content (first 100 chars): {chunk['content'][:100]}...")
    
    # Get sources
    response = supabase.table("document_chunks").select("source").execute()
    sources = set(chunk['source'] for chunk in response.data)
    print(f"\n📚 Unique sources: {len(sources)}")
    for source in sorted(sources)[:10]:
        print(f"   - {source}")
    if len(sources) > 10:
        print(f"   ... and {len(sources) - 10} more")
    
    print("\n" + "=" * 60)
    print("✅ Database check complete!")
    
    # Test search function
    print("\n🔎 Testing semantic search...")
    try:
        response = supabase.rpc('match_documents', {
            'query_embedding': [0.1] * 384,  # Dummy embedding
            'match_threshold': 0.0,
            'match_count': 3
        }).execute()
        print(f"✅ Search function works! Found {len(response.data)} results")
    except Exception as e:
        print(f"⚠️ Search function test failed: {e}")

if __name__ == "__main__":
    main()
