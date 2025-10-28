"""
Setup and Process Documents to Supabase Vector Store
Run this script to upload your NOTES documents to Supabase
"""

import os
import sys
from pathlib import Path
from supabase_vector_store import SupabaseVectorStore

def main():
    print("🚀 noteCHAT - Supabase Vector Store Setup")
    print("=" * 60)
    
    # Get Supabase credentials from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("\n❌ Error: Supabase credentials not found!")
        print("\n📋 Please set the following environment variables:")
        print("   export SUPABASE_URL='https://your-project.supabase.co'")
        print("   export SUPABASE_KEY='your-anon-key'")
        print("\n💡 Get these from your Supabase project:")
        print("   1. Go to https://supabase.com")
        print("   2. Select your project (or create a new one)")
        print("   3. Go to Settings > API")
        print("   4. Copy 'Project URL' and 'anon public' key")
        sys.exit(1)
    
    print(f"\n✅ Supabase URL: {supabase_url}")
    print(f"✅ API Key: {supabase_key[:20]}...")
    
    # Initialize vector store
    print("\n🔧 Initializing Supabase Vector Store...")
    vector_store = SupabaseVectorStore(supabase_url, supabase_key)
    
    # Step 1: Setup database schema
    print("\n" + "=" * 60)
    print("STEP 1: Database Schema Setup")
    print("=" * 60)
    
    sql_schema = vector_store.setup_database()
    
    print("\n⚠️  IMPORTANT: Run the SQL schema in Supabase SQL Editor")
    print("\nOptions:")
    print("  1. Copy the SQL above and run it in Supabase SQL Editor")
    print("  2. Press Enter when done to continue...")
    
    input("\nPress Enter after running the SQL in Supabase...")
    
    # Step 2: Process documents
    print("\n" + "=" * 60)
    print("STEP 2: Process Documents")
    print("=" * 60)
    
    notes_path = Path("../NOTES")
    if not notes_path.exists():
        print(f"❌ NOTES folder not found at: {notes_path.absolute()}")
        sys.exit(1)
    
    print(f"\n📁 Processing documents from: {notes_path.absolute()}")
    
    # Count files
    pdf_files = list(notes_path.glob("*.pdf"))
    docx_files = list(notes_path.glob("*.docx"))
    total_files = len(pdf_files) + len(docx_files)
    
    print(f"\n📊 Found:")
    print(f"   - {len(pdf_files)} PDF files")
    print(f"   - {len(docx_files)} DOCX files")
    print(f"   - {total_files} total files to process")
    
    if total_files == 0:
        print("\n❌ No PDF or DOCX files found in NOTES folder")
        sys.exit(1)
    
    # Check if documents already exist
    try:
        result = vector_store.supabase.table("document_chunks").select("id", count="exact").limit(1).execute()
        existing_count = result.count if hasattr(result, 'count') else 0
        
        if existing_count and existing_count > 0:
            print(f"\n⚠️  Database already contains {existing_count} chunks!")
            print("\nOptions:")
            print("  1. Clear existing data and reprocess everything")
            print("  2. Update/add new documents (skip duplicates)")
            print("  3. Cancel")
            
            choice = input("\nYour choice (1/2/3): ").strip()
            
            if choice == "1":
                print("\n🗑️  Clearing existing data...")
                vector_store.clear_all_chunks()
                print("✅ Database cleared")
            elif choice == "3":
                print("❌ Cancelled")
                sys.exit(0)
            # If choice == "2", continue with upsert
    except Exception as e:
        print(f"⚠️  Could not check existing data: {e}")
        print("Continuing anyway...")
    
    print("\n⚠️  This will:")
    print("   1. Extract text from all documents")
    print("   2. Split them into semantic chunks")
    print("   3. Generate vector embeddings")
    print("   4. Upload everything to Supabase")
    
    response = input("\n❓ Continue? (y/n): ").lower()
    if response != 'y':
        print("❌ Cancelled")
        sys.exit(0)
    
    # Process documents
    print("\n🔄 Processing documents...")
    success = vector_store.process_documents_folder(str(notes_path.absolute()))
    
    if success:
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print("\n🎉 All documents have been processed and uploaded to Supabase!")
        print("\n📊 Your vector database is ready for:")
        print("   - Semantic search")
        print("   - Question answering")
        print("   - RAG applications")
        
        print("\n🔗 Next Steps:")
        print("   1. Go to your Supabase Dashboard > Table Editor")
        print("   2. View the 'document_chunks' table")
        print("   3. See your document chunks and embeddings")
        print("   4. Use the search functionality in your app")
    else:
        print("\n❌ Failed to process documents")
        sys.exit(1)


if __name__ == "__main__":
    main()
