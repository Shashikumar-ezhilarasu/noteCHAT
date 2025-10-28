#!/usr/bin/env python3
"""
Quick upload script - processes and uploads documents to Supabase
"""

import os
from supabase_vector_store import SupabaseVectorStore

def main():
    print("🚀 noteCHAT - Uploading Documents to Supabase")
    print("=" * 60)
    
    # Get credentials from environment
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set")
        print("Run:")
        print("  export SUPABASE_URL='https://cnyajpokraljdcrufetk.supabase.co'")
        print("  export SUPABASE_KEY='your-key'")
        return
    
    # Initialize
    print("\n🔧 Initializing Supabase Vector Store...")
    vector_store = SupabaseVectorStore(supabase_url, supabase_key)
    
    # Process and upload
    print("\n🔄 Processing documents from ../NOTES ...")
    success = vector_store.process_documents_folder('../NOTES')
    
    if success:
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print("\n🎉 All documents have been processed and uploaded!")
        print("\n📊 Check your Supabase dashboard:")
        print("   https://supabase.com/dashboard")
        print("   Table: document_chunks")
    else:
        print("\n❌ Failed to process documents")

if __name__ == "__main__":
    main()
