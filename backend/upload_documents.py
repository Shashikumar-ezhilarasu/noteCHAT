#!/usr/bin/env python3
"""
Firebase Document Uploader
Uploads all documents from the NOTES folder to Firebase Storage
"""

import os
import sys
import firebase_admin
from firebase_admin import credentials, storage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase_admin_config.json")
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'notechat-26c38.firebasestorage.app'
            })
        
        bucket = storage.bucket()
        logger.info("Firebase initialized successfully")
        return bucket
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return None

def upload_documents(bucket, notes_folder="../NOTES"):
    """Upload all documents from NOTES folder to Firebase Storage"""
    try:
        if not os.path.exists(notes_folder):
            logger.error(f"NOTES folder not found at: {notes_folder}")
            return False
        
        uploaded_count = 0
        supported_extensions = {'.pdf', '.docx', '.txt', '.md', '.doc'}
        
        # List all files in NOTES folder
        for filename in os.listdir(notes_folder):
            file_path = os.path.join(notes_folder, filename)
            
            # Skip directories and unsupported files
            if os.path.isdir(file_path):
                continue
            
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in supported_extensions:
                logger.warning(f"Skipping unsupported file: {filename}")
                continue
            
            # Upload to Firebase Storage
            try:
                blob = bucket.blob(f"notebooks/{filename}")
                
                # Check if file already exists
                if blob.exists():
                    logger.info(f"File already exists, updating: {filename}")
                
                # Upload the file
                blob.upload_from_filename(file_path)
                uploaded_count += 1
                logger.info(f"✅ Uploaded: {filename}")
                
            except Exception as e:
                logger.error(f"❌ Failed to upload {filename}: {e}")
                continue
        
        logger.info(f"🎉 Upload complete! {uploaded_count} files uploaded to Firebase Storage")
        return uploaded_count > 0
        
    except Exception as e:
        logger.error(f"Error during upload: {e}")
        return False

def list_uploaded_files(bucket):
    """List all files in the notebooks/ folder"""
    try:
        blobs = bucket.list_blobs(prefix="notebooks/")
        files = []
        
        print("\n📚 Files in Firebase Storage (notebooks/ folder):")
        print("-" * 50)
        
        for blob in blobs:
            if not blob.name.endswith('/'):  # Skip directories
                filename = os.path.basename(blob.name)
                size = blob.size
                files.append(filename)
                print(f"📄 {filename} ({size} bytes)")
        
        print(f"\nTotal files: {len(files)}")
        return files
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []

def main():
    print("🔥 Firebase Document Uploader")
    print("=" * 40)
    
    # Check if firebase config exists
    if not os.path.exists("firebase_admin_config.json"):
        print("❌ firebase_admin_config.json not found!")
        print("Please add your Firebase service account credentials to this file.")
        sys.exit(1)
    
    # Initialize Firebase
    bucket = initialize_firebase()
    if not bucket:
        print("❌ Failed to initialize Firebase")
        sys.exit(1)
    
    # Upload documents
    print("\n📤 Starting upload...")
    success = upload_documents(bucket)
    
    if success:
        print("\n📋 Current files in storage:")
        list_uploaded_files(bucket)
        print("\n✅ All done! Your documents are now in Firebase Storage.")
        print("You can now start the backend and frontend to use your AI assistant.")
    else:
        print("❌ Upload failed. Please check the logs above.")

if __name__ == "__main__":
    main()
