import os
import shutil
import tempfile
from typing import List, Dict
import firebase_admin
from firebase_admin import credentials, storage
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.bucket = None
        
    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase app already exists
            if not firebase_admin._apps:
                cred = credentials.Certificate("firebase_admin_config.json")
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'notechat-26c38.firebasestorage.app'
                })
            
            self.bucket = storage.bucket()
            logger.info("Firebase initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            return False
    
    def download_files_from_firebase(self):
        """Download all files from Firebase Storage notebooks/ folder"""
        try:
            # Clear tmp_downloads directory
            if os.path.exists("tmp_downloads"):
                shutil.rmtree("tmp_downloads")
            os.makedirs("tmp_downloads", exist_ok=True)
            
            # List all blobs in the notebooks/ folder
            blobs = self.bucket.list_blobs(prefix="notebooks/")
            downloaded_files = []
            
            for blob in blobs:
                if blob.name.endswith('/'):  # Skip directories
                    continue
                    
                # Get the filename
                filename = os.path.basename(blob.name)
                local_path = os.path.join("tmp_downloads", filename)
                
                # Download the file
                blob.download_to_filename(local_path)
                downloaded_files.append(local_path)
                logger.info(f"Downloaded: {filename}")
            
            return downloaded_files
        except Exception as e:
            logger.error(f"Error downloading files from Firebase: {e}")
            return []
    
    def initialize_embeddings(self):
        """Initialize sentence transformers embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Embeddings initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
    
    def load_and_process_documents(self, file_paths: List[str]):
        """Load documents and create text chunks"""
        try:
            documents = []
            
            for file_path in file_paths:
                try:
                    # Use UnstructuredFileLoader for various file types
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()
                    
                    # Add source metadata
                    for doc in docs:
                        doc.metadata['source'] = os.path.basename(file_path)
                    
                    documents.extend(docs)
                    logger.info(f"Loaded document: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} text chunks")
            
            return chunks
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return []
    
    def create_vectorstore(self, documents):
        """Create and persist Chroma vectorstore"""
        try:
            # Remove existing database
            if os.path.exists("db"):
                shutil.rmtree("db")
            
            # Create new vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="db"
            )
            
            # Persist the database
            self.vectorstore.persist()
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            
            logger.info("Vectorstore created and persisted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create vectorstore: {e}")
            return False
    
    def initialize_llm(self):
        """Initialize local Hugging Face LLM"""
        try:
            # Use a smaller model that can run locally
            model_name = "microsoft/DialoGPT-medium"  # Smaller model for local use
            
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Create LangChain LLM
            llm = HuggingFacePipeline(pipeline=pipe)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            
            logger.info("LLM initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return False
    
    def load_existing_vectorstore(self):
        """Load existing vectorstore if available"""
        try:
            if os.path.exists("db"):
                self.vectorstore = Chroma(
                    persist_directory="db",
                    embedding_function=self.embeddings
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                logger.info("Loaded existing vectorstore")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load existing vectorstore: {e}")
            return False
    
    def initialize_db(self):
        """Initialize the complete RAG pipeline"""
        try:
            # Initialize Firebase
            if not self.initialize_firebase():
                return False
            
            # Initialize embeddings
            if not self.initialize_embeddings():
                return False
            
            # Try to load existing vectorstore first
            if self.load_existing_vectorstore():
                logger.info("Using existing vectorstore")
            else:
                # Download files from Firebase
                file_paths = self.download_files_from_firebase()
                if not file_paths:
                    logger.warning("No files downloaded from Firebase")
                    return False
                
                # Process documents
                documents = self.load_and_process_documents(file_paths)
                if not documents:
                    logger.warning("No documents processed")
                    return False
                
                # Create vectorstore
                if not self.create_vectorstore(documents):
                    return False
            
            # Initialize LLM
            if not self.initialize_llm():
                return False
            
            logger.info("RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False
    
    def query_rag(self, question: str) -> Dict:
        """Query the RAG system"""
        try:
            if not self.qa_chain:
                return {
                    "answer": "System not initialized. Please try again later.",
                    "sources": []
                }
            
            # Query the system
            result = self.qa_chain({"query": question})
            
            # Extract sources
            sources = []
            if "source_documents" in result:
                sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
            
            return {
                "answer": result["result"],
                "sources": list(set(sources))  # Remove duplicates
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": []
            }

# Global instance
rag_pipeline = RAGPipeline()
