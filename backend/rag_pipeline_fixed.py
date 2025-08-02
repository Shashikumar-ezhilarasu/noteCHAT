import os
import shutil
import tempfile
from typing import List, Dict, Optional
import firebase_admin
from firebase_admin import credentials, storage
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_community.document_loaders import UnstructuredFileLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain imports failed: {e}. Using fallback implementation.")
    LANGCHAIN_AVAILABLE = False

class SimpleRAGPipeline:
    """Simplified RAG pipeline that works without complex dependencies"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.vectorstore = None
        self.bucket = None
        self.initialized = False
        
    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase app already exists
            if not firebase_admin._apps:
                if os.path.exists("firebase_admin_config.json"):
                    cred = credentials.Certificate("firebase_admin_config.json")
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': 'notechat-26c38.firebasestorage.app'
                    })
                    self.bucket = storage.bucket()
                    logger.info("Firebase initialized successfully")
                    return True
                else:
                    logger.error("firebase_admin_config.json not found")
                    return False
            
            self.bucket = storage.bucket()
            logger.info("Firebase already initialized")
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
                if not filename:  # Skip if no filename
                    continue
                    
                local_path = os.path.join("tmp_downloads", filename)
                
                # Download the file
                blob.download_to_filename(local_path)
                downloaded_files.append(local_path)
                logger.info(f"Downloaded: {filename}")
            
            return downloaded_files
        except Exception as e:
            logger.error(f"Error downloading files from Firebase: {e}")
            return []
    
    def load_local_documents(self):
        """Fallback: Load documents from local NOTES folder if Firebase fails"""
        try:
            notes_path = "../NOTES"
            if not os.path.exists(notes_path):
                logger.warning("NOTES folder not found")
                return []
            
            # Clear tmp_downloads directory
            if os.path.exists("tmp_downloads"):
                shutil.rmtree("tmp_downloads")
            os.makedirs("tmp_downloads", exist_ok=True)
            
            files = []
            for filename in os.listdir(notes_path):
                if filename.endswith(('.pdf', '.docx', '.txt', '.md')):
                    source_path = os.path.join(notes_path, filename)
                    dest_path = os.path.join("tmp_downloads", filename)
                    shutil.copy2(source_path, dest_path)
                    files.append(dest_path)
                    logger.info(f"Copied local file: {filename}")
            
            return files
        except Exception as e:
            logger.error(f"Error loading local documents: {e}")
            return []
    
    def simple_text_extraction(self, file_path: str) -> str:
        """Simple text extraction without complex dependencies"""
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.endswith('.pdf'):
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text()
                        return text
                except ImportError:
                    logger.warning("PyPDF2 not available, skipping PDF")
                    return ""
            elif file_path.endswith('.docx'):
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    logger.warning("python-docx not available, skipping DOCX")
                    return ""
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def process_documents_simple(self, file_paths: List[str]):
        """Simple document processing without LangChain"""
        self.documents = []
        
        for file_path in file_paths:
            try:
                text = self.simple_text_extraction(file_path)
                if text.strip():
                    # Simple chunking - split by paragraphs or sentences
                    chunks = self.simple_chunk_text(text, file_path)
                    self.documents.extend(chunks)
                    logger.info(f"Processed: {os.path.basename(file_path)} -> {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        logger.info(f"Total documents processed: {len(self.documents)}")
        return len(self.documents) > 0
    
    def simple_chunk_text(self, text: str, source: str, chunk_size: int = 1000) -> List[Dict]:
        """Simple text chunking"""
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "source": os.path.basename(source),
                    "metadata": {"source": source}
                })
                current_chunk = []
                current_size = 0
        
        # Add remaining text
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "source": os.path.basename(source),
                "metadata": {"source": source}
            })
        
        return chunks
    
    def simple_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based search"""
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            # Simple scoring based on word overlap
            score = len(query_words.intersection(doc_words))
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:top_k]]
    
    def simple_answer_generation(self, query: str, context_docs: List[Dict]) -> str:
        """Simple rule-based answer generation"""
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Combine context
        context = "\n\n".join([doc["text"][:500] for doc in context_docs])
        
        # Simple template-based response
        answer = f"Based on the available documents, here's what I found:\n\n"
        
        # Add context with some processing
        if "clustering" in query.lower():
            answer += "Regarding clustering:\n"
        elif "machine learning" in query.lower():
            answer += "About machine learning:\n"
        elif "hmm" in query.lower():
            answer += "About Hidden Markov Models:\n"
        else:
            answer += "Relevant information:\n"
        
        answer += context[:1000] + "..."
        
        return answer
    
    def initialize_db(self):
        """Initialize the RAG pipeline"""
        try:
            # Try Firebase first
            if self.initialize_firebase():
                file_paths = self.download_files_from_firebase()
            else:
                logger.warning("Firebase initialization failed, using local files")
                file_paths = self.load_local_documents()
            
            if not file_paths:
                logger.error("No files found to process")
                return False
            
            # Process documents
            if self.process_documents_simple(file_paths):
                self.initialized = True
                logger.info("Simple RAG pipeline initialized successfully")
                return True
            else:
                logger.error("Document processing failed")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False
    
    def query_rag(self, question: str) -> Dict:
        """Query the RAG system"""
        try:
            if not self.initialized:
                return {
                    "answer": "System not initialized. Please try again later.",
                    "sources": []
                }
            
            # Search for relevant documents
            relevant_docs = self.simple_search(question)
            
            # Generate answer
            answer = self.simple_answer_generation(question, relevant_docs)
            
            # Extract sources
            sources = list(set([doc["source"] for doc in relevant_docs]))
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": []
            }

# Use the appropriate implementation
if LANGCHAIN_AVAILABLE:
    # Original implementation would go here
    logger.info("LangChain available - using full implementation")
    from rag_pipeline import RAGPipeline as FullRAGPipeline
    rag_pipeline = FullRAGPipeline()
else:
    logger.info("Using simplified RAG implementation")
    rag_pipeline = SimpleRAGPipeline()
