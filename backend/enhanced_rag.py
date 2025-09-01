import os
import shutil
import json
import pickle
from typing import List, Dict, Optional
import firebase_admin
from firebase_admin import credentials, storage
import requests
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from docx import Document
import PyPDF2

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with proper semantic search and knowledge extraction"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.vectorizer = None
        self.document_vectors = None
        self.bucket = None
        self.initialized = False
        self.knowledge_base = {}
        
        # Try to import sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_transformers = True
            logger.info("Using Sentence Transformers for embeddings")
        except ImportError:
            self.sentence_transformer = None
            self.use_transformers = False
            logger.info("Using TF-IDF for document similarity")
        
    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
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
            if os.path.exists("tmp_downloads"):
                shutil.rmtree("tmp_downloads")
            os.makedirs("tmp_downloads", exist_ok=True)
            
            blobs = self.bucket.list_blobs(prefix="notebooks/")
            downloaded_files = []
            
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue
                    
                filename = os.path.basename(blob.name)
                if not filename:
                    continue
                    
                local_path = os.path.join("tmp_downloads", filename)
                blob.download_to_filename(local_path)
                downloaded_files.append(local_path)
                logger.info(f"Downloaded: {filename}")
            
            return downloaded_files
        except Exception as e:
            logger.error(f"Error downloading files from Firebase: {e}")
            return []
    
    def load_local_documents(self):
        """Load documents from local NOTES folder"""
        try:
            notes_path = "../NOTES"
            if not os.path.exists(notes_path):
                logger.warning("NOTES folder not found")
                return []
            
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
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file with improved spacing"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            # Improve spacing for better readability
                            cleaned_page_text = self.fix_pdf_spacing(page_text)
                            text += f"\n--- Page {page_num + 1} ---\n{cleaned_page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} from {file_path}: {e}")
                        continue
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for para_num, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text.strip()
                if para_text:
                    text += f"{para_text}\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file types"""
        try:
            if file_path.endswith('.txt') or file_path.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.endswith('.pdf'):
                return self.extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                return self.extract_text_from_docx(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def fix_pdf_spacing(self, text: str) -> str:
        """Fix spacing issues common in PDF text extraction"""
        # Add space before capital letters that follow lowercase letters (camelCase fix)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Add space between letters and numbers
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # Add space after periods if not followed by space
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Add space after commas if not followed by space
        text = re.sub(r',([a-zA-Z])', r', \1', text)
        
        # Add space after colons if not followed by space
        text = re.sub(r':([a-zA-Z])', r': \1', text)
        
        # Fix common OCR issues with spacing
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
        
        # Handle specific cases like "performanceusingdifferent" -> "performance using different"
        text = re.sub(r'([a-z])(using|with|and|or|the|of|in|for|to|from|by)', r'\1 \2', text)
        
        return text

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better spacing"""
        # First fix PDF spacing issues
        text = self.fix_pdf_spacing(text)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.,;:!?()-]', ' ', text)
        
        # Fix common OCR issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence spacing
        text = re.sub(r'\.(\w)', r'. \1', text)
        text = re.sub(r'\?(\w)', r'? \1', text)
        text = re.sub(r'!(\w)', r'! \1', text)
        
        return text.strip()
    
    def create_chunks(self, text: str, source: str, chunk_size: int = 800, overlap: int = 200) -> List[Dict]:
        """Create overlapping text chunks with metadata"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) > 50:  # Only keep substantial chunks
                    chunks.append({
                        'text': chunk_text,
                        'source': os.path.basename(source),
                        'source_path': source,
                        'chunk_id': len(chunks),
                        'word_count': len(chunk_text.split())
                    })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    'text': chunk_text,
                    'source': os.path.basename(source),
                    'source_path': source,
                    'chunk_id': len(chunks),
                    'word_count': len(chunk_text.split())
                })
        
        return chunks
    
    def process_documents(self, file_paths: List[str]):
        """Process all documents and create knowledge base"""
        self.documents = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing: {os.path.basename(file_path)}")
                
                # Extract text
                text = self.extract_text_from_file(file_path)
                if not text.strip():
                    logger.warning(f"No text extracted from {file_path}")
                    continue
                
                # Clean text
                cleaned_text = self.clean_text(text)
                
                # Create chunks
                chunks = self.create_chunks(cleaned_text, file_path)
                self.documents.extend(chunks)
                
                logger.info(f"✅ Processed {os.path.basename(file_path)}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path}: {e}")
                continue
        
        logger.info(f"📚 Total processed: {len(self.documents)} chunks from {len(file_paths)} documents")
        return len(self.documents) > 0
    
    def create_embeddings(self):
        """Create embeddings for all document chunks"""
        if not self.documents:
            return False
        
        texts = [doc['text'] for doc in self.documents]
        
        if self.use_transformers:
            # Use sentence transformers for semantic embeddings
            logger.info("Creating semantic embeddings...")
            self.document_vectors = self.sentence_transformer.encode(texts)
            logger.info("✅ Semantic embeddings created")
        else:
            # Fallback to TF-IDF
            logger.info("Creating TF-IDF vectors...")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            self.document_vectors = self.vectorizer.fit_transform(texts)
            logger.info("✅ TF-IDF vectors created")
        
        return True
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search to find relevant documents"""
        if self.document_vectors is None:
            return []
        
        if self.use_transformers:
            # Encode query using sentence transformer
            query_vector = self.sentence_transformer.encode([query])
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        else:
            # Use TF-IDF
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                result = self.documents[idx].copy()
                result['similarity'] = float(similarities[idx])
                results.append(result)
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer based on query and relevant context"""
        if not context_docs:
            return "I couldn't find relevant information in the documents to answer your question. Please try rephrasing your question or ask about machine learning topics covered in the uploaded documents."
        
        # Extract key information based on query type
        query_lower = query.lower()
        
        # Build context from relevant documents
        context_parts = []
        sources = set()
        
        for doc in context_docs[:3]:  # Use top 3 most relevant
            context_parts.append(doc['text'])
            sources.add(doc['source'])
        
        combined_context = '\n\n'.join(context_parts)
        
        # Generate answer based on context
        if any(term in query_lower for term in ['k-means', 'k means', 'kmeans']):
            answer = self.extract_kmeans_info(combined_context)
        elif any(term in query_lower for term in ['hierarchical', 'hierarchy', 'agglomerative']):
            answer = self.extract_hierarchical_info(combined_context)
        elif any(term in query_lower for term in ['hmm', 'hidden markov', 'markov model']):
            answer = self.extract_hmm_info(combined_context)
        elif any(term in query_lower for term in ['clustering', 'cluster']):
            answer = self.extract_clustering_info(combined_context)
        elif any(term in query_lower for term in ['supervised', 'unsupervised', 'machine learning']):
            answer = self.extract_ml_info(combined_context)
        else:
            # General answer extraction
            answer = self.extract_general_info(query, combined_context)
        
        return answer
    
    def extract_kmeans_info(self, context: str) -> str:
        """Extract K-means specific information"""
        lines = context.split('\n')
        relevant_lines = []
        
        for line in lines:
            if any(term in line.lower() for term in ['k-means', 'kmeans', 'centroid', 'cluster center']):
                relevant_lines.append(line.strip())
        
        if relevant_lines:
            return f"**K-means Clustering:**\n\n" + '\n\n'.join(relevant_lines[:5])
        else:
            return f"**K-means Clustering Information:**\n\n{context[:1000]}..."
    
    def extract_hierarchical_info(self, context: str) -> str:
        """Extract hierarchical clustering information"""
        lines = context.split('\n')
        relevant_lines = []
        
        for line in lines:
            if any(term in line.lower() for term in ['hierarchical', 'dendrogram', 'agglomerative', 'divisive']):
                relevant_lines.append(line.strip())
        
        if relevant_lines:
            return f"**Hierarchical Clustering:**\n\n" + '\n\n'.join(relevant_lines[:5])
        else:
            return f"**Hierarchical Clustering Information:**\n\n{context[:1000]}..."
    
    def extract_hmm_info(self, context: str) -> str:
        """Extract HMM information"""
        lines = context.split('\n')
        relevant_lines = []
        
        for line in lines:
            if any(term in line.lower() for term in ['hmm', 'hidden markov', 'markov model', 'state', 'transition']):
                relevant_lines.append(line.strip())
        
        if relevant_lines:
            return f"**Hidden Markov Models (HMM):**\n\n" + '\n\n'.join(relevant_lines[:5])
        else:
            return f"**Hidden Markov Models Information:**\n\n{context[:1000]}..."
    
    def extract_clustering_info(self, context: str) -> str:
        """Extract general clustering information"""
        lines = context.split('\n')
        relevant_lines = []
        
        for line in lines:
            if any(term in line.lower() for term in ['cluster', 'clustering', 'centroid', 'distance']):
                relevant_lines.append(line.strip())
        
        if relevant_lines:
            return f"**Clustering:**\n\n" + '\n\n'.join(relevant_lines[:5])
        else:
            return f"**Clustering Information:**\n\n{context[:1000]}..."
    
    def extract_ml_info(self, context: str) -> str:
        """Extract machine learning information"""
        lines = context.split('\n')
        relevant_lines = []
        
        for line in lines:
            if any(term in line.lower() for term in ['machine learning', 'supervised', 'unsupervised', 'algorithm']):
                relevant_lines.append(line.strip())
        
        if relevant_lines:
            return f"**Machine Learning:**\n\n" + '\n\n'.join(relevant_lines[:5])
        else:
            return f"**Machine Learning Information:**\n\n{context[:1000]}..."
    
    def extract_general_info(self, query: str, context: str) -> str:
        """Extract general information based on query"""
        # Find sentences that contain query terms
        query_terms = query.lower().split()
        sentences = sent_tokenize(context)
        relevant_sentences = []
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in query_terms):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return f"**Relevant Information:**\n\n" + '\n\n'.join(relevant_sentences[:3])
        else:
            return f"**Information Found:**\n\n{context[:1000]}..."
    
    def initialize_db(self):
        """Initialize the complete RAG pipeline"""
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
            if not self.process_documents(file_paths):
                logger.error("Document processing failed")
                return False
            
            # Create embeddings
            if not self.create_embeddings():
                logger.error("Embedding creation failed")
                return False
            
            self.initialized = True
            logger.info("🎉 Enhanced RAG pipeline initialized successfully!")
            return True
            
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
            relevant_docs = self.semantic_search(question, top_k=5)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question. Please try asking about machine learning topics like clustering, HMM, or supervised/unsupervised learning.",
                    "sources": []
                }
            
            # Generate answer
            answer = self.generate_answer(question, relevant_docs)
            
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

# Global instance
rag_pipeline = EnhancedRAGPipeline()
