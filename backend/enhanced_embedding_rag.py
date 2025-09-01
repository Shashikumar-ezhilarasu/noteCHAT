"""
Enhanced RAG Pipeline with Hugging Face Models
Integrates modern transformer-based embeddings for superior semantic understanding
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Document processing
import PyPDF2
import docx
from docx import Document

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Hugging Face Models
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class DocumentChunk:
    """Enhanced document chunk with embeddings"""
    content: str
    source: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_id: str = ""
    confidence_score: float = 0.0
    embedding: Optional[np.ndarray] = None
    tfidf_score: float = 0.0

class EnhancedEmbeddingRAGPipeline:
    """RAG pipeline with Hugging Face transformer embeddings"""
    
    def __init__(self, use_gpu: bool = True):
        self.logger = self._setup_logging()
        self.chunks: List[DocumentChunk] = []
        
        # Device setup
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.logger.info(f"🔧 Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Traditional components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("enhanced_embedding_rag")
        return logger
    
    def _initialize_models(self):
        """Initialize Hugging Face models"""
        try:
            self.logger.info("🤗 Loading Hugging Face models...")
            
            # 1. Sentence Transformer for embeddings
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2',  # Fast and efficient
                device=self.device
            )
            self.logger.info("✅ Loaded SentenceTransformer: all-MiniLM-L6-v2")
            
            # 2. Question Answering Pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if self.device == "cuda" else -1
            )
            self.logger.info("✅ Loaded QA Pipeline: distilbert-base-cased-distilled-squad")
            
            # 3. Text Summarization Pipeline
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            self.logger.info("✅ Loaded Summarizer: facebook/bart-large-cnn")
            
            # 4. Alternative: More powerful embedding model (optional)
            # self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Hugging Face models: {e}")
            # Fallback to basic models
            self.sentence_model = None
            self.qa_pipeline = None
            self.summarizer = None
    
    def download_documents_from_local(self) -> List[str]:
        """Download documents from local directory"""
        local_files = []
        try:
            self.logger.info("🔄 Using local files")
            notes_dir = Path("../NOTES")
            if notes_dir.exists():
                os.makedirs("downloads", exist_ok=True)
                for file_path in notes_dir.glob("*"):
                    if file_path.suffix.lower() in ['.pdf', '.docx']:
                        dest_path = f"downloads/{file_path.name}"
                        shutil.copy2(file_path, dest_path)
                        local_files.append(dest_path)
                        self.logger.info(f"📁 Copied local: {file_path.name}")
            else:
                self.logger.warning("❌ NOTES directory not found")
                
        except Exception as e:
            self.logger.error(f"❌ Document download failed: {e}")
            
        return local_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Enhanced PDF text extraction"""
        text_data = {"pages": [], "full_text": "", "metadata": {}}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_data["metadata"]["total_pages"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            cleaned_text = self._clean_text(page_text)
                            if len(cleaned_text) > 50:
                                text_data["pages"].append({
                                    "page_number": page_num,
                                    "content": cleaned_text
                                })
                                text_data["full_text"] += f"\n{cleaned_text}"
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to extract page {page_num}: {e}")
                        
        except Exception as e:
            self.logger.error(f"❌ PDF extraction failed for {pdf_path}: {e}")
            
        return text_data
    
    def extract_text_from_docx(self, docx_path: str) -> Dict[str, Any]:
        """Enhanced DOCX text extraction"""
        text_data = {"pages": [], "full_text": "", "metadata": {}}
        
        try:
            doc = Document(docx_path)
            full_text = ""
            
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text += para.text + "\n"
            
            if full_text.strip():
                cleaned_text = self._clean_text(full_text)
                text_data["pages"].append({
                    "page_number": 1,
                    "content": cleaned_text
                })
                text_data["full_text"] = cleaned_text
                text_data["metadata"]["total_pages"] = 1
                
        except Exception as e:
            self.logger.error(f"❌ DOCX extraction failed for {docx_path}: {e}")
            
        return text_data
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\(\)]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def create_semantic_chunks_with_embeddings(self, text_data: Dict[str, Any], source: str) -> List[DocumentChunk]:
        """Create semantic chunks with both TF-IDF and embeddings"""
        chunks = []
        
        try:
            for page_data in text_data["pages"]:
                page_content = page_data["content"]
                page_number = page_data["page_number"]
                
                # Tokenize into sentences
                sentences = sent_tokenize(page_content)
                
                # Create overlapping chunks
                chunk_size = 3  # sentences per chunk
                overlap = 1     # sentence overlap
                
                for i in range(0, len(sentences), chunk_size - overlap):
                    chunk_sentences = sentences[i:i + chunk_size]
                    chunk_content = " ".join(chunk_sentences)
                    
                    if len(chunk_content.strip()) > 20:  # Minimum content length
                        # Generate embedding if model is available
                        embedding = None
                        if self.sentence_model:
                            embedding = self.sentence_model.encode(
                                chunk_content,
                                convert_to_tensor=False
                            )
                        
                        chunk = DocumentChunk(
                            content=chunk_content,
                            source=source,
                            page_number=page_number,
                            chunk_id=f"{source}_{len(chunks)}",
                            embedding=embedding
                        )
                        chunks.append(chunk)
                        
        except Exception as e:
            self.logger.error(f"❌ Chunking failed for {source}: {e}")
            
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> bool:
        """Process documents with enhanced extraction"""
        total_chunks = 0
        processed_docs = 0
        
        self.logger.info(f"🔄 Processing {len(file_paths)} documents...")
        
        for file_path in file_paths:
            try:
                source = os.path.basename(file_path)
                
                # Extract text based on file type
                if file_path.endswith('.pdf'):
                    text_data = self.extract_text_from_pdf(file_path)
                elif file_path.endswith('.docx'):
                    text_data = self.extract_text_from_docx(file_path)
                else:
                    continue
                
                if text_data["full_text"].strip():
                    # Create chunks with embeddings
                    doc_chunks = self.create_semantic_chunks_with_embeddings(text_data, source)
                    
                    if doc_chunks:
                        self.chunks.extend(doc_chunks)
                        total_chunks += len(doc_chunks)
                        processed_docs += 1
                        self.logger.info(f"✅ Processed {source}: {len(doc_chunks)} chunks")
                    else:
                        self.logger.warning(f"⚠️ No chunks created from {file_path}")
                else:
                    self.logger.warning(f"⚠️ No text extracted from {file_path}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to process {file_path}: {e}")
        
        self.logger.info(f"📚 Total: {total_chunks} chunks from {processed_docs} documents")
        return len(self.chunks) > 0
    
    def create_hybrid_search_index(self) -> bool:
        """Create hybrid search index (TF-IDF + Embeddings)"""
        try:
            if not self.chunks:
                self.logger.error("❌ No chunks to index")
                return False
            
            # 1. Create TF-IDF index
            texts = [chunk.content for chunk in self.chunks]
            
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=True
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.logger.info(f"✅ TF-IDF index created: {self.tfidf_matrix.shape}")
            
            # 2. Stack embeddings if available
            if self.sentence_model:
                embeddings = [chunk.embedding for chunk in self.chunks if chunk.embedding is not None]
                if embeddings:
                    self.embedding_matrix = np.vstack(embeddings)
                    self.logger.info(f"✅ Embedding matrix created: {self.embedding_matrix.shape}")
                else:
                    self.logger.warning("⚠️ No embeddings found")
                    self.embedding_matrix = None
            else:
                self.embedding_matrix = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create search index: {e}")
            return False
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Tuple[DocumentChunk, float]]:
        """Hybrid search combining TF-IDF and embeddings"""
        try:
            if not self.vectorizer or self.tfidf_matrix is None:
                self.logger.error("❌ Search index not available")
                return []
            
            self.logger.info(f"🔍 Hybrid searching for: {query}")
            
            # 1. TF-IDF similarity
            query_tfidf = self.vectorizer.transform([query])
            tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
            
            # 2. Embedding similarity (if available)
            embedding_similarities = np.zeros(len(self.chunks))
            if self.sentence_model and self.embedding_matrix is not None:
                query_embedding = self.sentence_model.encode([query])
                embedding_similarities = cosine_similarity(query_embedding, self.embedding_matrix).flatten()
            
            # 3. Combine scores (weighted)
            if self.embedding_matrix is not None:
                combined_similarities = (
                    alpha * embedding_similarities + 
                    (1 - alpha) * tfidf_similarities
                )
                self.logger.info(f"📊 Using hybrid scores (α={alpha})")
            else:
                combined_similarities = tfidf_similarities
                self.logger.info("📊 Using TF-IDF scores only")
            
            # Get top matches
            top_indices = combined_similarities.argsort()[-top_k * 2:][::-1]
            
            results = []
            for idx in top_indices:
                if combined_similarities[idx] > 0.01:
                    chunk = self.chunks[idx]
                    chunk.confidence_score = combined_similarities[idx]
                    chunk.tfidf_score = tfidf_similarities[idx]
                    results.append((chunk, combined_similarities[idx]))
                    self.logger.info(f"📋 Match {len(results)}: {combined_similarities[idx]:.3f} (TF-IDF: {tfidf_similarities[idx]:.3f}) - {chunk.content[:50]}...")
            
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Hybrid search failed: {e}")
            return []
    
    def generate_answer_with_qa_model(self, query: str, relevant_chunks: List[Tuple[DocumentChunk, float]]) -> Dict[str, Any]:
        """Generate answer using QA model + traditional approach"""
        if not relevant_chunks:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "confidence": 0.0,
                "method": "fallback"
            }
        
        # Traditional approach for source attribution
        answer_parts = []
        references = []
        sources = set()
        
        # Try QA model on best chunk
        qa_answer = None
        if self.qa_pipeline and relevant_chunks:
            try:
                best_chunk = relevant_chunks[0][0]
                qa_result = self.qa_pipeline(
                    question=query,
                    context=best_chunk.content
                )
                if qa_result['score'] > 0.1:  # Confidence threshold
                    qa_answer = qa_result['answer']
                    self.logger.info(f"🤖 QA Model answer (score: {qa_result['score']:.3f}): {qa_answer}")
            except Exception as e:
                self.logger.warning(f"⚠️ QA model failed: {e}")
        
        # Build comprehensive answer
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            reference_info = f"**Source {i}** (from {chunk.source}"
            if chunk.page_number:
                reference_info += f", Page {chunk.page_number}"
            reference_info += f", Confidence: {score:.2f})"
            
            # Use QA answer for first source if available
            if i == 1 and qa_answer:
                content = f"🤖 **AI Enhanced Answer**: {qa_answer}\n\n**Full Context**: {chunk.content}"
            else:
                content = chunk.content
            
            answer_parts.append(f"{reference_info}:\n{content}")
            
            references.append({
                "source": chunk.source,
                "page": chunk.page_number,
                "confidence": score,
                "tfidf_score": chunk.tfidf_score,
                "rank": i,
                "qa_enhanced": i == 1 and qa_answer is not None
            })
            sources.add(chunk.source)
        
        # Calculate overall confidence
        avg_confidence = sum(score for _, score in relevant_chunks) / len(relevant_chunks)
        
        answer = "\n\n".join(answer_parts)
        
        return {
            "answer": answer,
            "sources": list(sources),
            "confidence": avg_confidence,
            "references": references,
            "method": "hybrid_with_qa" if qa_answer else "hybrid",
            "qa_enhanced": qa_answer is not None
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query with enhanced models"""
        try:
            self.logger.info(f"🔍 Processing enhanced query: {question}")
            
            # Hybrid search
            relevant_chunks = self.hybrid_search(question, top_k=3)
            
            if not relevant_chunks:
                self.logger.warning("❌ No relevant chunks found")
                return {
                    "answer": "I couldn't find relevant information to answer this question.",
                    "sources": [],
                    "confidence": 0.0,
                    "method": "no_results"
                }
            
            # Generate enhanced answer
            result = self.generate_answer_with_qa_model(question, relevant_chunks)
            
            self.logger.info(f"✅ Enhanced answer generated with {len(result['sources'])} sources, confidence: {result['confidence']:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced query processing failed: {e}")
            return {
                "answer": "An error occurred while processing your question.",
                "sources": [],
                "confidence": 0.0,
                "method": "error"
            }
    
    def initialize(self) -> bool:
        """Initialize the enhanced RAG pipeline"""
        try:
            self.logger.info("🚀 Initializing Enhanced Embedding RAG Pipeline...")
            
            # Download documents
            file_paths = self.download_documents_from_local()
            if not file_paths:
                self.logger.error("❌ No documents found")
                return False
            
            # Process documents
            if not self.process_documents(file_paths):
                self.logger.error("❌ Document processing failed")
                return False
            
            # Create hybrid search index
            if not self.create_hybrid_search_index():
                self.logger.error("❌ Search index creation failed")
                return False
            
            self.logger.info("🎉 Enhanced Embedding RAG Pipeline initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced initialization failed: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize enhanced pipeline
    enhanced_rag = EnhancedEmbeddingRAGPipeline(use_gpu=False)  # Set True if you have GPU
    
    # Test initialization
    if enhanced_rag.initialize():
        print("🎉 Enhanced RAG initialized successfully!")
        
        # Test queries
        test_queries = [
            "What is supervised learning?",
            "Explain K-means clustering",
            "What are Hidden Markov Models?",
            "Difference between classification and regression"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            result = enhanced_rag.query(query)
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print(f"🔧 Method: {result['method']}")
            print(f"📚 Sources: {result['sources']}")
            print(f"🤖 QA Enhanced: {result.get('qa_enhanced', False)}")
            print(f"💬 Answer: {result['answer'][:200]}...")
    else:
        print("❌ Failed to initialize enhanced RAG")
