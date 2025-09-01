from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import os

# Import both RAG implementations
try:
    from enhanced_embedding_rag import EnhancedEmbeddingRAGPipeline
    ENHANCED_AVAILABLE = True
    print("🤗 Enhanced Hugging Face models available!")
except ImportError as e:
    print(f"⚠️ Enhanced models not available: {e}")
    ENHANCED_AVAILABLE = False

from robust_rag import RobustRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Notebook Assistant", 
    version="2.0.0",
    description="Enhanced RAG system with Hugging Face transformers"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    use_enhanced: Optional[bool] = True  # Use enhanced models by default

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: Optional[float] = None
    method: Optional[str] = None
    qa_enhanced: Optional[bool] = False
    model_type: str

class ModelInfo(BaseModel):
    current_model: str
    enhanced_available: bool
    hugging_face_models: List[str]

# Global variables
rag_pipeline = None
enhanced_rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize both RAG pipelines"""
    global rag_pipeline, enhanced_rag_pipeline
    
    try:
        logger.info("🚀 Starting server initialization...")
        
        # Initialize standard RAG pipeline
        logger.info("📚 Initializing standard RAG pipeline...")
        rag_pipeline = RobustRAGPipeline()
        success = rag_pipeline.initialize()
        
        if success:
            logger.info("✅ Standard RAG pipeline initialized!")
        else:
            raise Exception("Standard RAG initialization failed")
        
        # Initialize enhanced RAG pipeline if available
        if ENHANCED_AVAILABLE:
            try:
                logger.info("🤗 Initializing enhanced Hugging Face RAG pipeline...")
                enhanced_rag_pipeline = EnhancedEmbeddingRAGPipeline(use_gpu=False)
                enhanced_success = enhanced_rag_pipeline.initialize()
                
                if enhanced_success:
                    logger.info("✅ Enhanced RAG pipeline initialized!")
                else:
                    logger.warning("⚠️ Enhanced RAG initialization failed, using standard only")
                    enhanced_rag_pipeline = None
                    
            except Exception as e:
                logger.error(f"❌ Enhanced RAG initialization error: {e}")
                enhanced_rag_pipeline = None
        
        logger.info("🎉 Server initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.get("/health")
async def health_check():
    """Enhanced health check with model information"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "healthy",
        "initialized": True,
        "message": "Enhanced RAG pipeline ready",
        "chunks_count": len(rag_pipeline.chunks) if rag_pipeline else 0,
        "enhanced_available": enhanced_rag_pipeline is not None,
        "models": {
            "standard": "TF-IDF + Cosine Similarity",
            "enhanced": "Hugging Face Transformers" if enhanced_rag_pipeline else "Not available"
        }
    }

@app.get("/models", response_model=ModelInfo)
async def get_model_info():
    """Get information about available models"""
    hugging_face_models = []
    current_model = "TF-IDF + Cosine Similarity"
    
    if enhanced_rag_pipeline:
        current_model = "Hybrid (TF-IDF + Transformers)"
        hugging_face_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "distilbert-base-cased-distilled-squad", 
            "facebook/bart-large-cnn"
        ]
    
    return ModelInfo(
        current_model=current_model,
        enhanced_available=ENHANCED_AVAILABLE and enhanced_rag_pipeline is not None,
        hugging_face_models=hugging_face_models
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process query with model selection"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"📝 Processing query: {request.question}")
        logger.info(f"🔧 Use enhanced: {request.use_enhanced}")
        
        # Choose pipeline based on request and availability
        use_enhanced = (
            request.use_enhanced and 
            enhanced_rag_pipeline is not None and 
            ENHANCED_AVAILABLE
        )
        
        if use_enhanced:
            logger.info("🤗 Using enhanced Hugging Face pipeline")
            result = enhanced_rag_pipeline.query(request.question)
            model_type = "enhanced_transformers"
        else:
            logger.info("📚 Using standard TF-IDF pipeline")
            result = rag_pipeline.query(request.question)
            model_type = "standard_tfidf"
            # Add missing fields for consistency
            result.setdefault("method", "standard")
            result.setdefault("qa_enhanced", False)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            method=result.get("method", "standard"),
            qa_enhanced=result.get("qa_enhanced", False),
            model_type=model_type
        )
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/query/standard")
async def query_standard(request: QueryRequest):
    """Force use of standard TF-IDF model"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"📚 Processing query with standard model: {request.question}")
        result = rag_pipeline.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            method="standard",
            qa_enhanced=False,
            model_type="standard_tfidf"
        )
        
    except Exception as e:
        logger.error(f"❌ Standard query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Standard query processing failed: {str(e)}")

@app.post("/query/enhanced")
async def query_enhanced(request: QueryRequest):
    """Force use of enhanced Hugging Face models"""
    if not ENHANCED_AVAILABLE or enhanced_rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Enhanced models not available")
    
    try:
        logger.info(f"🤗 Processing query with enhanced models: {request.question}")
        result = enhanced_rag_pipeline.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            method=result.get("method", "enhanced"),
            qa_enhanced=result.get("qa_enhanced", False),
            model_type="enhanced_transformers"
        )
        
    except Exception as e:
        logger.error(f"❌ Enhanced query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced query processing failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List processed documents"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        sources = set()
        chunk_counts = {}
        
        for chunk in rag_pipeline.chunks:
            sources.add(chunk.source)
            chunk_counts[chunk.source] = chunk_counts.get(chunk.source, 0) + 1
        
        return {
            "total_documents": len(sources),
            "total_chunks": len(rag_pipeline.chunks),
            "documents": [
                {
                    "name": source,
                    "chunks": count
                } for source, count in chunk_counts.items()
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document listing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
