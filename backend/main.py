from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from robust_rag import RobustRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AI Notebook Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: Optional[float] = None

# Global variables
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global rag_pipeline
    
    try:
        logger.info("Starting server initialization...")
        
        # Initialize Robust RAG pipeline
        rag_pipeline = RobustRAGPipeline()
        success = rag_pipeline.initialize()
        
        if success:
            logger.info("Server initialized successfully!")
        else:
            logger.error("Failed to initialize RAG pipeline")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        rag_pipeline = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Notebook Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global rag_pipeline
    
    if rag_pipeline and rag_pipeline.chunks:
        return {
            "status": "healthy",
            "initialized": True,
            "message": "Robust RAG pipeline ready",
            "chunks_count": len(rag_pipeline.chunks)
        }
    elif rag_pipeline:
        return {
            "status": "initializing",
            "initialized": False,
            "message": "Robust RAG pipeline initializing"
        }
    else:
        return {
            "status": "error",
            "initialized": False,
            "message": "RAG pipeline failed to initialize"
        }

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the documents using the Premium RAG pipeline"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        logger.info(f"Processing query: {request.question}...")
        
        # Query using premium pipeline
        result = rag_pipeline.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List available documents"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized"
        )
    
    try:
        # Get unique sources from chunks
        sources = list(set(chunk.source for chunk in rag_pipeline.chunks))
        return {
            "documents": sources,
            "total_chunks": len(rag_pipeline.chunks)
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
