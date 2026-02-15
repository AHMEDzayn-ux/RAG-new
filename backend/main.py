"""
FastAPI Application for Multi-Tenant RAG Chatbot System
Phase 7: REST API Implementation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from logger import get_logger
from api.clients import router as clients_router
from api.documents import router as documents_router
from api.query import router as query_router
from config import settings

logger = get_logger(__name__)

# Log configuration at startup
logger.info(f"Starting RAG API...")
logger.info(f"Groq API Key configured: {bool(settings.groq_api_key)}")
logger.info(f"LLM Model: {settings.llm_model}")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Multi-tenant RAG system for customizable AI chatbots",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(clients_router)
app.include_router(documents_router)
app.include_router(query_router)


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "RAG Chatbot API is running"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG Chatbot API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
