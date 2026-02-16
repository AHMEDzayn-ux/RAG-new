"""
FastAPI Application for Multi-Tenant RAG Chatbot System
Phase 7: REST API Implementation with Security
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from logger import get_logger
from api.clients import router as clients_router
from api.documents import router as documents_router
from api.query import router as query_router
from config import settings
from security import SecurityMiddleware

logger = get_logger(__name__)

# Log configuration at startup
logger.info(f"Starting RAG API...")
logger.info(f"Environment: {settings.environment}")
logger.info(f"Groq API Key configured: {bool(settings.groq_api_key)}")
logger.info(f"LLM Model: {settings.llm_model}")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Multi-tenant RAG system with advanced security and retrieval optimization",
    version="2.0.0",
    docs_url="/docs" if settings.environment == "development" else None,  # Disable docs in production
    redoc_url="/redoc" if settings.environment == "development" else None
)

# Security Middleware (rate limiting, headers, validation)
app.add_middleware(SecurityMiddleware)

# Trusted Host Middleware (prevent host header attacks)
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.yourdomain.com", "localhost", "127.0.0.1"]
    )

# Configure CORS based on environment
if settings.environment == "production":
    # Production: Restrict to Vercel deployments and specific origins
    allowed_origins = []
    # Allow all Vercel deployments with regex
    allowed_origin_regex = r"https://rag-new-.*\.vercel\.app"
    logger.info(f"CORS restricted to Vercel deployments: {allowed_origin_regex}")
else:
    # Development: Allow localhost + Vercel deployments
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
    ]
    # Allow all Vercel preview deployments with regex
    allowed_origin_regex = r"https://rag-new-.*\.vercel\.app"
    logger.info("CORS enabled for local development + Vercel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=3600,
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
