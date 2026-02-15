"""
Query & Chat Router
Endpoints for RAG question answering and conversational chat
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Tuple

from api.models import QueryRequest, QueryResponse, ChatRequest, ChatResponse, Source
from api.clients import get_pipeline_manager
from logger import get_logger

logger = get_logger(__name__)

# Initialize router
router = APIRouter(prefix="/api/clients", tags=["query"])


@router.post("/{client_id}/query", response_model=QueryResponse)
async def query_documents(client_id: str, request: QueryRequest):
    """
    Query documents using RAG (Retrieval-Augmented Generation).
    
    - **client_id**: The client to query
    - **question**: The question to ask
    - **top_k**: Number of relevant documents to retrieve (1-10)
    - **include_sources**: Whether to include source documents in response
    
    Returns an AI-generated answer based on the retrieved document context.
    """
    try:
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(client_id)
        
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client '{client_id}' not found"
            )
        
        # Check if LLM is available
        if pipeline.llm_service.llm is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service not available. Please configure GROQ_API_KEY."
            )
        
        logger.info(f"Query for client '{client_id}': {request.question}")
        
        # Execute query
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            return_sources=request.include_sources
        )
        
        # Format sources
        sources = None
        if request.include_sources and result.get("sources"):
            sources = [
                Source(
                    text=source["text"],
                    metadata=source.get("metadata", {}),
                    distance=source.get("distance")
                )
                for source in result["sources"]
            ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying documents for client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query documents: {str(e)}"
        )


@router.post("/{client_id}/chat", response_model=ChatResponse)
async def chat_with_documents(client_id: str, request: ChatRequest):
    """
    Have a conversational chat with document-augmented responses.
    
    - **client_id**: The client to chat with
    - **message**: The user's message
    - **history**: Previous conversation messages
    - **use_retrieval**: Whether to retrieve relevant documents
    - **top_k**: Number of documents to retrieve if using retrieval
    
    Returns a conversational response, optionally augmented with document context.
    """
    try:
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(client_id)
        
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client '{client_id}' not found"
            )
        
        # Check if LLM is available
        if pipeline.llm_service.llm is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service not available. Please configure GROQ_API_KEY."
            )
        
        logger.info(f"Chat for client '{client_id}': {request.message}")
        
        # Convert ChatMessage objects to plain dicts
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.history
        ]
        
        # Execute chat
        result = pipeline.chat(
            message=request.message,
            conversation_history=history,
            use_retrieval=request.use_retrieval,
            top_k=request.top_k
        )
        
        # Format sources
        sources = None
        if result.get("sources"):
            sources = [
                Source(
                    text=source["text"],
                    metadata=source.get("metadata", {}),
                    distance=source.get("distance")
                )
                for source in result["sources"]
            ]
        
        return ChatResponse(
            response=result["answer"],
            used_retrieval=result.get("used_retrieval", False),
            sources=sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat for client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}"
        )
