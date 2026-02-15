"""
Client Management Router
Endpoints for creating, listing, and managing RAG chatbot clients
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
from api.models import ClientCreate, ClientResponse, ClientListResponse, MessageResponse
from services.rag_pipeline import MultiClientRAGPipeline
from logger import get_logger

logger = get_logger(__name__)

# Initialize router
router = APIRouter(prefix="/api/clients", tags=["clients"])

# Global pipeline manager (will be initialized on first request)
pipeline_manager: MultiClientRAGPipeline = None


def get_pipeline_manager() -> MultiClientRAGPipeline:
    """Get or initialize the pipeline manager."""
    global pipeline_manager
    if pipeline_manager is None:
        logger.info("Initializing MultiClientRAGPipeline manager")
        pipeline_manager = MultiClientRAGPipeline()
    return pipeline_manager


@router.post("", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
async def create_client(client: ClientCreate):
    """
    Create a new RAG chatbot client.
    
    - **client_id**: Unique identifier for the client
    - **description**: Optional description of the client
    """
    try:
        manager = get_pipeline_manager()
        
        # Check if client already exists
        if client.client_id in manager.list_clients():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Client '{client.client_id}' already exists"
            )
        
        # Create new pipeline
        logger.info(f"Creating new client: {client.client_id}")
        # Create pipeline with default settings
        pipeline = manager.create_pipeline(
            client_id=client.client_id,
            system_role="helpful assistant"
        )
        
        # Get stats
        stats = pipeline.get_stats()
        
        return ClientResponse(
            client_id=client.client_id,
            description=client.description,
            document_count=stats["document_count"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create client: {str(e)}"
        )


@router.get("", response_model=ClientListResponse)
async def list_clients():
    """
    List all active RAG chatbot clients.
    
    Returns a list of client objects with details.
    """
    try:
        manager = get_pipeline_manager()
        client_ids = manager.list_clients()
        
        client_list = []
        for client_id in client_ids:
            pipeline = manager.get_pipeline(client_id)
            if pipeline:
                stats = pipeline.get_stats()
                client_list.append(ClientResponse(
                    client_id=client_id,
                    description="",
                    document_count=stats["document_count"]
                ))
        
        return ClientListResponse(
            clients=client_list,
            total=len(client_list)
        )
        
    except Exception as e:
        logger.error(f"Error listing clients: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list clients: {str(e)}"
        )


@router.get("/{client_id}", response_model=ClientResponse)
async def get_client(client_id: str):
    """
    Get detailed information about a specific client.
    
    - **client_id**: The unique identifier of the client
    """
    try:
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(client_id)
        
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client '{client_id}' not found"
            )
        
        stats = pipeline.get_stats()
        
        return ClientResponse(
            client_id=client_id,
            description="",
            document_count=stats["document_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get client: {str(e)}"
        )


@router.delete("/{client_id}", response_model=MessageResponse)
async def delete_client(client_id: str):
    """
    Delete a RAG chatbot client and its associated data.
    
    - **client_id**: The unique identifier of the client to delete
    """
    try:
        manager = get_pipeline_manager()
        
        # Check if client exists
        if client_id not in manager.list_clients():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client '{client_id}' not found"
            )
        
        # Delete pipeline
        logger.info(f"Deleting client: {client_id}")
        manager.delete_pipeline(client_id)
        
        return MessageResponse(
            message=f"Client '{client_id}' deleted successfully",
            detail="All associated data has been removed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete client: {str(e)}"
        )
