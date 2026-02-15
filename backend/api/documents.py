"""
Document Management Router
Endpoints for uploading and managing PDF documents for RAG clients
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from typing import List, Optional
import tempfile
import os
from pathlib import Path

from api.models import DocumentUploadResponse, DocumentListResponse, DocumentInfo, MessageResponse
from api.clients import get_pipeline_manager
from logger import get_logger

logger = get_logger(__name__)

# Initialize router
router = APIRouter(prefix="/api/clients", tags=["documents"])


@router.post("/{client_id}/documents", response_model=DocumentUploadResponse)
async def upload_documents(
    client_id: str,
    file: UploadFile = File(..., description="PDF file to upload"),
    category: str = Form(default="general", description="Document category"),
    doc_type: str = Form(default="document", description="Document type")
):
    """
    Upload a PDF document to a client's collection.
    
    - **client_id**: The client to upload documents for
    - **file**: PDF file to upload
    - **category**: Category for organization (e.g., "admission", "policies")
    - **doc_type**: Document type (e.g., "guide", "faq", "policy")
    """
    try:
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(client_id)
        
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client '{client_id}' not found"
            )
        
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.filename}. Only PDF files are allowed."
            )
        
        # Save uploaded file temporarily and process
        temp_files = []
        try:
            # Create temp directory for this upload
            temp_dir = tempfile.mkdtemp()
            
            # Save to temp file
            temp_path = os.path.join(temp_dir, file.filename)
            content = await file.read()
            
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            temp_files.append(temp_path)
            logger.info(f"Saved uploaded file: {file.filename} ({len(content)} bytes)")
            
            # Get count before indexing
            stats_before = pipeline.get_stats()
            doc_count_before = stats_before["document_count"]
            
            # Index documents
            logger.info(f"Indexing document for client '{client_id}'")
            result = pipeline.index_documents(
                pdf_paths=temp_files,
                metadata={
                    "category": category,
                    "doc_type": doc_type
                }
            )
            
            # Get count after indexing
            stats_after = pipeline.get_stats()
            doc_count_after = stats_after["document_count"]
            chunks_created = doc_count_after - doc_count_before
            
            logger.info(f"Indexed file, created {chunks_created} chunks")
            
            return DocumentUploadResponse(
                message=f"Successfully uploaded document",
                files_processed=1,
                chunks_created=chunks_created,
                total_documents=doc_count_after
            )
            
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            
            # Cleanup temp directory
            try:
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to delete temp directory {temp_dir}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading documents for client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload documents: {str(e)}"
        )


@router.get("/{client_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    client_id: str,
    limit: int = 10,
    offset: int = 0
):
    """
    List documents in a client's collection.
    
    - **client_id**: The client to list documents for
    - **limit**: Maximum number of documents to return
    - **offset**: Number of documents to skip
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
        total_docs = stats["document_count"]
        
        # For now, return basic info since we don't have a method to list all documents
        # In a real implementation, you'd query the vector store
        documents = []
        
        # Return limited document info
        for i in range(offset, min(offset + limit, total_docs)):
            documents.append(DocumentInfo(
                chunk_index=i,
                text_preview="[Document content]",
                metadata={"note": "Full document listing requires vector store query implementation"}
            ))
        
        return DocumentListResponse(
            client_id=client_id,
            total_documents=total_docs,
            documents=documents
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents for client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/{client_id}/documents", response_model=MessageResponse)
async def clear_documents(client_id: str):
    """
    Clear all documents from a client's collection.
    
    - **client_id**: The client to clear documents for
    """
    try:
        manager = get_pipeline_manager()
        pipeline = manager.get_pipeline(client_id)
        
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client '{client_id}' not found"
            )
        
        # Get count before clearing
        stats_before = pipeline.get_stats()
        doc_count = stats_before["document_count"]
        
        # Clear collection
        pipeline.clear_collection()
        
        logger.info(f"Cleared {doc_count} documents from client '{client_id}'")
        
        return MessageResponse(
            message=f"Cleared all documents from client '{client_id}'",
            detail=f"Removed {doc_count} document(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing documents for client {client_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear documents: {str(e)}"
        )
