"""
Pydantic models for API request/response validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# Client Management Models
class ClientCreate(BaseModel):
    """Request model for creating a new client."""
    client_id: str = Field(..., description="Unique identifier for the client")
    description: str = Field(default="", description="Optional description of the client")
    
    class Config:
        json_schema_extra = {
            "example": {
                "client_id": "university",
                "description": "University admissions chatbot"
            }
        }


class ClientResponse(BaseModel):
    """Response model for client information."""
    client_id: str
    description: str = ""
    document_count: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "client_id": "university",
                "description": "University admissions chatbot",
                "document_count": 15
            }
        }


class ClientListResponse(BaseModel):
    """Response model for listing all clients."""
    clients: List[ClientResponse]
    total: int


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    detail: Optional[str] = None


# Document Management Models
class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    message: str
    files_processed: int
    chunks_created: int
    total_documents: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Documents uploaded successfully",
                "files_processed": 2,
                "chunks_created": 45,
                "total_documents": 47
            }
        }


class DocumentInfo(BaseModel):
    """Information about a document chunk."""
    chunk_index: int
    text_preview: str
    metadata: Dict[str, Any]


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    client_id: str
    total_documents: int
    documents: List[DocumentInfo]


# Query & Chat Models
class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    use_hybrid_search: bool = Field(default=True, description="Enable hybrid vector+keyword search")
    use_reranking: bool = Field(default=True, description="Enable cross-encoder re-ranking")
    use_query_rewriting: bool = Field(default=False, description="Enable LLM-based query rewriting")
    use_hyde: bool = Field(default=False, description="Enable HyDE (Hypothetical Document Embeddings)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the admission requirements?",
                "top_k": 3,
                "include_sources": True,
                "use_hybrid_search": True,
                "use_reranking": True,
                "use_query_rewriting": False,
                "use_hyde": False
            }
        }


class Source(BaseModel):
    """Source document information."""
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    sources: Optional[List[Source]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The admission requirements include...",
                "sources": [
                    {
                        "text": "Excerpt from document...",
                        "metadata": {"source": "admission_guide.pdf"},
                        "distance": 0.85
                    }
                ]
            }
        }


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for conversational chat."""
    message: str = Field(..., description="The user's message")
    history: List[ChatMessage] = Field(default=[], description="Previous conversation history")
    use_retrieval: bool = Field(default=True, description="Whether to use document retrieval")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")
    use_hybrid_search: bool = Field(default=True, description="Enable hybrid vector+keyword search")
    use_reranking: bool = Field(default=True, description="Enable cross-encoder re-ranking")
    use_query_rewriting: bool = Field(default=False, description="Enable LLM-based query rewriting")
    use_hyde: bool = Field(default=False, description="Enable HyDE (Hypothetical Document Embeddings)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Tell me more about that",
                "history": [
                    {"role": "user", "content": "What are the requirements?"},
                    {"role": "assistant", "content": "The requirements include..."}
                ],
                "use_retrieval": True,
                "top_k": 3
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat."""
    response: str
    used_retrieval: bool
    sources: Optional[List[Source]] = None
