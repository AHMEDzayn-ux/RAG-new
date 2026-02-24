"""
Configuration Management for RAG System

Centralizes all configuration settings with environment variable support.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses .env file for local development, environment variables for production.
    """
    
    # API Keys
    groq_api_key: str = ""
    openai_api_key: str = ""  # For voice TTS (text-to-speech)
    
    # Database
    database_url: str = "sqlite:///./rag_system.db"
    
    # Admin
    admin_password: str = "changeme"
    
    # Application
    environment: str = "development"
    log_level: str = "INFO"
    
    # Paths
    documents_dir: Path = Path(__file__).parent.parent / "documents"
    vector_stores_dir: Path = Path(__file__).parent.parent / "vector_stores"
    
    # Document Processing
    default_chunk_size: int = 1200
    default_chunk_overlap: int = 200
    
    # Vector Store
    vector_store_type: str = "chromadb"
    chromadb_persist_directory: Optional[str] = None
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # Dimension for all-MiniLM-L6-v2
    
    # LLM
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048  # Increased from 512 to allow complete responses
    
    # Retrieval
    retrieval_top_k: int = 10  # Increased from 6 to 10 for more comprehensive answers
    
    # Distance-based Relevance Filtering
    enable_distance_filtering: bool = True  # Filter out irrelevant results based on distance threshold
    distance_threshold: float = 1.3  # L2 distance threshold (0=identical, 2=very different)
    min_results_after_filter: int = 1  # Minimum results to return even if below threshold
    
    # Advanced Retrieval Features (ordered by cost-effectiveness)
    use_query_normalization: bool = True  # NEW - lightweight, enabled by default
    use_hybrid_search: bool = True
    use_reranking: bool = True
    use_query_rewriting: bool = False  # More expensive alternative to normalization
    use_hyde: bool = False
    use_multi_query: bool = False  # Most comprehensive but expensive - use for critical queries
    multi_query_variations: int = 3
    multi_query_boost_original: float = 1.5
    
    # Semantic Metadata Settings
    enable_semantic_metadata: bool = True  # Embed metadata values into searchable content
    metadata_synonym_expansion: bool = True  # Add synonyms for metadata values
    semantic_metadata_position: str = "prefix"  # "prefix", "suffix", or "both"
    
    # WhatsApp Integration
    whatsapp_verify_token: str = "your_verify_token_here"
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_business_account_id: str = ""
    whatsapp_client_id: str = "customer_care_test"  # Default client for WhatsApp
    whatsapp_bot_name: str = "Customer Support"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set ChromaDB directory if not specified
        if self.chromadb_persist_directory is None:
            self.chromadb_persist_directory = str(self.vector_stores_dir / "chromadb")
        
        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vector_stores_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings instance
    """
    return settings
