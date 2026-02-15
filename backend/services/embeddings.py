"""
Embeddings Service

Handles text embedding generation using HuggingFace Sentence Transformers.
Converts text chunks into vector representations for semantic search.
"""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from logger import get_logger

logger = get_logger(__name__)


class EmbeddingsService:
    """
    Service for generating text embeddings using Sentence Transformers.
    
    Uses HuggingFace models to convert text into dense vector representations
    suitable for semantic similarity search.
    
    Attributes:
        model_name: Name of the HuggingFace model to use
        model: Loaded SentenceTransformer model
        dimension: Dimension of the embedding vectors
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the embeddings service.
        
        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2)
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If text is empty or None
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            raise ValueError("All texts are empty")
        
        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Convert to list of lists
            result = embeddings.tolist()
            
            logger.info(f"Successfully generated {len(result)} embeddings")
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def embed_documents(
        self, 
        documents: List[dict],
        text_key: str = "text",
        batch_size: int = 32
    ) -> List[dict]:
        """
        Generate embeddings for document chunks with metadata.
        
        Args:
            documents: List of document dictionaries (must contain text_key)
            text_key: Key in document dict containing text to embed
            batch_size: Batch size for processing
            
        Returns:
            List of documents with added 'embedding' field
            
        Raises:
            ValueError: If documents list is empty or text_key not found
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # Extract texts
        texts = []
        for i, doc in enumerate(documents):
            if text_key not in doc:
                raise ValueError(f"Document at index {i} missing '{text_key}' key")
            texts.append(doc[text_key])
        
        # Generate embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        # Add embeddings to documents
        result = []
        for doc, embedding in zip(documents, embeddings):
            doc_with_embedding = doc.copy()
            doc_with_embedding['embedding'] = embedding
            result.append(doc_with_embedding)
        
        return result
    
    def compute_similarity(
        self, 
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
            
        Raises:
            ValueError: If embeddings have different dimensions
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if vec1.shape != vec2.shape:
            raise ValueError(
                f"Embedding dimensions don't match: {vec1.shape} vs {vec2.shape}"
            )
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.dimension,
            "max_sequence_length": self.model.max_seq_length,
            "device": str(self.model.device)
        }
