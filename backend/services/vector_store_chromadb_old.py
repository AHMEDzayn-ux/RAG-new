"""
Vector Store Service

Handles vector storage and retrieval using ChromaDB.
Stores document embeddings and performs similarity search.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from pathlib import Path
from logger import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """
    Service for managing vector storage with ChromaDB.
    
    Stores document embeddings and provides similarity search functionality.
    Supports multiple collections for multi-client scenarios.
    
    Attributes:
        persist_directory: Directory for persistent storage
        client: ChromaDB client instance
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        client_settings: Optional[Settings] = None
    ):
        """
        Initialize the vector store service.
        
        Args:
            persist_directory: Path to store the vector database
            client_settings: Optional ChromaDB client settings
        """
        if persist_directory is None:
            persist_directory = str(Path("../vector_stores/chromadb"))
        
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {persist_directory}")
        
        try:
            if client_settings:
                self.client = chromadb.Client(client_settings)
            else:
                self.client = chromadb.PersistentClient(path=persist_directory)
            
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Create or get a collection in the vector store.
        
        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection
            
        Returns:
            ChromaDB collection instance
        """
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            logger.info(f"Collection '{collection_name}' ready")
            return collection
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embeddings_key: str = "embedding"
    ) -> int:
        """
        Add documents with embeddings to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents with embeddings and metadata
            embeddings_key: Key in document dict containing embeddings
            
        Returns:
            Number of documents added
            
        Raises:
            ValueError: If documents are invalid
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        collection = self.create_collection(collection_name)
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        texts = []
        
        for i, doc in enumerate(documents):
            if embeddings_key not in doc:
                raise ValueError(f"Document {i} missing '{embeddings_key}' key")
            
            # Generate ID if not present
            doc_id = doc.get('id', f"{collection_name}_{i}")
            ids.append(str(doc_id))
            
            # Extract embedding
            embeddings.append(doc[embeddings_key])
            
            # Extract text
            text = doc.get('text', '')
            texts.append(text)
            
            # Prepare metadata (exclude embedding and text)
            metadata = {k: v for k, v in doc.items() 
                       if k not in [embeddings_key, 'text', 'id']}
            metadatas.append(metadata)
        
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to '{collection_name}'")
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            collection_name: Name of the collection to query
            query_embedding: Query vector embedding
            n_results: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter
            
        Returns:
            Query results with ids, documents, metadatas, and distances
        """
        try:
            collection = self.client.get_collection(collection_name)
        except Exception:
            logger.warning(f"Collection '{collection_name}' not found")
            return {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.info(f"Query returned {len(results['ids'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector store.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection: {str(e)}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the vector store.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            names = [col.name for col in collections]
            logger.info(f"Found {len(names)} collections")
            return names
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
    
    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of documents in the collection
        """
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            return count
        except Exception:
            return 0
    
    def update_document(
        self,
        collection_name: str,
        document_id: str,
        embedding: Optional[List[float]] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing document in a collection.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to update
            embedding: Optional new embedding
            text: Optional new text
            metadata: Optional new metadata
            
        Returns:
            True if updated successfully
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            update_data = {'ids': [document_id]}
            if embedding:
                update_data['embeddings'] = [embedding]
            if text:
                update_data['documents'] = [text]
            if metadata:
                update_data['metadatas'] = [metadata]
            
            collection.update(**update_data)
            logger.info(f"Updated document '{document_id}' in '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            return False
    
    def delete_documents(
        self,
        collection_name: str,
        document_ids: List[str]
    ) -> bool:
        """
        Delete documents from a collection.
        
        Args:
            collection_name: Name of the collection
            document_ids: List of document IDs to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            collection = self.client.get_collection(collection_name)
            collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False
