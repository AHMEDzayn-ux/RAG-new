"""
Vector Store Service for RAG System
Handles storing and retrieving document embeddings using FAISS
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import pickle
import numpy as np
import faiss

from logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class VectorStoreService:
    """
    Service for managing document embeddings in a vector database.
    Uses FAISS for efficient similarity search with persistent storage.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the Vector Store Service.
        
        Args:
            persist_directory: Directory path for persistent storage.
                             If None, uses the default from settings.
        """
        self.persist_directory = persist_directory or str(settings.vector_stores_dir / "faiss")
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store FAISS indexes for different collections
        self.collections: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"VectorStoreService initialized with persist_directory: {self.persist_directory}")
    
    def create_collection(
        self,
        name: str,
        embedding_dimension: Optional[int] = None
    ) -> None:
        """
        Create a new collection for storing embeddings.
        
        Args:
            name: Name of the collection
            embedding_dimension: Dimension of embeddings (default from settings)
        """
        if name in self.collections:
            logger.warning(f"Collection '{name}' already exists")
            return
        
        dimension = embedding_dimension or settings.embedding_dimension
        
        # Create FAISS index (L2 distance)
        index = faiss.IndexFlatL2(dimension)
        
        self.collections[name] = {
            'index': index,
            'documents': [],
            'metadatas': [],
            'ids': [],
            'dimension': dimension
        }
        
        logger.info(f"Created collection '{name}' with dimension {dimension}")
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents with their embeddings to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs (auto-generated if not provided)
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        collection = self.collections[collection_name]
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Validate embedding dimension
        if embeddings_array.shape[1] != collection['dimension']:
            raise ValueError(
                f"Embedding dimension {embeddings_array.shape[1]} "
                f"does not match collection dimension {collection['dimension']}"
            )
        
        # Generate IDs if not provided
        if ids is None:
            start_id = len(collection['ids'])
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Add to FAISS index
        collection['index'].add(embeddings_array)
        
        # Store documents, metadatas, and ids
        collection['documents'].extend(documents)
        collection['metadatas'].extend(metadatas)
        collection['ids'].extend(ids)
        
        logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
    
    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the collection for similar documents with optional metadata filtering.
        
        Args:
            collection_name: Name of the collection to query
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return per query
            metadata_filter: Optional dictionary to filter results by metadata
                           Example: {"section": "work_experience", "user_tier": "enterprise"}
        
        Returns:
            Dictionary containing:
                - documents: List of lists of matching document texts
                - metadatas: List of lists of metadata dictionaries
                - distances: List of lists of distance scores
                - ids: List of lists of document IDs
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        collection = self.collections[collection_name]
        
        # Convert query embeddings to numpy array
        query_array = np.array(query_embeddings, dtype=np.float32)
        
        # Validate dimension
        if query_array.shape[1] != collection['dimension']:
            raise ValueError(
                f"Query embedding dimension {query_array.shape[1]} "
                f"does not match collection dimension {collection['dimension']}"
            )
        
        # If metadata filter provided, retrieve more results and filter
        search_n = n_results * 3 if metadata_filter else n_results
        
        # Perform search
        distances, indices = collection['index'].search(query_array, min(search_n, len(collection['documents'])))
        
        # Prepare results
        results = {
            'documents': [],
            'metadatas': [],
            'distances': [],
            'ids': []
        }
        
        for i, query_indices in enumerate(indices):
            query_docs = []
            query_metas = []
            query_ids = []
            query_dists = []
            
            for j, idx in enumerate(query_indices):
                if idx < len(collection['documents']):
                    metadata = collection['metadatas'][idx]
                    
                    # Apply metadata filter if provided
                    if metadata_filter:
                        if not self._match_metadata_filter(metadata, metadata_filter):
                            continue
                    
                    query_docs.append(collection['documents'][idx])
                    query_metas.append(metadata)
                    query_ids.append(collection['ids'][idx])
                    # Cast to native float — numpy float32 isn't JSON-serializable
                    # and downstream callers/tests expect Python floats.
                    query_dists.append(float(distances[i][j]))
                    
                    # Stop once we have enough results
                    if len(query_docs) >= n_results:
                        break
            
            results['documents'].append(query_docs)
            results['metadatas'].append(query_metas)
            results['distances'].append(query_dists)
            results['ids'].append(query_ids)
        
        filter_msg = f" with filter {metadata_filter}" if metadata_filter else ""
        logger.info(
            f"Queried collection '{collection_name}' with {len(query_embeddings)} queries{filter_msg}, "
            f"returning {n_results} results each"
        )
        
        return results
    
    def _match_metadata_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if metadata matches all filter criteria.
        
        Args:
            metadata: Document metadata dictionary
            filter_dict: Filter criteria dictionary
            
        Returns:
            True if all filter criteria match, False otherwise
        """
        for key, value in filter_dict.items():
            # Support nested keys with dot notation (e.g., "user.tier")
            if '.' in key:
                current = metadata
                for part in key.split('.'):
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return False
                if current != value:
                    return False
            else:
                if metadata.get(key) != value:
                    return False
        return True
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection and its associated data.
        
        Args:
            name: Name of the collection to delete
        """
        if name not in self.collections:
            logger.warning(f"Collection '{name}' does not exist")
            return
        
        del self.collections[name]
        
        # Delete persisted files if they exist (index + JSON and any legacy pkl)
        index_path = Path(self.persist_directory) / f"{name}.index"
        for sidecar in (
            index_path,
            Path(self.persist_directory) / f"{name}_metadata.json",
            Path(self.persist_directory) / f"{name}_metadata.pkl",
        ):
            if sidecar.exists():
                sidecar.unlink()

        logger.info(f"Deleted collection '{name}'")
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        return list(self.collections.keys())
    
    def get_collection_count(self, name: str) -> int:
        """
        Get the number of documents in a collection.
        
        Args:
            name: Name of the collection
        
        Returns:
            Number of documents in the collection
        """
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' does not exist")
        
        return len(self.collections[name]['documents'])
    
    def persist(self) -> None:
        """
        Save all collections to disk.

        Metadata is written as JSON (safe, portable, human-inspectable) rather
        than pickle. A legacy ``*_metadata.pkl`` sidecar, if present, is removed
        once the JSON has been written so the two can't drift out of sync.
        """
        for name, collection in self.collections.items():
            # Save FAISS index
            index_path = Path(self.persist_directory) / f"{name}.index"
            faiss.write_index(collection['index'], str(index_path))

            # Save metadata as JSON
            metadata = {
                'documents': collection['documents'],
                'metadatas': collection['metadatas'],
                'ids': collection['ids'],
                'dimension': collection['dimension']
            }
            metadata_path = Path(self.persist_directory) / f"{name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False)

            # Retire any old pickle sidecar so stale data isn't loaded later.
            legacy_pkl = Path(self.persist_directory) / f"{name}_metadata.pkl"
            if legacy_pkl.exists():
                legacy_pkl.unlink()

            logger.info(f"Persisted collection '{name}'")
    
    def load_collection(self, name: str) -> None:
        """
        Load a collection from disk.
        
        Args:
            name: Name of the collection to load
        """
        index_path = Path(self.persist_directory) / f"{name}.index"
        json_path = Path(self.persist_directory) / f"{name}_metadata.json"
        legacy_pkl_path = Path(self.persist_directory) / f"{name}_metadata.pkl"

        if not index_path.exists() or not (json_path.exists() or legacy_pkl_path.exists()):
            raise ValueError(f"Collection '{name}' does not exist on disk")

        # Load FAISS index
        index = faiss.read_index(str(index_path))

        # Load metadata — prefer JSON; fall back to a legacy pickle sidecar for
        # collections indexed before the JSON switch (loaded once, then rewritten
        # as JSON on the next persist()).
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            with open(legacy_pkl_path, 'rb') as f:
                metadata = pickle.load(f)

        self.collections[name] = {
            'index': index,
            'documents': metadata['documents'],
            'metadatas': metadata['metadatas'],
            'ids': metadata['ids'],
            'dimension': metadata['dimension']
        }
        
        logger.info(f"Loaded collection '{name}' with {len(metadata['documents'])} documents")
    
    def update_document(
        self,
        collection_name: str,
        document_id: str,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update an existing document in the collection.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to update
            document: New document text (optional)
            embedding: New embedding vector (optional)
            metadata: New metadata (optional)
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        collection = self.collections[collection_name]
        
        # Find document index
        try:
            idx = collection['ids'].index(document_id)
        except ValueError:
            raise ValueError(f"Document with ID '{document_id}' not found")
        
        # Update document
        if document is not None:
            collection['documents'][idx] = document

        # Update metadata
        if metadata is not None:
            collection['metadatas'][idx] = metadata

        # FAISS has no in-place vector update, but IndexFlatL2 supports
        # reconstruct(), so we can pull every stored vector back out, swap the
        # one row, and rebuild — no second copy of the embeddings kept in RAM.
        if embedding is not None:
            new_vec = np.array(embedding, dtype=np.float32)
            if new_vec.shape[0] != collection['dimension']:
                raise ValueError(
                    f"Embedding dimension {new_vec.shape[0]} does not match "
                    f"collection dimension {collection['dimension']}"
                )
            vectors = self._reconstruct_all(collection['index'])
            vectors[idx] = new_vec
            collection['index'] = self._build_index_from_vectors(
                collection['dimension'], vectors
            )

        logger.info(f"Updated document '{document_id}' in collection '{collection_name}'")
    
    def delete_documents(
        self,
        collection_name: str,
        document_ids: List[str]
    ) -> None:
        """
        Delete documents from a collection by their IDs.
        
        Args:
            collection_name: Name of the collection
            document_ids: List of document IDs to delete
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        collection = self.collections[collection_name]

        # FAISS IndexFlatL2 has no efficient in-place delete, so we rebuild the
        # index from the surviving vectors. IndexFlatL2 supports reconstruct(),
        # so we recover the kept vectors straight from the index rather than
        # keeping a second copy of every embedding resident in RAM.
        delete_set = set(document_ids)
        indices_to_keep = [
            i for i, doc_id in enumerate(collection['ids'])
            if doc_id not in delete_set
        ]

        if len(indices_to_keep) == len(collection['ids']):
            logger.warning(f"No documents found with IDs: {document_ids}")
            return

        # Reconstruct only the surviving vectors and rebuild the index so the
        # FAISS positions stay aligned with the parallel document/metadata lists.
        all_vectors = self._reconstruct_all(collection['index'])
        kept_vectors = all_vectors[indices_to_keep] if indices_to_keep else all_vectors[:0]
        collection['index'] = self._build_index_from_vectors(
            collection['dimension'], kept_vectors
        )

        collection['documents'] = [collection['documents'][i] for i in indices_to_keep]
        collection['metadatas'] = [collection['metadatas'][i] for i in indices_to_keep]
        collection['ids'] = [collection['ids'][i] for i in indices_to_keep]

        logger.info(
            f"Deleted {len(document_ids)} document(s) from collection '{collection_name}'; "
            f"index rebuilt with {len(indices_to_keep)} remaining vectors"
        )

    @staticmethod
    def _reconstruct_all(index: "faiss.Index") -> np.ndarray:
        """Return every stored vector as a (n, dim) float32 array.

        IndexFlatL2 stores raw vectors, so reconstruct_n recovers them exactly
        without us having to keep a parallel copy of the embeddings in memory.
        """
        n = index.ntotal
        if n == 0:
            return np.empty((0, index.d), dtype=np.float32)
        return index.reconstruct_n(0, n)

    @staticmethod
    def _build_index_from_vectors(dimension: int, vectors: np.ndarray) -> "faiss.Index":
        """Create a fresh IndexFlatL2 populated with the given vectors."""
        new_index = faiss.IndexFlatL2(dimension)
        if vectors is not None and len(vectors) > 0:
            new_index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return new_index
