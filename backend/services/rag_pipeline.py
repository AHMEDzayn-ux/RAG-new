"""
RAG Pipeline Service
Orchestrates the complete RAG workflow: document processing, retrieval, and generation
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from services.document_loader import DocumentLoader
from services.embeddings import EmbeddingsService
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline.
    
    Orchestrates:
    1. Document loading and chunking
    2. Embedding generation
    3. Vector storage and retrieval
    4. LLM-based response generation
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        api_key: Optional[str] = None,
        system_role: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            collection_name: Name of the vector store collection
            api_key: Groq API key (optional, uses settings if not provided)
            system_role: System role for LLM responses (e.g., "university advisor")
        """
        self.collection_name = collection_name
        self.system_role = system_role or "helpful assistant"
        
        # Initialize all services
        self.doc_loader = DocumentLoader()
        self.embeddings_service = EmbeddingsService()
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService(api_key=api_key)
        
        logger.info(f"RAGPipeline initialized for collection: {collection_name}")
    
    def index_documents(
        self,
        pdf_paths: List[str],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index documents into the vector store.
        
        Complete workflow:
        1. Load PDFs and chunk text
        2. Generate embeddings for chunks
        3. Store in vector database
        
        Args:
            pdf_paths: List of PDF file paths to index
            chunk_size: Custom chunk size (optional)
            chunk_overlap: Custom chunk overlap (optional)
            metadata: Additional metadata to attach to all documents
        
        Returns:
            Dictionary with indexing statistics
        """
        logger.info(f"Starting document indexing for {len(pdf_paths)} PDFs")
        
        all_chunks = []
        all_texts = []
        all_metadatas = []
        
        # Step 1: Load and chunk all documents
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Processing: {pdf_path}")
                
                # Load and chunk
                if chunk_size and chunk_overlap:
                    chunks = self.doc_loader.load_and_chunk_pdf(
                        pdf_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                else:
                    chunks = self.doc_loader.load_and_chunk_pdf(pdf_path)
                
                # Prepare data
                for i, chunk in enumerate(chunks):
                    text = chunk['text']
                    chunk_metadata = chunk.get('metadata', {})
                    
                    # Add custom metadata
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    # Add document source
                    chunk_metadata['source'] = pdf_path
                    chunk_metadata['chunk_index'] = i
                    
                    all_chunks.append(chunk)
                    all_texts.append(text)
                    all_metadatas.append(chunk_metadata)
                
                logger.info(f"Processed {len(chunks)} chunks from {pdf_path}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                raise
        
        # Step 2: Generate embeddings
        logger.info(f"Generating embeddings for {len(all_texts)} chunks")
        embeddings = self.embeddings_service.embed_batch(all_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 3: Create collection if it doesn't exist
        if self.collection_name not in self.vector_store.list_collections():
            self.vector_store.create_collection(
                self.collection_name,
                embedding_dimension=settings.embedding_dimension
            )
            logger.info(f"Created collection: {self.collection_name}")
        
        # Step 4: Add documents to vector store
        self.vector_store.add_documents(
            collection_name=self.collection_name,
            documents=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas
        )
        
        # Step 5: Persist to disk
        self.vector_store.persist()
        logger.info("Vector store persisted to disk")
        
        # Return statistics
        stats = {
            'pdfs_processed': len(pdf_paths),
            'total_chunks': len(all_chunks),
            'total_embeddings': len(embeddings),
            'collection_name': self.collection_name,
            'vector_store_count': self.vector_store.get_collection_count(self.collection_name)
        }
        
        logger.info(f"Indexing complete: {stats}")
        return stats
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Complete workflow:
        1. Generate embedding for question
        2. Retrieve relevant documents from vector store
        3. Generate response using LLM with context
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve (default from settings)
            return_sources: Whether to include source documents in response
        
        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Generate query embedding
        query_embedding = self.embeddings_service.embed_text(question)
        logger.info("Generated query embedding")
        
        # Step 2: Retrieve relevant documents
        top_k = top_k or settings.retrieval_top_k
        
        try:
            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_embeddings=[query_embedding],
                n_results=top_k
            )
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise
        
        # Extract retrieved documents
        retrieved_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        
        logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        # Step 3: Generate response using LLM
        try:
            answer = self.llm_service.generate_rag_response(
                query=question,
                retrieved_docs=retrieved_docs,
                system_role=self.system_role
            )
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
        
        logger.info("Generated response")
        
        # Prepare response
        response = {
            'question': question,
            'answer': answer,
            'num_sources': len(retrieved_docs)
        }
        
        if return_sources:
            response['sources'] = retrieved_docs
        
        return response
    
    def chat(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        use_retrieval: bool = True,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Conversational query with history.
        
        Args:
            message: Current user message
            conversation_history: Previous conversation messages (list of dicts with 'role' and 'content')
            use_retrieval: Whether to retrieve context from vector store
            top_k: Number of documents to retrieve if using retrieval
        
        Returns:
            Dictionary with response and conversation info
        """
        logger.info(f"Processing chat message: {message}")
        
        context = None
        retrieved_docs = None
        
        # Retrieve context if requested
        if use_retrieval:
            try:
                query_embedding = self.embeddings_service.embed_text(message)
                top_k = top_k or settings.retrieval_top_k
                
                results = self.vector_store.query(
                    collection_name=self.collection_name,
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                # Extract context
                if results and results.get('documents') and len(results['documents']) > 0 and results['documents'][0]:
                    context = results['documents'][0]
                    retrieved_docs = [
                        {
                            'text': doc,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                            'distance': results['distances'][0][i] if results.get('distances') else 0
                        }
                        for i, doc in enumerate(context)
                    ]
                    logger.info(f"Retrieved {len(retrieved_docs)} documents")
                else:
                    logger.info("No documents found in collection for retrieval")
            except Exception as e:
                logger.warning(f"Error during retrieval: {e}. Proceeding without retrieval.")
        
        # Generate response
        answer = self.llm_service.generate_chat_response(
            query=message,
            conversation_history=conversation_history,
            context=context
        )
        
        response = {
            'message': message,
            'answer': answer,
            'used_retrieval': use_retrieval
        }
        
        if retrieved_docs:
            response['sources'] = retrieved_docs
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            doc_count = self.vector_store.get_collection_count(self.collection_name)
        except:
            doc_count = 0
        
        return {
            'collection_name': self.collection_name,
            'document_count': doc_count,
            'embedding_model': self.embeddings_service.get_model_info(),
            'llm_model': self.llm_service.get_model_info(),
            'system_role': self.system_role
        }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the current collection.
        """
        logger.info(f"Clearing collection: {self.collection_name}")
        self.vector_store.delete_collection(self.collection_name)
        
        # Recreate empty collection immediately
        self.vector_store.create_collection(
            self.collection_name,
            embedding_dimension=settings.embedding_dimension
        )
        self.vector_store.persist()
        logger.info("Collection cleared and recreated")
    
    def load_existing_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Load an existing collection from disk.
        
        Args:
            collection_name: Name of collection to load (uses current if not provided)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        name = collection_name or self.collection_name
        
        try:
            self.vector_store.load_collection(name)
            if collection_name:
                self.collection_name = collection_name
            logger.info(f"Loaded collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load collection {name}: {str(e)}")
            return False


class MultiClientRAGPipeline:
    """
    Manages multiple RAG pipelines for different clients.
    Each client gets their own collection and optional custom configuration.
    """
    
    def __init__(self):
        """Initialize the multi-client RAG manager."""
        self.pipelines: Dict[str, RAGPipeline] = {}
        logger.info("MultiClientRAGPipeline initialized")
        
        # Automatically restore existing clients from persisted data
        self._restore_clients_from_disk()
    
    def create_pipeline(
        self,
        client_id: str,
        system_role: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> RAGPipeline:
        """
        Create a new RAG pipeline for a client.
        
        Args:
            client_id: Unique identifier for the client
            system_role: Custom system role for this client
            api_key: Optional Groq API key
        
        Returns:
            RAGPipeline instance for the client
        """
        if client_id in self.pipelines:
            logger.warning(f"Pipeline for client '{client_id}' already exists")
            return self.pipelines[client_id]
        
        pipeline = RAGPipeline(
            collection_name=f"client_{client_id}",
            api_key=api_key,
            system_role=system_role
        )
        
        self.pipelines[client_id] = pipeline
        logger.info(f"Created pipeline for client: {client_id}")
        
        return pipeline
    
    def get_pipeline(self, client_id: str) -> Optional[RAGPipeline]:
        """
        Get a client's RAG pipeline.
        
        Args:
            client_id: Client identifier
        
        Returns:
            RAGPipeline instance or None if not found
        """
        return self.pipelines.get(client_id)
    
    def delete_pipeline(self, client_id: str) -> bool:
        """
        Delete a client's pipeline and data.
        
        Args:
            client_id: Client identifier
        
        Returns:
            True if deleted, False if not found
        """
        if client_id not in self.pipelines:
            return False
        
        # Clear the collection
        self.pipelines[client_id].clear_collection()
        
        # Remove from dictionary
        del self.pipelines[client_id]
        
        logger.info(f"Deleted pipeline for client: {client_id}")
        return True
    
    def list_clients(self) -> List[str]:
        """
        List all client IDs with active pipelines.
        
        Returns:
            List of client IDs
        """
        return list(self.pipelines.keys())
    
    def _restore_clients_from_disk(self) -> None:
        """
        Restore client pipelines from persisted vector store files.
        Automatically loads existing clients on server startup.
        """
        from pathlib import Path
        
        vector_store_dir = Path(settings.vector_stores_dir) / "faiss"
        if not vector_store_dir.exists():
            logger.info("No vector store directory found, starting fresh")
            return
        
        # Find all .index files
        index_files = list(vector_store_dir.glob("client_*.index"))
        
        if not index_files:
            logger.info("No existing clients found")
            return
        
        logger.info(f"Found {len(index_files)} persisted client(s), restoring...")
        
        for index_file in index_files:
            # Extract client_id from filename (e.g., "client_acme.index" -> "acme")
            collection_name = index_file.stem  # "client_acme"
            if not collection_name.startswith("client_"):
                continue
            
            client_id = collection_name[7:]  # Remove "client_" prefix
            
            try:
                # Create pipeline instance
                pipeline = RAGPipeline(
                    collection_name=collection_name,
                    system_role="helpful assistant"
                )
                
                # Load the persisted collection
                pipeline.vector_store.load_collection(collection_name)
                
                # Add to pipelines dictionary
                self.pipelines[client_id] = pipeline
                
                logger.info(f"Restored client: {client_id}")
                
            except Exception as e:
                logger.error(f"Failed to restore client {client_id}: {e}")
        
        logger.info(f"Successfully restored {len(self.pipelines)} client(s)")
