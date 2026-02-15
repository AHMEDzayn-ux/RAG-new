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
    
    # Query intent patterns to filter by section
    QUERY_SECTION_MAP = {
        "work_experience": [
            r"(?i)(work\s+experience|job|employment|professional\s+experience|career)",
            r"(?i)(jobs\s+done|worked\s+as|working\s+at)"
        ],
        "education": [
            r"(?i)(education|degree|university|college|academic|studied|student)"
        ],
        "skills": [
            r"(?i)(skills|technical|competencies|expertise|proficient)"
        ],
        "volunteer": [
            r"(?i)(volunteer|extracurricular|organizing|committee|leadership\s+role)"
        ],
        "projects": [
            r"(?i)(projects|portfolio|built|developed)"
        ]
    }
    
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
        metadata: Optional[Dict[str, Any]] = None,
        use_parent_child: bool = False,
        generate_qa_pairs: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents into the vector store with advanced strategies.
        
        Complete workflow:
        1. Load PDFs and chunk text (with optional parent-child strategy)
        2. Generate QA pairs if enabled (for better search alignment)
        3. Generate embeddings for chunks/questions
        4. Store in vector database with rich metadata
        
        Args:
            pdf_paths: List of PDF file paths to index
            chunk_size: Custom chunk size (optional)
            chunk_overlap: Custom chunk overlap (optional)
            metadata: Additional metadata to attach to all documents
            use_parent_child: Use parent-child chunking strategy
            generate_qa_pairs: Generate hypothetical QA pairs for better search
        
        Returns:
            Dictionary with indexing statistics
        """
        logger.info(f"Starting document indexing for {len(pdf_paths)} PDFs")
        logger.info(f"Parent-child: {use_parent_child}, QA generation: {generate_qa_pairs}")
        
        all_chunks = []
        all_texts = []
        all_metadatas = []
        parent_lookup = {}  # Maps child_id to parent_text
        
        # Step 1: Load and chunk all documents
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Processing: {pdf_path}")
                
                # Load PDF content
                text = self.doc_loader.load_pdf(pdf_path)
                
                if use_parent_child:
                    # Use parent-child strategy
                    result = self.doc_loader.chunk_with_parent_child(text, metadata)
                    chunks = result['child_chunks']  # Index children for search
                    
                    # Store parent lookup for retrieval
                    for parent in result['parent_chunks']:
                        parent_id = parent['metadata']['parent_id']
                        parent_lookup[parent_id] = parent['text']
                    
                    logger.info(f"Created {len(result['parent_chunks'])} parents, {len(chunks)} children")
                    
                elif generate_qa_pairs:
                    # Generate QA pairs for better search
                    chunks = self.doc_loader.chunk_with_qa_generation(
                        text, 
                        metadata,
                        llm_service=self.llm_service,
                        generate_qa=True
                    )
                    logger.info(f"Generated QA pairs for {len(chunks)} chunks")
                else:
                    # Standard chunking
                    if chunk_size and chunk_overlap:
                        self.doc_loader.chunk_size = chunk_size
                        self.doc_loader.chunk_overlap = chunk_overlap
                    
                    chunks = self.doc_loader.chunk_text(text, metadata)
                
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
                    
                    # Store parent lookup if using parent-child
                    if use_parent_child and 'parent_id' in chunk_metadata:
                        chunk_metadata['has_parent'] = True
                    
                    all_chunks.append(chunk)
                    all_texts.append(text)
                    all_metadatas.append(chunk_metadata)
                    
                    # If QA pairs generated, also index the questions
                    if generate_qa_pairs and 'generated_questions' in chunk:
                        for q_idx, question in enumerate(chunk['generated_questions']):
                            qa_metadata = chunk_metadata.copy()
                            qa_metadata['content_type'] = 'question'
                            qa_metadata['original_chunk_index'] = i
                            qa_metadata['question_index'] = q_idx
                            
                            all_texts.append(question)
                            all_metadatas.append(qa_metadata)
                            logger.debug(f"Added question: {question[:50]}...")
                
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
        return_sources: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system with advanced retrieval strategies.
        
        Complete workflow:
        1. Generate embedding for question
        2. Retrieve relevant documents from vector store (with metadata filtering)
        3. Filter by section if query intent detected
        4. Retrieve parent chunks if using parent-child strategy
        5. Generate response using LLM with context
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve (default from settings)
            return_sources: Whether to include source documents in response
            metadata_filter: Optional metadata filter (e.g., {"user_tier": "enterprise"})
        
        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing query: {question}")
        
        # Detect query intent for section filtering
        target_section = self._detect_query_section(question)
        
        # Build metadata filter
        combined_filter = metadata_filter.copy() if metadata_filter else {}
        if target_section:
            combined_filter['section'] = target_section
            logger.info(f"Detected query intent: {target_section}")
        
        # Step 1: Generate query embedding
        query_embedding = self.embeddings_service.embed_text(question)
        logger.info("Generated query embedding")
        # Step 2: Retrieve relevant documents with metadata filter
        top_k = top_k or settings.retrieval_top_k
        
        try:
            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_embeddings=[query_embedding],
                n_results=top_k,
                metadata_filter=combined_filter if combined_filter else None
            )
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise
        
        # Extract retrieved documents
        retrieved_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                doc_metadata = results['metadatas'][0][i]
                
                # Skip questions, retrieve original content
                if doc_metadata.get('content_type') == 'question':
                    continue
                
                retrieved_docs.append({
                    'text': doc,
                    'metadata': doc_metadata,
                    'distance': results['distances'][0][i]
                })
        
        # Step 3: Retrieve parent chunks if using parent-child strategy
        if retrieved_docs and any(doc['metadata'].get('has_parent') for doc in retrieved_docs):
            retrieved_docs = self._retrieve_parent_chunks(retrieved_docs)
            logger.info("Replaced child chunks with parent chunks for full context")
        
        logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        # Step 4: Generate response using LLM
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
    
    def _detect_query_section(self, query: str) -> Optional[str]:
        """
        Detect which CV section the query is asking about.
        
        Args:
            query: User's question
            
        Returns:
            Section name or None if no specific section detected
        """
        import re
        for section, patterns in self.QUERY_SECTION_MAP.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return section
        return None
    
    def _filter_by_section(self, documents: List[Dict[str, Any]], section: str) -> List[Dict[str, Any]]:
        """
        Filter documents by section metadata.
        
        Args:
            documents: List of retrieved documents
            section: Target section name
            
        Returns:
            Filtered list of documents from the target section
        """
        filtered = []
        for doc in documents:
            doc_section = doc.get('metadata', {}).get('section')
            if doc_section == section:
                filtered.append(doc)
        return filtered
    
    def _retrieve_parent_chunks(self, child_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replace child chunks with their parent chunks for full context.
        
        Args:
            child_docs: List of retrieved child documents
            
        Returns:
            List of parent documents (deduplicated)
        """
        parent_docs = []
        seen_parents = set()
        
        for doc in child_docs:
            parent_id = doc['metadata'].get('parent_id')
            if not parent_id or parent_id in seen_parents:
                # No parent or already retrieved, keep original
                if parent_id not in seen_parents:
                    parent_docs.append(doc)
                    if parent_id:
                        seen_parents.add(parent_id)
                continue
            
            # Try to retrieve parent chunk from vector store
            # Search for parent by parent_id in metadata
            try:
                parent_filter = {'parent_id': parent_id, 'chunk_type': 'parent'}
                results = self.vector_store.query(
                    collection_name=self.collection_name,
                    query_embeddings=[doc['metadata'].get('embedding', [0.0] * 384)],  # Dummy query
                    n_results=1,
                    metadata_filter=parent_filter
                )
                
                if results['documents'] and results['documents'][0]:
                    parent_docs.append({
                        'text': results['documents'][0][0],
                        'metadata': results['metadatas'][0][0],
                        'distance': doc['distance']  # Keep child's relevance score
                    })
                    seen_parents.add(parent_id)
                else:
                    # Parent not found, keep child
                    parent_docs.append(doc)
            except:
                # Error retrieving parent, keep child
                parent_docs.append(doc)
        
        return parent_docs
    
    def chat(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        use_retrieval: bool = True,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Conversational query with history and advanced retrieval.
        
        Args:
            message: Current user message
            conversation_history: Previous conversation messages (list of dicts with 'role' and 'content')
            use_retrieval: Whether to retrieve context from vector store
            top_k: Number of documents to retrieve if using retrieval
            metadata_filter: Optional metadata filter for retrieval
        
        Returns:
            Dictionary with response and conversation info
        """
        logger.info(f"Processing chat message: {message}")
        
        # Detect query intent for section filtering
        target_section = self._detect_query_section(message)
        
        # Build combined metadata filter
        combined_filter = metadata_filter.copy() if metadata_filter else {}
        if target_section:
            combined_filter['section'] = target_section
            logger.info(f"Detected chat query intent: {target_section}")
        
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
                    n_results=top_k,
                    metadata_filter=combined_filter if combined_filter else None
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
                        if not results['metadatas'][0][i].get('content_type') == 'question'  # Skip questions
                    ]
                    
                    # Retrieve parent chunks if using parent-child
                    if retrieved_docs and any(doc['metadata'].get('has_parent') for doc in retrieved_docs):
                        retrieved_docs = self._retrieve_parent_chunks(retrieved_docs)
                        context = [doc['text'] for doc in retrieved_docs]
                        logger.info(f"Retrieved parent chunks for full context")
                    
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
