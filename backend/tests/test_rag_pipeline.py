"""
Tests for RAG Pipeline Service
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from services.rag_pipeline import RAGPipeline, MultiClientRAGPipeline


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_services():
    """Create mocked service instances."""
    with patch('services.rag_pipeline.DocumentLoader') as mock_doc_loader, \
         patch('services.rag_pipeline.EmbeddingsService') as mock_embeddings, \
         patch('services.rag_pipeline.VectorStoreService') as mock_vector_store, \
         patch('services.rag_pipeline.LLMService') as mock_llm:
        
        yield {
            'doc_loader': mock_doc_loader,
            'embeddings': mock_embeddings,
            'vector_store': mock_vector_store,
            'llm': mock_llm
        }


@pytest.fixture
def rag_pipeline(mock_services):
    """Create a RAG pipeline instance with mocked services."""
    pipeline = RAGPipeline(collection_name="test_collection")
    return pipeline


@pytest.fixture
def sample_chunks():
    """Sample document chunks."""
    return [
        {
            'text': 'Machine learning is a subset of AI.',
            'metadata': {'page': 1}
        },
        {
            'text': 'Python is popular for data science.',
            'metadata': {'page': 2}
        }
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings."""
    return [[0.1] * 384, [0.2] * 384]


class TestRAGPipelineInitialization:
    """Test RAG pipeline initialization."""
    
    def test_init_default(self, mock_services):
        """Test initialization with defaults."""
        pipeline = RAGPipeline()
        assert pipeline.collection_name == "default"
        assert pipeline.system_role == "helpful assistant"
    
    def test_init_custom_parameters(self, mock_services):
        """Test initialization with custom parameters."""
        pipeline = RAGPipeline(
            collection_name="custom_collection",
            system_role="university advisor"
        )
        assert pipeline.collection_name == "custom_collection"
        assert pipeline.system_role == "university advisor"
    
    def test_init_with_api_key(self, mock_services):
        """Test initialization with API key."""
        pipeline = RAGPipeline(api_key="test_key")
        # Verify LLM service was initialized with API key
        mock_services['llm'].assert_called_once()


class TestIndexDocuments:
    """Test document indexing workflow."""
    
    def test_index_documents_basic(self, rag_pipeline, sample_chunks, sample_embeddings):
        """Test basic document indexing."""
        # Mock document loader
        rag_pipeline.doc_loader.load_and_chunk_pdf = Mock(return_value=sample_chunks)
        
        # Mock embeddings service
        rag_pipeline.embeddings_service.embed_batch = Mock(return_value=sample_embeddings)
        
        # Mock vector store
        rag_pipeline.vector_store.list_collections = Mock(return_value=[])
        rag_pipeline.vector_store.create_collection = Mock()
        rag_pipeline.vector_store.add_documents = Mock()
        rag_pipeline.vector_store.persist = Mock()
        rag_pipeline.vector_store.get_collection_count = Mock(return_value=2)
        
        # Index documents
        stats = rag_pipeline.index_documents(["test.pdf"])
        
        # Verify workflow
        assert rag_pipeline.doc_loader.load_and_chunk_pdf.called
        assert rag_pipeline.embeddings_service.embed_batch.called
        assert rag_pipeline.vector_store.create_collection.called
        assert rag_pipeline.vector_store.add_documents.called
        assert rag_pipeline.vector_store.persist.called
        
        # Verify stats
        assert stats['pdfs_processed'] == 1
        assert stats['total_chunks'] == len(sample_chunks)
        assert stats['collection_name'] == "test_collection"
    
    def test_index_multiple_pdfs(self, rag_pipeline, sample_chunks, sample_embeddings):
        """Test indexing multiple PDFs."""
        rag_pipeline.doc_loader.load_and_chunk_pdf = Mock(return_value=sample_chunks)
        rag_pipeline.embeddings_service.embed_batch = Mock(return_value=sample_embeddings * 3)
        rag_pipeline.vector_store.list_collections = Mock(return_value=[])
        rag_pipeline.vector_store.create_collection = Mock()
        rag_pipeline.vector_store.add_documents = Mock()
        rag_pipeline.vector_store.persist = Mock()
        rag_pipeline.vector_store.get_collection_count = Mock(return_value=6)
        
        stats = rag_pipeline.index_documents(["test1.pdf", "test2.pdf", "test3.pdf"])
        
        assert stats['pdfs_processed'] == 3
        assert rag_pipeline.doc_loader.load_and_chunk_pdf.call_count == 3
    
    def test_index_with_custom_chunk_size(self, rag_pipeline, sample_chunks, sample_embeddings):
        """Test indexing with custom chunk size."""
        rag_pipeline.doc_loader.load_and_chunk_pdf = Mock(return_value=sample_chunks)
        rag_pipeline.embeddings_service.embed_batch = Mock(return_value=sample_embeddings)
        rag_pipeline.vector_store.list_collections = Mock(return_value=[])
        rag_pipeline.vector_store.create_collection = Mock()
        rag_pipeline.vector_store.add_documents = Mock()
        rag_pipeline.vector_store.persist = Mock()
        rag_pipeline.vector_store.get_collection_count = Mock(return_value=2)
        
        stats = rag_pipeline.index_documents(
            ["test.pdf"],
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Verify custom parameters were passed
        call_kwargs = rag_pipeline.doc_loader.load_and_chunk_pdf.call_args[1]
        assert call_kwargs['chunk_size'] == 500
        assert call_kwargs['chunk_overlap'] == 50
    
    def test_index_with_custom_metadata(self, rag_pipeline, sample_chunks, sample_embeddings):
        """Test indexing with custom metadata."""
        rag_pipeline.doc_loader.load_and_chunk_pdf = Mock(return_value=sample_chunks)
        rag_pipeline.embeddings_service.embed_batch = Mock(return_value=sample_embeddings)
        rag_pipeline.vector_store.list_collections = Mock(return_value=[])
        rag_pipeline.vector_store.create_collection = Mock()
        rag_pipeline.vector_store.add_documents = Mock()
        rag_pipeline.vector_store.persist = Mock()
        rag_pipeline.vector_store.get_collection_count = Mock(return_value=2)
        
        custom_metadata = {"client_id": "test_client", "category": "documentation"}
        stats = rag_pipeline.index_documents(["test.pdf"], metadata=custom_metadata)
        
        # Verify metadata was added
        call_args = rag_pipeline.vector_store.add_documents.call_args
        metadatas = call_args[1]['metadatas']
        assert all('client_id' in m for m in metadatas)
        assert all(m['client_id'] == "test_client" for m in metadatas)
    
    def test_index_existing_collection(self, rag_pipeline, sample_chunks, sample_embeddings):
        """Test indexing into existing collection."""
        rag_pipeline.doc_loader.load_and_chunk_pdf = Mock(return_value=sample_chunks)
        rag_pipeline.embeddings_service.embed_batch = Mock(return_value=sample_embeddings)
        rag_pipeline.vector_store.list_collections = Mock(return_value=["test_collection"])
        rag_pipeline.vector_store.add_documents = Mock()
        rag_pipeline.vector_store.persist = Mock()
        rag_pipeline.vector_store.get_collection_count = Mock(return_value=2)
        
        stats = rag_pipeline.index_documents(["test.pdf"])
        
        # Verify collection was not created
        rag_pipeline.vector_store.create_collection.assert_not_called()


class TestQuery:
    """Test query workflow."""
    
    def test_query_basic(self, rag_pipeline):
        """Test basic query."""
        # Mock embeddings
        query_embedding = [0.5] * 384
        rag_pipeline.embeddings_service.embed_text = Mock(return_value=query_embedding)
        
        # Mock vector store results
        mock_results = {
            'documents': [['Doc 1 text', 'Doc 2 text']],
            'metadatas': [[{'source': 'test.pdf'}, {'source': 'test.pdf'}]],
            'distances': [[0.1, 0.2]],
            'ids': [['id1', 'id2']]
        }
        rag_pipeline.vector_store.query = Mock(return_value=mock_results)
        
        # Mock LLM response
        rag_pipeline.llm_service.generate_rag_response = Mock(
            return_value="This is the answer."
        )
        
        # Query
        response = rag_pipeline.query("What is AI?")
        
        # Verify workflow
        assert rag_pipeline.embeddings_service.embed_text.called
        assert rag_pipeline.vector_store.query.called
        assert rag_pipeline.llm_service.generate_rag_response.called
        
        # Verify response
        assert response['question'] == "What is AI?"
        assert response['answer'] == "This is the answer."
        assert response['num_sources'] == 2
        assert 'sources' in response
    
    def test_query_with_custom_top_k(self, rag_pipeline):
        """Test query with custom top_k."""
        rag_pipeline.embeddings_service.embed_text = Mock(return_value=[0.5] * 384)
        rag_pipeline.vector_store.query = Mock(return_value={
            'documents': [['Doc 1']],
            'metadatas': [[{'source': 'test.pdf'}]],
            'distances': [[0.1]],
            'ids': [['id1']]
        })
        rag_pipeline.llm_service.generate_rag_response = Mock(return_value="Answer")
        
        response = rag_pipeline.query("Test question", top_k=5)
        
        # Verify top_k was passed
        call_kwargs = rag_pipeline.vector_store.query.call_args[1]
        assert call_kwargs['n_results'] == 5
    
    def test_query_without_sources(self, rag_pipeline):
        """Test query without returning sources."""
        rag_pipeline.embeddings_service.embed_text = Mock(return_value=[0.5] * 384)
        rag_pipeline.vector_store.query = Mock(return_value={
            'documents': [['Doc 1']],
            'metadatas': [[{'source': 'test.pdf'}]],
            'distances': [[0.1]],
            'ids': [['id1']]
        })
        rag_pipeline.llm_service.generate_rag_response = Mock(return_value="Answer")
        
        response = rag_pipeline.query("Test question", return_sources=False)
        
        assert 'sources' not in response
        assert 'answer' in response


class TestChat:
    """Test conversational chat."""
    
    def test_chat_with_retrieval(self, rag_pipeline):
        """Test chat with retrieval enabled."""
        conversation_history = [
            {"role": "user", "content": "What is ML?"},
            {"role": "assistant", "content": "ML is..."}
        ]
        
        rag_pipeline.embeddings_service.embed_text = Mock(return_value=[0.5] * 384)
        rag_pipeline.vector_store.query = Mock(return_value={
            'documents': [['ML context']],
            'metadatas': [[{'source': 'ml.pdf'}]],
            'distances': [[0.1]],
            'ids': [['id1']]
        })
        rag_pipeline.llm_service.generate_chat_response = Mock(return_value="Chat response")
        
        response = rag_pipeline.chat(
            "Tell me more",
            conversation_history=conversation_history,
            use_retrieval=True
        )
        
        assert response['message'] == "Tell me more"
        assert response['answer'] == "Chat response"
        assert response['used_retrieval'] is True
        assert 'sources' in response
    
    def test_chat_without_retrieval(self, rag_pipeline):
        """Test chat without retrieval."""
        conversation_history = []
        
        rag_pipeline.llm_service.generate_chat_response = Mock(return_value="Chat response")
        
        response = rag_pipeline.chat(
            "Hello",
            conversation_history=conversation_history,
            use_retrieval=False
        )
        
        assert response['used_retrieval'] is False
        assert 'sources' not in response
        rag_pipeline.embeddings_service.embed_text.assert_not_called()


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_get_stats(self, rag_pipeline):
        """Test getting pipeline statistics."""
        rag_pipeline.vector_store.get_collection_count = Mock(return_value=10)
        rag_pipeline.embeddings_service.get_model_info = Mock(return_value={
            'model': 'all-MiniLM-L6-v2'
        })
        rag_pipeline.llm_service.get_model_info = Mock(return_value={
            'model': 'llama3-70b'
        })
        
        stats = rag_pipeline.get_stats()
        
        assert stats['collection_name'] == "test_collection"
        assert stats['document_count'] == 10
        assert 'embedding_model' in stats
        assert 'llm_model' in stats
    
    def test_clear_collection(self, rag_pipeline):
        """Test clearing collection."""
        rag_pipeline.vector_store.delete_collection = Mock()
        
        rag_pipeline.clear_collection()
        
        rag_pipeline.vector_store.delete_collection.assert_called_once_with("test_collection")
    
    def test_load_existing_collection(self, rag_pipeline):
        """Test loading existing collection."""
        rag_pipeline.vector_store.load_collection = Mock()
        
        result = rag_pipeline.load_existing_collection("existing_collection")
        
        assert result is True
        rag_pipeline.vector_store.load_collection.assert_called_once_with("existing_collection")
        assert rag_pipeline.collection_name == "existing_collection"
    
    def test_load_collection_failure(self, rag_pipeline):
        """Test loading collection that doesn't exist."""
        rag_pipeline.vector_store.load_collection = Mock(
            side_effect=Exception("Collection not found")
        )
        
        result = rag_pipeline.load_existing_collection("nonexistent")
        
        assert result is False


class TestMultiClientRAGPipeline:
    """Test multi-client RAG pipeline manager."""
    
    def test_init(self, mock_services):
        """Test initialization."""
        manager = MultiClientRAGPipeline()
        assert len(manager.pipelines) == 0
    
    def test_create_pipeline(self, mock_services):
        """Test creating a client pipeline."""
        manager = MultiClientRAGPipeline()
        
        pipeline = manager.create_pipeline(
            client_id="university",
            system_role="university advisor"
        )
        
        assert pipeline is not None
        assert "university" in manager.pipelines
        assert pipeline.collection_name == "client_university"
    
    def test_create_duplicate_pipeline(self, mock_services):
        """Test creating pipeline for existing client."""
        manager = MultiClientRAGPipeline()
        
        pipeline1 = manager.create_pipeline("client1")
        pipeline2 = manager.create_pipeline("client1")
        
        assert pipeline1 is pipeline2
        assert len(manager.pipelines) == 1
    
    def test_get_pipeline(self, mock_services):
        """Test getting a client's pipeline."""
        manager = MultiClientRAGPipeline()
        created_pipeline = manager.create_pipeline("client1")
        
        retrieved_pipeline = manager.get_pipeline("client1")
        
        assert retrieved_pipeline is created_pipeline
    
    def test_get_nonexistent_pipeline(self, mock_services):
        """Test getting pipeline that doesn't exist."""
        manager = MultiClientRAGPipeline()
        
        pipeline = manager.get_pipeline("nonexistent")
        
        assert pipeline is None
    
    def test_delete_pipeline(self, mock_services):
        """Test deleting a client's pipeline."""
        manager = MultiClientRAGPipeline()
        pipeline = manager.create_pipeline("client1")
        pipeline.clear_collection = Mock()
        
        result = manager.delete_pipeline("client1")
        
        assert result is True
        assert "client1" not in manager.pipelines
        assert pipeline.clear_collection.called
    
    def test_delete_nonexistent_pipeline(self, mock_services):
        """Test deleting pipeline that doesn't exist."""
        manager = MultiClientRAGPipeline()
        
        result = manager.delete_pipeline("nonexistent")
        
        assert result is False
    
    def test_list_clients(self, mock_services):
        """Test listing all clients."""
        manager = MultiClientRAGPipeline()
        manager.create_pipeline("client1")
        manager.create_pipeline("client2")
        manager.create_pipeline("client3")
        
        clients = manager.list_clients()
        
        assert len(clients) == 3
        assert "client1" in clients
        assert "client2" in clients
        assert "client3" in clients


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_complete_rag_workflow(self, rag_pipeline, sample_chunks, sample_embeddings):
        """Test complete RAG workflow from indexing to querying."""
        # Setup mocks
        rag_pipeline.doc_loader.load_and_chunk_pdf = Mock(return_value=sample_chunks)
        rag_pipeline.embeddings_service.embed_batch = Mock(return_value=sample_embeddings)
        rag_pipeline.embeddings_service.embed_text = Mock(return_value=[0.5] * 384)
        rag_pipeline.vector_store.list_collections = Mock(return_value=[])
        rag_pipeline.vector_store.create_collection = Mock()
        rag_pipeline.vector_store.add_documents = Mock()
        rag_pipeline.vector_store.persist = Mock()
        rag_pipeline.vector_store.get_collection_count = Mock(return_value=2)
        rag_pipeline.vector_store.query = Mock(return_value={
            'documents': [['ML is AI']],
            'metadatas': [[{'source': 'test.pdf'}]],
            'distances': [[0.1]],
            'ids': [['id1']]
        })
        rag_pipeline.llm_service.generate_rag_response = Mock(return_value="AI answer")
        
        # Index documents
        index_stats = rag_pipeline.index_documents(["test.pdf"])
        assert index_stats['pdfs_processed'] == 1
        
        # Query
        query_response = rag_pipeline.query("What is AI?")
        assert query_response['answer'] == "AI answer"
        assert query_response['num_sources'] > 0
