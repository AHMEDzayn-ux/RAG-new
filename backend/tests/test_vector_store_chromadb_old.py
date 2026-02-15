"""
Unit tests for Vector Store Service
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from services.vector_store import VectorStoreService


class TestVectorStoreService:
    """Test cases for VectorStoreService class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for vector store"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Cleanup
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create a VectorStoreService instance for testing"""
        return VectorStoreService(persist_directory=temp_dir)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents with embeddings"""
        return [
            {
                "id": "doc1",
                "text": "Python is a programming language",
                "embedding": [0.1] * 384,
                "metadata": {"source": "test.pdf", "page": 1}
            },
            {
                "id": "doc2",
                "text": "Machine learning is a subset of AI",
                "embedding": [0.2] * 384,
                "metadata": {"source": "test.pdf", "page": 2}
            },
            {
                "id": "doc3",
                "text": "ChromaDB is a vector database",
                "embedding": [0.3] * 384,
                "metadata": {"source": "test2.pdf", "page": 1}
            }
        ]
    
    def test_initialization(self, temp_dir):
        """Test that VectorStoreService initializes correctly"""
        store = VectorStoreService(persist_directory=temp_dir)
        
        assert store.persist_directory == temp_dir
        assert store.client is not None
        assert Path(temp_dir).exists()
    
    def test_create_collection(self, vector_store):
        """Test collection creation"""
        collection = vector_store.create_collection("test_collection")
        
        assert collection is not None
        assert collection.name == "test_collection"
    
    def test_create_collection_with_metadata(self, vector_store):
        """Test collection creation with metadata"""
        metadata = {"client_id": "test_client", "type": "documents"}
        collection = vector_store.create_collection(
            "test_collection",
            metadata=metadata
        )
        
        assert collection is not None
        assert collection.metadata == metadata
    
    def test_add_documents(self, vector_store, sample_documents):
        """Test adding documents to collection"""
        count = vector_store.add_documents("test_collection", sample_documents)
        
        assert count == 3
        assert vector_store.get_collection_count("test_collection") == 3
    
    def test_add_documents_empty_list(self, vector_store):
        """Test adding empty document list"""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            vector_store.add_documents("test_collection", [])
    
    def test_add_documents_missing_embedding(self, vector_store):
        """Test adding documents without embeddings"""
        docs = [{"text": "Test document", "id": "doc1"}]
        
        with pytest.raises(ValueError, match="missing 'embedding' key"):
            vector_store.add_documents("test_collection", docs)
    
    def test_query_basic(self, vector_store, sample_documents):
        """Test basic vector similarity query"""
        # Add documents first
        vector_store.add_documents("test_collection", sample_documents)
        
        # Query with similar embedding to doc1
        query_embedding = [0.1] * 384
        results = vector_store.query("test_collection", query_embedding, n_results=2)
        
        assert len(results['ids'][0]) == 2
        assert len(results['documents'][0]) == 2
        assert len(results['distances'][0]) == 2
        # First result should be doc1 (exact match)
        assert results['ids'][0][0] == "doc1"
    
    def test_query_with_n_results(self, vector_store, sample_documents):
        """Test query with different n_results"""
        vector_store.add_documents("test_collection", sample_documents)
        
        query_embedding = [0.2] * 384
        results = vector_store.query("test_collection", query_embedding, n_results=1)
        
        assert len(results['ids'][0]) == 1
        assert results['ids'][0][0] == "doc2"
    
    def test_query_nonexistent_collection(self, vector_store):
        """Test querying non-existent collection"""
        query_embedding = [0.1] * 384
        results = vector_store.query("nonexistent", query_embedding)
        
        assert results['ids'] == [[]]
        assert results['documents'] == [[]]
    
    def test_list_collections_empty(self, vector_store):
        """Test listing collections when empty"""
        collections = vector_store.list_collections()
        
        assert isinstance(collections, list)
        assert len(collections) == 0
    
    def test_list_collections(self, vector_store):
        """Test listing multiple collections"""
        vector_store.create_collection("collection1")
        vector_store.create_collection("collection2")
        vector_store.create_collection("collection3")
        
        collections = vector_store.list_collections()
        
        assert len(collections) == 3
        assert "collection1" in collections
        assert "collection2" in collections
        assert "collection3" in collections
    
    def test_get_collection_count(self, vector_store, sample_documents):
        """Test getting document count in collection"""
        vector_store.add_documents("test_collection", sample_documents)
        
        count = vector_store.get_collection_count("test_collection")
        
        assert count == 3
    
    def test_get_collection_count_empty(self, vector_store):
        """Test count of non-existent collection"""
        count = vector_store.get_collection_count("nonexistent")
        
        assert count == 0
    
    def test_delete_collection(self, vector_store, sample_documents):
        """Test deleting a collection"""
        vector_store.add_documents("test_collection", sample_documents)
        
        result = vector_store.delete_collection("test_collection")
        
        assert result is True
        assert "test_collection" not in vector_store.list_collections()
    
    def test_delete_nonexistent_collection(self, vector_store):
        """Test deleting non-existent collection"""
        result = vector_store.delete_collection("nonexistent")
        
        assert result is False
    
    def test_update_document(self, vector_store, sample_documents):
        """Test updating a document"""
        vector_store.add_documents("test_collection", sample_documents)
        
        new_metadata = {"source": "updated.pdf", "page": 99}
        result = vector_store.update_document(
            "test_collection",
            "doc1",
            metadata=new_metadata
        )
        
        assert result is True
    
    def test_delete_documents(self, vector_store, sample_documents):
        """Test deleting specific documents"""
        vector_store.add_documents("test_collection", sample_documents)
        
        result = vector_store.delete_documents("test_collection", ["doc1", "doc2"])
        
        assert result is True
        assert vector_store.get_collection_count("test_collection") == 1
    
    def test_documents_with_auto_ids(self, vector_store):
        """Test adding documents without explicit IDs"""
        docs = [
            {
                "text": "Document without ID",
                "embedding": [0.5] * 384
            }
        ]
        
        count = vector_store.add_documents("test_collection", docs)
        
        assert count == 1
        assert vector_store.get_collection_count("test_collection") == 1
    
    def test_query_results_structure(self, vector_store, sample_documents):
        """Test that query results have correct structure"""
        vector_store.add_documents("test_collection", sample_documents)
        
        query_embedding = [0.15] * 384
        results = vector_store.query("test_collection", query_embedding, n_results=3)
        
        assert 'ids' in results
        assert 'documents' in results
        assert 'metadatas' in results
        assert 'distances' in results
        assert len(results['ids']) == 1  # One query
        assert len(results['ids'][0]) == 3  # Three results
    
    def test_persistence(self, temp_dir, sample_documents):
        """Test that data persists across client restarts"""
        # Create store and add documents
        store1 = VectorStoreService(persist_directory=temp_dir)
        store1.add_documents("test_collection", sample_documents)
        del store1
        
        # Create new store with same directory
        store2 = VectorStoreService(persist_directory=temp_dir)
        count = store2.get_collection_count("test_collection")
        
        assert count == 3


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests for VectorStoreService"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_full_workflow(self, temp_dir):
        """Test complete workflow: add, query, update, delete"""
        store = VectorStoreService(persist_directory=temp_dir)
        
        # Add documents
        docs = [
            {
                "id": f"doc{i}",
                "text": f"Document {i}",
                "embedding": [float(i)/10] * 384,
                "metadata": {"index": i}
            }
            for i in range(5)
        ]
        
        store.add_documents("workflow_test", docs)
        assert store.get_collection_count("workflow_test") == 5
        
        # Query
        results = store.query("workflow_test", [0.0] * 384, n_results=2)
        assert len(results['ids'][0]) == 2
        
        # Update
        store.update_document("workflow_test", "doc0", metadata={"updated": True})
        
        # Delete some documents
        store.delete_documents("workflow_test", ["doc3", "doc4"])
        assert store.get_collection_count("workflow_test") == 3
        
        # Delete collection
        store.delete_collection("workflow_test")
        assert store.get_collection_count("workflow_test") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
