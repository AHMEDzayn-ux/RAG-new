"""
Tests for Vector Store Service (FAISS implementation)
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from services.vector_store_faiss import VectorStoreService


@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for testing persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_store(temp_persist_dir):
    """Create a VectorStoreService instance for testing."""
    return VectorStoreService(persist_directory=temp_persist_dir)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return np.random.rand(5, 384).tolist()


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by the human brain.",
        "Deep learning uses multiple layers of neural networks.",
        "Natural language processing helps computers understand text."
    ]


@pytest.fixture
def sample_metadatas():
    """Sample metadata for testing."""
    return [
        {"source": "ml_basics.pdf", "page": 1},
        {"source": "python_guide.pdf", "page": 5},
        {"source": "nn_intro.pdf", "page": 10},
        {"source": "deep_learning.pdf", "page": 3},
        {"source": "nlp_guide.pdf", "page": 7}
    ]


class TestVectorStoreServiceInitialization:
    """Test vector store service initialization."""
    
    def test_init_default(self, temp_persist_dir):
        """Test initialization with default settings."""
        service = VectorStoreService(persist_directory=temp_persist_dir)
        assert service.persist_directory == temp_persist_dir
        assert len(service.collections) == 0
        assert Path(temp_persist_dir).exists()
    
    def test_init_creates_directory(self, temp_persist_dir):
        """Test that initialization creates the persist directory."""
        subdir = Path(temp_persist_dir) / "subdir"
        service = VectorStoreService(persist_directory=str(subdir))
        assert subdir.exists()


class TestCollectionManagement:
    """Test collection creation and management."""
    
    def test_create_collection(self, vector_store):
        """Test creating a new collection."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        assert "test_collection" in vector_store.collections
        assert vector_store.collections["test_collection"]["dimension"] == 384
    
    def test_create_duplicate_collection(self, vector_store):
        """Test creating a collection that already exists."""
        vector_store.create_collection("test_collection")
        # Should not raise an error, just log a warning
        vector_store.create_collection("test_collection")
        assert len(vector_store.collections) == 1
    
    def test_list_collections(self, vector_store):
        """Test listing all collections."""
        vector_store.create_collection("collection1")
        vector_store.create_collection("collection2")
        collections = vector_store.list_collections()
        assert len(collections) == 2
        assert "collection1" in collections
        assert "collection2" in collections
    
    def test_delete_collection(self, vector_store):
        """Test deleting a collection."""
        vector_store.create_collection("test_collection")
        vector_store.delete_collection("test_collection")
        assert "test_collection" not in vector_store.collections
    
    def test_delete_nonexistent_collection(self, vector_store):
        """Test deleting a collection that doesn't exist."""
        # Should not raise an error, just log a warning
        vector_store.delete_collection("nonexistent")


class TestDocumentOperations:
    """Test document addition and retrieval."""
    
    def test_add_documents(self, vector_store, sample_documents, sample_embeddings, sample_metadatas):
        """Test adding documents to a collection."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas
        )
        
        count = vector_store.get_collection_count("test_collection")
        assert count == len(sample_documents)
    
    def test_add_documents_with_auto_ids(self, vector_store, sample_documents, sample_embeddings):
        """Test adding documents without providing IDs."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        collection = vector_store.collections["test_collection"]
        assert len(collection["ids"]) == len(sample_documents)
        assert all(id.startswith("doc_") for id in collection["ids"])
    
    def test_add_documents_with_custom_ids(self, vector_store, sample_documents, sample_embeddings):
        """Test adding documents with custom IDs."""
        custom_ids = [f"custom_{i}" for i in range(len(sample_documents))]
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings,
            ids=custom_ids
        )
        
        collection = vector_store.collections["test_collection"]
        assert collection["ids"] == custom_ids
    
    def test_add_documents_wrong_dimension(self, vector_store, sample_documents):
        """Test adding documents with wrong embedding dimension."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        wrong_embeddings = [[0.1, 0.2, 0.3]] * len(sample_documents)  # Wrong dimension
        
        with pytest.raises(ValueError, match="does not match collection dimension"):
            vector_store.add_documents(
                collection_name="test_collection",
                documents=sample_documents,
                embeddings=wrong_embeddings
            )
    
    def test_add_documents_to_nonexistent_collection(self, vector_store, sample_documents, sample_embeddings):
        """Test adding documents to a collection that doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            vector_store.add_documents(
                collection_name="nonexistent",
                documents=sample_documents,
                embeddings=sample_embeddings
            )
    
    def test_get_collection_count(self, vector_store, sample_documents, sample_embeddings):
        """Test getting the count of documents in a collection."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        count = vector_store.get_collection_count("test_collection")
        assert count == len(sample_documents)
    
    def test_get_collection_count_nonexistent(self, vector_store):
        """Test getting count for a nonexistent collection."""
        with pytest.raises(ValueError, match="does not exist"):
            vector_store.get_collection_count("nonexistent")


class TestQueryOperations:
    """Test similarity search queries."""
    
    def test_query_basic(self, vector_store, sample_documents, sample_embeddings):
        """Test basic query operation."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        # Query with the first embedding
        query_embedding = [sample_embeddings[0]]
        results = vector_store.query(
            collection_name="test_collection",
            query_embeddings=query_embedding,
            n_results=3
        )
        
        assert len(results["documents"]) == 1
        assert len(results["documents"][0]) == 3
        assert len(results["distances"][0]) == 3
        assert len(results["ids"][0]) == 3
    
    def test_query_multiple(self, vector_store, sample_documents, sample_embeddings):
        """Test querying with multiple query vectors."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        # Query with two embeddings
        query_embeddings = sample_embeddings[:2]
        results = vector_store.query(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            n_results=2
        )
        
        assert len(results["documents"]) == 2
        assert len(results["documents"][0]) == 2
        assert len(results["documents"][1]) == 2
    
    def test_query_with_n_results(self, vector_store, sample_documents, sample_embeddings):
        """Test query with different n_results values."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        query_embedding = [sample_embeddings[0]]
        results = vector_store.query(
            collection_name="test_collection",
            query_embeddings=query_embedding,
            n_results=2
        )
        
        assert len(results["documents"][0]) == 2
    
    def test_query_nonexistent_collection(self, vector_store, sample_embeddings):
        """Test querying a collection that doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            vector_store.query(
                collection_name="nonexistent",
                query_embeddings=[sample_embeddings[0]],
                n_results=5
            )
    
    def test_query_wrong_dimension(self, vector_store, sample_documents, sample_embeddings):
        """Test querying with wrong embedding dimension."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        wrong_query = [[0.1, 0.2, 0.3]]  # Wrong dimension
        with pytest.raises(ValueError, match="does not match collection dimension"):
            vector_store.query(
                collection_name="test_collection",
                query_embeddings=wrong_query,
                n_results=5
            )


class TestPersistence:
    """Test persistence and loading of collections."""
    
    def test_persist_and_load(self, temp_persist_dir, sample_documents, sample_embeddings, sample_metadatas):
        """Test persisting and loading a collection."""
        # Create and populate collection
        service1 = VectorStoreService(persist_directory=temp_persist_dir)
        service1.create_collection("test_collection", embedding_dimension=384)
        service1.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas
        )
        service1.persist()
        
        # Create new service and load collection
        service2 = VectorStoreService(persist_directory=temp_persist_dir)
        service2.load_collection("test_collection")
        
        assert "test_collection" in service2.collections
        assert service2.get_collection_count("test_collection") == len(sample_documents)
        
        # Verify documents are retrievable
        query_embedding = [sample_embeddings[0]]
        results = service2.query(
            collection_name="test_collection",
            query_embeddings=query_embedding,
            n_results=3
        )
        assert len(results["documents"][0]) == 3
    
    def test_load_nonexistent_collection(self, vector_store):
        """Test loading a collection that doesn't exist."""
        with pytest.raises(ValueError, match="does not exist on disk"):
            vector_store.load_collection("nonexistent")
    
    def test_persist_multiple_collections(self, temp_persist_dir, sample_documents, sample_embeddings):
        """Test persisting multiple collections."""
        service = VectorStoreService(persist_directory=temp_persist_dir)
        
        # Create two collections
        service.create_collection("collection1", embedding_dimension=384)
        service.create_collection("collection2", embedding_dimension=384)
        
        service.add_documents("collection1", sample_documents[:2], sample_embeddings[:2])
        service.add_documents("collection2", sample_documents[2:4], sample_embeddings[2:4])
        
        service.persist()
        
        # Verify files were created
        assert (Path(temp_persist_dir) / "collection1.index").exists()
        assert (Path(temp_persist_dir) / "collection1_metadata.pkl").exists()
        assert (Path(temp_persist_dir) / "collection2.index").exists()
        assert (Path(temp_persist_dir) / "collection2_metadata.pkl").exists()


class TestUpdateOperations:
    """Test document update operations."""
    
    def test_update_document_text(self, vector_store, sample_documents, sample_embeddings):
        """Test updating document text."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        custom_ids = ["doc_1", "doc_2", "doc_3"]
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents[:3],
            embeddings=sample_embeddings[:3],
            ids=custom_ids
        )
        
        new_text = "Updated document text"
        vector_store.update_document(
            collection_name="test_collection",
            document_id="doc_2",
            document=new_text
        )
        
        collection = vector_store.collections["test_collection"]
        idx = collection["ids"].index("doc_2")
        assert collection["documents"][idx] == new_text
    
    def test_update_document_metadata(self, vector_store, sample_documents, sample_embeddings, sample_metadatas):
        """Test updating document metadata."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        custom_ids = ["doc_1", "doc_2", "doc_3"]
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents[:3],
            embeddings=sample_embeddings[:3],
            metadatas=sample_metadatas[:3],
            ids=custom_ids
        )
        
        new_metadata = {"source": "updated.pdf", "page": 99}
        vector_store.update_document(
            collection_name="test_collection",
            document_id="doc_2",
            metadata=new_metadata
        )
        
        collection = vector_store.collections["test_collection"]
        idx = collection["ids"].index("doc_2")
        assert collection["metadatas"][idx] == new_metadata
    
    def test_update_nonexistent_document(self, vector_store, sample_documents, sample_embeddings):
        """Test updating a document that doesn't exist."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents,
            embeddings=sample_embeddings
        )
        
        with pytest.raises(ValueError, match="not found"):
            vector_store.update_document(
                collection_name="test_collection",
                document_id="nonexistent_id",
                document="New text"
            )
    
    def test_update_embedding_not_supported(self, vector_store, sample_documents, sample_embeddings):
        """Test that updating embeddings raises NotImplementedError."""
        vector_store.create_collection("test_collection", embedding_dimension=384)
        custom_ids = ["doc_1"]
        vector_store.add_documents(
            collection_name="test_collection",
            documents=sample_documents[:1],
            embeddings=sample_embeddings[:1],
            ids=custom_ids
        )
        
        new_embedding = sample_embeddings[1]
        with pytest.raises(NotImplementedError, match="rebuilding the FAISS index"):
            vector_store.update_document(
                collection_name="test_collection",
                document_id="doc_1",
                embedding=new_embedding
            )


class TestIntegrationWorkflow:
    """Test complete workflow integration."""
    
    def test_complete_workflow(self, temp_persist_dir, sample_documents, sample_embeddings, sample_metadatas):
        """Test a complete workflow from creation to querying."""
        service = VectorStoreService(persist_directory=temp_persist_dir)
        
        # Create collection
        service.create_collection("ml_docs", embedding_dimension=384)
        
        # Add documents
        service.add_documents(
            collection_name="ml_docs",
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadatas=sample_metadatas
        )
        
        # Query
        query_embedding = [sample_embeddings[0]]
        results = service.query(
            collection_name="ml_docs",
            query_embeddings=query_embedding,
            n_results=3
        )
        
        # Verify results
        assert len(results["documents"][0]) == 3
        assert len(results["metadatas"][0]) == 3
        assert all(isinstance(d, float) for d in results["distances"][0])
        
        # The closest result should be the same document
        assert results["documents"][0][0] == sample_documents[0]
        
        # Persist
        service.persist()
        
        # Verify persistence
        assert (Path(temp_persist_dir) / "ml_docs.index").exists()
        assert (Path(temp_persist_dir) / "ml_docs_metadata.pkl").exists()
