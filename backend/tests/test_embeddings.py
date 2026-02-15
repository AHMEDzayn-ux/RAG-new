"""
Unit tests for Embeddings Service
"""

import pytest
import numpy as np
from services.embeddings import EmbeddingsService


class TestEmbeddingsService:
    """Test cases for EmbeddingsService class"""
    
    @pytest.fixture
    def embeddings_service(self):
        """Create an EmbeddingsService instance for testing"""
        # Use the lightweight model for faster tests
        return EmbeddingsService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
    
    def test_service_initialization(self, embeddings_service):
        """Test that EmbeddingsService initializes correctly"""
        assert embeddings_service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embeddings_service.model is not None
        assert embeddings_service.dimension == 384  # all-MiniLM-L6-v2 dimension
    
    def test_embed_text_basic(self, embeddings_service):
        """Test basic text embedding"""
        text = "This is a test sentence for embedding."
        embedding = embeddings_service.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_text_empty_string(self, embeddings_service):
        """Test embedding with empty string"""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            embeddings_service.embed_text("")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            embeddings_service.embed_text("   ")
    
    def test_embed_text_consistency(self, embeddings_service):
        """Test that same text produces same embedding"""
        text = "Consistent embedding test"
        embedding1 = embeddings_service.embed_text(text)
        embedding2 = embeddings_service.embed_text(text)
        
        # Should be identical
        assert embedding1 == embedding2
    
    def test_embed_batch_basic(self, embeddings_service):
        """Test batch embedding generation"""
        texts = [
            "First test sentence",
            "Second test sentence",
            "Third test sentence"
        ]
        
        embeddings = embeddings_service.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)
    
    def test_embed_batch_empty_list(self, embeddings_service):
        """Test batch embedding with empty list"""
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            embeddings_service.embed_batch([])
    
    def test_embed_batch_with_empty_strings(self, embeddings_service):
        """Test batch embedding filters empty strings"""
        texts = ["Valid text", "", "Another valid text"]
        
        embeddings = embeddings_service.embed_batch(texts)
        
        # Should only generate embeddings for non-empty texts
        assert len(embeddings) == 2
    
    def test_embed_batch_all_empty(self, embeddings_service):
        """Test batch embedding with all empty strings"""
        texts = ["", "   ", ""]
        
        with pytest.raises(ValueError, match="All texts are empty"):
            embeddings_service.embed_batch(texts)
    
    def test_embed_documents_basic(self, embeddings_service):
        """Test embedding documents with metadata"""
        documents = [
            {"text": "First document", "metadata": {"page": 1}},
            {"text": "Second document", "metadata": {"page": 2}}
        ]
        
        result = embeddings_service.embed_documents(documents)
        
        assert len(result) == 2
        assert all("embedding" in doc for doc in result)
        assert all(len(doc["embedding"]) == 384 for doc in result)
        assert all("metadata" in doc for doc in result)
    
    def test_embed_documents_custom_text_key(self, embeddings_service):
        """Test embedding documents with custom text key"""
        documents = [
            {"content": "Document content", "id": 1},
            {"content": "More content", "id": 2}
        ]
        
        result = embeddings_service.embed_documents(documents, text_key="content")
        
        assert len(result) == 2
        assert all("embedding" in doc for doc in result)
        assert all("id" in doc for doc in result)
    
    def test_embed_documents_missing_text_key(self, embeddings_service):
        """Test embedding documents with missing text key"""
        documents = [
            {"content": "Document content"},
            {"different_key": "Content"}  # Missing 'text' key
        ]
        
        with pytest.raises(ValueError, match="missing 'text' key"):
            embeddings_service.embed_documents(documents)
    
    def test_embed_documents_empty_list(self, embeddings_service):
        """Test embedding empty documents list"""
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            embeddings_service.embed_documents([])
    
    def test_compute_similarity_identical(self, embeddings_service):
        """Test similarity between identical embeddings"""
        text = "Test sentence"
        embedding = embeddings_service.embed_text(text)
        
        similarity = embeddings_service.compute_similarity(embedding, embedding)
        
        assert 0.99 <= similarity <= 1.01  # Should be very close to 1.0
    
    def test_compute_similarity_different(self, embeddings_service):
        """Test similarity between different texts"""
        embedding1 = embeddings_service.embed_text("Machine learning")
        embedding2 = embeddings_service.embed_text("Artificial intelligence")
        embedding3 = embeddings_service.embed_text("Banana recipe")
        
        # Similar texts should have higher similarity
        sim_related = embeddings_service.compute_similarity(embedding1, embedding2)
        sim_unrelated = embeddings_service.compute_similarity(embedding1, embedding3)
        
        assert sim_related > sim_unrelated
        assert -1 <= sim_related <= 1
        assert -1 <= sim_unrelated <= 1
    
    def test_compute_similarity_with_numpy(self, embeddings_service):
        """Test similarity computation with numpy arrays"""
        embedding1 = embeddings_service.embed_text("Test text")
        embedding2 = np.array(embedding1)
        
        similarity = embeddings_service.compute_similarity(embedding1, embedding2)
        
        assert 0.99 <= similarity <= 1.01
    
    def test_compute_similarity_dimension_mismatch(self, embeddings_service):
        """Test similarity with mismatched dimensions"""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.0, 2.0]
        
        with pytest.raises(ValueError, match="dimensions don't match"):
            embeddings_service.compute_similarity(embedding1, embedding2)
    
    def test_get_model_info(self, embeddings_service):
        """Test model information retrieval"""
        info = embeddings_service.get_model_info()
        
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert "max_sequence_length" in info
        assert "device" in info
        
        assert info["embedding_dimension"] == 384
        assert info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_different_texts_different_embeddings(self, embeddings_service):
        """Test that different texts produce different embeddings"""
        embedding1 = embeddings_service.embed_text("First sentence")
        embedding2 = embeddings_service.embed_text("Completely different")
        
        assert embedding1 != embedding2
    
    @pytest.mark.parametrize("text", [
        "Short text",
        "This is a longer sentence with more words in it.",
        "A" * 100,  # Long repeated character
    ])
    def test_embed_various_lengths(self, embeddings_service, text):
        """Test embedding texts of various lengths"""
        embedding = embeddings_service.embed_text(text)
        
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)


@pytest.mark.integration
class TestEmbeddingsServiceIntegration:
    """Integration tests for EmbeddingsService"""
    
    @pytest.fixture
    def embeddings_service(self):
        """Create an EmbeddingsService instance for integration testing"""
        return EmbeddingsService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
    
    def test_embed_with_real_chunks(self, embeddings_service, sample_text):
        """Test embedding real document chunks"""
        # Simulate document chunks
        chunks = [
            {"text": sample_text[:100], "chunk_index": 0},
            {"text": sample_text[100:200], "chunk_index": 1}
        ]
        
        result = embeddings_service.embed_documents(chunks)
        
        assert len(result) == 2
        assert all("embedding" in chunk for chunk in result)
        assert all("chunk_index" in chunk for chunk in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
