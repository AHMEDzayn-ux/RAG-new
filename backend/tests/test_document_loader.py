"""
Unit tests for Document Loader Service
"""

import pytest
from services.document_loader import DocumentLoader


class TestDocumentLoader:
    """Test cases for DocumentLoader class"""
    
    @pytest.fixture
    def loader(self):
        """Create a DocumentLoader instance for testing"""
        return DocumentLoader(chunk_size=100, chunk_overlap=20)
    
    def test_loader_initialization(self, loader):
        """Test that DocumentLoader initializes correctly"""
        assert loader.chunk_size == 100
        assert loader.chunk_overlap == 20
        assert loader.text_splitter is not None
    
    def test_loader_default_values(self):
        """Test default chunk size and overlap values"""
        loader = DocumentLoader()
        assert loader.chunk_size == 1000
        assert loader.chunk_overlap == 200
    
    def test_chunk_text_basic(self, loader, sample_text):
        """Test basic text chunking functionality"""
        chunks = loader.chunk_text(sample_text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)
        assert all("chunk_size" in chunk for chunk in chunks)
    
    def test_chunk_text_with_metadata(self, loader, sample_text, sample_metadata):
        """Test chunking with custom metadata"""
        chunks = loader.chunk_text(sample_text, sample_metadata)
        
        assert all(chunk["metadata"] == sample_metadata for chunk in chunks)
    
    def test_chunk_text_empty_string(self, loader):
        """Test chunking with empty string"""
        chunks = loader.chunk_text("")
        assert len(chunks) == 0
        
        chunks = loader.chunk_text("   ")
        assert len(chunks) == 0
    
    def test_chunk_text_size_limits(self, loader, sample_text):
        """Test that chunks respect size limits"""
        chunks = loader.chunk_text(sample_text)
        
        for chunk in chunks:
            assert chunk["chunk_size"] <= loader.chunk_size + 50  # Allow some tolerance
            assert chunk["chunk_size"] > 0
    
    def test_chunk_indices(self, loader, sample_text):
        """Test that chunk indices are sequential"""
        chunks = loader.chunk_text(sample_text)
        
        for idx, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == idx
    
    def test_load_pdf_file_not_found(self, loader):
        """Test loading a non-existent PDF file"""
        with pytest.raises(FileNotFoundError):
            loader.load_pdf("nonexistent.pdf")
    
    def test_load_pdf_invalid_extension(self, loader, temp_pdf_path):
        """Test loading a file that's not a PDF"""
        # Create a temporary text file
        txt_file = temp_pdf_path / "test.txt"
        txt_file.write_text("Not a PDF")
        
        with pytest.raises(ValueError, match="File must be a PDF"):
            loader.load_pdf(str(txt_file))
    
    def test_get_chunk_stats_empty(self, loader):
        """Test statistics for empty chunk list"""
        stats = loader.get_chunk_stats([])
        
        assert stats["total_chunks"] == 0
        assert stats["total_characters"] == 0
        assert stats["avg_chunk_size"] == 0
        assert stats["min_chunk_size"] == 0
        assert stats["max_chunk_size"] == 0
    
    def test_get_chunk_stats(self, loader, sample_text):
        """Test chunk statistics calculation"""
        chunks = loader.chunk_text(sample_text)
        stats = loader.get_chunk_stats(chunks)
        
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_characters"] > 0
        assert stats["avg_chunk_size"] > 0
        assert stats["min_chunk_size"] > 0
        assert stats["max_chunk_size"] > 0
        assert stats["min_chunk_size"] <= stats["avg_chunk_size"] <= stats["max_chunk_size"]
    
    def test_chunk_text_preserves_content(self, loader):
        """Test that chunking preserves all content"""
        text = "A" * 500  # Simple text to verify no content loss
        chunks = loader.chunk_text(text)
        
        # Reconstruct text from chunks (accounting for overlap)
        assert len(chunks) > 0
        total_chars = sum(chunk["chunk_size"] for chunk in chunks)
        assert total_chars >= len(text)  # Should be >= due to overlap
    
    @pytest.mark.parametrize("chunk_size,overlap", [
        (500, 100),
        (1000, 200),
        (2000, 400),
    ])
    def test_different_chunk_sizes(self, chunk_size, overlap, sample_text):
        """Test chunking with different size parameters"""
        loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = loader.chunk_text(sample_text)
        
        assert all(chunk["chunk_size"] <= chunk_size + 50 for chunk in chunks)


# Mark tests that require actual PDF files
@pytest.mark.integration
class TestDocumentLoaderIntegration:
    """Integration tests that require actual PDF files"""
    
    def test_load_and_chunk_pdf_requires_file(self):
        """Test that load_and_chunk_pdf requires valid PDF file"""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_and_chunk_pdf("missing.pdf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
