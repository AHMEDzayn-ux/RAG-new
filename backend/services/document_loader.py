"""
Document Loader Service

Handles PDF loading and text chunking for RAG system.
Supports loading PDFs and splitting them into manageable chunks.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """
    Loads PDF documents and splits them into chunks for processing.
    
    Attributes:
        chunk_size: Maximum size of each text chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Maximum characters per chunk (default: 1000)
            chunk_overlap: Overlap between chunks in characters (default: 200)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> str:
        """
        Load text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content from all pages
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF: {file_path}")
        
        try:
            reader = PdfReader(str(path))
            text_content = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    text_content.append(text)
            
            return "\n\n".join(text_content)
        
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text content to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        
        result = []
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "chunk_index": idx,
                "chunk_size": len(chunk),
                "metadata": metadata or {}
            }
            result.append(chunk_data)
        
        return result
    
    def load_and_chunk_pdf(
        self, 
        file_path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load a PDF and split it into chunks in one operation.
        
        Args:
            file_path: Path to the PDF file
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks with metadata
        """
        # Add file path to metadata
        path = Path(file_path)
        file_metadata = {
            "source": str(path),
            "filename": path.name,
            **(metadata or {})
        }
        
        # Load PDF content
        text = self.load_pdf(file_path)
        
        # Chunk the text
        chunks = self.chunk_text(text, file_metadata)
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        sizes = [chunk["chunk_size"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(sizes),
            "avg_chunk_size": sum(sizes) // len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes)
        }
