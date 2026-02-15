"""
Document Loader Service

Handles PDF loading and text chunking for RAG system.
Supports loading PDFs and splitting them into manageable chunks.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """
    Loads PDF documents and splits them into chunks for processing.
    
    Attributes:
        chunk_size: Maximum size of each text chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    """
    
    # CV section patterns for auto-detection
    SECTION_PATTERNS = {
        "work_experience": [
            r"(?i)^(work\s+experience|employment\s+history|professional\s+experience|career\s+history)",
            r"(?i)^(experience|employment)"
        ],
        "education": [
            r"(?i)^(education|academic\s+background|qualifications|educational\s+background)"
        ],
        "skills": [
            r"(?i)^(skills|technical\s+skills|competencies|expertise)"
        ],
        "volunteer": [
            r"(?i)^(volunteer|volunteering|community\s+service|extracurricular|leadership|organizing\s+committee)"
        ],
        "projects": [
            r"(?i)^(projects|key\s+projects|portfolio)"
        ],
        "certifications": [
            r"(?i)^(certifications|certificates|licenses|credentials)"
        ]
    }
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        enable_section_detection: bool = True,
        parent_chunk_size: int = 3000,
        child_chunk_size: int = 400
    ):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Maximum characters per chunk (default: 1000)
            chunk_overlap: Overlap between chunks in characters (default: 200)
            enable_section_detection: Enable CV section detection (default: True)
            parent_chunk_size: Size of parent chunks for parent-child retrieval (default: 3000)
            child_chunk_size: Size of child chunks for search (default: 400)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_section_detection = enable_section_detection
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        
        # Main text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Parent chunk splitter (larger chunks for context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap * 2,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Child chunk splitter (smaller chunks for precise matching)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
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
        
        # Detect sections if enabled
        if self.enable_section_detection:
            return self._chunk_with_sections(text, metadata)
        
        # Standard chunking without section detection
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
    
    def _detect_section(self, text_line: str) -> Optional[str]:
        """
        Detect which CV section a line belongs to.
        
        Args:
            text_line: A line of text to check
            
        Returns:
            Section name or None if no match
        """
        text_line = text_line.strip()
        if not text_line:
            return None
        
        for section, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, text_line):
                    return section
        
        return None
    
    def _chunk_with_sections(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks while detecting and tagging CV sections.
        
        Args:
            text: Text content to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text, section tags, and metadata
        """
        lines = text.split('\n')
        current_section = None
        chunks_text = self.text_splitter.split_text(text)
        
        result = []
        for idx, chunk in enumerate(chunks_text):
            # Try to detect section from the chunk's first meaningful line
            chunk_lines = chunk.strip().split('\n')
            detected_section = None
            
            for line in chunk_lines[:5]:  # Check first 5 lines
                detected = self._detect_section(line)
                if detected:
                    detected_section = detected
                    break
            
            # Build metadata with section tag
            chunk_metadata = {**(metadata or {})}
            if detected_section:
                chunk_metadata["section"] = detected_section
                current_section = detected_section
            elif current_section:
                # Continue with previous section if no new section detected
                chunk_metadata["section"] = current_section
            
            chunk_data = {
                "text": chunk,
                "chunk_index": idx,
                "chunk_size": len(chunk),
                "metadata": chunk_metadata
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
    def chunk_with_parent_child(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parent-Child Chunking Strategy for better retrieval.
        
        Creates:
        - Parent chunks (large): Full context for LLM
        - Child chunks (small): Precise search matching
        
        Search happens on child chunks, but parent chunks are retrieved for LLM.
        
        Args:
            text: Text content to split
            metadata: Optional metadata
            
        Returns:
            Dictionary with parent_chunks and child_chunks lists
        """
        if not text or not text.strip():
            return {"parent_chunks": [], "child_chunks": []}
        
        # Create parent chunks
        parent_texts = self.parent_splitter.split_text(text)
        parent_chunks = []
        child_chunks = []
        
        for parent_idx, parent_text in enumerate(parent_texts):
            parent_id = f"parent_{parent_idx}"
            
            # Detect section for parent
            parent_section = None
            if self.enable_section_detection:
                for line in parent_text.split('\n')[:5]:
                    parent_section = self._detect_section(line)
                    if parent_section:
                        break
            
            parent_metadata = {
                **(metadata or {}),
                "parent_id": parent_id,
                "chunk_type": "parent",
                "parent_index": parent_idx
            }
            if parent_section:
                parent_metadata["section"] = parent_section
            
            parent_chunk = {
                "text": parent_text,
                "chunk_id": parent_id,
                "metadata": parent_metadata
            }
            parent_chunks.append(parent_chunk)
            
            # Create child chunks from this parent
            child_texts = self.child_splitter.split_text(parent_text)
            current_section = parent_section
            
            for child_idx, child_text in enumerate(child_texts):
                child_id = f"{parent_id}_child_{child_idx}"
                
                # Detect section for child (might differ within parent)
                if self.enable_section_detection:
                    for line in child_text.split('\n')[:3]:
                        detected = self._detect_section(line)
                        if detected:
                            current_section = detected
                            break
                
                child_metadata = {
                    **(metadata or {}),
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "parent_index": parent_idx,
                    "child_index": child_idx
                }
                if current_section:
                    child_metadata["section"] = current_section
                
                child_chunk = {
                    "text": child_text,
                    "chunk_id": child_id,
                    "metadata": child_metadata
                }
                child_chunks.append(child_chunk)
        
        return {
            "parent_chunks": parent_chunks,
            "child_chunks": child_chunks
        }
    
    def generate_qa_pairs(
        self,
        chunk_text: str,
        num_questions: int = 3,
        llm_service: Optional[Any] = None
    ) -> List[str]:
        """
        Generate hypothetical questions for a chunk.
        
        Strategy: Index questions alongside content for better search alignment.
        User queries are questions; indexing questions improves matching.
        
        Args:
            chunk_text: Text content
            num_questions: Number of questions to generate
            llm_service: LLMService instance for generation
            
        Returns:
            List of generated questions
        """
        if not llm_service:
            # Fallback: Generate simple keyword-based questions
            return self._generate_simple_questions(chunk_text, num_questions)
        
        try:
            prompt = f"""Based on the following text, generate {num_questions} specific questions that this text can answer. 
Each question should be concise and directly answerable by the text.
Format: One question per line, no numbering.

Text:
{chunk_text[:1000]}

Questions:"""
            
            response = llm_service.generate_response(
                query=prompt,
                system_prompt="You are a helpful assistant that generates relevant questions from documentation."
            )
            
            # Parse questions from response
            questions = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('#')]
            return questions[:num_questions]
        
        except Exception as e:
            # Fallback to simple questions on error
            return self._generate_simple_questions(chunk_text, num_questions)
    
    def _generate_simple_questions(self, text: str, num: int = 3) -> List[str]:
        """
        Generate simple keyword-based questions as fallback.
        
        Args:
            text: Text content
            num: Number of questions
            
        Returns:
            List of simple questions
        """
        # Extract key phrases (simple heuristic)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        questions = []
        
        templates = [
            "What is {}?",
            "How does {} work?",
            "Tell me about {}",
            "Explain {}",
            "What are the details of {}?"
        ]
        
        # Extract first few words from sentences as topics
        for i, sentence in enumerate(sentences[:num]):
            words = sentence.split()[:5]
            if len(words) >= 2:
                topic = ' '.join(words)
                template = templates[i % len(templates)]
                questions.append(template.format(topic))
        
        return questions[:num]
    
    def chunk_with_qa_generation(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        llm_service: Optional[Any] = None,
        generate_qa: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and optionally generate QA pairs for each chunk.
        
        Args:
            text: Text content
            metadata: Optional metadata
            llm_service: LLMService for QA generation
            generate_qa: Whether to generate QA pairs
            
        Returns:
            List of chunks with optional generated questions
        """
        chunks = self.chunk_text(text, metadata)
        
        if generate_qa and chunks:
            for chunk in chunks:
                chunk["generated_questions"] = self.generate_qa_pairs(
                    chunk["text"],
                    num_questions=2,
                    llm_service=llm_service
                )
        
        return chunks