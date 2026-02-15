"""
Pytest configuration and shared fixtures.
"""

import pytest
from pathlib import Path
import sys

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


@pytest.fixture
def sample_text():
    """Sample text for testing text processing"""
    return """This is a test document for RAG system testing.
    
This document contains multiple paragraphs to test text chunking functionality.
Each paragraph should be processed correctly by the document loader.

The chunking algorithm should split this text into manageable pieces while
maintaining context through overlapping sections between chunks.

This helps ensure that important information is not lost at chunk boundaries.
The RAG system will use these chunks to retrieve relevant context for queries."""


@pytest.fixture
def sample_faq_text():
    """Sample FAQ text for customer support testing"""
    return """
University Undergraduate Studies - Frequently Asked Questions

Q: What are the admission requirements?
A: Students must have completed high school with a minimum GPA of 3.0. 
SAT or ACT scores are required for all applicants.

Q: What is the application deadline?
A: The regular decision deadline is January 15th. Early decision applicants 
must submit by November 1st.

Q: What financial aid options are available?
A: We offer merit-based scholarships, need-based grants, and federal student loans.
"""


@pytest.fixture
def temp_pdf_path(tmp_path):
    """Temporary path for PDF files"""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    return pdf_dir


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {
        "source": "test_document.pdf",
        "client_id": "test_client",
        "category": "faq"
    }
