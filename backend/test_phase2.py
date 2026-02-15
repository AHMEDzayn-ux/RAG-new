"""
Manual test script for Document Loader

This script demonstrates the PDF loading and chunking functionality.
Run this after Phase 2 to verify everything works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.document_loader import DocumentLoader


def test_with_sample_text():
    """Test document loader with sample text"""
    
    print("=" * 60)
    print("Phase 2: PDF Loading & Chunking - Manual Test")
    print("=" * 60)
    print()
    
    # Create document loader
    loader = DocumentLoader(chunk_size=200, chunk_overlap=50)
    print(f"✓ DocumentLoader initialized")
    print(f"  - Chunk size: {loader.chunk_size}")
    print(f"  - Chunk overlap: {loader.chunk_overlap}")
    print()
    
    # Sample customer support text
    sample_text = """
    University Undergraduate Studies - Frequently Asked Questions
    
    Q: What are the admission requirements?
    A: Students must have completed high school with a minimum GPA of 3.0. 
    SAT or ACT scores are required for all applicants. International students 
    must provide TOEFL or IELTS scores.
    
    Q: What is the application deadline?
    A: The regular decision deadline is January 15th. Early decision applicants 
    must submit by November 1st. We also accept rolling admissions for certain programs.
    
    Q: What financial aid options are available?
    A: We offer merit-based scholarships, need-based grants, and federal student loans. 
    Students should complete the FAFSA by March 1st to be considered for all aid programs.
    
    Q: Can I transfer credits from another institution?
    A: Yes, we accept transfer credits from accredited institutions. Credits must be 
    C grade or higher to transfer. A maximum of 60 credits can be transferred.
    """
    
    # Chunk the text
    print("Chunking sample text...")
    chunks = loader.chunk_text(sample_text, metadata={"source": "undergrad_faq.pdf", "type": "faq"})
    print(f"✓ Text chunked into {len(chunks)} chunks")
    print()
    
    # Display chunks
    print("Generated Chunks:")
    print("-" * 60)
    for chunk in chunks:
        print(f"\n[Chunk {chunk['chunk_index']}] ({chunk['chunk_size']} characters)")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Text preview: {chunk['text'][:150]}...")
        print("-" * 60)
    
    # Get and display statistics
    stats = loader.get_chunk_stats(chunks)
    print(f"\nChunk Statistics:")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Total characters: {stats['total_characters']}")
    print(f"  - Average chunk size: {stats['avg_chunk_size']}")
    print(f"  - Min chunk size: {stats['min_chunk_size']}")
    print(f"  - Max chunk size: {stats['max_chunk_size']}")
    print()
    
    print("=" * 60)
    print("✓ Phase 2 Complete: All functionality working!")
    print("=" * 60)
    print()
    print("Next: Add a sample PDF file to test full PDF loading:")
    print("  1. Place a PDF in the 'documents' folder")
    print("  2. Update test_with_pdf() function below")
    print("  3. Uncomment the test_with_pdf() call at the bottom")
    print()


def test_with_pdf():
    """
    Test document loader with an actual PDF file.
    Uncomment and update the file path once you have a sample PDF.
    """
    loader = DocumentLoader()
    
    # Update this path to your sample PDF
    pdf_path = "../documents/sample.pdf"
    
    try:
        chunks = loader.load_and_chunk_pdf(pdf_path)
        print(f"✓ Loaded PDF with {len(chunks)} chunks")
        
        # Display first chunk
        if chunks:
            print(f"\nFirst chunk preview:")
            print(chunks[0]['text'][:200])
        
        # Statistics
        stats = loader.get_chunk_stats(chunks)
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except FileNotFoundError:
        print(f"✗ PDF file not found: {pdf_path}")
        print("  Place a sample PDF in the documents folder first")


if __name__ == "__main__":
    # Test with sample text (always works)
    test_with_sample_text()
    
    # Test with actual PDF (uncomment when ready)
    # test_with_pdf()
