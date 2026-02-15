"""Test PDF loading with real file"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from services.document_loader import DocumentLoader

loader = DocumentLoader()
pdf_path = '../documents/university_guide.pdf'

print(f"Loading PDF: {pdf_path}")
print("=" * 60)

chunks = loader.load_and_chunk_pdf(pdf_path)

print(f"✓ Successfully loaded PDF!")
print(f"  Total chunks: {len(chunks)}")
print(f"  First chunk preview:")
print(f"  {chunks[0]['text'][:300]}...")
print()

stats = loader.get_chunk_stats(chunks)
print("Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")
