"""Test full PDF to embeddings pipeline"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from services.document_loader import DocumentLoader
from services.embeddings import EmbeddingsService

print("=" * 60)
print("Full Pipeline Test: PDF → Chunks → Embeddings")
print("=" * 60)
print()

# Step 1: Load and chunk PDF
print("Step 1: Loading PDF...")
loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
pdf_path = '../documents/university_guide.pdf'
chunks = loader.load_and_chunk_pdf(pdf_path)
print(f"✓ Loaded {len(chunks)} chunks from PDF")
print()

# Step 2: Generate embeddings
print("Step 2: Generating embeddings...")
embeddings_service = EmbeddingsService(device="cpu")
embedded_chunks = embeddings_service.embed_documents(chunks)
print(f"✓ Generated embeddings for {len(embedded_chunks)} chunks")
print()

# Step 3: Test semantic search
print("Step 3: Testing semantic search...")
query = "What is this document about?"
print(f"Query: '{query}'")
print()

query_embedding = embeddings_service.embed_text(query)

# Find most similar chunks
similarities = []
for chunk in embedded_chunks:
    similarity = embeddings_service.compute_similarity(
        query_embedding, 
        chunk['embedding']
    )
    similarities.append((chunk, similarity))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

print("Top 3 most relevant chunks:")
print("-" * 60)
for i, (chunk, score) in enumerate(similarities[:3], 1):
    print(f"\n{i}. Similarity: {score:.4f}")
    print(f"   Chunk {chunk['chunk_index']}: {chunk['text'][:150]}...")

print()
print("=" * 60)
print("✓ Full Pipeline Working Successfully!")
print("=" * 60)
