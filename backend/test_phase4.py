"""
Phase 4 Manual Test: Vector Storage with FAISS

Demonstrates:
1. Creating a vector store collection
2. Adding document embeddings
3. Performing similarity search
4. Persisting and loading collections
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.document_loader import DocumentLoader
from services.embeddings import EmbeddingsService
from services.vector_store import VectorStoreService
from logger import get_logger

logger = get_logger(__name__)


def main():
    """Run Phase 4 demonstration."""
    print("=" * 80)
    print("PHASE 4: Vector Storage with FAISS")
    print("=" * 80)
    print()
    
    # Initialize services
    print("1. Initializing services...")
    doc_loader = DocumentLoader()
    embeddings_service = EmbeddingsService()
    vector_store = VectorStoreService()
    print("   ✓ Services initialized")
    print()
    
    # Load and process a PDF
    print("2. Loading PDF document...")
    pdf_path = Path("../documents/university_guide.pdf")
    
    if not pdf_path.exists():
        print(f"   ✗ PDF not found at {pdf_path}")
        print("   Please ensure the PDF file exists")
        return
    
    # Load and chunk
    chunks = doc_loader.load_and_chunk_pdf(str(pdf_path))
    print(f"   ✓ Loaded and chunked into {len(chunks)} chunks")
    print(f"   First chunk preview: {chunks[0]['text'][:100]}...")
    print()
    
    # Generate embeddings
    print("3. Generating embeddings...")
    texts = [chunk['text'] for chunk in chunks[:10]]  # Use first 10 chunks
    chunk_embeddings = embeddings_service.embed_batch(texts)
    print(f"   ✓ Generated {len(chunk_embeddings)} embeddings")
    print(f"   Embedding dimension: {len(chunk_embeddings[0])}")
    print()
    
    # Create collection and add documents
    print("4. Creating vector store collection...")
    collection_name = "university_docs"
    vector_store.create_collection(collection_name, embedding_dimension=384)
    print(f"   ✓ Created collection: {collection_name}")
    
    # Prepare metadatas
    metadatas = [{"source": "university_guide.pdf", "chunk_id": i} for i in range(len(texts))]
    
    # Add documents
    print("   Adding documents to collection...")
    vector_store.add_documents(
        collection_name=collection_name,
        documents=texts,
        embeddings=chunk_embeddings,
        metadatas=metadatas
    )
    count = vector_store.get_collection_count(collection_name)
    print(f"   ✓ Added {count} documents to collection")
    print()
    
    # Perform similarity search
    print("5. Performing similarity search...")
    query = "What programs are available?"
    print(f"   Query: '{query}'")
    
    # Generate query embedding
    query_embedding = embeddings_service.embed_text(query)
    
    # Search
    results = vector_store.query(
        collection_name=collection_name,
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(f"   ✓ Found {len(results['documents'][0])} similar documents")
    print()
    print("   Top 3 Results:")
    print("   " + "-" * 76)
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    ), 1):
        print(f"   Result {i} (Distance: {distance:.4f}):")
        print(f"   Source: {metadata.get('source', 'N/A')}, Chunk: {metadata.get('chunk_id', 'N/A')}")
        print(f"   Text: {doc[:150]}...")
        print()
    
    # Persist collection
    print("6. Persisting collection to disk...")
    vector_store.persist()
    print(f"   ✓ Collection persisted")
    print()
    
    # Test loading
    print("7. Testing collection loading...")
    new_store = VectorStoreService()
    new_store.load_collection(collection_name)
    loaded_count = new_store.get_collection_count(collection_name)
    print(f"   ✓ Loaded collection with {loaded_count} documents")
    print()
    
    # List all collections
    print("8. Listing all collections...")
    collections = new_store.list_collections()
    print(f"   Available collections: {collections}")
    print()
    
    print("=" * 80)
    print("PHASE 4 COMPLETE! ✓")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  • Created collection: {collection_name}")
    print(f"  • Stored {count} document chunks with embeddings")
    print(f"  • Performed similarity search")
    print(f"  • Persisted and reloaded collection")
    print()
    print("What's working:")
    print("  ✓ FAISS-based vector storage")
    print("  ✓ Multi-collection support")
    print("  ✓ Similarity search with distance scores")
    print("  ✓ Persistent storage and loading")
    print("  ✓ Metadata support")
    print()
    print("Next Phase: LLM Integration with Groq API")


if __name__ == "__main__":
    main()
