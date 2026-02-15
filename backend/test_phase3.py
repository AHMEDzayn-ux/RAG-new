"""
Manual test script for Embeddings Service

Demonstrates the embeddings functionality with real examples.
Run this after Phase 3 to verify everything works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.embeddings import EmbeddingsService


def test_embeddings():
    """Test embeddings service with sample text"""
    
    print("=" * 60)
    print("Phase 3: Embeddings Generation - Manual Test")
    print("=" * 60)
    print()
    
    # Initialize service
    print("Initializing EmbeddingsService...")
    service = EmbeddingsService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    print("✓ Service initialized")
    print()
    
    # Get model info
    info = service.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  - {key}: {value}")
    print()
    
    # Test single text embedding
    print("-" * 60)
    print("Test 1: Single Text Embedding")
    print("-" * 60)
    
    text = "What are the admission requirements for undergraduate studies?"
    print(f"Text: {text}")
    
    embedding = service.embed_text(text)
    print(f"✓ Generated embedding with {len(embedding)} dimensions")
    print(f"  First 5 values: {embedding[:5]}")
    print()
    
    # Test batch embeddings
    print("-" * 60)
    print("Test 2: Batch Embeddings")
    print("-" * 60)
    
    texts = [
        "What is the application deadline?",
        "How can I apply for financial aid?",
        "Can I transfer credits from another university?",
        "What documents do I need for admission?"
    ]
    
    print(f"Processing {len(texts)} texts...")
    embeddings = service.embed_batch(texts, show_progress=True)
    print(f"✓ Generated {len(embeddings)} embeddings")
    print()
    
    # Test document chunks embedding
    print("-" * 60)
    print("Test 3: Document Chunks with Metadata")
    print("-" * 60)
    
    documents = [
        {
            "text": "Students must have completed high school with a minimum GPA of 3.0.",
            "metadata": {"source": "admissions.pdf", "page": 1},
            "chunk_index": 0
        },
        {
            "text": "The application deadline is January 15th for regular decision.",
            "metadata": {"source": "admissions.pdf", "page": 2},
            "chunk_index": 1
        },
        {
            "text": "Financial aid is available including scholarships and grants.",
            "metadata": {"source": "financial_aid.pdf", "page": 1},
            "chunk_index": 0
        }
    ]
    
    print(f"Processing {len(documents)} document chunks...")
    embedded_docs = service.embed_documents(documents)
    print(f"✓ Embedded {len(embedded_docs)} documents")
    
    for i, doc in enumerate(embedded_docs):
        print(f"\n  Document {i}:")
        print(f"    Text: {doc['text'][:50]}...")
        print(f"    Source: {doc['metadata']['source']}")
        print(f"    Embedding dimensions: {len(doc['embedding'])}")
    print()
    
    # Test similarity computation
    print("-" * 60)
    print("Test 4: Semantic Similarity")
    print("-" * 60)
    
    query = "How do I apply for scholarships?"
    
    print(f"Query: {query}")
    print("\nFinding most similar document...")
    
    query_embedding = service.embed_text(query)
    
    similarities = []
    for doc in embedded_docs:
        similarity = service.compute_similarity(query_embedding, doc['embedding'])
        similarities.append((doc['text'], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\nResults (ranked by similarity):")
    for i, (text, score) in enumerate(similarities, 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     Text: {text[:60]}...")
    print()
    
    # Test semantic understanding
    print("-" * 60)
    print("Test 5: Semantic Understanding")
    print("-" * 60)
    
    pairs = [
        ("machine learning", "artificial intelligence"),
        ("machine learning", "cooking recipes"),
        ("university admission", "college application"),
        ("university admission", "car repair")
    ]
    
    print("Computing semantic similarities:\n")
    for text1, text2 in pairs:
        emb1 = service.embed_text(text1)
        emb2 = service.embed_text(text2)
        similarity = service.compute_similarity(emb1, emb2)
        
        relation = "similar" if similarity > 0.5 else "different"
        print(f"  '{text1}' vs '{text2}'")
        print(f"    Similarity: {similarity:.4f} ({relation})")
        print()
    
    print("=" * 60)
    print("✓ Phase 3 Complete: All functionality working!")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("  • Single text embedding generation")
    print("  • Batch processing for efficiency")
    print("  • Document chunks with metadata")
    print("  • Semantic similarity computation")
    print("  • Understanding semantic relationships")
    print()


if __name__ == "__main__":
    test_embeddings()
