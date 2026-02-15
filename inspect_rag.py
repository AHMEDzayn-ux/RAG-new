"""
RAG System Inspection Tool
-------------------------
Visualize and verify your RAG system's operation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.vector_store import VectorStoreService
from services.embeddings import EmbeddingsService
from config import settings
from pathlib import Path
import json

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_section(text):
    print(f"\n{'─'*70}")
    print(f"  {text}")
    print('─'*70)

def inspect_rag_system(client_id=None):
    """Inspect the RAG system and show what's stored."""
    
    print_header("🔍 RAG SYSTEM INSPECTION")
    
    # Initialize services
    vector_store = VectorStoreService()
    embeddings = EmbeddingsService()
    
    # List all collections
    print_section("📚 Available Collections (Clients)")
    collections = vector_store.list_collections()
    
    if not collections:
        print("❌ No collections found! Upload some documents first.")
        return
    
    for i, coll_name in enumerate(collections, 1):
        count = vector_store.get_collection_count(coll_name)
        client_name = coll_name.replace("client_", "") if coll_name.startswith("client_") else coll_name
        print(f"  {i}. {client_name} ({coll_name})")
        print(f"     └─ {count} document chunks stored")
    
    # Select collection to inspect
    if client_id:
        target_collection = f"client_{client_id}"
        if target_collection not in collections:
            print(f"\n❌ Client '{client_id}' not found!")
            return
    else:
        if len(collections) == 1:
            target_collection = collections[0]
        else:
            print("\n🎯 Select a collection to inspect (or press Enter for first):")
            choice = input(f"   Enter number [1-{len(collections)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(collections):
                target_collection = collections[int(choice)-1]
            else:
                target_collection = collections[0]
    
    client_name = target_collection.replace("client_", "")
    
    print_header(f"📊 INSPECTING CLIENT: {client_name}")
    
    # Get collection stats
    print_section("📈 Collection Statistics")
    count = vector_store.get_collection_count(target_collection)
    print(f"  Total Chunks: {count}")
    print(f"  Collection Name: {target_collection}")
    print(f"  Embedding Dimension: {settings.embedding_dimension}")
    print(f"  Embedding Model: {settings.embedding_model}")
    
    # Show sample documents
    print_section("📄 Sample Document Chunks (First 5)")
    
    # Query with a generic embedding to get some documents
    sample_query = "information"
    query_embedding = embeddings.embed_text(sample_query)
    results = vector_store.query(
        collection_name=target_collection,
        query_embeddings=[query_embedding],
        n_results=min(5, count)
    )
    
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0], 1):
            metadata = results['metadatas'][0][i-1] if results.get('metadatas') else {}
            print(f"\n  Chunk {i}:")
            print(f"  Source: {metadata.get('source', 'N/A')}")
            print(f"  Page: {metadata.get('page', 'N/A')}")
            print(f"  Preview: {doc[:150]}...")
            print(f"  Length: {len(doc)} characters")
    else:
        print("  No documents found in collection")
    
    # Interactive RAG testing
    print_section("🧪 Test RAG Retrieval")
    print("  Enter questions to test document retrieval (or 'quit' to exit)")
    
    while True:
        print("\n" + "─"*70)
        question = input("  ❓ Your question: ").strip()
        
        if not question or question.lower() in ['quit', 'exit', 'q']:
            break
        
        print("\n  🔎 Searching vector store...")
        
        # Generate embedding for question
        q_embedding = embeddings.embed_text(question)
        
        # Retrieve relevant documents
        results = vector_store.query(
            collection_name=target_collection,
            query_embeddings=[q_embedding],
            n_results=3
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"\n  ✅ Found {len(results['documents'][0])} relevant chunks:\n")
            
            for i, doc in enumerate(results['documents'][0], 1):
                metadata = results['metadatas'][0][i-1]
                distance = results['distances'][0][i-1]
                similarity = 1 / (1 + distance)  # Convert distance to similarity score
                
                print(f"  [{i}] Relevance: {similarity:.2%} | Distance: {distance:.4f}")
                print(f"      Source: {metadata.get('source', 'N/A')}")
                print(f"      Page: {metadata.get('page', 'N/A')}")
                print(f"      Content: {doc[:200]}...")
                print()
        else:
            print("  ❌ No relevant documents found!")
    
    print_section("✅ Inspection Complete")
    print("\n  Summary:")
    print(f"  • Client: {client_name}")
    print(f"  • Total Chunks: {count}")
    print(f"  • Ready for RAG queries: {'Yes' if count > 0 else 'No'}")
    print()

def show_flow_diagram():
    """Show the RAG system flow diagram."""
    
    print_header("🔄 RAG SYSTEM FLOW")
    
    flow = """
    
    📤 DOCUMENT UPLOAD FLOW
    ─────────────────────────────────────────────────────────────
    
    1. User uploads PDF
       ↓
    2. PDF Loader extracts text
       ↓
    3. Text Chunker splits into chunks (500 chars, 100 overlap)
       ↓
    4. Embeddings Service generates vectors (384 dimensions)
       ↓
    5. FAISS Vector Store saves chunks + embeddings
       ↓
    6. Persist to disk (vector_stores/faiss/)
    
    
    💬 CHAT/QUERY FLOW
    ─────────────────────────────────────────────────────────────
    
    1. User asks question
       ↓
    2. Embeddings Service converts question → vector
       ↓
    3. FAISS searches for similar vectors (similarity search)
       ↓
    4. Retrieve top K most relevant chunks (default: 3)
       ↓
    5. LLM Service receives:
       • Question
       • Retrieved context chunks
       • Conversation history (if chat mode)
       ↓
    6. Groq LLM generates answer using context
       ↓
    7. Return answer + source citations to user
    
    
    🗄️ DATA STORAGE
    ─────────────────────────────────────────────────────────────
    
    Location: F:\\My projects\\RAG\\vector_stores\\faiss\\
    
    Files per client:
    • client_<name>.index        - FAISS vector index
    • client_<name>_metadata.pkl - Document metadata
    
    Content stored:
    • Original text chunks
    • 384-dimensional embeddings
    • Metadata (source file, page number, etc.)
    
    
    🔍 RETRIEVAL MECHANISM
    ─────────────────────────────────────────────────────────────
    
    Method: Cosine Similarity Search
    
    1. Question embedding: [0.23, 0.45, ..., 0.12] (384 dims)
    2. Compare with all stored embeddings
    3. Calculate similarity scores
    4. Return top K closest matches
    5. Lower distance = higher similarity = more relevant
    
    Example:
    Question: "What are the admission requirements?"
    
    Search Results:
    ├─ Chunk 1 (distance: 0.23) ← Most relevant
    │  "Admission requirements include..."
    ├─ Chunk 2 (distance: 0.45)
    │  "Application process for admissions..."
    └─ Chunk 3 (distance: 0.67)
       "Important admission deadlines..."
    
    """
    
    print(flow)

def main():
    """Main inspection function."""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--flow":
            show_flow_diagram()
            return
        elif sys.argv[1] == "--help":
            print("\nRAG System Inspection Tool")
            print("=" * 70)
            print("\nUsage:")
            print("  python inspect_rag.py              - Interactive inspection")
            print("  python inspect_rag.py <client_id>  - Inspect specific client")
            print("  python inspect_rag.py --flow       - Show system flow diagram")
            print("  python inspect_rag.py --help       - Show this help")
            print()
            return
        else:
            inspect_rag_system(client_id=sys.argv[1])
    else:
        # Show flow first
        show_flow_diagram()
        
        # Then do inspection
        input("\n📋 Press Enter to start interactive inspection...")
        inspect_rag_system()

if __name__ == "__main__":
    main()
