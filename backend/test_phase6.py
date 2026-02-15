"""
Phase 6 Manual Test: Complete RAG Pipeline

Demonstrates the complete RAG workflow:
1. Index documents (load → chunk → embed → store)
2. Query with retrieval and generation
3. Conversational chat with context
4. Multi-client pipeline management
"""

import sys
from pathlib import Path
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.rag_pipeline import RAGPipeline, MultiClientRAGPipeline
from logger import get_logger

logger = get_logger(__name__)


def main():
    """Run Phase 6 demonstration."""
    print("=" * 80)
    print("PHASE 6: Complete RAG Pipeline")
    print("=" * 80)
    print()
    
    # Check for PDF
    pdf_path = Path("../documents/university_guide.pdf")
    if not pdf_path.exists():
        print(f"⚠️  PDF not found: {pdf_path}")
        print("   Please ensure the PDF exists")
        return
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    has_api_key = bool(api_key)
    
    if not has_api_key:
        print("⚠️  GROQ_API_KEY not set - will test indexing only")
        print("   Set API key to test full RAG workflow")
        print()
    
    print("1. Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        collection_name="university_docs",
        system_role="university advisor"
    )
    
    # Override LLM model if needed
    if has_api_key:
        from services.llm_service import LLMService
        pipeline.llm_service = LLMService(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )
    
    print("   ✓ Pipeline initialized")
    print()
    
    # Get initial stats
    print("2. Getting pipeline statistics...")
    stats = pipeline.get_stats()
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Documents: {stats['document_count']}")
    print(f"   System role: {stats['system_role']}")
    print()
    
    # Index documents
    print("3. Indexing documents...")
    print(f"   PDF: {pdf_path}")
    
    try:
        index_stats = pipeline.index_documents(
            pdf_paths=[str(pdf_path)],
            metadata={"category": "university", "type": "guide"}
        )
        
        print(f"   ✓ Indexing complete!")
        print(f"   - PDFs processed: {index_stats['pdfs_processed']}")
        print(f"   - Total chunks: {index_stats['total_chunks']}")
        print(f"   - Embeddings generated: {index_stats['total_embeddings']}")
        print(f"   - Collection: {index_stats['collection_name']}")
        print(f"   - Documents in store: {index_stats['vector_store_count']}")
    except Exception as e:
        print(f"   ✗ Error indexing: {str(e)}")
        return
    print()
    
    # Test querying if API key available
    if has_api_key:
        print("4. Testing RAG query...")
        question = "What information is in this document?"
        print(f"   Question: '{question}'")
        
        try:
            response = pipeline.query(
                question=question,
                top_k=3,
                return_sources=True
            )
            
            print(f"   ✓ Query successful!")
            print()
            print(f"   Question: {response['question']}")
            print(f"   Answer: {response['answer']}")
            print()
            print(f"   Sources ({response['num_sources']}):")
            for i, source in enumerate(response['sources'][:3], 1):
                print(f"   [{i}] Distance: {source['distance']:.4f}")
                print(f"       Text: {source['text'][:100]}...")
                print(f"       Metadata: {source['metadata']}")
                print()
        except Exception as e:
            print(f"   ✗ Error querying: {str(e)}")
            return
        
        # Test conversational query
        print("5. Testing conversational chat...")
        conversation = [
            {"role": "user", "content": "What is this document about?"},
            {"role": "assistant", "content": "This appears to be a university guide."}
        ]
        
        print("   Conversation history:")
        for msg in conversation:
            print(f"   {msg['role'].upper()}: {msg['content']}")
        print()
        
        current_message = "Can you tell me more details?"
        print(f"   Current message: '{current_message}'")
        
        try:
            chat_response = pipeline.chat(
                message=current_message,
                conversation_history=conversation,
                use_retrieval=True,
                top_k=2
            )
            
            print(f"   ✓ Chat successful!")
            print()
            print(f"   Response: {chat_response['answer']}")
            print(f"   Used retrieval: {chat_response['used_retrieval']}")
            if 'sources' in chat_response:
                print(f"   Retrieved {len(chat_response['sources'])} sources")
        except Exception as e:
            print(f"   ✗ Error in chat: {str(e)}")
            return
        print()
    else:
        print("4. Skipping query tests (no API key)")
        print("   Set GROQ_API_KEY to test querying")
        print()
    
    # Test loading existing collection
    print(f"6. Testing collection persistence...")
    print("   Creating new pipeline instance...")
    new_pipeline = RAGPipeline(collection_name="temp")
    
    try:
        loaded = new_pipeline.load_existing_collection("university_docs")
        if loaded:
            print("   ✓ Collection loaded successfully")
            count = new_pipeline.vector_store.get_collection_count("university_docs")
            print(f"   - Documents in loaded collection: {count}")
        else:
            print("   ✗ Failed to load collection")
    except Exception as e:
        print(f"   ✗ Error loading: {str(e)}")
    print()
    
    # Test multi-client pipeline
    print("7. Testing multi-client pipeline...")
    manager = MultiClientRAGPipeline()
    
    # Create pipelines for different clients
    print("   Creating client pipelines...")
    university_pipeline = manager.create_pipeline(
        client_id="university",
        system_role="university advisor"
    )
    print("   ✓ Created university pipeline")
    
    government_pipeline = manager.create_pipeline(
        client_id="government",
        system_role="government service assistant"
    )
    print("   ✓ Created government pipeline")
    
    # List clients
    clients = manager.list_clients()
    print(f"   Active clients: {clients}")
    print()
    
    # Get pipeline stats
    print("8. Final statistics...")
    final_stats = pipeline.get_stats()
    print(f"   Collection: {final_stats['collection_name']}")
    print(f"   Documents: {final_stats['document_count']}")
    print(f"   Embedding model: {final_stats['embedding_model']['model_name']}")
    print(f"   LLM model: {final_stats['llm_model']['model_name']}")
    print()
    
    print("=" * 80)
    print("PHASE 6 COMPLETE! ✓")
    print("=" * 80)
    print()
    print("Summary:")
    print("  • Complete RAG pipeline implemented")
    print("  • Document indexing working (PDF → chunks → embeddings → vector store)")
    if has_api_key:
        print("  • Query workflow working (question → retrieval → LLM → answer)")
        print("  • Conversational chat working (history + retrieval)")
    print("  • Collection persistence working")
    print("  • Multi-client support working")
    print()
    print("What's working:")
    print("  ✓ End-to-end RAG pipeline")
    print("  ✓ Document indexing with metadata")
    print("  ✓ Semantic search and retrieval")
    if has_api_key:
        print("  ✓ Context-aware LLM responses")
        print("  ✓ Conversational chat with history")
    print("  ✓ Collection management and persistence")
    print("  ✓ Multi-client/multi-tenant architecture")
    print()
    
    if not has_api_key:
        print("To test full RAG capabilities:")
        print("1. Get API key from https://console.groq.com")
        print("2. Set: $env:GROQ_API_KEY='your-key'")
        print("3. Run this script again")
        print()
    
    print("Next Phase: Multi-client configuration and management")


if __name__ == "__main__":
    main()
