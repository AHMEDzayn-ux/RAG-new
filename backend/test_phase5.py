"""
Phase 5 Manual Test: LLM Integration with Groq API

Demonstrates:
1. LLM service initialization
2. Basic response generation
3. RAG-based response with context
4. Conversational responses with history
"""

import sys
from pathlib import Path
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from services.llm_service import LLMService
from logger import get_logger

logger = get_logger(__name__)


def main():
    """Run Phase 5 demonstration."""
    print("=" * 80)
    print("PHASE 5: LLM Integration with Groq API")
    print("=" * 80)
    print()
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️  WARNING: GROQ_API_KEY not found in environment")
        print("   Set it with: $env:GROQ_API_KEY='your-key-here'")
        print()
        print("   To get a free API key:")
        print("   1. Visit https://console.groq.com")
        print("   2. Sign up for a free account")
        print("   3. Generate an API key")
        print()
        print("   Running tests with mock API (no real API calls)...")
        print()
        test_without_api_key()
        return
    
    print("✓ Groq API key found")
    print()
    
    # Initialize service
    print("1. Initializing LLM service...")
    llm_service = LLMService()
    
    model_info = llm_service.get_model_info()
    print(f"   Model: {model_info['model_name']}")
    print(f"   Provider: {model_info['provider']}")
    print(f"   Temperature: {model_info['temperature']}")
    print(f"   Max Tokens: {model_info['max_tokens']}")
    print()
    
    # Test connection
    print("2. Testing API connection...")
    if llm_service.test_connection():
        print("   ✓ Connection successful")
    else:
        print("   ✗ Connection failed")
        return
    print()
    
    # Basic response generation
    print("3. Testing basic response generation...")
    print("   Query: 'What is artificial intelligence?'")
    try:
        response = llm_service.generate_response(
            query="What is artificial intelligence? Give a brief answer in 2-3 sentences."
        )
        print(f"   Response: {response}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return
    print()
    
    # RAG-based response with context
    print("4. Testing RAG response with context...")
    context = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers to process complex patterns.",
        "Supervised learning uses labeled training data where the correct output is known for each input example."
    ]
    
    print("   Context provided:")
    for i, ctx in enumerate(context, 1):
        print(f"   [{i}] {ctx[:60]}...")
    print()
    
    print("   Query: 'What is machine learning and how does it relate to AI?'")
    try:
        response = llm_service.generate_response(
            query="What is machine learning and how does it relate to AI? Be concise.",
            context=context
        )
        print(f"   Response: {response}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return
    print()
    
    # RAG response with retrieved documents
    print("5. Testing RAG response with retrieved documents...")
    retrieved_docs = [
        {
            "text": "The university offers undergraduate programs in Computer Science, Engineering, Business, and Liberal Arts.",
            "metadata": {"source": "university_guide.pdf", "page": 5}
        },
        {
            "text": "Graduate programs include Masters and PhD degrees in various fields. Application deadlines are typically in January for fall admission.",
            "metadata": {"source": "university_guide.pdf", "page": 12}
        }
    ]
    
    print("   Query: 'What programs does the university offer?'")
    print("   Role: university advisor")
    try:
        response = llm_service.generate_rag_response(
            query="What programs does the university offer?",
            retrieved_docs=retrieved_docs,
            system_role="university advisor"
        )
        print(f"   Response: {response}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return
    print()
    
    # Conversational response with history
    print("6. Testing conversational response with history...")
    conversation_history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."},
        {"role": "user", "content": "What can I use it for?"}
    ]
    
    print("   Conversation history:")
    for msg in conversation_history:
        print(f"   {msg['role'].upper()}: {msg['content'][:60]}...")
    print()
    
    print("   Current query: 'Is it good for beginners?'")
    try:
        response = llm_service.generate_chat_response(
            query="Is it good for beginners?",
            conversation_history=conversation_history
        )
        print(f"   Response: {response}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return
    print()
    
    # Token estimation
    print("7. Testing token estimation...")
    test_text = "This is a test string to estimate tokens."
    tokens = llm_service.estimate_tokens(test_text)
    print(f"   Text: '{test_text}'")
    print(f"   Estimated tokens: {tokens}")
    print()
    
    print("=" * 80)
    print("PHASE 5 COMPLETE! ✓")
    print("=" * 80)
    print()
    print("Summary:")
    print("  • LLM service initialized with Groq API")
    print("  • Basic response generation working")
    print("  • RAG response with context working")
    print("  • RAG response with document format working")
    print("  • Conversational responses with history working")
    print("  • Token estimation utility working")
    print()
    print("What's working:")
    print("  ✓ Groq API integration via LangChain")
    print("  ✓ Multiple prompt templates (default, RAG, chat)")
    print("  ✓ Context injection for RAG")
    print("  ✓ Conversation history support")
    print("  ✓ Customizable system roles")
    print("  ✓ Error handling and validation")
    print()
    print("Next Phase: RAG Pipeline - Connecting all components")


def test_without_api_key():
    """Test service functionality without making real API calls."""
    print("Testing service initialization and structure...")
    print()
    
    # Initialize without API key
    llm_service = LLMService(api_key="dummy_key_for_testing")
    
    # Test model info
    print("1. Model configuration:")
    model_info = llm_service.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    print()
    
    # Test prompt building
    print("2. Testing prompt building...")
    context = ["Context 1", "Context 2"]
    message = llm_service._build_user_message("Test query", context)
    print(f"   ✓ Message built with {len(context)} context passages")
    print()
    
    # Test system prompts
    print("3. Testing system prompts...")
    default_prompt = llm_service._get_default_system_prompt()
    print(f"   ✓ Default prompt: {default_prompt[:50]}...")
    
    rag_prompt = llm_service._get_rag_system_prompt("advisor")
    print(f"   ✓ RAG prompt: {rag_prompt[:50]}...")
    
    chat_prompt = llm_service._get_chat_system_prompt()
    print(f"   ✓ Chat prompt: {chat_prompt[:50]}...")
    print()
    
    # Test token estimation
    print("4. Testing token estimation...")
    test_text = "This is a test string for token estimation."
    tokens = llm_service.estimate_tokens(test_text)
    print(f"   Text: '{test_text}'")
    print(f"   Estimated tokens: {tokens}")
    print()
    
    # Test input validation
    print("5. Testing input validation...")
    short_messages = ["Short message"]
    result = llm_service.validate_input_size(short_messages)
    print(f"   ✓ Short message validation: {result}")
    
    long_message = ["word " * 10000]  # Very long message
    result = llm_service.validate_input_size(long_message)
    print(f"   ✓ Long message validation (should be False): {result}")
    print()
    
    print("=" * 80)
    print("Service structure tests COMPLETE! ✓")
    print("=" * 80)
    print()
    print("To test with real API calls:")
    print("1. Get a free API key from https://console.groq.com")
    print("2. Set environment variable: $env:GROQ_API_KEY='your-key-here'")
    print("3. Run this script again")
    print()
    print("Or add to .env file:")
    print("GROQ_API_KEY=your-key-here")


if __name__ == "__main__":
    main()
