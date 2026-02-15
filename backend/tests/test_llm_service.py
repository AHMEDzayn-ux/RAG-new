"""
Tests for LLM Service
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage

from services.llm_service import LLMService


@pytest.fixture
def mock_groq_response():
    """Create a mock Groq API response."""
    mock_response = Mock(spec=AIMessage)
    mock_response.content = "This is a test response from the LLM."
    return mock_response


@pytest.fixture
def llm_service():
    """Create an LLMService instance for testing."""
    with patch('services.llm_service.ChatGroq'):
        service = LLMService(api_key="test_api_key")
        return service


@pytest.fixture
def sample_context():
    """Sample context passages for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular language for data science.",
        "Neural networks are inspired by biological neurons."
    ]


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a branch of AI."},
        {"role": "user", "content": "Can you explain more?"}
    ]


@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved documents for RAG."""
    return [
        {
            "text": "Machine learning algorithms learn from data.",
            "metadata": {"source": "ml_intro.pdf", "page": 1}
        },
        {
            "text": "Supervised learning uses labeled training data.",
            "metadata": {"source": "ml_intro.pdf", "page": 2}
        }
    ]


class TestLLMServiceInitialization:
    """Test LLM service initialization."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch('services.llm_service.ChatGroq') as mock_groq:
            service = LLMService(api_key="test_key")
            assert service.api_key == "test_key"
            mock_groq.assert_called_once()
    
    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch('services.llm_service.ChatGroq') as mock_groq:
            service = LLMService(
                api_key="test_key",
                model_name="llama3-8b-8192",
                temperature=0.5,
                max_tokens=512
            )
            
            assert service.model_name == "llama3-8b-8192"
            assert service.temperature == 0.5
            assert service.max_tokens == 512
            
            # Verify ChatGroq was initialized with correct params
            call_kwargs = mock_groq.call_args[1]
            assert call_kwargs["model_name"] == "llama3-8b-8192"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 512
    
    def test_init_without_api_key_warns(self):
        """Test initialization without API key sets llm to None."""
        with patch('services.llm_service.ChatGroq') as mock_groq:
            # Patch settings to have empty API key
            with patch('services.llm_service.settings') as mock_settings:
                mock_settings.groq_api_key = ""
                mock_settings.llm_model = "llama3-70b-8192"
                mock_settings.llm_temperature = 0.7
                mock_settings.llm_max_tokens = 1024
                
                service = LLMService(api_key="")
                # Should not initialize ChatGroq when no API key
                mock_groq.assert_not_called()
                # LLM should be None
                assert service.llm is None


class TestGenerateResponse:
    """Test response generation."""
    
    def test_generate_response_basic(self, llm_service, mock_groq_response):
        """Test basic response generation."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        response = llm_service.generate_response("What is AI?")
        
        assert response == "This is a test response from the LLM."
        assert llm_service.llm.invoke.called
    
    def test_generate_response_with_context(self, llm_service, mock_groq_response, sample_context):
        """Test response generation with context."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        response = llm_service.generate_response(
            query="What is machine learning?",
            context=sample_context
        )
        
        assert response == "This is a test response from the LLM."
        
        # Verify context was included in the message
        call_args = llm_service.llm.invoke.call_args[0][0]
        user_message = call_args[-1].content
        assert "Context" in user_message
        assert sample_context[0] in user_message
    
    def test_generate_response_with_custom_system_prompt(self, llm_service, mock_groq_response):
        """Test response generation with custom system prompt."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        custom_prompt = "You are a math tutor."
        response = llm_service.generate_response(
            query="Explain calculus",
            system_prompt=custom_prompt
        )
        
        assert response == "This is a test response from the LLM."
        
        # Verify custom system prompt was used
        call_args = llm_service.llm.invoke.call_args[0][0]
        system_message = call_args[0].content
        assert system_message == custom_prompt
    
    def test_generate_response_with_conversation_history(
        self, llm_service, mock_groq_response, sample_conversation_history
    ):
        """Test response generation with conversation history."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        response = llm_service.generate_response(
            query="Tell me more",
            conversation_history=sample_conversation_history
        )
        
        assert response == "This is a test response from the LLM."
        
        # Verify conversation history was included
        call_args = llm_service.llm.invoke.call_args[0][0]
        # Should have system message + history messages + current query
        assert len(call_args) > len(sample_conversation_history)
    
    def test_generate_response_error_handling(self, llm_service):
        """Test error handling in response generation."""
        llm_service.llm.invoke = Mock(side_effect=Exception("API Error"))
        
        with pytest.raises(Exception, match="API Error"):
            llm_service.generate_response("Test query")


class TestGenerateRAGResponse:
    """Test RAG-specific response generation."""
    
    def test_generate_rag_response(self, llm_service, mock_groq_response, sample_retrieved_docs):
        """Test RAG response generation."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        response = llm_service.generate_rag_response(
            query="What is supervised learning?",
            retrieved_docs=sample_retrieved_docs
        )
        
        assert response == "This is a test response from the LLM."
        
        # Verify retrieved docs were used as context
        call_args = llm_service.llm.invoke.call_args[0][0]
        user_message = call_args[-1].content
        assert sample_retrieved_docs[0]["text"] in user_message
        assert sample_retrieved_docs[1]["text"] in user_message
    
    def test_generate_rag_response_with_role(self, llm_service, mock_groq_response, sample_retrieved_docs):
        """Test RAG response with custom role."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        response = llm_service.generate_rag_response(
            query="What programs are available?",
            retrieved_docs=sample_retrieved_docs,
            system_role="university advisor"
        )
        
        assert response == "This is a test response from the LLM."
        
        # Verify role was used in system prompt
        call_args = llm_service.llm.invoke.call_args[0][0]
        system_message = call_args[0].content
        assert "university advisor" in system_message


class TestGenerateChatResponse:
    """Test chat response generation."""
    
    def test_generate_chat_response(
        self, llm_service, mock_groq_response, sample_conversation_history, sample_context
    ):
        """Test chat response generation."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        response = llm_service.generate_chat_response(
            query="What about deep learning?",
            conversation_history=sample_conversation_history,
            context=sample_context
        )
        
        assert response == "This is a test response from the LLM."
        
        # Verify conversation history and context were used
        call_args = llm_service.llm.invoke.call_args[0][0]
        assert len(call_args) > 2  # System + history + current


class TestPromptBuilding:
    """Test prompt building methods."""
    
    def test_build_user_message_without_context(self, llm_service):
        """Test building user message without context."""
        message = llm_service._build_user_message("What is AI?")
        assert message == "What is AI?"
    
    def test_build_user_message_with_context(self, llm_service, sample_context):
        """Test building user message with context."""
        message = llm_service._build_user_message("What is AI?", sample_context)
        
        assert "Context" in message
        assert "What is AI?" in message
        assert sample_context[0] in message
        assert sample_context[1] in message
    
    def test_get_default_system_prompt(self, llm_service):
        """Test getting default system prompt."""
        prompt = llm_service._get_default_system_prompt()
        assert "helpful" in prompt.lower()
        assert "assistant" in prompt.lower()
    
    def test_get_rag_system_prompt_default(self, llm_service):
        """Test getting RAG system prompt with default role."""
        prompt = llm_service._get_rag_system_prompt()
        assert "context" in prompt.lower()
        assert "helpful assistant" in prompt.lower()
    
    def test_get_rag_system_prompt_custom_role(self, llm_service):
        """Test getting RAG system prompt with custom role."""
        prompt = llm_service._get_rag_system_prompt("university advisor")
        assert "university advisor" in prompt.lower()
        assert "context" in prompt.lower()
    
    def test_get_chat_system_prompt(self, llm_service):
        """Test getting chat system prompt."""
        prompt = llm_service._get_chat_system_prompt()
        assert "conversation" in prompt.lower()
        assert "friendly" in prompt.lower()


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_get_model_info(self, llm_service):
        """Test getting model information."""
        info = llm_service.get_model_info()
        
        assert "model_name" in info
        assert "temperature" in info
        assert "max_tokens" in info
        assert "provider" in info
        assert info["provider"] == "Groq"
    
    def test_test_connection_success(self, llm_service, mock_groq_response):
        """Test successful connection test."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        result = llm_service.test_connection()
        assert result is True
    
    def test_test_connection_failure(self, llm_service):
        """Test failed connection test."""
        llm_service.llm.invoke = Mock(side_effect=Exception("Connection failed"))
        
        result = llm_service.test_connection()
        assert result is False
    
    def test_estimate_tokens(self, llm_service):
        """Test token estimation."""
        text = "This is a test string with multiple words."
        tokens = llm_service.estimate_tokens(text)
        
        # Rough estimate: ~4 chars per token
        expected = len(text) // 4
        assert tokens == expected
    
    def test_estimate_tokens_empty(self, llm_service):
        """Test token estimation with empty string."""
        tokens = llm_service.estimate_tokens("")
        assert tokens == 0
    
    def test_validate_input_size_within_limits(self, llm_service):
        """Test input size validation within limits."""
        messages = ["Short message", "Another short message"]
        result = llm_service.validate_input_size(messages)
        assert result is True
    
    def test_validate_input_size_exceeds_limits(self, llm_service):
        """Test input size validation exceeding limits."""
        # Create a very long message
        long_message = "word " * 10000  # ~40k characters, ~10k tokens
        result = llm_service.validate_input_size([long_message])
        assert result is False


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    def test_rag_workflow(self, llm_service, mock_groq_response, sample_retrieved_docs):
        """Test complete RAG workflow."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        # Simulate RAG: retrieve docs -> generate response
        query = "Explain supervised learning"
        response = llm_service.generate_rag_response(
            query=query,
            retrieved_docs=sample_retrieved_docs,
            system_role="AI tutor"
        )
        
        assert response is not None
        assert isinstance(response, str)
        assert llm_service.llm.invoke.called
    
    def test_conversational_workflow(
        self, llm_service, mock_groq_response, sample_conversation_history
    ):
        """Test conversational workflow with history."""
        llm_service.llm.invoke = Mock(return_value=mock_groq_response)
        
        # Simulate conversation with history
        response = llm_service.generate_chat_response(
            query="Can you give an example?",
            conversation_history=sample_conversation_history
        )
        
        assert response is not None
        assert isinstance(response, str)
        
        # Verify history was considered
        call_args = llm_service.llm.invoke.call_args[0][0]
        assert len(call_args) > 2  # More than just system + user message
