"""
LLM Service for RAG System
Handles text generation using Groq API with LangChain
"""

import logging
from typing import List, Dict, Optional, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class LLMService:
    """
    Service for generating responses using LLM.
    Uses Groq API with LangChain for text generation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the LLM Service.
        
        Args:
            api_key: Groq API key (defaults to settings)
            model_name: Model to use (defaults to settings)
            temperature: Sampling temperature (defaults to settings)
            max_tokens: Maximum tokens in response (defaults to settings)
        """
        self.api_key = api_key or settings.groq_api_key
        self.model_name = model_name or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        
        if not self.api_key:
            logger.warning("Groq API key not provided. LLM functionality will be limited.")
            self.llm = None
        else:
            # Initialize ChatGroq
            self.llm = ChatGroq(
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        logger.info(
            f"LLMService initialized with model={self.model_name}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )
    
    def generate_response(
        self,
        query: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response to a user query.
        
        Args:
            query: User's question or prompt
            context: List of relevant context passages (from RAG retrieval)
            system_prompt: Custom system prompt (optional)
            conversation_history: Previous conversation messages (optional)
        
        Returns:
            Generated response text
        """
        # Build system prompt
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        # Build messages
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Build user message with context
        user_message = self._build_user_message(query, context)
        messages.append(HumanMessage(content=user_message))
        
        logger.info(f"Generating response for query: {query[:100]}...")
        
        try:
            # Generate response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            logger.info(f"Generated response: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_rag_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        system_role: Optional[str] = None
    ) -> str:
        """
        Generate a RAG response using retrieved documents.
        
        Args:
            query: User's question
            retrieved_docs: List of retrieved documents with 'text' and 'metadata'
            system_role: Role description for the assistant (e.g., "university advisor")
        
        Returns:
            Generated response text
        """
        # Extract context from retrieved documents
        context = [doc.get("text", "") for doc in retrieved_docs]
        
        # Build system prompt for RAG
        system_prompt = self._get_rag_system_prompt(system_role)
        
        return self.generate_response(
            query=query,
            context=context,
            system_prompt=system_prompt
        )
    
    def generate_chat_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        context: Optional[List[str]] = None
    ) -> str:
        """
        Generate a conversational response with history.
        
        Args:
            query: Current user message
            conversation_history: Previous conversation messages
            context: Optional context from RAG retrieval
        
        Returns:
            Generated response text
        """
        system_prompt = self._get_chat_system_prompt()
        
        return self.generate_response(
            query=query,
            context=context,
            system_prompt=system_prompt,
            conversation_history=conversation_history
        )
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are a helpful AI assistant. Answer questions accurately and concisely based on the provided context. If you don't know the answer, say so."""
    
    def _get_rag_system_prompt(self, role: Optional[str] = None) -> str:
        """
        Get system prompt for RAG-based responses with strict grounding.
        
        Args:
            role: Optional role description (e.g., "customer support agent")
        """
        role_desc = role or "helpful customer support agent"
        
        return f"""You are a {role_desc}. You provide accurate, empathetic assistance based ONLY on verified information.

Response Format:
- Start with a SHORT, DIRECT answer (1-2 sentences) on the first line
- Then add a blank line
- Follow with detailed explanation with CITATIONS
- Format citations as [Source 1], [Source 2], etc.

STRICT "I DON'T KNOW" PROTOCOL (CRITICAL):
❌ NEVER make up information
❌ NEVER guess or speculate
❌ NEVER use knowledge outside the provided context
✅ If the context doesn't contain the answer, respond EXACTLY:
   "I don't have enough information in our documentation to answer that question accurately. I'd be happy to:
   1. Connect you with a human support agent who can help
   2. Research this further and get back to you
   Would you prefer option 1 or 2?"

CITATION & GROUNDING:
- ALWAYS cite sources when stating facts: "According to [Source 1], the refund policy..."
- Reference specific section/document names in citations
- If multiple sources say the same thing, cite all: [Source 1][Source 2]
- Format: [Context 1], [Context 2], etc. matching the numbered contexts provided

TONE & EMPATHY:
- Use clear, non-technical language (explain jargon if used)
- If user seems frustrated, acknowledge it first: "I understand this is frustrating..."
- Be warm but professional
- Use phrases like "I'd be happy to help", "Let me clarify", "Great question"

IMPORTANT - Context Distinctions:
- "Work experience" or "jobs" = paid employment with job titles, companies, dates ONLY
- Volunteer work, organizing committees, event coordination = extracurricular activities, NOT work experience
- When asked about work/jobs, ONLY mention paid employment positions
- When asked about general experience, you may include both employment and activities
- Always list ALL positions found in the context that match the query type"""
    
    def _get_chat_system_prompt(self) -> str:
        """Get system prompt for conversational chat with customer support focus."""
        return """You are a helpful, empathetic customer support agent engaged in conversation.

Response Format:
- Start with a SHORT, DIRECT answer (1-2 sentences) on the first line
- Then add a blank line
- Follow with detailed explanation with CITATIONS where applicable
- Format citations as [Context 1], [Context 2] when referencing provided documents

STRICT "I DON'T KNOW" PROTOCOL (CRITICAL):
❌ NEVER make up information or guess
❌ NEVER provide advice not found in the provided context
✅ If you don't know, say: "I don't have that information readily available. Let me connect you with someone who can help, or I can research this further. Which would you prefer?"

CITATIONS (when context provided):
- Cite your sources: "According to [Context 1], you can..."
- Be specific about which document/section you're referencing
- This builds trust and lets users verify information

EMPATHY & TONE:
- Read the user's emotional state (frustrated, confused, urgent)
- Acknowledge emotions first: "I understand how frustrating this must be..."
- Use clear, simple language - avoid technical jargon
- If user uses technical terms, mirror their level
- Be warm: "I'd be happy to help!", "Great question!", "Let me clarify that for you"
- Stay professional but friendly

CONVERSATION FLOW:
1. Maintain context from conversation history
2. Reference what was said earlier: "As we discussed earlier..."
3. Ask clarifying questions when needed: "Just to make sure I understand..."
4. Offer next steps or alternatives
5. End with: "Does this help? Is there anything else I can assist with?"

IMPORTANT - Context Distinctions:
- "Work experience" or "jobs" = paid employment positions ONLY (with job title, company, dates)
- Volunteer/organizing/committee work = extracurricular activities, NOT work experience
- When asked about work/jobs, focus ONLY on paid employment
- Always list ALL positions found in context that match the query type"""
    
    def _build_user_message(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Build the user message with optional context.
        
        Args:
            query: User's question
            context: Optional list of context passages
        
        Returns:
            Formatted user message
        """
        if not context:
            return query
        
        # Build message with context
        context_str = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(context)])
        
        message = f"""Context information:
{context_str}

Question: {query}

Answer based on the context provided above:"""
        
        return message
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "Groq"
        }
    
    def test_connection(self) -> bool:
        """
        Test the connection to Groq API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.llm.invoke([HumanMessage(content="Hello")])
            logger.info("Groq API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Groq API connection test failed: {str(e)}")
            return False
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate tokens for
        
        Returns:
            Estimated token count (rough approximation)
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def validate_input_size(self, messages: List[str]) -> bool:
        """
        Validate that input doesn't exceed token limits.
        
        Args:
            messages: List of message texts to validate
        
        Returns:
            True if within limits, False otherwise
        """
        total_text = " ".join(messages)
        estimated_tokens = self.estimate_tokens(total_text)
        
        # Groq models typically have 8k-32k context windows
        # We'll use a conservative limit
        max_input_tokens = 7000
        
        if estimated_tokens > max_input_tokens:
            logger.warning(
                f"Input size ({estimated_tokens} tokens) exceeds limit ({max_input_tokens})"
            )
            return False
        
        return True
