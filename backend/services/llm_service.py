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
        
        response = self.generate_response(
            query=query,
            context=context,
            system_prompt=system_prompt
        )
        
        # Clean citation phrases from response
        return self._clean_citation_phrases(response)
    
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
        
        response = self.generate_response(
            query=query,
            context=context,
            system_prompt=system_prompt,
            conversation_history=conversation_history
        )
        
        # Clean citation phrases from response
        return self._clean_citation_phrases(response)
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are a helpful AI assistant. Answer questions accurately and concisely based on the provided context. If you don't know the answer, say so."""
    
    def _get_rag_system_prompt(self, role: Optional[str] = None) -> str:
        """
        Get system prompt for RAG-based responses with strict grounding.
        
        Args:
            role: Optional role description (e.g., "assistant")
        """
        role_desc = role or "helpful assistant"
        
        return f"""You are a {role_desc}. 

📋 CONTENT RULES:
• Answer naturally, as if you know it personally
• Never say "According to", "Based on", or cite sources
• When context is provided, use it fully but speak naturally
• Always give complete answers—include all relevant details, projects, skills, or experiences
• Only omit information if truly irrelevant

✨ FORMATTING RULES (VERY IMPORTANT):
• Use bullet points (•) for lists - NOT numbered lists unless specific order matters
• Break long content into SHORT paragraphs (2-3 sentences max)
• Add line breaks between different topics/sections
• Use clear, scannable structure - avoid huge text blocks
• Make it visually appealing and easy to read at a glance

Examples:

❌ BAD (poor formatting):
"He has worked on several projects including a database with fast retrieval and ACID operations and a 4-bit Nano Processor using VHDL and Basys 3 Board and an indoor sports court booking system with SMS alerts and a disaster management platform and an e-commerce platform."

✅ GOOD (well formatted):
"He has worked on several projects:

• Database with fast retrieval and ACID-compliant operations
• 4-bit Nano Processor using VHDL and Basys 3 Board
• Indoor sports court booking system with SMS alerts
• Disaster management platform with real-time reporting
• E-commerce platform with optimized database design

Each project showcases his skills in database design and system development."

Always format responses for easy scanning and readability!"""
    
    def _get_chat_system_prompt(self) -> str:
        """Get system prompt for conversational chat."""
        return """You are a helpful assistant having a natural conversation.

📋 CONTENT RULES:
1. Answer naturally - no formal citations
2. NEVER write "According to [Context X]" or cite sources
3. NEVER say "Based on the context provided"
4. When context is provided: use it to answer, but write naturally
5. Include ALL relevant details - don't hide information
6. Only exclude info if truly not relevant

✨ FORMATTING RULES (CRITICAL FOR READABILITY):
• Use bullet points (•) for lists of items
• Break responses into SHORT paragraphs (2-3 sentences each)
• Add blank lines between different topics
• Make it easy to scan - avoid huge text blocks
• Use clear visual structure

Examples:

❌ BAD (wall of text):
"He has worked on several projects including database design and optimization for fast retrieval and ACID operations, a 4-bit Nano Processor using VHDL and Basys 3 Board for arithmetic operations, an indoor sports court booking system with convenient booking and SMS alerts and admin dashboard, a disaster management platform for reporting disasters and missing persons, and an e-commerce platform with database optimization."

✅ GOOD (well formatted):
"He has worked on several projects:

• Database system with fast retrieval and ACID-compliant operations
• 4-bit Nano Processor using VHDL and Basys 3 Board
• Indoor sports court booking system with SMS alerts and admin dashboard
• Disaster management platform for real-time reporting
• E-commerce platform with optimized database design

These projects demonstrate his expertise in database design, system development, and optimization."

Always format for easy reading and quick comprehension!"""
    
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
        
        # Build message with context - NO LABELS to avoid citations
        context_str = "\n\n---\n\n".join(context)
        
        message = f"""Information:

{context_str}

---

Question: {query}

IMPORTANT Instructions:
1. Use ALL the information provided above to give a COMPLETE and COMPREHENSIVE answer
2. Do NOT summarize or hide important details - include everything relevant
3. If there are multiple items (projects, skills, experiences, etc.), mention ALL of them
4. Answer naturally without mentioning sources or contexts
5. FORMAT properly: Use bullet points for lists, short paragraphs, and line breaks - NO huge text blocks!

Answer:"""
        
        return message
    
    def _clean_citation_phrases(self, response: str) -> str:
        """
        Remove citation phrases that mention context sources.
        
        Args:
            response: Raw LLM response
        
        Returns:
            Cleaned response without citation phrases
        """
        import re
        
        # Patterns to remove (case-insensitive)
        patterns = [
            r"According to \[Context \d+\],?\s*",
            r"Based on \[Context \d+\],?\s*",
            r"As mentioned in \[Context \d+\],?\s*",
            r"From \[Context \d+\],?\s*",
            r"In \[Context \d+\],?\s*",
            r"\[Context \d+\] states that\s*",
            r"\[Context \d+\] mentions that\s*",
            r"\[Context \d+\] indicates that\s*",
            r"\[Source \d+\],?\s*",
            r"According to the (provided )?context,?\s*",
            r"Based on the (provided )?context,?\s*",
            r"From the (provided )?context,?\s*",
        ]
        
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Clean up double spaces and leading spaces
        cleaned = re.sub(r"  +", " ", cleaned)
        cleaned = cleaned.strip()
        
        # Capitalize first letter if needed
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        logger.debug(f"Cleaned citations: '{response[:100]}...' → '{cleaned[:100]}...'")
        return cleaned
    
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
