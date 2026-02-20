"""
WhatsApp Message Formatter
Formats RAG responses for WhatsApp display with proper markdown
"""

from typing import List
import re


class WhatsAppFormatter:
    """
    Formats messages for WhatsApp Business API
    Converts RAG responses to WhatsApp-friendly markdown
    """
    
    # WhatsApp message length limit
    MAX_MESSAGE_LENGTH = 4096
    
    @staticmethod
    def format_response(content: str, sources: List[str] = None) -> str:
        """
        Format RAG response for WhatsApp
        
        Args:
            content: Response content from RAG
            sources: Optional list of source documents
            
        Returns:
            Formatted message string with WhatsApp markdown
        """
        # Clean up the response
        formatted = content.strip()
        
        # Add sources if provided
        if sources and len(sources) > 0:
            formatted += "\n\n📚 *Sources:*"
            for i, source in enumerate(sources[:3], 1):  # Limit to 3 sources
                # Extract just the filename if it's a path
                source_name = source.split('/')[-1] if '/' in source else source
                formatted += f"\n  {i}. {source_name}"
        
        return formatted
    
    @staticmethod
    def format_welcome_message(client_name: str = None) -> str:
        """
        Generate welcome message for new conversations
        
        Args:
            client_name: Optional client name for personalization
            
        Returns:
            Welcome message string
        """
        if client_name:
            return (
                f"👋 *Welcome to {client_name} Support!*\n\n"
                "I'm here to help you with any questions.\n\n"
                "Just send me your question, and I'll provide you with the information you need.\n\n"
                "💡 _Type 'help' for tips on how to ask questions._"
            )
        else:
            return (
                "👋 *Welcome!*\n\n"
                "I'm your AI assistant. How can I help you today?\n\n"
                "Just send me your question!\n\n"
                "💡 _Type 'help' for more information._"
            )
    
    @staticmethod
    def format_error_message(error_type: str = "general") -> str:
        """
        Generate user-friendly error messages
        
        Args:
            error_type: Type of error ('general', 'timeout', 'rate_limit')
            
        Returns:
            Error message string
        """
        if error_type == "timeout":
            return (
                "⏱️ *Request Timeout*\n\n"
                "Sorry, the request took too long to process.\n"
                "Please try again with a simpler question."
            )
        elif error_type == "rate_limit":
            return (
                "⚠️ *Too Many Requests*\n\n"
                "You're sending messages too quickly.\n"
                "Please wait a moment and try again."
            )
        else:
            return (
                "❌ *Something went wrong*\n\n"
                "I encountered an error while processing your request.\n"
                "Please try again or rephrase your question."
            )
    
    @staticmethod
    def format_help_message() -> str:
        """
        Generate help message with usage tips
        
        Returns:
            Help message string
        """
        return (
            "ℹ️ *How to Use This Bot*\n\n"
            "*Ask Questions:*\n"
            "Just type your question naturally. I'll search our knowledge base and provide the best answer.\n\n"
            "*Examples:*\n"
            "• How do I track my order?\n"
            "• What are your business hours?\n"
            "• Tell me about your products\n\n"
            "*Commands:*\n"
            "• *help* - Show this message\n"
            "• *clear* - Start a new conversation\n\n"
            "💬 I remember our conversation context, so feel free to ask follow-up questions!"
        )
    
    @staticmethod
    def format_clear_confirmation() -> str:
        """
        Generate confirmation message for cleared session
        
        Returns:
            Confirmation message string
        """
        return (
            "🔄 *Conversation Cleared*\n\n"
            "Your conversation history has been cleared.\n"
            "You can start a fresh conversation now!"
        )
    
    @staticmethod
    def split_long_message(message: str) -> List[str]:
        """
        Split message into chunks if it exceeds WhatsApp limit
        
        Args:
            message: Message to split
            
        Returns:
            List of message chunks
        """
        if len(message) <= WhatsAppFormatter.MAX_MESSAGE_LENGTH:
            return [message]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = message.split('\n\n')
        
        for para in paragraphs:
            # If adding this paragraph exceeds limit, save current chunk
            if len(current_chunk) + len(para) + 2 > WhatsAppFormatter.MAX_MESSAGE_LENGTH:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If single paragraph is too long, split by sentences
                if len(para) > WhatsAppFormatter.MAX_MESSAGE_LENGTH:
                    sentences = re.split(r'([.!?]\s+)', para)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > WhatsAppFormatter.MAX_MESSAGE_LENGTH:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += sentence
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add continuation markers
        if len(chunks) > 1:
            for i in range(len(chunks)):
                chunks[i] = f"*[Part {i+1}/{len(chunks)}]*\n\n{chunks[i]}"
        
        return chunks
    
    @staticmethod
    def sanitize_markdown(text: str) -> str:
        """
        Sanitize text to work with WhatsApp markdown
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # WhatsApp supports: *bold*, _italic_, ~strikethrough~, ```code```
        # Ensure proper formatting doesn't break
        
        # Fix common markdown issues
        text = text.replace('**', '*')  # Convert double asterisks to single
        text = text.replace('__', '_')  # Convert double underscores to single
        
        return text


# Create formatter instance
formatter = WhatsAppFormatter()
