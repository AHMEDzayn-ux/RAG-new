"""
Session Manager for WhatsApp Conversations
Manages conversation history and context for WhatsApp users
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import threading


class SessionManager:
    """
    Manages conversation sessions for WhatsApp users
    Stores conversation history and handles session timeouts
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        """
        Initialize the session manager
        
        Args:
            session_timeout_minutes: Minutes of inactivity before session expires
        """
        self.sessions: Dict[str, List[Dict]] = defaultdict(list)
        self.last_activity: Dict[str, datetime] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.lock = threading.Lock()
    
    def add_message(self, phone_number: str, role: str, content: str) -> None:
        """
        Add a message to the conversation history
        
        Args:
            phone_number: WhatsApp phone number (unique identifier)
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        with self.lock:
            # Clean up old sessions first
            self._cleanup_expired_sessions()
            
            # Add new message
            self.sessions[phone_number].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now()
            })
            
            # Update last activity
            self.last_activity[phone_number] = datetime.now()
            
            # Limit history to last 20 messages to prevent memory issues
            if len(self.sessions[phone_number]) > 20:
                self.sessions[phone_number] = self.sessions[phone_number][-20:]
    
    def get_history(self, phone_number: str, max_messages: int = 10) -> List[Dict]:
        """
        Get conversation history for a user
        
        Args:
            phone_number: WhatsApp phone number
            max_messages: Maximum number of recent messages to return
            
        Returns:
            List of message dictionaries with role and content
        """
        with self.lock:
            # Check if session has expired
            if phone_number in self.last_activity:
                if datetime.now() - self.last_activity[phone_number] > self.session_timeout:
                    # Session expired, clear history
                    self._clear_session(phone_number)
                    return []
            
            # Return recent messages (without timestamp for API compatibility)
            messages = self.sessions.get(phone_number, [])
            return [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in messages[-max_messages:]
            ]
    
    def clear_session(self, phone_number: str) -> None:
        """
        Manually clear a user's session
        
        Args:
            phone_number: WhatsApp phone number
        """
        with self.lock:
            self._clear_session(phone_number)
    
    def _clear_session(self, phone_number: str) -> None:
        """Internal method to clear session without lock"""
        if phone_number in self.sessions:
            del self.sessions[phone_number]
        if phone_number in self.last_activity:
            del self.last_activity[phone_number]
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions to free memory"""
        now = datetime.now()
        expired = [
            phone for phone, last_time in self.last_activity.items()
            if now - last_time > self.session_timeout
        ]
        
        for phone in expired:
            self._clear_session(phone)
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        with self.lock:
            self._cleanup_expired_sessions()
            return len(self.sessions)


# Global session manager instance
session_manager = SessionManager(session_timeout_minutes=30)
