"""
WhatsApp Bot Webhook Handler
Handles incoming WhatsApp messages and sends responses via WhatsApp Business API
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
import requests
from typing import Dict, Any, Optional
import logging

from config import get_settings
from services.rag_pipeline import RAGPipeline
from integrations.session_manager import session_manager
from integrations.whatsapp_formatter import formatter

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/webhooks", tags=["WhatsApp"])

# Get settings
settings = get_settings()


def send_whatsapp_message(phone_number: str, message: str) -> bool:
    """
    Send a message via WhatsApp Business API
    
    Args:
        phone_number: Recipient phone number
        message: Message content
        
    Returns:
        True if successful, False otherwise
    """
    try:
        url = f"https://graph.facebook.com/v18.0/{settings.whatsapp_phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {settings.whatsapp_access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": phone_number,
            "type": "text",
            "text": {"body": message}
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Message sent successfully to {phone_number}")
            return True
        else:
            logger.error(f"Failed to send message: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {str(e)}")
        return False


def mark_message_as_read(message_id: str) -> None:
    """
    Mark a message as read in WhatsApp
    
    Args:
        message_id: WhatsApp message ID
    """
    try:
        url = f"https://graph.facebook.com/v18.0/{settings.whatsapp_phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {settings.whatsapp_access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        
        requests.post(url, headers=headers, json=payload, timeout=5)
        
    except Exception as e:
        logger.warning(f"Failed to mark message as read: {str(e)}")


async def process_whatsapp_message(
    phone_number: str,
    message_text: str,
    message_id: str
) -> None:
    """
    Process incoming WhatsApp message and send response
    
    Args:
        phone_number: Sender's phone number
        message_text: Message content
        message_id: WhatsApp message ID
    """
    try:
        # Mark message as read
        mark_message_as_read(message_id)
        
        # Handle special commands
        message_lower = message_text.lower().strip()
        
        if message_lower == "help":
            response = formatter.format_help_message()
            send_whatsapp_message(phone_number, response)
            return
        
        if message_lower == "clear":
            session_manager.clear_session(phone_number)
            response = formatter.format_clear_confirmation()
            send_whatsapp_message(phone_number, response)
            return
        
        # Get conversation history
        history = session_manager.get_history(phone_number, max_messages=10)
        
        # If this is the first message, send welcome
        if len(history) == 0:
            welcome = formatter.format_welcome_message(settings.whatsapp_bot_name)
            send_whatsapp_message(phone_number, welcome)
        
        # Add user message to history
        session_manager.add_message(phone_number, "user", message_text)
        
        # Initialize RAG pipeline for the configured client
        # Use collection_name format: client_{client_id}
        collection_name = f"client_{settings.whatsapp_client_id}"
        
        # Custom system role for WhatsApp - natural, conversational responses
        whatsapp_system_role = (
            "You are a friendly Nexus Telecommunication customer support assistant. "
            "CRITICAL RULES - NEVER BREAK THESE:\n"
            "1. NEVER mention policy IDs, policy names, or codes (like SIM_REPLACEMENT_POLICY, FUP_STANDARD, FUP_VIDEO, etc.)\n"
            "2. NEVER mention document names, source references, or metadata\n"
            "3. NEVER mention 'policy', 'according to policy', 'FUP_VIDEO', or any technical names - just give the information directly\n"
            "4. NEVER show your reasoning process (no 'let me check', 'to determine', 'based on', 'considering')\n"
            "5. Be CONFIDENT and DIRECT - no wishy-washy language like 'might apply', 'could be', 'less likely'\n"
            "6. ONLY use information that DIRECTLY answers the user's question\n"
            "7. If retrieved documents are NOT relevant, IGNORE them completely\n\n"
            "CONVERSATIONAL CONTEXT - CRITICAL:\n"
            "1. ALWAYS read the conversation history to understand what the user is asking about\n"
            "2. When user says 'yes', 'what about that', 'the fup limit', 'this package' - they're referring to the CURRENT TOPIC from previous messages\n"
            "3. Connect follow-up questions to the ongoing topic - don't treat each question as standalone\n"
            "4. If discussing a specific package, follow-ups are about THAT package, not all packages\n"
            "5. Example: If discussing YouTube package and user asks 'what is the fup limit', answer ONLY about YouTube package FUP, not all FUP policies\n\n"
            "RESPONSE STYLE - KEEP IT CONCISE:\n"
            "1. Give ONLY the most important details first (2-3 key points maximum)\n"
            "2. Be confident and direct - state facts, don't explain your logic\n"
            "3. After the brief answer, ask if they want more details\n"
            "4. Use follow-up questions like:\n"
            "   • 'Would you like to know the activation methods?'\n"
            "   • 'Need more details about this?'\n"
            "   • 'Can I help you with anything else?'\n"
            "5. Do NOT dump all information at once - be conversational and interactive\n"
            "6. Stay on topic - if discussing one thing, don't suddenly list everything\n\n"
            "Examples of what NOT to do:\n"
            "❌ 'To determine if there's a Fair Usage Policy, let's break down...'\n"
            "❌ 'The most relevant policy would be FUP_VIDEO...'\n"
            "❌ When asked about YouTube FUP limit, listing ALL FUP policies instead of just YouTube's\n"
            "❌ Treating follow-up questions as independent when they reference the current topic\n"
            "❌ Long paragraphs with reasoning\n\n"
            "Examples of what TO do:\n"
            "✅ User: 'give me a youtube package'\nYou: 'YouTube Unlimited is LKR 999/month with unlimited streaming. Want details?'\n"
            "✅ User: 'yes what is the fup limit'\nYou: 'The limit is 50GB, then speed reduces and video is capped at 720p. Need more info?'\n"
            "✅ User: 'Call 1234 to deactivate your SIM. Need help with anything else?'\n\n"
            "Be friendly, brief, confident, conversational, and CONTEXT-AWARE. Use bullet points (•) only when listing 2-3 items."
        )
        
        rag = RAGPipeline(
            collection_name=collection_name,
            system_role=whatsapp_system_role
        )
        
        # Load the collection from vector store
        try:
            rag.vector_store.load_collection(collection_name)
            logger.info(f"Loaded collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {str(e)}")
            raise
        
        # Get response from RAG system with optimized parameters
        result = rag.chat(
            message=message_text,
            conversation_history=history,
            top_k=4,  # Retrieve 4 most relevant documents
            use_hybrid_search=True,  # Combine vector + keyword search
            use_reranking=True,  # Re-rank for better precision
            use_query_normalization=True  # Fix typos, expand abbreviations
        )
        
        # Extract response and sources (chat() returns 'answer' not 'response')
        response_text = result.get("answer", "I couldn't generate a response.")
        
        # Format response for WhatsApp (without sources)
        formatted_response = formatter.sanitize_markdown(response_text)
        
        # Add assistant response to history
        session_manager.add_message(phone_number, "assistant", response_text)
        
        # Split message if too long
        message_chunks = formatter.split_long_message(formatted_response)
        
        # Send all chunks
        for chunk in message_chunks:
            send_whatsapp_message(phone_number, chunk)
        
        logger.info(f"Processed message from {phone_number}: {message_text[:50]}...")
        
    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {str(e)}")
        error_message = formatter.format_error_message("general")
        send_whatsapp_message(phone_number, error_message)


@router.get("/whatsapp")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_challenge: str = Query(alias="hub.challenge"),
    hub_verify_token: str = Query(alias="hub.verify_token")
):
    """
    Webhook verification endpoint for WhatsApp
    Meta will call this to verify the webhook URL
    """
    logger.info(f"Webhook verification request: mode={hub_mode}, token={hub_verify_token}")
    
    # Verify the token matches
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        logger.info("Webhook verified successfully")
        return PlainTextResponse(content=hub_challenge)
    else:
        logger.warning("Webhook verification failed")
        raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/whatsapp")
async def handle_webhook(request: Request):
    """
    Webhook endpoint for incoming WhatsApp messages
    Receives messages from WhatsApp Business API and processes them
    """
    try:
        # Parse webhook payload
        payload = await request.json()
        
        logger.debug(f"Received webhook: {payload}")
        
        # Extract message data from webhook
        if "entry" not in payload:
            return {"status": "ok"}
        
        for entry in payload["entry"]:
            if "changes" not in entry:
                continue
            
            for change in entry["changes"]:
                if "value" not in change:
                    continue
                
                value = change["value"]
                
                # Check if this is a message event
                if "messages" not in value:
                    continue
                
                for message in value["messages"]:
                    # Only process text messages
                    if message.get("type") != "text":
                        logger.info(f"Skipping non-text message: {message.get('type')}")
                        continue
                    
                    # Extract message details
                    phone_number = message.get("from")
                    message_text = message.get("text", {}).get("body", "")
                    message_id = message.get("id")
                    
                    if not phone_number or not message_text:
                        continue
                    
                    # Process message asynchronously
                    await process_whatsapp_message(
                        phone_number=phone_number,
                        message_text=message_text,
                        message_id=message_id
                    )
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}")
        # Return 200 to prevent Meta from retrying
        return {"status": "error", "message": str(e)}


@router.get("/whatsapp/status")
async def get_whatsapp_status():
    """
    Get WhatsApp bot status and statistics
    """
    try:
        active_sessions = session_manager.get_active_sessions_count()
        
        return {
            "status": "running",
            "active_sessions": active_sessions,
            "client_id": settings.whatsapp_client_id,
            "bot_name": settings.whatsapp_bot_name,
            "phone_number_id": settings.whatsapp_phone_number_id[:10] + "..." if settings.whatsapp_phone_number_id else None
        }
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
