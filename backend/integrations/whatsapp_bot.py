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
        rag = RAGPipeline(collection_name=collection_name)
        
        # Load the collection from vector store
        try:
            rag.vector_store.load_collection(collection_name)
            logger.info(f"Loaded collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {str(e)}")
            raise
        
        # Get response from RAG system
        result = rag.chat(
            message=message_text,
            conversation_history=history
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
