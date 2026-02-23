"""
Voice Chat API - Real-time voice interaction with RAG system
Handles speech-to-text, RAG processing, and text-to-speech
"""

import io
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import httpx

from config import get_settings
from services.rag_pipeline import RAGPipeline
from logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/voice", tags=["voice"])


class VoiceChatRequest(BaseModel):
    """Request for voice chat with conversation history"""
    client_id: str
    conversation_history: list = []
    voice: str = "alloy"  # Voice options: alloy, echo, fable, onyx, nova, shimmer


async def transcribe_audio(audio_file: UploadFile) -> str:
    """
    Transcribe audio using Groq Whisper API
    
    Args:
        audio_file: Audio file (webm, mp3, wav, etc.)
        
    Returns:
        Transcribed text
    """
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Use Groq Whisper API for transcription
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {
                "file": (audio_file.filename or "audio.webm", audio_data, audio_file.content_type),
            }
            data = {
                "model": "whisper-large-v3",
                "language": "en",
                "response_format": "json"
            }
            headers = {
                "Authorization": f"Bearer {settings.groq_api_key}"
            }
            
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Whisper API error: {response.text}")
                raise HTTPException(status_code=500, detail="Speech transcription failed")
            
            result = response.json()
            transcription = result.get("text", "")
            
            logger.info(f"Transcription successful: {transcription[:50]}...")
            return transcription
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")


async def generate_speech(text: str, voice: str = "alloy") -> bytes:
    """
    Generate speech from text using OpenAI TTS API
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        
    Returns:
        Audio bytes (MP3)
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "tts-1",  # or tts-1-hd for higher quality
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",
                    "speed": 1.0
                }
            )
            
            if response.status_code != 200:
                logger.error(f"TTS API error: {response.text}")
                raise HTTPException(status_code=500, detail="Speech generation failed")
            
            return response.content
            
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@router.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Transcribe audio to text
    
    Returns:
        JSON with transcribed text
    """
    transcription = await transcribe_audio(audio)
    return {"text": transcription}


@router.post("/chat")
async def voice_chat(
    audio: UploadFile = File(..., description="Audio file with user's question"),
    client_id: str = Form(default="default")
):
    """
    Voice chat with text response (client-side TTS):
    1. Transcribe user's audio using Groq Whisper
    2. Process through RAG pipeline
    3. Return text response for client-side text-to-speech
    
    Returns:
        JSON with transcription and response text
    """
    try:
        # Step 1: Transcribe audio to text
        logger.info(f"Voice chat request for client: {client_id}")
        user_text = await transcribe_audio(audio)
        logger.info(f"User said: {user_text}")
        
        # Step 2: Process through RAG
        collection_name = f"client_{client_id}"
        rag = RAGPipeline(collection_name=collection_name)
        
        try:
            rag.vector_store.load_collection(collection_name)
        except Exception as e:
            logger.warning(f"Could not load collection {collection_name}: {e}")
        
        result = rag.chat(
            message=user_text,
            conversation_history=[]  # TODO: Implement session-based history
        )
        
        response_text = result.get("answer", "I couldn't generate a response.")
        logger.info(f"RAG response: {response_text[:100]}...")
        
        # Return JSON response for client-side TTS
        return JSONResponse({
            "transcription": user_text,
            "response": response_text,
            "success": True
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize")
async def synthesize_speech(
    text: str,
    voice: str = "alloy"
):
    """
    Convert text to speech
    
    Args:
        text: Text to convert
        voice: Voice to use
        
    Returns:
        Audio file (MP3)
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(text) > 4096:
        raise HTTPException(status_code=400, detail="Text too long (max 4096 characters)")
    
    audio_bytes = await generate_speech(text, voice)
    
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"}
    )
