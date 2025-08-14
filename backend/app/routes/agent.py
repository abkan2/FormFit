""" Router for agent endpoint """

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Body
from typing import List, Any, Dict, Optional
import httpx
from httpx import Timeout
import os
from app.prompts import AGENT_PROMPT_ALEX
from app.services.firebase_service import firebase_service
from dotenv import load_dotenv
from pydantic import BaseModel
import re
from datetime import datetime

router = APIRouter()
load_dotenv()

# Simple in-memory session storage for exercise selection
user_sessions = {}

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    user_id: Optional[str] = None


class SessionRequest(BaseModel):
    user_id: Optional[str] = None

@router.post("/sessions")
async def create_session(request: SessionRequest):
    if os.getenv("OPENAI_API_KEY") is None:
        raise HTTPException(status_code=500, detail="OPENAI API Key not found")
    
    # Get user context if user_id is provided
    user_context = ""
    if request.user_id:
        user_data = await firebase_service.get_user_data(request.user_id)
        if user_data:
            user_context = firebase_service.format_user_context(user_data)
    
    # Create personalized prompt
    personalized_prompt = AGENT_PROMPT_ALEX
    if user_context:
        personalized_prompt += f"\n\n[CURRENT USER CONTEXT]\n{user_context}"
    
    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini-realtime-preview",
        "voice": "verse",
        "instructions": [
            {"role": "system", "content": personalized_prompt}
        ],
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.7,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 300,
            "create_response": True,
        },
        # "input_audio_transcription": {"model": "whisper-1"},
    }

    # TODO: Add error handling for failed API requests
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        data = response.json()

    # Send back the JSON we received from the OpenAI REST API
    return data


@router.get("/user/{user_id}")
async def get_user_context(user_id: str):
    """Get user context for ALEX personalization"""
    try:
        user_data = await firebase_service.get_user_data(user_id)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_context = firebase_service.format_user_context(user_data)
        return {
            "user_data": user_data,
            "formatted_context": user_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user data: {str(e)}")


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI API Key not found")

    content = await file.read()
    files = {
        "file": (file.filename, content, file.content_type)
    }
    data = {
        "model": "whisper-1",
        "language": "en"
    }
    timeout = Timeout(60.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:

            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
                timeout =timeout
            )
        except:
            raise HTTPException(status_code=504, detail="OpenAI API request timed out")

    # If Whisper returns a JSON error, raise it
    if resp.status_code != 200:
        detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=f"Whisper error: {detail}")

    result = resp.json()
    return {"text": result["text"]}


@router.get("/user/{user_id}")
async def get_user_context(user_id: str):
    """Get user context for testing purposes"""
    try:
        user_data = await firebase_service.get_user_data(user_id)
        if user_data:
            formatted_context = firebase_service.format_user_context(user_data)
            return {
                "user_id": user_id,
                "user_data": user_data,
                "formatted_context": formatted_context
            }
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user data: {str(e)}")


@router.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """
    Enhanced chat endpoint with user context and exercise selection detection
    """
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI API Key not found")
    
    # Get user context if user_id is provided
    user_context = ""
    if request.user_id:
        user_data = await firebase_service.get_user_data(request.user_id)
        if user_data:
            user_context = firebase_service.format_user_context(user_data)
    
    # Check for exercise selection in the latest user message
    if request.messages and request.user_id:
        latest_user_message = None
        # Find the latest user message
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                latest_user_message = msg.get("content", "").lower()
                break
        
        if latest_user_message:
            # Detect exercise selection
            selected_exercise = None
            if any(word in latest_user_message for word in ["squat", "squats"]):
                selected_exercise = "squat"
            elif any(word in latest_user_message for word in ["pushup", "push-up", "push up", "pushups"]):
                selected_exercise = "pushup"
            elif any(word in latest_user_message for word in ["plank", "planks"]):
                selected_exercise = "plank"
            
            if selected_exercise:
                # Store the selected exercise for this user
                user_sessions[request.user_id] = {
                    "selected_exercise": selected_exercise,
                    "timestamp": str(datetime.now())
                }
                print(f"🎯 User {request.user_id} selected exercise: {selected_exercise}")
    
    # Create personalized prompt
    personalized_prompt = AGENT_PROMPT_ALEX
    if user_context:
        personalized_prompt += f"\n\n[CURRENT USER CONTEXT]\n{user_context}"
    
    system_msg = {"role": "system", "content": personalized_prompt}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [system_msg, *request.messages]
    }
    
    timeout = Timeout(60.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post("https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=timeout)
       
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="OpenAI API request timed out")
    
    data = resp.json()
    choice = data.get("choices", [{}])[0].get("message", {}).get("content")
    if choice is None:
        raise HTTPException(status_code=500, detail="Chat completion failed")
    return {"text": choice}


@router.get("/session/{user_id}")
async def get_user_session(user_id: str):
    """Get user session data including selected exercise"""
    session_data = user_sessions.get(user_id, {})
    return {
        "user_id": user_id,
        "session_data": session_data,
        "has_selected_exercise": "selected_exercise" in session_data
    }