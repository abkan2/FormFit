from fastapi import WebSocket, WebSocketDisconnect
import websockets
import json
import asyncio
from fastapi.responses import Response

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Body
from typing import List, Any, Dict, Optional
import httpx
from httpx import Timeout
import os
from app.prompts import AGENT_PROMPT_ALEX, Demo_prompt_ALEX
from app.services.firebase_service import firebase_service
from dotenv import load_dotenv
from pydantic import BaseModel
import re
from datetime import datetime
import base64
router = APIRouter()
load_dotenv()

# Simple in-memory session storage for exercise selection
user_sessions = {}

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    user_id: Optional[str] = None


class SessionRequest(BaseModel):
    user_id: Optional[str] = None

class TTSRequest(BaseModel):
    text: str

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "ALEX Agent is healthy!"}

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
    
    # Build personalized instructions string
    instructions_text = Demo_prompt_ALEX
    if user_context:
        instructions_text += f"\n\n[CURRENT USER CONTEXT]\n{user_context}"

    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini-realtime-preview",
        "voice": "verse",
        # instructions must be a plain string, NOT a list
        "instructions": instructions_text,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        # Enable transcription so Unity can read captions / detect workout triggers
        "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,           # lower = more sensitive
            "prefix_padding_ms": 200,
            "silence_duration_ms": 500, # wait 500ms of silence before responding
            "create_response": True,
        },
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI Realtime session error: {response.text}"
            )
        data = response.json()

    # Return the full session object — Unity needs client_secret.value as the ephemeral token
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


@router.websocket("/ws/chat_text")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        await ws.send_json({"error": "OPENAI API Key not found"})
        await ws.close(code=1011)
        return

    # Receive initial message: { user_id, text, history }
    try:
        init_data = await ws.receive_json()
        user_id = init_data.get("user_id")
        user_text = init_data.get("text", "")
        history = init_data.get("history", [])
    except Exception:
        await ws.send_json({"error": "Invalid initial payload"})
        await ws.close(code=1003)
        return

    if not user_text:
        await ws.send_json({"error": "Missing 'text' in payload"})
        await ws.close(code=1003)
        return

    # ----- USER CONTEXT -----
    user_context = ""
    if user_id:
        try:
            user_data = await firebase_service.get_user_data(user_id)
            if user_data:
                user_context = firebase_service.format_user_context(user_data)
        except Exception as e:
            print(f"⚠️ Failed to fetch user context: {e}")


    # ----- BUILD CHAT MESSAGES -----
    personalized_prompt = AGENT_PROMPT_ALEX
    if user_context:
        personalized_prompt += f"\n\n[CURRENT USER CONTEXT]\n{user_context}"

    messages = [
        {"role": "system", "content": Demo_prompt_ALEX},
    ]

    # Existing convo history from client
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    # Latest user message (from speech)
    messages.append({"role": "user", "content": user_text})

    # ----- CALL OPENAI CHAT WITH STREAMING -----
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "stream": True,
    }

    full_text = ""

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    await ws.send_json({
                        "error": f"OpenAI error: {err_text.decode('utf-8', 'ignore')}"
                    })
                    await ws.close(code=1011)
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        await ws.send_json({"event": "done", "full_text": full_text})
                        break

                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        full_text += delta
                        await ws.send_json({"event": "delta", "content": delta})
    except WebSocketDisconnect:
        print("📴 Client disconnected from /ws/chat")
    except Exception as e:
        print("❌ Error in websocket_chat:", e)
        try:
            await ws.send_json({"error": f"Server error: {str(e)}"})
        except Exception:
            pass
        await ws.close(code=1011)


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
    Ultra-fast chat endpoint using OpenAI's chat API (gpt-4o-mini)
    with user context and exercise selection detection
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

    # Detect exercise selection in latest user message
    if request.messages and request.user_id:
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                for ex, keys in {
                    "squat": ["squat", "squats"],
                    "pushup": ["pushup", "push-up", "push up", "pushups"],
                    "plank": ["plank", "planks"],
                }.items():
                    if any(k in content for k in keys):
                        user_sessions[request.user_id] = {
                            "selected_exercise": ex,
                            "timestamp": str(datetime.now()),
                        }
                        print(f"🎯 User {request.user_id} selected exercise: {ex}")
                        break
                break

    # Build system + user messages for chat API
    messages = []

    # System / personalization
    personalized_prompt = AGENT_PROMPT_ALEX
    if user_context:
        personalized_prompt += f"\n\n[CURRENT USER CONTEXT]\n{user_context}"
    messages.append({"role": "system", "content": personalized_prompt})

    # Append conversation messages from client (assumed already in {role, content} format)
    for msg in request.messages:
        # defensively only keep role/content
        role = msg.get("role", "user")
        content = msg.get("content", "")
        messages.append({"role": role, "content": content})

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "stream": False,
    }

    timeout = Timeout(30.0, read=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="OpenAI API request timed out")

    if resp.status_code != 200:
        try:
            err_json = resp.json()
            err_detail = err_json.get("error", {}).get("message", resp.text)
        except Exception:
            err_detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=f"OpenAI API error: {err_detail}")

    data = resp.json()
    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        raise HTTPException(status_code=500, detail="Malformed response from OpenAI API")

    content = choices[0].get("message", {}).get("content")
    if not content:
        # Optional: log full response for debugging
        print("⚠️ OpenAI response had no message content:", data)
        raise HTTPException(status_code=500, detail="Chat completion failed: No content returned")

    return {"text": content}


@router.get("/session/{user_id}")
async def get_user_session(user_id: str):
    """Get user session data including selected exercise"""
    session_data = user_sessions.get(user_id, {})
    return {
        "user_id": user_id,
        "session_data": session_data,
        "has_selected_exercise": "selected_exercise" in session_data
    }


class TTSRequest(BaseModel):
    text: str

@router.post("/tts")
async def tts_endpoint(body: TTSRequest):
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI API Key not found")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "tts-1-hd",
        "voice": "alloy",
        "input": text,
        "format": "mp3",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

    # Encode raw bytes as base64 and return JSON (easy for RN clients)
    audio_b64 = base64.b64encode(resp.content).decode("utf-8")
    return {"audio_base64": audio_b64, "mime": "audio/mpeg"}
