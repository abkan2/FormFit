from fastapi import WebSocket, WebSocketDisconnect
import websockets
import json
import asyncio


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


# WebSocket endpoint for realtime chat with OpenAI voice streaming
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        await websocket.send_json({"error": "OpenAI API key not found"})
        await websocket.close(code=1011)
        return

    try:
        # Receive initial message with user_id and messages
        init_data = await websocket.receive_json()
        user_id = init_data.get("user_id")
        messages = init_data.get("messages", [])
        use_voice = init_data.get("use_voice", True)  # Enable voice by default
        
        # Get user context
        user_context = ""
        if user_id:
            user_data = await firebase_service.get_user_data(user_id)
            if user_data:
                user_context = firebase_service.format_user_context(user_data)

        # Detect exercise selection
        if messages and user_id:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "").lower()
                    for ex, keys in {"squat": ["squat", "squats"], "pushup": ["pushup", "push-up", "push up", "pushups"], "plank": ["plank", "planks"]}.items():
                        if any(k in content for k in keys):
                            user_sessions[user_id] = {"selected_exercise": ex, "timestamp": str(datetime.now())}
                            print(f"🎯 User {user_id} selected exercise: {ex}")
                            break
                    break

        # Create personalized prompt
        personalized_prompt = AGENT_PROMPT_ALEX
        if user_context:
            personalized_prompt += f"\n\n[CURRENT USER CONTEXT]\n{user_context}"

        if use_voice:
            # Use OpenAI's TTS endpoint for realistic voice
            # First get text response
            chat_url = "https://api.openai.com/v1/chat/completions"
            chat_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            chat_payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "system", "content": personalized_prompt}, *messages],
                "stream": False
            }

            timeout = Timeout(30.0, read=30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                chat_resp = await client.post(chat_url, headers=chat_headers, json=chat_payload, timeout=timeout)
                
                if chat_resp.status_code != 200:
                    error_detail = chat_resp.text
                    await websocket.send_json({"error": f"OpenAI Chat API error: {error_detail}"})
                    return
                    
                chat_data = chat_resp.json()
                choices = chat_data.get("choices")
                if not choices or not isinstance(choices, list):
                    await websocket.send_json({"error": "Malformed response from OpenAI Chat API"})
                    return
                    
                ai_text = choices[0].get("message", {}).get("content")
                if not ai_text:
                    await websocket.send_json({"error": "No content returned from OpenAI"})
                    return

                # Send text response first
                await websocket.send_json({"text": ai_text})

                # Now convert to speech using OpenAI TTS
                tts_url = "https://api.openai.com/v1/audio/speech"
                tts_payload = {
                    "model": "tts-1-hd",  # High quality model
                    "input": ai_text,
                    "voice": "nova",  # More realistic voice (nova, alloy, echo, fable, onyx, shimmer)
                    "response_format": "mp3",
                    "speed": 1.0
                }

                tts_resp = await client.post(tts_url, headers=chat_headers, json=tts_payload, timeout=timeout)
                
                if tts_resp.status_code == 200:
                    # Stream audio data in chunks
                    audio_data = tts_resp.content
                    import base64
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # Send audio data to client
                    await websocket.send_json({
                        "audio": audio_base64,
                        "format": "mp3",
                        "voice_ready": True
                    })
                else:
                    # Fallback to text-only if TTS fails
                    await websocket.send_json({"tts_error": "TTS failed, using text only"})
        else:
            # Text-only response (fallback)
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "system", "content": personalized_prompt}, *messages],
                "stream": False
            }

            timeout = Timeout(30.0, read=30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
                
                if resp.status_code != 200:
                    error_detail = resp.text
                    await websocket.send_json({"error": f"OpenAI API error: {error_detail}"})
                    return
                    
                data = resp.json()
                choices = data.get("choices")
                if not choices or not isinstance(choices, list):
                    await websocket.send_json({"error": "Malformed response from OpenAI API"})
                    return
                    
                choice = choices[0].get("message", {}).get("content")
                if not choice:
                    await websocket.send_json({"error": "No content returned from OpenAI"})
                    return
                    
                # Send response back to client
                await websocket.send_json({"text": choice})
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.send_json({"error": f"Server error: {str(e)}"})
    finally:
        await websocket.close()


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


# WebSocket endpoint for realtime chat with OpenAI
@router.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        await ws.close(code=1011)
        return

    # Receive initial message with user_id and messages
    try:
        init_data = await ws.receive_json()
        user_id = init_data.get("user_id")
        messages = init_data.get("messages", [])
    except Exception:
        await ws.send_json({"error": "Invalid initial payload"})
        await ws.close(code=1003)
        return

    # Get user context
    user_context = ""
    if user_id:
        user_data = await firebase_service.get_user_data(user_id)
        if user_data:
            user_context = firebase_service.format_user_context(user_data)

    # Detect exercise selection
    if messages and user_id:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                for ex, keys in {"squat": ["squat", "squats"], "pushup": ["pushup", "push-up", "push up", "pushups"], "plank": ["plank", "planks"]}.items():
                    if any(k in content for k in keys):
                        user_sessions[user_id] = {"selected_exercise": ex, "timestamp": str(datetime.now())}
                        print(f"🎯 User {user_id} selected exercise: {ex}")
                        break
                break

    # Prepare OpenAI WebSocket connection
    openai_url = "wss://api.openai.com/v1/realtime"
    openai_headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Compose initial payload for OpenAI
    personalized_prompt = AGENT_PROMPT_ALEX
    if user_context:
        personalized_prompt += f"\n\n[CURRENT USER CONTEXT]\n{user_context}"

    openai_payload = {
        "model": "gpt-4o-mini-realtime-preview",
        "instructions": [
            {"role": "system", "content": personalized_prompt}
        ],
        # Add other required fields for OpenAI realtime API here
    }

    async def openai_ws_handler():
        try:
            async with websockets.connect(openai_url, extra_headers=openai_headers) as openai_ws:
                # Send initial payload to OpenAI
                await openai_ws.send(json.dumps(openai_payload))

                async def forward_user_to_openai():
                    while True:
                        try:
                            user_msg = await ws.receive_json()
                            await openai_ws.send(json.dumps(user_msg))
                        except WebSocketDisconnect:
                            break
                        except Exception:
                            continue

                async def forward_openai_to_user():
                    while True:
                        try:
                            openai_msg = await openai_ws.recv()
                            await ws.send_text(openai_msg)
                        except Exception:
                            break

                await asyncio.gather(forward_user_to_openai(), forward_openai_to_user())
        except Exception as e:
            await ws.send_json({"error": f"OpenAI WebSocket error: {str(e)}"})
            await ws.close(code=1011)

    await openai_ws_handler()


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





@router.get("/session/{user_id}")
async def get_user_session(user_id: str):
    """Get user session data including selected exercise"""
    session_data = user_sessions.get(user_id, {})
    return {
        "user_id": user_id,
        "session_data": session_data,
        "has_selected_exercise": "selected_exercise" in session_data
    }