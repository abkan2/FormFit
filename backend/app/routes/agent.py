""" Router for agent endpoint """

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Body
from typing import List, Any, Dict
import httpx
from httpx import Timeout
import os
from app.prompts import AGENT_PROMPT_ALEX
from dotenv import load_dotenv
from pydantic import BaseModel

router = APIRouter()
load_dotenv()


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

@router.get("/sessions")
async def create_session():
    if os.getenv("OPENAI_API_KEY") is None:
        raise HTTPException(status_code=500, detail="OPENAI API Key not found")
    
    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini-realtime-preview",
        "voice": "verse",
            "instructions": [
      {"role": "system", "content": AGENT_PROMPT_ALEX}
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


@router.post("/chat")
async def chat_with_ai(
    messages: List[Dict[str, Any]] = Body(...,embed=True, example=[
      {"role": "system",    "content": "You are a helpful assistant."},
      {"role": "user",      "content": "Hello!"},
      {"role": "assistant", "content": "Hi there!"}
    ])
):
    """
    messages: [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
    """
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI API Key not found")
    system_msg = {"role": "system", "content": AGENT_PROMPT_ALEX}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [system_msg, *messages]
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