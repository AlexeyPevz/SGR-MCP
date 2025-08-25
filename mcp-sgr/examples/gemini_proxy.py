#!/usr/bin/env python3
"""
Simple proxy server to adapt Gemini API to OpenAI-compatible format for MCP-SGR.

This allows using Google's Gemini models with MCP-SGR by converting between APIs.

Usage:
    1. Set your Gemini API key: export GEMINI_API_KEY=your-key
    2. Run this proxy: python gemini_proxy.py
    3. Configure MCP-SGR to use custom backend:
       - CUSTOM_LLM_URL=http://localhost:8001/v1/chat/completions
       - LLM_BACKENDS=custom
"""

import os
import json
import logging
from typing import List, Dict, Any

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Create FastAPI app
app = FastAPI(title="Gemini Proxy for MCP-SGR")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemini-pro"
    messages: List[Message]
    temperature: float = 0.1
    max_tokens: int = 2000

class ChatCompletionResponse(BaseModel):
    choices: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Convert OpenAI-style chat completion to Gemini API call."""
    try:
        # Select Gemini model
        if "gemini" not in request.model:
            # Map common models to Gemini equivalents
            model_map = {
                "gpt-4": "gemini-pro",
                "gpt-3.5-turbo": "gemini-pro",
                "default": "gemini-pro"
            }
            model_name = model_map.get(request.model, "gemini-pro")
        else:
            model_name = request.model
        
        # Initialize Gemini model
        model = genai.GenerativeModel(model_name)
        
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                # Gemini doesn't have system role, prepend to first user message
                prompt_parts.append(f"System: {msg.content}\n\n")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )
        
        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Format response in OpenAI style
        return ChatCompletionResponse(
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text
                },
                "finish_reason": "stop"
            }],
            model=model_name,
            usage={
                "prompt_tokens": len(full_prompt) // 4,  # Rough estimate
                "completion_tokens": len(response.text) // 4,
                "total_tokens": (len(full_prompt) + len(response.text)) // 4
            }
        )
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "backend": "gemini"}

if __name__ == "__main__":
    port = int(os.getenv("PROXY_PORT", "8001"))
    logger.info(f"Starting Gemini proxy on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)