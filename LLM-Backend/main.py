import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# 1. Initialize FastAPI App
app = FastAPI(title="Offline RAG Knowledge Portal")

# Add CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup Connection to Local LLM (llama.cpp)
# We point base_url to port 8000 where your llama engine will run
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-no-key-required"
)

# 3. Define the Request Model (Data format)
class ChatRequest(BaseModel):
    query: str
    context: str = "No context provided."
    stream: bool = False
    system_instruction: str | None = None  # Optional override
    max_tokens: int = 2048  # Prevent cutoff by default


# 4. Create the Chat Endpoint (non-streaming)
@app.post("/generate")
async def generate_response(request: ChatRequest):
    try:
        # Determine strictness based on whether it's a RAG query or Utility task
        if request.system_instruction:
            # Custom instruction (Utility Task) - trust the caller
            system_txt = request.system_instruction
        else:
            # Default RAG behavior - Strict Document Mode
            system_txt = (
                "You are a document-based assistant for an internal knowledge portal. "
                "STRICT RULES:\n"
                "1. Answer ONLY using information from the provided context below.\n"
                "2. If the context does not contain the answer, say 'I could not find this information in the uploaded documents.'\n"
                "3. NEVER use your own knowledge or training data. This is an OFFLINE system.\n"
                "4. Cite sources using the exact document names from the context, like [Source: filename.pdf, Page X].\n"
                "5. Do not mention Wikipedia, web sources, or external references.\n"
                "6. If multiple documents contain relevant info, cite all of them."
            )

        # User Prompt: The actual data
        formatted_prompt = f"""
        ### Context:
        {request.context}

        ### Question:
        {request.query}
        """

        # If streaming requested, use streaming endpoint
        if request.stream:
            return StreamingResponse(
                stream_response(system_txt, formatted_prompt, max_tokens=request.max_tokens),
                media_type="text/event-stream",
            )

        print(f"DEBUG: Sending query to llama.cpp...")

        # Call the llama.cpp server (non-streaming)
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": system_txt},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.1,
            max_tokens=request.max_tokens,
            stream=False
        )

        # Extract the answer
        ai_answer = response.choices[0].message.content
        return {"response": ai_answer}

    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        raise HTTPException(status_code=500, detail=f"LLM Server Error: {str(e)}")


# 5. Streaming response generator - synchronous
def stream_response(system_instruction: str, formatted_prompt: str, max_tokens: int = 2048):
    """Stream tokens from llama.cpp as Server-Sent Events.
    
    Uses synchronous generator which FastAPI handles correctly.
    """
    try:
        print(f"[LLM] Starting stream...")
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            stream=True  # Enable streaming
        )
        
        token_count = 0
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                token_count += 1
                # Send as SSE format
                yield f"data: {token}\n\n"
        
        print(f"[LLM] Stream complete. {token_count} tokens sent.")
        # Signal end of stream
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"[LLM] Error: {e}")
        yield f"data: [ERROR] {str(e)}\n\n"


# 6. Dedicated streaming endpoint
@app.post("/generate/stream")
async def generate_stream(request: ChatRequest):
    """Streaming endpoint for real-time token output."""
    if request.system_instruction:
        system_txt = request.system_instruction
    else:
        system_txt = (
            "You are a document-based assistant for an internal knowledge portal. "
            "STRICT RULES:\n"
            "1. Answer ONLY using information from the provided context below.\n"
            "2. If the context does not contain the answer, say 'I could not find this information in the uploaded documents.'\n"
            "3. NEVER use your own knowledge or training data. This is an OFFLINE system.\n"
            "4. Cite sources using the exact document names from the context, like [Source: filename.pdf, Page X].\n"
            "5. Do not mention Wikipedia, web sources, or external references.\n"
            "6. If multiple documents contain relevant info, cite all of them."
        )
    
    formatted_prompt = f"""
    ### Context:
    {request.context}

    ### Question:
    {request.query}
    """
    
    return StreamingResponse(
        stream_response(system_txt, formatted_prompt, max_tokens=request.max_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )


# 7. Run the Server
if __name__ == "__main__":
    # Runs on Port 8080 to avoid conflict with llama.cpp (Port 8000)
    uvicorn.run(app, host="0.0.0.0", port=8080)