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
    stream: bool = False  # New: streaming option


# 4. Create the Chat Endpoint (non-streaming)
@app.post("/generate")
async def generate_response(request: ChatRequest):
    try:
        # System Prompt: Strict instructions for the AI
        system_instruction = (
            "You are a helpful and precise assistant for an internal knowledge portal. "
            "Answer the user's question based ONLY on the provided context below. "
            "If the answer is not in the context, state that you do not know. "
            "Do not hallucinate facts."
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
                stream_response(system_instruction, formatted_prompt),
                media_type="text/event-stream",
            )

        print(f"DEBUG: Sending query to llama.cpp...")

        # Call the llama.cpp server (non-streaming)
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=False
        )

        # Extract the answer
        ai_answer = response.choices[0].message.content
        return {"response": ai_answer}

    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        raise HTTPException(status_code=500, detail=f"LLM Server Error: {str(e)}")


# 5. Streaming response generator - synchronous
def stream_response(system_instruction: str, formatted_prompt: str):
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
            max_tokens=1024,
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
    system_instruction = (
        "You are a helpful and precise assistant for an internal knowledge portal. "
        "Answer the user's question based ONLY on the provided context below. "
        "If the answer is not in the context, state that you do not know. "
        "Do not hallucinate facts."
    )
    
    formatted_prompt = f"""
    ### Context:
    {request.context}

    ### Question:
    {request.query}
    """
    
    return StreamingResponse(
        stream_response(system_instruction, formatted_prompt),
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