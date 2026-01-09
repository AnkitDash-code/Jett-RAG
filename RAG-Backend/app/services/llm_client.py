"""
LLM client for interacting with the LLM-Backend service.
Supports both streaming and non-streaming generation.
"""
import json
import logging
import time
from typing import Optional, AsyncIterator, Dict, Any

import httpx

from app.config import settings
from app.services.token_counter import get_token_counter
from app.services.metrics_service import get_metrics_service

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for the LLM-Backend service.
    Connects to the FastAPI wrapper on port 8080 that talks to llama.cpp.
    """
    
    def __init__(
        self,
        base_url: str = settings.LLM_API_BASE_URL,
        model: str = settings.LLM_MODEL_NAME,
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._token_counter = get_token_counter()
        self._metrics = get_metrics_service()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def generate(
        self,
        messages: list[Dict[str, str]],
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        stop: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a completion from the LLM (non-streaming).
        
        Calls the LLM-Backend /generate endpoint with query and context.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature (passed to LLM-Backend)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
        
        Returns:
            Dict with 'content', 'tokens_used', 'prompt_tokens', 'completion_tokens', 'latency_ms'
        """
        client = await self._get_client()
        
        start_time = time.perf_counter()
        
        # Extract query and context from messages for LLM-Backend format
        query, context = self._messages_to_query_context(messages)
        
        # Count prompt tokens
        prompt_tokens = self._token_counter.count_messages(messages)
        
        try:
            response = await client.post(
                "/generate",
                json={
                    "query": query,
                    "context": context,
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # LLM-Backend returns {"response": "..."}
            content = data.get("response", "")
            
            # Count completion tokens
            completion_tokens = self._token_counter.count(content)
            
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            latency_seconds = latency_ms / 1000.0
            
            # Record metrics
            self._metrics.record_llm_request(
                llm_type="main",
                status="success",
                latency=latency_seconds
            )
            
            return {
                "content": content,
                "tokens_used": prompt_tokens + completion_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
            }
            
        except httpx.HTTPError as e:
            logger.error(f"LLM API error: {e}")
            self._metrics.record_llm_request(llm_type="main", status="error")
            raise RuntimeError(f"LLM generation failed: {e}")
    
    def _messages_to_query_context(
        self,
        messages: list[Dict[str, str]],
    ) -> tuple[str, str]:
        """
        Convert chat messages to query/context format for LLM-Backend.
        
        The LLM-Backend expects:
        - query: The user's question
        - context: The RAG context (from system message)
        """
        query = ""
        context = "No context provided."
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # System message contains the context
                context = content
            elif role == "user":
                # Last user message is the query
                query = content
        
        return query, context
    
    async def generate_stream(
        self,
        messages: list[Dict[str, str]],
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        stop: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion from the LLM.
        
        Uses the /generate/stream endpoint for real token-by-token streaming.
        Falls back to simulated streaming if the streaming endpoint is not available.
        """
        client = await self._get_client()
        
        # Extract query and context from messages
        query, context = self._messages_to_query_context(messages)
        
        logger.info(f"[LLMClient] Starting stream request to {self.base_url}/generate/stream")
        token_count = 0
        
        try:
            # Use streaming endpoint
            async with client.stream(
                "POST",
                "/generate/stream",
                json={
                    "query": query,
                    "context": context,
                    "stream": True,
                },
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            ) as response:
                response.raise_for_status()
                logger.info(f"[LLMClient] Stream connection established, status: {response.status_code}")
                
                # Read SSE stream
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            logger.info(f"[LLMClient] Stream complete. Total tokens: {token_count}")
                            break
                        elif data.startswith("[ERROR]"):
                            logger.error(f"LLM streaming error: {data}")
                            raise RuntimeError(f"LLM streaming error: {data}")
                        else:
                            token_count += 1
                            if token_count <= 5:
                                logger.debug(f"[LLMClient] Token {token_count}: {repr(data)}")
                            yield data
                            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Streaming endpoint not available, fall back to simulated streaming
                logger.warning("Streaming endpoint not available, using simulated streaming")
                result = await self.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                content = result.get("content", "")
                words = content.split(" ")
                for i, word in enumerate(words):
                    if i > 0:
                        yield " "
                    yield word
            else:
                raise
        except httpx.HTTPError as e:
            logger.error(f"LLM streaming error: {e}")
            raise RuntimeError(f"LLM streaming failed: {e}")


class PromptBuilder:
    """
    Builds prompts for RAG generation.
    Handles context formatting, citation instructions, and conversation history.
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions clearly using the provided context when relevant.

CONTEXT:
{context}

Use the context above to answer when applicable, and cite sources with [Source: document_name]. If the context doesn't fully answer the question, provide a helpful response based on what's available."""

    def build_messages(
        self,
        query: str,
        context_chunks: list[Dict[str, Any]],
        conversation_history: Optional[list[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> list[Dict[str, str]]:
        """
        Build the message list for LLM completion.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks with metadata
            conversation_history: Previous messages in conversation
            system_prompt: Optional custom system prompt
        """
        # Format context
        context_text = self._format_context(context_chunks)
        
        # Build system prompt
        system = (system_prompt or self.SYSTEM_PROMPT).format(context=context_text)
        
        messages = [{"role": "system", "content": system}]
        
        # Add conversation history (limited to last N turns)
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 3 turns (6 messages)
                messages.append(msg)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _format_context(self, chunks: list[Dict[str, Any]]) -> str:
        """Format chunks into context string."""
        if not chunks:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number")
            section = chunk.get("section_path", "")
            text = chunk.get("text", "")
            
            header = f"[Source {i}: {source}"
            if page:
                header += f", Page {page}"
            if section:
                header += f", Section: {section}"
            header += "]"
            
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def extract_citations(
        self,
        response: str,
        chunks: list[Dict[str, Any]],
        max_sources: int = 2,
    ) -> list[Dict[str, Any]]:
        """
        Extract and resolve citations from LLM response.
        Maps [Source: X] references to actual chunk metadata.
        Returns only the top N most relevant sources.
        """
        citations = []
        
        # Sort chunks by score (already reranked if reranker is enabled)
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
        
        # Return only top max_sources as citations
        for chunk in sorted_chunks[:max_sources]:
            doc_name = chunk.get("document_name", "")
            chunk_id = chunk.get("chunk_id", "")
            citations.append({
                "id": chunk_id,
                "name": doc_name,
                "page": chunk.get("page_number"),
                "section": chunk.get("section_path"),
                "relevance": self._score_to_relevance(chunk.get("score", 0.5)),
                "snippet": chunk.get("text", "")[:500],  # First 500 chars of chunk
            })
        
        return citations
    
    def _score_to_relevance(self, score: float) -> str:
        """Convert numeric score to relevance label."""
        if score >= 0.8:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"
