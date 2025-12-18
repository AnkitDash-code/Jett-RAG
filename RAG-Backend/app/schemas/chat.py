"""
Chat and RAG schemas for request/response validation.
"""
from typing import Optional, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Source citation for RAG responses."""
    id: str
    name: str  # Document filename
    page: Optional[int] = None
    section: Optional[str] = None
    relevance: Literal["High", "Medium", "Low"] = "Medium"
    snippet: Optional[str] = None  # Relevant text snippet


class ChatRequest(BaseModel):
    """Chat message request."""
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: Optional[str] = None  # None for new conversation
    
    # Optional filters for retrieval
    document_ids: Optional[List[str]] = None  # Limit to specific documents
    date_range: Optional[dict] = None  # {"start": "2024-01-01", "end": "2024-12-31"}
    tags: Optional[List[str]] = None
    
    # Generation settings (optional overrides)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What are the main compliance requirements?",
                "conversation_id": None,
                "document_ids": None,
            }
        }
    }


class ChatResponse(BaseModel):
    """Non-streaming chat response."""
    conversation_id: str
    message_id: str
    content: str
    sources: List[SourceCitation]
    
    # Metadata
    model: str
    tokens_used: int
    latency_ms: int
    
    model_config = {"from_attributes": True}


class ChatStreamEvent(BaseModel):
    """Server-Sent Event for streaming responses."""
    event: Literal["token", "sources", "done", "error"]
    data: str  # Token text, JSON sources, or error message
    
    # Metadata (in final event)
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


class MessageResponse(BaseModel):
    """Single message in conversation history."""
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    sources: Optional[List[SourceCitation]] = None
    created_at: datetime
    
    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    """Conversation with messages."""
    id: str
    title: Optional[str]
    messages: List[MessageResponse]
    created_at: datetime
    updated_at: datetime
    message_count: int
    
    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    """List of conversations."""
    conversations: List[dict]  # Simplified conversation objects
    total: int
    page: int
    page_size: int


class FeedbackRequest(BaseModel):
    """User feedback on a response."""
    message_id: str
    rating: int = Field(ge=1, le=5)
    is_helpful: Optional[bool] = None
    is_accurate: Optional[bool] = None
    comment: Optional[str] = Field(default=None, max_length=1000)
    issues: Optional[List[str]] = None  # ["hallucination", "incomplete", etc.]
