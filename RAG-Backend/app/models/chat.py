"""
Chat and conversation related models.
Includes: Conversation, Message, QueryLog, Feedback
"""
import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlmodel import SQLModel, Field, Relationship


class MessageRoleEnum(str, Enum):
    """Message sender role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class QueryTypeEnum(str, Enum):
    """Classification of query type for routing."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    ENTITY = "entity"
    DOCUMENT = "document"
    CONVERSATIONAL = "conversational"


class Conversation(SQLModel, table=True):
    """Chat conversation/session model."""
    __tablename__ = "conversations"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(index=True)
    
    # Conversation metadata
    title: Optional[str] = Field(default=None, max_length=255)
    summary: Optional[str] = Field(default=None)  # Auto-generated summary
    
    # Settings for this conversation
    settings_json: str = Field(default="{}")  # Temperature, model preferences, etc.
    
    # Status
    is_active: bool = Field(default=True)
    is_archived: bool = Field(default=False)
    
    # Statistics
    message_count: int = Field(default=0)
    total_tokens_used: int = Field(default=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    messages: List["Message"] = Relationship(back_populates="conversation")


class Message(SQLModel, table=True):
    """Individual chat message model."""
    __tablename__ = "messages"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    conversation_id: uuid.UUID = Field(foreign_key="conversations.id", index=True)
    
    # Message content
    role: MessageRoleEnum = Field(default=MessageRoleEnum.USER)
    content: str  # The message text
    
    # RAG sources (for assistant messages)
    sources_json: str = Field(default="[]")  # JSON array of source citations
    
    # Token usage
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    
    # Performance metrics
    retrieval_latency_ms: Optional[int] = Field(default=None)
    llm_latency_ms: Optional[int] = Field(default=None)
    total_latency_ms: Optional[int] = Field(default=None)
    
    # Query classification (for user messages)
    query_type: Optional[QueryTypeEnum] = Field(default=None)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    conversation: Optional[Conversation] = Relationship(back_populates="messages")
    query_log: Optional["QueryLog"] = Relationship(back_populates="message")
    feedback: Optional["Feedback"] = Relationship(back_populates="message")


class QueryLog(SQLModel, table=True):
    """Detailed logging of RAG queries for analysis and debugging."""
    __tablename__ = "query_logs"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    message_id: uuid.UUID = Field(foreign_key="messages.id", unique=True, index=True)
    user_id: uuid.UUID = Field(index=True)
    
    # Original query
    raw_query: str
    normalized_query: str
    
    # Query classification
    query_type: QueryTypeEnum = Field(default=QueryTypeEnum.SIMPLE)
    query_difficulty: str = Field(default="simple", max_length=20)  # simple, complex
    
    # Query expansion
    expanded_queries_json: str = Field(default="[]")  # Multi-query RAG variants
    
    # Retrieval details
    retrieval_mode: str = Field(default="hybrid", max_length=50)  # vector, bm25, hybrid, graph
    chunks_retrieved: int = Field(default=0)
    chunks_after_rerank: int = Field(default=0)
    
    # Phase timings (ms)
    normalization_ms: Optional[int] = Field(default=None)
    embedding_ms: Optional[int] = Field(default=None)
    vector_search_ms: Optional[int] = Field(default=None)
    graph_search_ms: Optional[int] = Field(default=None)
    reranking_ms: Optional[int] = Field(default=None)
    
    # Quality metrics
    top_chunk_score: Optional[float] = Field(default=None)
    mean_chunk_score: Optional[float] = Field(default=None)
    relevance_grade: Optional[str] = Field(default=None)  # high, medium, low
    
    # Retry information
    retry_count: int = Field(default=0)
    final_answer_type: str = Field(default="answered", max_length=50)  # answered, no_answer, error
    
    # Full context for debugging (can be large)
    context_used: Optional[str] = Field(default=None)  # The actual context sent to LLM
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    message: Optional[Message] = Relationship(back_populates="query_log")


class Feedback(SQLModel, table=True):
    """User feedback on chat responses for evaluation."""
    __tablename__ = "feedback"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: uuid.UUID = Field(foreign_key="messages.id", unique=True, index=True)
    user_id: uuid.UUID = Field(index=True)
    
    # Rating
    rating: int = Field(ge=1, le=5)  # 1-5 stars
    
    # Detailed feedback
    is_helpful: Optional[bool] = Field(default=None)
    is_accurate: Optional[bool] = Field(default=None)
    is_complete: Optional[bool] = Field(default=None)
    
    # Free-form feedback
    comment: Optional[str] = Field(default=None)
    
    # Specific issues
    issues_json: str = Field(default="[]")  # ["hallucination", "incomplete", "wrong_source", etc.]
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    message: Optional[Message] = Relationship(back_populates="feedback")
