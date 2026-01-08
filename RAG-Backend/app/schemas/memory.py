"""
Memory schemas for the Supermemory System (Phase 6).
Request/response models for memory API endpoints.
"""
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class CreateMemoryRequest(BaseModel):
    """Request to create a new memory."""
    content: str = Field(min_length=1, max_length=10000, description="The memory content")
    memory_type: Literal["episodic", "semantic"] = Field(
        description="Type of memory: episodic (events) or semantic (concepts)"
    )
    conversation_id: Optional[str] = Field(
        default=None, 
        description="Associated conversation ID"
    )
    summary: Optional[str] = Field(
        default=None, 
        max_length=500,
        description="Optional condensed summary"
    )
    entity_ids: Optional[List[str]] = Field(
        default=None,
        description="Related entity UUIDs for hybrid search"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "User discussed project requirements for Q1 2026",
                "memory_type": "episodic",
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }
    }


class MemoryRetrievalRequest(BaseModel):
    """Request to retrieve relevant memories."""
    query: str = Field(min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum memories to return")
    memory_type: Optional[Literal["episodic", "semantic"]] = Field(
        default=None,
        description="Filter by memory type"
    )
    relevance_threshold: float = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0,
        description="Minimum hybrid relevance score"
    )
    include_archived: bool = Field(
        default=False,
        description="Include archived memories in results"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "project requirements",
                "top_k": 5,
                "memory_type": None,
                "relevance_threshold": 0.3,
            }
        }
    }


class MemoryFeedbackRequest(BaseModel):
    """Request to update memory feedback score."""
    feedback_delta: float = Field(
        ge=-1.0, 
        le=1.0,
        description="Value to add to feedback score (-1 to +1)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "feedback_delta": 0.2,
            }
        }
    }


class CreateMemoryLinkRequest(BaseModel):
    """Request to create a link between memories."""
    target_memory_id: str = Field(description="Target memory UUID")
    link_type: Literal["related", "causes", "follows", "contradicts", "supports"] = Field(
        default="related",
        description="Type of relationship"
    )
    strength: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Link strength"
    )


class EvictMemoriesRequest(BaseModel):
    """Request to evict low-importance memories from cache."""
    importance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Memories below this importance will be archived"
    )
    age_threshold_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Only archive memories older than this many days"
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class MemoryResponse(BaseModel):
    """Single memory in response."""
    id: str
    memory_type: str
    content: str
    summary: Optional[str] = None
    
    # Importance scoring
    importance_score: float
    interaction_count: int
    feedback_score: float
    days_old: float
    
    # Status
    status: str
    is_cached: bool
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    last_accessed_at: Optional[datetime] = None
    
    # Metadata
    entity_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    
    model_config = {"from_attributes": True}
    
    @classmethod
    def from_memory(cls, memory) -> "MemoryResponse":
        """Create response from Memory model."""
        return cls(
            id=str(memory.id),
            memory_type=memory.memory_type.value,
            content=memory.content,
            summary=memory.summary,
            importance_score=memory.importance_score,
            interaction_count=memory.interaction_count,
            feedback_score=memory.feedback_score,
            days_old=memory.days_old,
            status=memory.status.value,
            is_cached=memory.is_cached,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            last_accessed_at=memory.last_accessed_at,
            entity_ids=memory.entity_ids if memory.entity_ids else None,
            conversation_id=str(memory.conversation_id) if memory.conversation_id else None,
        )


class MemoryCreationResponse(BaseModel):
    """Response from memory creation."""
    memory: MemoryResponse
    embedding_ms: int
    total_ms: int
    was_duplicate: bool = False


class MemoryRetrievalResponse(BaseModel):
    """Response from memory retrieval."""
    memories: List[MemoryResponse]
    hybrid_scores: Dict[str, float]  # memory_id -> score
    retrieval_ms: int
    method_used: str
    total_candidates: int


class MemoryListResponse(BaseModel):
    """Paginated list of memories."""
    memories: List[MemoryResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class ImportanceUpdateResponse(BaseModel):
    """Response from importance update/eviction."""
    updated_count: int
    archived_count: int
    update_ms: int
    message: str


class MemoryStatsResponse(BaseModel):
    """Memory statistics."""
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    cache_size: int
    total_active: int
    total_archived: int
    avg_importance: float


class MemoryLinkResponse(BaseModel):
    """Memory link in response."""
    id: str
    source_memory_id: str
    target_memory_id: str
    link_type: str
    strength: float
    created_at: datetime
    
    model_config = {"from_attributes": True}
