"""
Memory models for the Supermemory System (Phase 6).
Includes: Episodic and Semantic memory storage with importance-based retention.

Key Concepts:
- Episodic Memory: Specific events/interactions (time-bound, decays faster)
- Semantic Memory: Concepts/knowledge (persistent, strengthens with use)
- Importance Formula: (1/days_old) × interaction_count × feedback_score
- Smart Forgetting: Cache eviction only, DB records archived not deleted
"""
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlmodel import SQLModel, Field, Relationship


class MemoryType(str, Enum):
    """Type of memory storage."""
    EPISODIC = "episodic"   # Specific events/interactions
    SEMANTIC = "semantic"   # Concepts/knowledge


class MemoryStatus(str, Enum):
    """Memory lifecycle status."""
    ACTIVE = "active"       # In cache & DB, fully accessible
    ARCHIVED = "archived"   # In DB only, removed from cache
    CONSOLIDATED = "consolidated"  # Merged into semantic memory


class Memory(SQLModel, table=True):
    """
    Long-term memory storage for user interactions.
    
    Supports both episodic (event-based) and semantic (concept-based) memory
    with importance scoring and smart cache eviction.
    """
    __tablename__ = "memories"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(index=True)
    conversation_id: Optional[uuid.UUID] = Field(
        default=None, 
        foreign_key="conversations.id", 
        index=True
    )
    
    # Memory classification
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC, index=True)
    
    # Content
    content: str  # The memory text/description
    summary: Optional[str] = Field(default=None, max_length=500)  # Condensed version
    
    # Importance scoring components
    interaction_count: int = Field(default=1, ge=0)  # Times referenced/accessed
    feedback_score: float = Field(default=0.5, ge=0.0, le=1.0)  # User feedback 0-1
    days_old: float = Field(default=0.0, ge=0.0)  # Calculated on access
    importance_score: float = Field(default=1.0, ge=0.0, index=True)
    # Formula: (1 / max(days_old, 0.1)) × interaction_count × feedback_score
    
    # Retrieval tracking
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    access_count: int = Field(default=0, ge=0)
    
    # Entity linking for hybrid relevance scoring
    entity_ids_json: str = Field(default="[]")  # JSON array of entity UUIDs
    related_chunk_ids_json: str = Field(default="[]")  # Source document chunks
    
    # Vector storage reference
    vector_id: Optional[str] = Field(default=None, max_length=100)  # ID in FAISS/vector store
    embedding_model: str = Field(default="all-MiniLM-L6-v2", max_length=100)
    
    # Status and caching
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE, index=True)
    is_cached: bool = Field(default=True, index=True)
    
    # Multi-tenancy
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    
    # Flexible metadata storage
    metadata_json: str = Field(default="{}")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    archived_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    accesses: List["MemoryAccess"] = Relationship(back_populates="memory")
    
    # --- Helper properties ---
    
    @property
    def entity_ids(self) -> List[str]:
        """Get entity IDs as list."""
        try:
            return json.loads(self.entity_ids_json)
        except (json.JSONDecodeError, TypeError):
            return []
    
    @entity_ids.setter
    def entity_ids(self, value: List[str]):
        """Set entity IDs from list."""
        self.entity_ids_json = json.dumps(value)
    
    @property
    def related_chunk_ids(self) -> List[str]:
        """Get related chunk IDs as list."""
        try:
            return json.loads(self.related_chunk_ids_json)
        except (json.JSONDecodeError, TypeError):
            return []
    
    @related_chunk_ids.setter
    def related_chunk_ids(self, value: List[str]):
        """Set related chunk IDs from list."""
        self.related_chunk_ids_json = json.dumps(value)
    
    @property
    def extra_metadata(self) -> Dict[str, Any]:
        """Get metadata as dict (named extra_metadata to avoid SQLAlchemy conflict)."""
        try:
            return json.loads(self.metadata_json)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    @extra_metadata.setter
    def extra_metadata(self, value: Dict[str, Any]):
        """Set metadata from dict."""
        self.metadata_json = json.dumps(value)
    
    def calculate_importance(self) -> float:
        """
        Calculate importance score using the formula:
        importance = (1 / max(days_old, 0.1)) × interaction_count × feedback_score
        """
        days = max(self.days_old, 0.1)  # Prevent division by zero
        recency_factor = 1.0 / days
        importance = recency_factor * self.interaction_count * self.feedback_score
        return round(importance, 4)
    
    def update_days_old(self) -> None:
        """Update days_old based on current time."""
        delta = datetime.utcnow() - self.created_at
        self.days_old = delta.total_seconds() / 86400  # Convert to days


class MemoryAccess(SQLModel, table=True):
    """
    Audit trail for memory retrieval.
    Tracks when and how memories are accessed for analytics and debugging.
    """
    __tablename__ = "memory_accesses"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    memory_id: uuid.UUID = Field(foreign_key="memories.id", index=True)
    
    # Access context
    query: str  # The query that triggered retrieval
    relevance_score: float = Field(ge=0.0, le=1.0)  # How relevant was this memory
    
    # Retrieval method used
    retrieval_method: str = Field(default="hybrid", max_length=50)
    # Options: "vector", "entity", "hybrid", "recency"
    
    # Performance
    retrieval_latency_ms: Optional[int] = Field(default=None)
    
    # Timestamp
    accessed_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Relationships
    memory: Optional[Memory] = Relationship(back_populates="accesses")


class MemoryLink(SQLModel, table=True):
    """
    Links between memories for associative recall.
    Enables traversing related memories in a graph-like manner.
    """
    __tablename__ = "memory_links"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Source and target memories
    source_memory_id: uuid.UUID = Field(foreign_key="memories.id", index=True)
    target_memory_id: uuid.UUID = Field(foreign_key="memories.id", index=True)
    
    # Link metadata
    link_type: str = Field(default="related", max_length=50)
    # Options: "related", "causes", "follows", "contradicts", "supports"
    
    strength: float = Field(default=0.5, ge=0.0, le=1.0)  # Link strength 0-1
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
