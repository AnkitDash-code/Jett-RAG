"""
Job Management Models for Phase 5 Background Processing.

Provides unified job tracking for:
- Document ingestion
- Graph rebuilding
- Community detection
- Cache cleanup
- OCR processing
- Maintenance tasks
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from sqlmodel import SQLModel, Field, Column, JSON


class JobType(str, Enum):
    """Types of background jobs."""
    DOCUMENT_INGESTION = "document_ingestion"
    GRAPH_REBUILD = "graph_rebuild"
    COMMUNITY_DETECTION = "community_detection"
    CACHE_CLEANUP = "cache_cleanup"
    OCR_PROCESSING = "ocr_processing"
    EMBEDDING_REFRESH = "embedding_refresh"
    MAINTENANCE = "maintenance"
    DATABASE_VACUUM = "database_vacuum"


class JobStatus(str, Enum):
    """Job lifecycle states."""
    PENDING = "pending"          # Queued, waiting for worker
    RUNNING = "running"          # Currently being processed
    COMPLETED = "completed"      # Successfully finished
    FAILED = "failed"            # Error occurred
    CANCELLED = "cancelled"      # User or system cancelled
    PAUSED = "paused"            # Temporarily halted
    PENDING_EXTERNAL = "pending_external"  # Waiting for external service (e.g., OCR)


class JobPriority(int, Enum):
    """Job priority levels (higher = more urgent)."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class Job(SQLModel, table=True):
    """
    Unified job tracking for all background tasks.
    
    Provides:
    - Status tracking with progress percentage
    - Priority-based scheduling
    - Result/error storage
    - Timestamps for monitoring
    """
    __tablename__ = "jobs"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Job identification
    job_type: JobType = Field(index=True)
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    priority: int = Field(default=JobPriority.NORMAL, index=True)
    
    # Ownership
    user_id: Optional[uuid.UUID] = Field(default=None, index=True)
    tenant_id: Optional[str] = Field(default=None, max_length=50, index=True)
    
    # Target resource (e.g., document_id for ingestion jobs)
    resource_type: Optional[str] = Field(default=None, max_length=50)
    resource_id: Optional[uuid.UUID] = Field(default=None, index=True)
    
    # Progress tracking
    progress_percent: int = Field(default=0, ge=0, le=100)
    progress_message: Optional[str] = Field(default=None, max_length=500)
    current_step: Optional[str] = Field(default=None, max_length=100)
    total_steps: int = Field(default=1)
    current_step_number: int = Field(default=0)
    
    # Results
    payload: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    result_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    error_message: Optional[str] = Field(default=None)
    error_details: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Retry tracking
    attempt_count: int = Field(default=0)
    max_attempts: int = Field(default=3)
    
    # Worker info
    worker_id: Optional[str] = Field(default=None, max_length=100)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    cancelled_at: Optional[datetime] = Field(default=None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Estimated time
    estimated_duration_seconds: Optional[int] = Field(default=None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to API response dict."""
        return {
            "id": str(self.id),
            "job_type": self.job_type.value,
            "status": self.status.value,
            "priority": self.priority,
            "user_id": str(self.user_id) if self.user_id else None,
            "resource_type": self.resource_type,
            "resource_id": str(self.resource_id) if self.resource_id else None,
            "progress": {
                "percent": self.progress_percent,
                "message": self.progress_message,
                "current_step": self.current_step,
                "step_number": self.current_step_number,
                "total_steps": self.total_steps,
            },
            "result": self.result_data,
            "error": {
                "message": self.error_message,
                "details": self.error_details,
            } if self.error_message else None,
            "attempts": {
                "count": self.attempt_count,
                "max": self.max_attempts,
            },
            "timestamps": {
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            },
            "estimated_duration_seconds": self.estimated_duration_seconds,
        }


class JobHistory(SQLModel, table=True):
    """
    Audit trail for job state changes.
    
    Records every status transition for debugging and analytics.
    """
    __tablename__ = "job_history"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Reference to job
    job_id: uuid.UUID = Field(foreign_key="jobs.id", index=True)
    
    # State transition
    from_status: Optional[JobStatus] = Field(default=None)
    to_status: JobStatus
    
    # Context
    message: Optional[str] = Field(default=None, max_length=500)
    event_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Actor (user or system)
    actor_type: str = Field(default="system", max_length=20)  # "user", "system", "worker"
    actor_id: Optional[str] = Field(default=None, max_length=100)
    
    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class IngestionCheckpoint(SQLModel, table=True):
    """
    Checkpoint for resumable document ingestion.
    
    Saves progress after each major step to allow resumption
    from failure without reprocessing completed work.
    """
    __tablename__ = "ingestion_checkpoints"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Reference to job
    job_id: uuid.UUID = Field(foreign_key="jobs.id", index=True)
    document_id: uuid.UUID = Field(index=True)
    
    # Step tracking
    step_name: str = Field(max_length=50, index=True)  # e.g., "parsing", "chunking", "embedding"
    step_order: int = Field(default=0)
    is_completed: bool = Field(default=False)
    
    # Step-specific data for resumption
    step_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Metrics
    items_processed: int = Field(default=0)
    items_total: int = Field(default=0)
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Idempotency
    idempotency_key: Optional[str] = Field(default=None, max_length=100, index=True)


# =============================================================================
# Ingestion Step Constants
# =============================================================================

class IngestionStep:
    """Constants for ingestion pipeline steps."""
    PARSING = "parsing"
    CHUNKING = "chunking"
    HIERARCHY = "hierarchy"  # Parent summary generation
    EMBEDDING = "embedding"
    VECTOR_INDEX = "vector_index"
    GRAPH_EXTRACT = "graph_extract"
    GRAPH_INDEX = "graph_index"
    FINALIZE = "finalize"
    
    # Step order for resumption
    ORDER = {
        PARSING: 1,
        CHUNKING: 2,
        HIERARCHY: 3,
        EMBEDDING: 4,
        VECTOR_INDEX: 5,
        GRAPH_EXTRACT: 6,
        GRAPH_INDEX: 7,
        FINALIZE: 8,
    }
    
    # Progress percentages for each step
    PROGRESS = {
        PARSING: 10,
        CHUNKING: 25,
        HIERARCHY: 35,
        EMBEDDING: 55,
        VECTOR_INDEX: 70,
        GRAPH_EXTRACT: 85,
        GRAPH_INDEX: 95,
        FINALIZE: 100,
    }
    
    # Human-readable messages
    MESSAGES = {
        PARSING: "Parsing document...",
        CHUNKING: "Splitting into chunks...",
        HIERARCHY: "Generating summaries...",
        EMBEDDING: "Generating embeddings...",
        VECTOR_INDEX: "Indexing in vector store...",
        GRAPH_EXTRACT: "Extracting entities...",
        GRAPH_INDEX: "Building knowledge graph...",
        FINALIZE: "Finalizing...",
    }
