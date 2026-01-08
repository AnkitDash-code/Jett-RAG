"""
Memory API endpoints for the Supermemory System (Phase 6).
Provides CRUD and retrieval operations for episodic/semantic memories.
"""
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.api.deps import get_current_user, CurrentUser
from app.services.memory_service import get_memory_service
from app.models.memory import MemoryType, MemoryStatus
from app.schemas.memory import (
    CreateMemoryRequest,
    MemoryRetrievalRequest,
    MemoryFeedbackRequest,
    CreateMemoryLinkRequest,
    EvictMemoriesRequest,
    MemoryResponse,
    MemoryCreationResponse,
    MemoryRetrievalResponse,
    MemoryListResponse,
    ImportanceUpdateResponse,
    MemoryStatsResponse,
    MemoryLinkResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# CREATE MEMORIES
# ============================================================================

@router.post("", response_model=MemoryCreationResponse, status_code=201)
async def create_memory(
    request: CreateMemoryRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new memory (episodic or semantic).
    
    Memories are automatically indexed for vector similarity search
    and can be retrieved using hybrid scoring (vector + entity overlap).
    """
    service = get_memory_service(db)
    
    # Convert string type to enum
    memory_type = MemoryType.EPISODIC if request.memory_type == "episodic" else MemoryType.SEMANTIC
    
    # Parse conversation_id if provided
    conversation_id = None
    if request.conversation_id:
        try:
            conversation_id = uuid.UUID(request.conversation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation_id format")
    
    result = await service.create_memory(
        user_id=current_user.id,
        content=request.content,
        memory_type=memory_type,
        conversation_id=conversation_id,
        summary=request.summary,
        entity_ids=request.entity_ids,
        metadata=request.metadata,
        tenant_id=current_user.tenant_id,
    )
    
    await db.commit()
    
    return MemoryCreationResponse(
        memory=MemoryResponse.from_memory(result.memory),
        embedding_ms=result.embedding_ms,
        total_ms=result.total_ms,
        was_duplicate=result.was_duplicate,
    )


# ============================================================================
# LIST & RETRIEVE MEMORIES
# ============================================================================

@router.get("", response_model=MemoryListResponse)
async def list_memories(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    memory_type: Optional[str] = Query(None, description="Filter by type: episodic or semantic"),
    status: Optional[str] = Query(None, description="Filter by status: active, archived, consolidated"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List user's memories with pagination.
    
    Optionally filter by memory type (episodic/semantic) and status (active/archived).
    """
    from sqlmodel import select, func
    from app.models.memory import Memory
    
    # Build base query
    base_query = select(Memory).where(Memory.user_id == current_user.id)
    
    if memory_type:
        mem_type = MemoryType.EPISODIC if memory_type == "episodic" else MemoryType.SEMANTIC
        base_query = base_query.where(Memory.memory_type == mem_type)
    
    if status:
        try:
            mem_status = MemoryStatus(status)
            base_query = base_query.where(Memory.status == mem_status)
        except ValueError:
            pass  # Invalid status, ignore filter
    
    # Get total count
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination
    offset = (page - 1) * page_size
    paginated_query = base_query.order_by(Memory.created_at.desc()).offset(offset).limit(page_size)
    
    result = await db.execute(paginated_query)
    memories = list(result.scalars().all())
    
    return MemoryListResponse(
        memories=[MemoryResponse.from_memory(m) for m in memories],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific memory by ID."""
    from sqlmodel import select
    from app.models.memory import Memory
    
    try:
        mem_uuid = uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")
    
    stmt = select(Memory).where(
        Memory.id == mem_uuid,
        Memory.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    memory = result.scalar_one_or_none()
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return MemoryResponse.from_memory(memory)


@router.post("/retrieve", response_model=MemoryRetrievalResponse)
async def retrieve_memories(
    request: MemoryRetrievalRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve relevant memories using hybrid search.
    
    Hybrid Score = 0.6 × vector_similarity + 0.4 × entity_overlap
    
    Only returns memories with score >= relevance_threshold (default 0.3).
    """
    service = get_memory_service(db)
    
    # Convert memory_type if provided
    memory_type = None
    if request.memory_type:
        memory_type = MemoryType.EPISODIC if request.memory_type == "episodic" else MemoryType.SEMANTIC
    
    result = await service.retrieve_memories(
        query=request.query,
        user_id=current_user.id,
        top_k=request.top_k,
        memory_type=memory_type,
        relevance_threshold=request.relevance_threshold,
        include_archived=request.include_archived,
    )
    
    return MemoryRetrievalResponse(
        memories=[MemoryResponse.from_memory(m) for m in result.memories],
        hybrid_scores=result.hybrid_scores,
        retrieval_ms=result.retrieval_ms,
        method_used=result.method_used,
        total_candidates=result.total_candidates,
    )


# ============================================================================
# UPDATE MEMORIES
# ============================================================================

@router.post("/{memory_id}/feedback", response_model=MemoryResponse)
async def update_memory_feedback(
    memory_id: str,
    request: MemoryFeedbackRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update memory feedback score.
    
    Positive feedback (+delta) increases importance, negative (-delta) decreases it.
    Feedback score is clamped to 0-1 range.
    """
    service = get_memory_service(db)
    
    try:
        mem_uuid = uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")
    
    memory = await service.update_feedback(mem_uuid, request.feedback_delta)
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return MemoryResponse.from_memory(memory)


@router.post("/{memory_id}/restore", response_model=MemoryResponse)
async def restore_memory(
    memory_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Restore an archived memory to active status.
    
    Returns the memory to the cache and makes it available for retrieval.
    """
    service = get_memory_service(db)
    
    try:
        mem_uuid = uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")
    
    memory = await service.restore_memory(mem_uuid)
    
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return MemoryResponse.from_memory(memory)


# ============================================================================
# MEMORY LINKS
# ============================================================================

@router.post("/{memory_id}/links", response_model=MemoryLinkResponse, status_code=201)
async def create_memory_link(
    memory_id: str,
    request: CreateMemoryLinkRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a link between two memories for associative recall.
    
    Link types: related, causes, follows, contradicts, supports
    """
    service = get_memory_service(db)
    
    try:
        source_uuid = uuid.UUID(memory_id)
        target_uuid = uuid.UUID(request.target_memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")
    
    link = await service.create_link(
        source_memory_id=source_uuid,
        target_memory_id=target_uuid,
        link_type=request.link_type,
        strength=request.strength,
    )
    
    await db.commit()
    
    return MemoryLinkResponse(
        id=str(link.id),
        source_memory_id=str(link.source_memory_id),
        target_memory_id=str(link.target_memory_id),
        link_type=link.link_type,
        strength=link.strength,
        created_at=link.created_at,
    )


@router.get("/{memory_id}/linked", response_model=list[MemoryResponse])
async def get_linked_memories(
    memory_id: str,
    link_type: Optional[str] = Query(None, description="Filter by link type"),
    min_strength: float = Query(0.0, ge=0.0, le=1.0, description="Minimum link strength"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get memories linked to a given memory."""
    service = get_memory_service(db)
    
    try:
        mem_uuid = uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")
    
    memories = await service.get_linked_memories(
        memory_id=mem_uuid,
        link_type=link_type,
        min_strength=min_strength,
    )
    
    return [MemoryResponse.from_memory(m) for m in memories]


# ============================================================================
# IMPORTANCE & EVICTION
# ============================================================================

@router.post("/update-importance", response_model=ImportanceUpdateResponse)
async def update_importance_scores(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Recalculate importance scores for all user memories.
    
    Formula: importance = (1 / days_old) × interaction_count × feedback_score
    
    This is typically called by a maintenance job daily.
    """
    service = get_memory_service(db)
    
    result = await service.update_importance_scores(user_id=current_user.id)
    
    return ImportanceUpdateResponse(
        updated_count=result.updated_count,
        archived_count=result.archived_count,
        update_ms=result.update_ms,
        message=f"Updated {result.updated_count} memories",
    )


@router.post("/evict", response_model=ImportanceUpdateResponse)
async def evict_memories(
    request: EvictMemoriesRequest = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Archive low-importance memories (cache eviction only).
    
    Criteria: importance < threshold AND days_old > age_threshold
    
    Memories are NOT deleted, only archived from the cache.
    This is typically called by a maintenance job weekly.
    """
    service = get_memory_service(db)
    
    # Use defaults if no request body
    threshold = 0.3
    age_days = 30
    if request:
        threshold = request.importance_threshold
        age_days = request.age_threshold_days
    
    result = await service.evict_low_importance_memories(
        user_id=current_user.id,
        importance_threshold=threshold,
        age_threshold_days=age_days,
    )
    
    return ImportanceUpdateResponse(
        updated_count=result.updated_count,
        archived_count=result.archived_count,
        update_ms=result.update_ms,
        message=f"Archived {result.archived_count} low-importance memories",
    )


# ============================================================================
# STATISTICS
# ============================================================================

@router.get("/stats/summary", response_model=MemoryStatsResponse)
async def get_memory_stats(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get memory statistics for the current user."""
    service = get_memory_service(db)
    
    stats = await service.get_memory_stats(user_id=current_user.id)
    
    return MemoryStatsResponse(**stats)
