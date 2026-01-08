"""
Job Status API Endpoints for Phase 5.

Provides:
- GET /jobs - List user's jobs with filters
- GET /jobs/{job_id} - Get job details
- POST /jobs/{job_id}/cancel - Cancel a job
- GET /jobs/{job_id}/stream - SSE progress stream
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.api.deps import get_current_user, CurrentUser, require_admin
from app.services.job_service import JobService, get_job_service, get_event_publisher
from app.models.job import JobType, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


# =============================================================================
# Response Models
# =============================================================================

class JobProgressResponse(BaseModel):
    """Job progress details."""
    percent: int
    message: Optional[str]
    current_step: Optional[str]
    step_number: int
    total_steps: int


class JobTimestampsResponse(BaseModel):
    """Job timestamps."""
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    cancelled_at: Optional[str]


class JobErrorResponse(BaseModel):
    """Job error details."""
    message: Optional[str]
    details: Optional[dict]


class JobResponse(BaseModel):
    """Job details response."""
    id: str
    job_type: str
    status: str
    priority: int
    user_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    progress: JobProgressResponse
    result: Optional[dict]
    error: Optional[JobErrorResponse]
    attempts: dict
    timestamps: JobTimestampsResponse
    estimated_duration_seconds: Optional[int]
    
    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Paginated job list."""
    jobs: List[JobResponse]
    total: int
    limit: int
    offset: int


class CancelJobRequest(BaseModel):
    """Cancel job request."""
    reason: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=JobListResponse)
async def list_jobs(
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List the current user's jobs.
    
    Supports filtering by job type and status.
    """
    job_service = JobService(db)
    
    # Parse filters
    job_type_enum = None
    if job_type:
        try:
            job_type_enum = JobType(job_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job_type. Valid values: {[t.value for t in JobType]}"
            )
    
    status_enum = None
    if status:
        try:
            status_enum = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid values: {[s.value for s in JobStatus]}"
            )
    
    # Get jobs
    jobs = await job_service.list_jobs(
        user_id=current_user.id,
        job_type=job_type_enum,
        status=status_enum,
        limit=limit,
        offset=offset,
    )
    
    total = await job_service.count_jobs(user_id=current_user.id)
    
    return JobListResponse(
        jobs=[JobResponse(**job.to_dict()) for job in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get details for a specific job.
    """
    job_service = JobService(db)
    job = await job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership (unless admin)
    if job.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return JobResponse(**job.to_dict())


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: uuid.UUID,
    request: CancelJobRequest = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a running or pending job.
    """
    job_service = JobService(db)
    job = await job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership (unless admin)
    if job.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if cancellable
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in status: {job.status.value}"
        )
    
    reason = request.reason if request else None
    job = await job_service.cancel_job(
        job_id=job_id,
        user_id=current_user.id,
        reason=reason,
    )
    
    return {"status": "cancelled", "job_id": str(job_id)}


@router.get("/{job_id}/stream")
async def stream_job_progress(
    job_id: uuid.UUID,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Stream job progress updates via Server-Sent Events (SSE).
    
    Returns events in format:
        data: {"progress": 45, "message": "Chunking document...", "status": "running"}
    
    Stream ends when job completes, fails, or is cancelled.
    """
    job_service = JobService(db)
    job = await job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if job.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # If job already terminal, return immediately
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        async def terminal_stream():
            data = {
                "type": "complete",
                "job_id": str(job_id),
                "status": job.status.value,
                "progress": job.progress_percent,
                "message": job.progress_message,
            }
            yield f"data: {__import__('json').dumps(data)}\n\n"
        
        return StreamingResponse(
            terminal_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    # Subscribe to job events
    publisher = get_event_publisher()
    queue = await publisher.subscribe(job_id)
    
    async def event_stream():
        import json
        
        # Send initial state
        initial = {
            "type": "initial",
            "job_id": str(job_id),
            "status": job.status.value,
            "progress": job.progress_percent,
            "message": job.progress_message,
            "current_step": job.current_step,
        }
        yield f"data: {json.dumps(initial)}\n\n"
        
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                try:
                    # Wait for event with timeout (for heartbeat)
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # End stream if job is terminal
                    if event.get("status") in ("completed", "failed", "cancelled"):
                        break
                        
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f": heartbeat\n\n"
                    
        finally:
            await publisher.unsubscribe(job_id, queue)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/{job_id}/history")
async def get_job_history(
    job_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get status change history for a job.
    """
    job_service = JobService(db)
    job = await job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if job.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    history = await job_service.get_job_history(job_id)
    
    return {
        "job_id": str(job_id),
        "history": [
            {
                "from_status": h.from_status.value if h.from_status else None,
                "to_status": h.to_status.value,
                "message": h.message,
                "actor_type": h.actor_type,
                "created_at": h.created_at.isoformat(),
            }
            for h in history
        ],
    }


@router.get("/{job_id}/checkpoints")
async def get_job_checkpoints(
    job_id: uuid.UUID,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get ingestion checkpoints for a job.
    
    Useful for seeing which steps completed before a failure.
    """
    job_service = JobService(db)
    job = await job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if job.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    checkpoints = await job_service.get_checkpoints(job_id)
    
    return {
        "job_id": str(job_id),
        "checkpoints": [
            {
                "step_name": c.step_name,
                "step_order": c.step_order,
                "is_completed": c.is_completed,
                "items_processed": c.items_processed,
                "items_total": c.items_total,
                "started_at": c.started_at.isoformat(),
                "completed_at": c.completed_at.isoformat() if c.completed_at else None,
            }
            for c in checkpoints
        ],
    }
