"""
Admin API endpoints for Job Management (Phase 5).

Provides admin-level control over:
- Job monitoring and statistics
- Manual job triggering
- Task queue management
- Maintenance job control
"""
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database import get_db
from app.api.deps import get_current_user, CurrentUser
from app.services.job_service import JobService
from app.services.task_queue import get_task_queue
from app.models.job import JobType, JobStatus, JobPriority

router = APIRouter()


# ============================================================================
# Request/Response Schemas
# ============================================================================

class JobStats(BaseModel):
    """Job statistics response."""
    total_jobs: int
    pending: int
    running: int
    completed: int
    failed: int
    cancelled: int
    avg_duration_ms: Optional[float]
    oldest_pending_job: Optional[datetime]


class QueueStats(BaseModel):
    """Task queue statistics."""
    queue_size: int
    active_workers: int
    total_workers: int
    tasks_processed: int
    is_running: bool


class TriggerJobRequest(BaseModel):
    """Request to trigger a maintenance job."""
    job_type: str  # job_cleanup, graph_rebuild, vector_compaction, etc.
    priority: int = 1
    params: dict = {}


class TriggerJobResponse(BaseModel):
    """Response from triggering a job."""
    job_id: str
    job_type: str
    status: str
    message: str


# ============================================================================
# Admin Job Management Endpoints
# ============================================================================

@router.get("/jobs/stats", response_model=JobStats)
async def get_job_stats(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get job statistics for admin dashboard.
    
    Requires admin role.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    job_service = JobService(db)
    stats = await job_service.get_stats()
    
    return JobStats(
        total_jobs=stats.get("total", 0),
        pending=stats.get("pending", 0),
        running=stats.get("running", 0),
        completed=stats.get("completed", 0),
        failed=stats.get("failed", 0),
        cancelled=stats.get("cancelled", 0),
        avg_duration_ms=stats.get("avg_duration_ms"),
        oldest_pending_job=stats.get("oldest_pending"),
    )


@router.get("/jobs/queue", response_model=QueueStats)
async def get_queue_stats(
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get task queue statistics.
    
    Requires admin role.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    task_queue = get_task_queue()
    
    if not task_queue:
        return QueueStats(
            queue_size=0,
            active_workers=0,
            total_workers=0,
            tasks_processed=0,
            is_running=False,
        )
    
    stats = task_queue.get_stats()
    
    return QueueStats(
        queue_size=stats.get("queue_size", 0),
        active_workers=stats.get("active_workers", 0),
        total_workers=stats.get("total_workers", 0),
        tasks_processed=stats.get("tasks_processed", 0),
        is_running=stats.get("is_running", False),
    )


@router.get("/jobs/all")
async def list_all_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    job_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all jobs with filtering (admin view).
    
    Requires admin role.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    job_service = JobService(db)
    
    # Convert status string to enum
    status_enum = None
    if status:
        try:
            status_enum = JobStatus(status)
        except ValueError:
            pass
    
    type_enum = None
    if job_type:
        try:
            type_enum = JobType(job_type)
        except ValueError:
            pass
    
    jobs = await job_service.list_jobs(
        status=status_enum,
        job_type=type_enum,
        limit=limit,
        offset=offset,
    )
    
    return {
        "jobs": [
            {
                "id": str(j.id),
                "job_type": j.job_type.value if hasattr(j.job_type, 'value') else str(j.job_type),
                "status": j.status.value if hasattr(j.status, 'value') else str(j.status),
                "priority": j.priority,
                "progress_percent": j.progress_percent,
                "progress_message": j.progress_message,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "error_message": j.error_message,
            }
            for j in jobs
        ],
        "count": len(jobs),
        "limit": limit,
        "offset": offset,
    }


@router.post("/jobs/trigger", response_model=TriggerJobResponse)
async def trigger_maintenance_job(
    request: TriggerJobRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger a maintenance job.
    
    Supported job types:
    - job_cleanup: Clean up old completed/failed jobs
    - graph_rebuild: Rebuild knowledge graph communities
    - vector_compaction: Compact vector store indices
    - stream_cleanup: Clean up stale streams
    - orphan_cleanup: Clean up orphaned data
    
    Requires admin role.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    valid_types = {
        "job_cleanup", "graph_rebuild", "vector_compaction",
        "stream_cleanup", "orphan_cleanup",
    }
    
    if request.job_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job type. Must be one of: {', '.join(valid_types)}"
        )
    
    job_service = JobService(db)
    task_queue = get_task_queue()
    
    if not task_queue or not task_queue._running:
        raise HTTPException(
            status_code=503,
            detail="Task queue is not running"
        )
    
    # Create job record
    job = await job_service.create_job(
        job_type=JobType.MAINTENANCE,
        priority=request.priority,
        payload={
            "task_type": request.job_type,
            **request.params,
        },
        created_by=current_user.id,
    )
    
    # Enqueue task
    await task_queue.enqueue(
        task_type=request.job_type,
        priority=request.priority,
        job_id=job.id,
        **request.params,
    )
    
    return TriggerJobResponse(
        job_id=str(job.id),
        job_type=request.job_type,
        status="pending",
        message=f"Maintenance job '{request.job_type}' scheduled",
    )


@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retry a failed job.
    
    Creates a new job with the same parameters.
    Requires admin role.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    job_service = JobService(db)
    
    # Get original job
    original = await job_service.get_job(uuid.UUID(job_id))
    if not original:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if original.status not in [JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail="Only failed or cancelled jobs can be retried"
        )
    
    task_queue = get_task_queue()
    if not task_queue or not task_queue._running:
        raise HTTPException(
            status_code=503,
            detail="Task queue is not running"
        )
    
    # Create new job with same parameters
    payload = original.payload_json
    import json
    payload_dict = json.loads(payload) if payload else {}
    
    new_job = await job_service.create_job(
        job_type=original.job_type,
        priority=original.priority,
        payload=payload_dict,
        created_by=current_user.id,
    )
    
    # Determine task type from original
    task_type = payload_dict.get("task_type", "document_ingestion")
    
    # Enqueue
    await task_queue.enqueue(
        task_type=task_type,
        priority=original.priority,
        job_id=new_job.id,
        **payload_dict,
    )
    
    return {
        "original_job_id": job_id,
        "new_job_id": str(new_job.id),
        "status": "pending",
        "message": "Job retry scheduled",
    }


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a job record.
    
    Only completed, failed, or cancelled jobs can be deleted.
    Requires admin role.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    job_service = JobService(db)
    
    job = await job_service.get_job(uuid.UUID(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete pending or running jobs. Cancel first."
        )
    
    # Delete job and related records
    from app.models.job import JobHistory, IngestionCheckpoint
    from sqlmodel import delete
    
    await db.execute(
        delete(JobHistory).where(JobHistory.job_id == job.id)
    )
    await db.execute(
        delete(IngestionCheckpoint).where(IngestionCheckpoint.job_id == job.id)
    )
    await db.delete(job)
    await db.commit()
    
    return {"message": "Job deleted successfully", "job_id": job_id}


@router.post("/queue/pause")
async def pause_queue(
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Pause the task queue (stop processing new tasks).
    
    Requires admin role.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    task_queue = get_task_queue()
    if not task_queue:
        raise HTTPException(status_code=503, detail="Task queue not available")
    
    task_queue.pause()
    
    return {"message": "Task queue paused", "status": "paused"}


@router.post("/queue/resume")
async def resume_queue(
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Resume the task queue.
    
    Requires admin role.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    task_queue = get_task_queue()
    if not task_queue:
        raise HTTPException(status_code=503, detail="Task queue not available")
    
    task_queue.resume()
    
    return {"message": "Task queue resumed", "status": "running"}
