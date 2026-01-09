"""
Job Service for Phase 5 Background Processing.

Provides job lifecycle management:
- Create, update, cancel jobs
- Progress tracking with SSE events
- Retry logic with exponential backoff
- Job history and cleanup
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, AsyncGenerator

from sqlalchemy import select, and_, or_, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.job import (
    Job, JobHistory, IngestionCheckpoint,
    JobType, JobStatus, JobPriority, IngestionStep
)
from app.database import get_db

logger = logging.getLogger(__name__)


# =============================================================================
# SSE Event Publisher
# =============================================================================

class JobEventPublisher:
    """
    In-memory pub/sub for job progress events.
    
    Allows SSE endpoints to subscribe to job updates.
    """
    
    def __init__(self):
        self._subscribers: Dict[uuid.UUID, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, job_id: uuid.UUID) -> asyncio.Queue:
        """Subscribe to events for a job."""
        async with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = []
            
            queue = asyncio.Queue(maxsize=100)
            self._subscribers[job_id].append(queue)
            logger.debug(f"New subscriber for job {job_id}")
            return queue
    
    async def unsubscribe(self, job_id: uuid.UUID, queue: asyncio.Queue):
        """Unsubscribe from job events."""
        async with self._lock:
            if job_id in self._subscribers:
                try:
                    self._subscribers[job_id].remove(queue)
                    if not self._subscribers[job_id]:
                        del self._subscribers[job_id]
                except ValueError:
                    pass
    
    async def publish(self, job_id: uuid.UUID, event: Dict[str, Any]):
        """Publish an event to all subscribers."""
        async with self._lock:
            if job_id not in self._subscribers:
                return
            
            dead_queues = []
            for queue in self._subscribers[job_id]:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    dead_queues.append(queue)
            
            # Clean up dead queues
            for queue in dead_queues:
                self._subscribers[job_id].remove(queue)


# Global event publisher
_event_publisher = JobEventPublisher()


def get_event_publisher() -> JobEventPublisher:
    """Get the global event publisher."""
    return _event_publisher


# =============================================================================
# Job Service
# =============================================================================

class JobService:
    """
    Service for job lifecycle management.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._publisher = get_event_publisher()
    
    # -------------------------------------------------------------------------
    # Create
    # -------------------------------------------------------------------------
    
    async def create_job(
        self,
        job_type: JobType,
        user_id: Optional[uuid.UUID] = None,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[uuid.UUID] = None,
        priority: int = JobPriority.NORMAL,
        total_steps: int = 1,
        estimated_duration_seconds: Optional[int] = None,
        max_attempts: int = 3,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """
        Create a new job.
        
        Returns:
            Created Job instance
        """
        job = Job(
            job_type=job_type,
            status=JobStatus.PENDING,
            priority=priority,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            total_steps=total_steps,
            estimated_duration_seconds=estimated_duration_seconds,
            max_attempts=max_attempts,
            payload=payload,
        )
        
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        
        # Record history
        await self._record_history(
            job_id=job.id,
            to_status=JobStatus.PENDING,
            message="Job created",
        )
        
        logger.info(f"Created job {job.id} (type={job_type.value})")
        return job
    
    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------
    
    async def get_job(self, job_id: uuid.UUID) -> Optional[Job]:
        """Get a job by ID."""
        result = await self.db.execute(
            select(Job).where(Job.id == job_id)
        )
        return result.scalar_one_or_none()
    
    async def list_jobs(
        self,
        user_id: Optional[uuid.UUID] = None,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with filters."""
        query = select(Job)
        
        conditions = []
        if user_id:
            conditions.append(Job.user_id == user_id)
        if job_type:
            conditions.append(Job.job_type == job_type)
        if status:
            conditions.append(Job.status == status)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(Job.created_at.desc()).limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def count_jobs(
        self,
        user_id: Optional[uuid.UUID] = None,
        status: Optional[JobStatus] = None,
    ) -> int:
        """Count jobs matching filters."""
        query = select(func.count(Job.id))
        
        conditions = []
        if user_id:
            conditions.append(Job.user_id == user_id)
        if status:
            conditions.append(Job.status == status)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        result = await self.db.execute(query)
        return result.scalar() or 0
    
    async def get_pending_jobs(self, limit: int = 10) -> List[Job]:
        """Get pending jobs ordered by priority and creation time."""
        result = await self.db.execute(
            select(Job)
            .where(Job.status == JobStatus.PENDING)
            .order_by(Job.priority.desc(), Job.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    # -------------------------------------------------------------------------
    # Update Status
    # -------------------------------------------------------------------------
    
    async def mark_running(
        self,
        job_id: uuid.UUID,
        worker_id: Optional[str] = None,
    ) -> Optional[Job]:
        """Mark a job as running."""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        old_status = job.status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        job.attempt_count += 1
        job.worker_id = worker_id
        
        await self.db.commit()
        await self.db.refresh(job)
        
        await self._record_history(
            job_id=job_id,
            from_status=old_status,
            to_status=JobStatus.RUNNING,
            message=f"Started by worker {worker_id}" if worker_id else "Started",
            actor_type="worker",
            actor_id=worker_id,
        )
        
        await self._publish_progress(job)
        
        logger.info(f"Job {job_id} marked as running (attempt {job.attempt_count})")
        return job
    
    async def mark_completed(
        self,
        job_id: uuid.UUID,
        result_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Job]:
        """Mark a job as completed."""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        old_status = job.status
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        job.progress_percent = 100
        job.progress_message = "Completed"
        job.result_data = result_data
        
        await self.db.commit()
        await self.db.refresh(job)
        
        await self._record_history(
            job_id=job_id,
            from_status=old_status,
            to_status=JobStatus.COMPLETED,
            message="Job completed successfully",
            metadata={"result_summary": str(result_data)[:200] if result_data else None},
        )
        
        await self._publish_progress(job)
        
        logger.info(f"Job {job_id} completed")
        return job
    
    async def mark_failed(
        self,
        job_id: uuid.UUID,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Job]:
        """Mark a job as failed."""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        old_status = job.status
        job.status = JobStatus.FAILED
        job.updated_at = datetime.utcnow()
        job.error_message = error_message
        job.error_details = error_details
        
        await self.db.commit()
        await self.db.refresh(job)
        
        await self._record_history(
            job_id=job_id,
            from_status=old_status,
            to_status=JobStatus.FAILED,
            message=f"Failed: {error_message}",
            metadata=error_details,
        )
        
        await self._publish_progress(job)
        
        logger.error(f"Job {job_id} failed: {error_message}")
        return job
    
    async def cancel_job(
        self,
        job_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        reason: Optional[str] = None,
    ) -> Optional[Job]:
        """Cancel a job."""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            logger.warning(f"Cannot cancel job {job_id} in status {job.status}")
            return job
        
        old_status = job.status
        job.status = JobStatus.CANCELLED
        job.cancelled_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        job.progress_message = reason or "Cancelled by user"
        
        await self.db.commit()
        await self.db.refresh(job)
        
        await self._record_history(
            job_id=job_id,
            from_status=old_status,
            to_status=JobStatus.CANCELLED,
            message=reason or "Cancelled",
            actor_type="user" if user_id else "system",
            actor_id=str(user_id) if user_id else None,
        )
        
        await self._publish_progress(job)
        
        logger.info(f"Job {job_id} cancelled")
        return job
    
    # -------------------------------------------------------------------------
    # Progress Tracking
    # -------------------------------------------------------------------------
    
    async def update_progress(
        self,
        job_id: uuid.UUID,
        percent: int,
        message: Optional[str] = None,
        current_step: Optional[str] = None,
        current_step_number: Optional[int] = None,
    ) -> Optional[Job]:
        """Update job progress."""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.progress_percent = min(100, max(0, percent))
        job.updated_at = datetime.utcnow()
        
        if message:
            job.progress_message = message
        if current_step:
            job.current_step = current_step
        if current_step_number is not None:
            job.current_step_number = current_step_number
        
        await self.db.commit()
        await self.db.refresh(job)
        
        await self._publish_progress(job)
        
        logger.debug(f"Job {job_id} progress: {percent}% - {message or current_step}")
        return job
    
    async def _publish_progress(self, job: Job):
        """Publish progress event to subscribers."""
        event = {
            "type": "progress",
            "job_id": str(job.id),
            "status": job.status.value,
            "progress": job.progress_percent,
            "message": job.progress_message,
            "current_step": job.current_step,
            "step_number": job.current_step_number,
            "total_steps": job.total_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self._publisher.publish(job.id, event)
    
    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    
    async def save_checkpoint(
        self,
        job_id: uuid.UUID,
        document_id: uuid.UUID,
        step_name: str,
        step_data: Optional[Dict[str, Any]] = None,
        items_processed: int = 0,
        items_total: int = 0,
    ) -> IngestionCheckpoint:
        """Save an ingestion checkpoint."""
        checkpoint = IngestionCheckpoint(
            job_id=job_id,
            document_id=document_id,
            step_name=step_name,
            step_order=IngestionStep.ORDER.get(step_name, 0),
            is_completed=True,
            step_data=step_data,
            items_processed=items_processed,
            items_total=items_total,
            completed_at=datetime.utcnow(),
            idempotency_key=f"{job_id}-{step_name}",
        )
        
        self.db.add(checkpoint)
        await self.db.commit()
        
        # Refresh with error handling for detached instances
        try:
            await self.db.refresh(checkpoint)
        except Exception as e:
            logger.debug(f"Could not refresh checkpoint (not critical): {e}")
            # Re-fetch from DB if refresh fails
            from sqlmodel import select
            stmt = select(IngestionCheckpoint).where(
                IngestionCheckpoint.job_id == job_id,
                IngestionCheckpoint.step_name == step_name
            )
            result = await self.db.execute(stmt)
            refreshed = result.scalars().first()
            if refreshed:
                checkpoint = refreshed
        
        logger.debug(f"Saved checkpoint: job={job_id}, step={step_name}")
        return checkpoint
    
    async def get_last_checkpoint(
        self,
        job_id: uuid.UUID,
    ) -> Optional[IngestionCheckpoint]:
        """Get the last completed checkpoint for a job."""
        result = await self.db.execute(
            select(IngestionCheckpoint)
            .where(
                and_(
                    IngestionCheckpoint.job_id == job_id,
                    IngestionCheckpoint.is_completed == True,
                )
            )
            .order_by(IngestionCheckpoint.step_order.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_checkpoints(
        self,
        job_id: uuid.UUID,
    ) -> List[IngestionCheckpoint]:
        """Get all checkpoints for a job."""
        result = await self.db.execute(
            select(IngestionCheckpoint)
            .where(IngestionCheckpoint.job_id == job_id)
            .order_by(IngestionCheckpoint.step_order.asc())
        )
        return list(result.scalars().all())
    
    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------
    
    async def _record_history(
        self,
        job_id: uuid.UUID,
        to_status: JobStatus,
        from_status: Optional[JobStatus] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        actor_type: str = "system",
        actor_id: Optional[str] = None,
    ):
        """Record a job history entry."""
        history = JobHistory(
            job_id=job_id,
            from_status=from_status,
            to_status=to_status,
            message=message,
            event_metadata=metadata,
            actor_type=actor_type,
            actor_id=actor_id,
        )
        
        self.db.add(history)
        await self.db.commit()
    
    async def get_job_history(
        self,
        job_id: uuid.UUID,
    ) -> List[JobHistory]:
        """Get history for a job."""
        result = await self.db.execute(
            select(JobHistory)
            .where(JobHistory.job_id == job_id)
            .order_by(JobHistory.created_at.asc())
        )
        return list(result.scalars().all())
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    async def cleanup_old_jobs(
        self,
        days: int = 7,
        statuses: Optional[List[JobStatus]] = None,
    ) -> int:
        """Delete old completed/failed jobs."""
        statuses = statuses or [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Delete history first (foreign key constraint)
        subquery = select(Job.id).where(
            and_(
                Job.status.in_(statuses),
                Job.created_at < cutoff,
            )
        )
        
        await self.db.execute(
            delete(JobHistory).where(JobHistory.job_id.in_(subquery))
        )
        
        await self.db.execute(
            delete(IngestionCheckpoint).where(IngestionCheckpoint.job_id.in_(subquery))
        )
        
        # Then delete jobs
        result = await self.db.execute(
            delete(Job).where(
                and_(
                    Job.status.in_(statuses),
                    Job.created_at < cutoff,
                )
            )
        )
        
        await self.db.commit()
        
        deleted = result.rowcount
        logger.info(f"Cleaned up {deleted} old jobs")
        return deleted
    
    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        # Count by status
        status_counts = {}
        for status in JobStatus:
            result = await self.db.execute(
                select(func.count(Job.id)).where(Job.status == status)
            )
            status_counts[status.value] = result.scalar() or 0
        
        # Count by type
        type_counts = {}
        for job_type in JobType:
            result = await self.db.execute(
                select(func.count(Job.id)).where(Job.job_type == job_type)
            )
            type_counts[job_type.value] = result.scalar() or 0
        
        # Average completion time for completed jobs
        result = await self.db.execute(
            select(func.avg(
                func.julianday(Job.completed_at) - func.julianday(Job.started_at)
            ))
            .where(
                and_(
                    Job.status == JobStatus.COMPLETED,
                    Job.started_at.isnot(None),
                    Job.completed_at.isnot(None),
                )
            )
        )
        avg_duration_days = result.scalar() or 0
        avg_duration_seconds = int(avg_duration_days * 86400) if avg_duration_days else 0
        
        # Failure rate
        total_result = await self.db.execute(
            select(func.count(Job.id)).where(
                Job.status.in_([JobStatus.COMPLETED, JobStatus.FAILED])
            )
        )
        total = total_result.scalar() or 0
        failure_rate = (status_counts.get("failed", 0) / total * 100) if total > 0 else 0
        
        return {
            "by_status": status_counts,
            "by_type": type_counts,
            "avg_duration_seconds": avg_duration_seconds,
            "failure_rate_percent": round(failure_rate, 2),
            "total_jobs": sum(status_counts.values()),
        }


# =============================================================================
# Dependency
# =============================================================================

async def get_job_service(db: AsyncSession) -> JobService:
    """Get job service instance."""
    return JobService(db)
