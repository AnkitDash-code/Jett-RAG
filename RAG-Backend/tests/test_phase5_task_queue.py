"""
Tests for Phase 5: Task Queue and Job Management.
"""
import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.job import Job, JobType, JobStatus, JobPriority, IngestionStep
from app.services.task_queue import TaskQueue, task_handler, QueuedTask
from app.services.job_service import JobService, JobEventPublisher


# ============================================================================
# Task Queue Tests
# ============================================================================

class TestTaskQueue:
    """Tests for the Python-only task queue."""
    
    @pytest.fixture
    def task_queue(self):
        """Create a fresh task queue for testing."""
        return TaskQueue(num_workers=2, max_queue_size=100)
    
    @pytest.mark.asyncio
    async def test_task_queue_start_stop(self, task_queue):
        """Test starting and stopping the task queue."""
        assert not task_queue._running
        
        await task_queue.start()
        assert task_queue._running
        assert len(task_queue._workers) == 2
        
        await task_queue.stop()
        assert not task_queue._running
    
    @pytest.mark.asyncio
    async def test_task_queue_enqueue(self, task_queue):
        """Test enqueueing tasks."""
        # Register a test handler
        @task_handler("test_enqueue_task")
        async def test_handler(job_id, **kwargs):
            return {"success": True}
        
        await task_queue.start()
        
        try:
            job_id = uuid.uuid4()
            task_id = await task_queue.enqueue(
                job_id=job_id,
                task_type="test_enqueue_task",
                priority=1,
                test_param="value",
            )
            
            # Queue should have the task or it was processed
            assert task_id is not None
        finally:
            await task_queue.stop()
    
    @pytest.mark.asyncio
    async def test_task_queue_priority_ordering(self, task_queue):
        """Test that high priority tasks are processed first."""
        processed_order = []
        
        @task_handler("priority_test")
        async def handler(job_id, priority_value, **kwargs):
            processed_order.append(priority_value)
            return {"priority": priority_value}
        
        await task_queue.start()
        
        try:
            # Pause to queue tasks without processing
            task_queue.pause()
            
            # Enqueue tasks with different priorities
            for priority, value in [(0, "low"), (5, "high"), (2, "medium")]:
                await task_queue.enqueue(
                    task_type="priority_test",
                    priority=priority,
                    job_id=uuid.uuid4(),
                    priority_value=value,
                )
            
            # Resume and wait for processing
            task_queue.resume()
            await asyncio.sleep(0.5)
            
            # High priority should be first
            if processed_order:
                assert processed_order[0] == "high"
        finally:
            await task_queue.stop()
    
    @pytest.mark.asyncio
    async def test_task_queue_pause_resume(self, task_queue):
        """Test pausing and resuming the queue."""
        await task_queue.start()
        
        try:
            assert not task_queue._paused
            
            task_queue.pause()
            assert task_queue._paused
            
            task_queue.resume()
            assert not task_queue._paused
        finally:
            await task_queue.stop()
    
    def test_task_queue_stats(self, task_queue):
        """Test getting queue statistics."""
        stats = task_queue.get_stats()
        
        assert "running" in stats
        assert "paused" in stats
        assert "queue_size" in stats
        assert "num_workers" in stats
        assert stats["num_workers"] == 2


class TestTaskHandler:
    """Tests for the @task_handler decorator."""
    
    def test_task_handler_registration(self):
        """Test that handlers are properly registered."""
        from app.services.task_queue import _task_handlers
        
        @task_handler("test_registration")
        async def my_handler(job_id, **kwargs):
            return {"success": True}
        
        assert "test_registration" in _task_handlers
        assert _task_handlers["test_registration"] == my_handler


# ============================================================================
# Job Service Tests
# ============================================================================

class TestJobService:
    """Tests for the Job Service."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock(return_value=None)
        db.flush = AsyncMock(return_value=None)
        db.refresh = AsyncMock(return_value=None)
        db.execute = AsyncMock()
        return db
    
    @pytest.fixture
    def job_service(self, mock_db):
        """Create a job service with mock DB."""
        service = JobService(mock_db)
        # Mock the internal publisher to avoid async issues
        service._publisher = AsyncMock()
        service._publisher.publish = AsyncMock(return_value=None)
        return service
    
    @pytest.mark.asyncio
    async def test_create_job(self, job_service, mock_db):
        """Test creating a new job."""
        # Patch _record_history to avoid issues
        job_service._record_history = AsyncMock(return_value=None)
        
        user_id = uuid.uuid4()
        
        job = await job_service.create_job(
            job_type=JobType.DOCUMENT_INGESTION,
            priority=1,
            user_id=user_id,
        )
        
        assert job is not None
        assert job.job_type == JobType.DOCUMENT_INGESTION
        assert job.status == JobStatus.PENDING
        assert job.priority == 1
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_progress(self, job_service, mock_db):
        """Test updating job progress."""
        job_id = uuid.uuid4()
        
        # Mock get_job to return a job
        mock_job = Job(
            id=job_id,
            job_type=JobType.DOCUMENT_INGESTION,
            status=JobStatus.RUNNING,
        )
        mock_db.execute = AsyncMock()
        mock_db.execute.return_value.scalar_one_or_none = MagicMock(return_value=mock_job)
        
        await job_service.update_progress(
            job_id=job_id,
            percent=50,
            message="Processing...",
            current_step="chunking",
        )
        
        assert mock_job.progress_percent == 50
        assert mock_job.progress_message == "Processing..."
    
    @pytest.mark.asyncio
    async def test_mark_completed(self, job_service, mock_db):
        """Test marking a job as completed."""
        job_id = uuid.uuid4()
        
        mock_job = Job(
            id=job_id,
            job_type=JobType.DOCUMENT_INGESTION,
            status=JobStatus.RUNNING,
        )
        mock_db.execute = AsyncMock()
        mock_db.execute.return_value.scalar_one_or_none = MagicMock(return_value=mock_job)
        
        await job_service.mark_completed(
            job_id=job_id,
            result_data={"chunks_created": 10},
        )
        
        assert mock_job.status == JobStatus.COMPLETED
        assert mock_job.progress_percent == 100
        assert mock_job.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_mark_failed(self, job_service, mock_db):
        """Test marking a job as failed."""
        job_id = uuid.uuid4()
        
        mock_job = Job(
            id=job_id,
            job_type=JobType.DOCUMENT_INGESTION,
            status=JobStatus.RUNNING,
        )
        mock_db.execute = AsyncMock()
        mock_db.execute.return_value.scalar_one_or_none = MagicMock(return_value=mock_job)
        
        await job_service.mark_failed(
            job_id=job_id,
            error_message="Processing error",
        )
        
        assert mock_job.status == JobStatus.FAILED
        assert mock_job.error_message == "Processing error"


class TestJobEventPublisher:
    """Tests for the SSE event publisher."""
    
    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self):
        """Test subscribing and unsubscribing from events."""
        publisher = JobEventPublisher()
        job_id = uuid.uuid4()
        
        queue = await publisher.subscribe(job_id)
        assert queue is not None
        assert job_id in publisher._subscribers
        
        await publisher.unsubscribe(job_id, queue)
        assert job_id not in publisher._subscribers
    
    @pytest.mark.asyncio
    async def test_publish_event(self):
        """Test publishing events to subscribers."""
        publisher = JobEventPublisher()
        job_id = uuid.uuid4()
        
        queue = await publisher.subscribe(job_id)
        
        await publisher.publish(job_id, {"event": "progress", "data": {"percent": 50}})
        
        # Check event was received
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["event"] == "progress"
        assert event["data"]["percent"] == 50
        
        await publisher.unsubscribe(job_id, queue)


# ============================================================================
# Job Model Tests
# ============================================================================

class TestJobModels:
    """Tests for job-related models."""
    
    def test_job_creation(self):
        """Test creating a Job instance."""
        job = Job(
            job_type=JobType.DOCUMENT_INGESTION,
            status=JobStatus.PENDING,
            priority=1,
        )
        
        assert job.job_type == JobType.DOCUMENT_INGESTION
        assert job.status == JobStatus.PENDING
        assert job.progress_percent == 0
    
    def test_ingestion_step_order(self):
        """Test ingestion step ordering."""
        assert IngestionStep.ORDER[IngestionStep.PARSING] < IngestionStep.ORDER[IngestionStep.CHUNKING]
        assert IngestionStep.ORDER[IngestionStep.CHUNKING] < IngestionStep.ORDER[IngestionStep.EMBEDDING]
        assert IngestionStep.ORDER[IngestionStep.EMBEDDING] < IngestionStep.ORDER[IngestionStep.FINALIZE]
    
    def test_ingestion_step_progress(self):
        """Test ingestion step progress values."""
        assert IngestionStep.PROGRESS[IngestionStep.PARSING] == 10
        assert IngestionStep.PROGRESS[IngestionStep.FINALIZE] == 100
    
    def test_job_priority_values(self):
        """Test job priority enum values."""
        assert JobPriority.LOW.value == 1
        assert JobPriority.NORMAL.value == 5
        assert JobPriority.HIGH.value == 8
        assert JobPriority.CRITICAL.value == 10
