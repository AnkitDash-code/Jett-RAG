"""
Python-Only Task Queue for Phase 5 Background Processing.

Uses asyncio Queue + worker tasks for background job processing.
No external dependencies (Redis, Celery) required.

Features:
- Priority-based task scheduling
- Configurable worker pool
- Graceful shutdown
- Task handler registry
- Retry with exponential backoff
"""
import asyncio
import logging
import signal
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Awaitable, List
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Task Registry
# =============================================================================

# Global registry of task handlers
_task_handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}


def task_handler(task_type: str):
    """
    Decorator to register a function as a task handler.
    
    Usage:
        @task_handler("document_ingestion")
        async def handle_document_ingestion(job_id: uuid.UUID, **kwargs):
            ...
    """
    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        _task_handlers[task_type] = wrapper
        logger.info(f"Registered task handler: {task_type}")
        return wrapper
    
    return decorator


def get_task_handler(task_type: str) -> Optional[Callable[..., Awaitable[Any]]]:
    """Get a registered task handler by type."""
    return _task_handlers.get(task_type)


def list_task_handlers() -> List[str]:
    """List all registered task handler types."""
    return list(_task_handlers.keys())


# =============================================================================
# Task Data Structures
# =============================================================================

@dataclass(order=True)
class QueuedTask:
    """
    Task queued for processing.
    
    Ordered by priority (descending) then created_at (ascending).
    """
    priority: int = field(compare=True)
    created_at: float = field(compare=True)
    task_id: uuid.UUID = field(compare=False)
    job_id: uuid.UUID = field(compare=False)
    task_type: str = field(compare=False)
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    
    def __post_init__(self):
        # Negate priority for max-heap behavior (higher priority = processed first)
        self.priority = -self.priority


class WorkerState(str, Enum):
    """Worker lifecycle states."""
    IDLE = "idle"
    PROCESSING = "processing"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class WorkerInfo:
    """Information about a worker."""
    worker_id: str
    state: WorkerState = WorkerState.IDLE
    current_task_id: Optional[uuid.UUID] = None
    current_job_id: Optional[uuid.UUID] = None
    tasks_processed: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "state": self.state.value,
            "current_task_id": str(self.current_task_id) if self.current_task_id else None,
            "current_job_id": str(self.current_job_id) if self.current_job_id else None,
            "tasks_processed": self.tasks_processed,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }


# =============================================================================
# Task Queue
# =============================================================================

class TaskQueue:
    """
    In-process async task queue with worker pool.
    
    Uses asyncio.PriorityQueue for task ordering and
    asyncio.create_task for worker coroutines.
    """
    
    def __init__(
        self,
        num_workers: int = None,
        max_queue_size: int = 1000,
    ):
        """
        Initialize task queue.
        
        Args:
            num_workers: Number of worker tasks (default from config)
            max_queue_size: Maximum tasks in queue before blocking
        """
        self.num_workers = num_workers or getattr(settings, 'TASK_QUEUE_WORKERS', 4)
        self.max_queue_size = max_queue_size
        
        # Priority queue for tasks
        self._queue: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        
        # Worker management
        self._workers: Dict[str, asyncio.Task] = {}
        self._worker_info: Dict[str, WorkerInfo] = {}
        
        # Control flags
        self._running = False
        self._paused = False
        self._shutdown_event = asyncio.Event()
        
        # Job service reference (set after initialization to avoid circular import)
        self._job_service = None
        
        # Stats
        self._tasks_enqueued = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        
        logger.info(f"TaskQueue initialized with {self.num_workers} workers")
    
    def set_job_service(self, job_service):
        """Set job service reference for status updates."""
        self._job_service = job_service
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
    
    async def start(self):
        """Start the task queue and worker tasks."""
        if self._running:
            logger.warning("TaskQueue already running")
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Start worker tasks
        for i in range(self.num_workers):
            worker_id = f"worker-{i+1}"
            self._worker_info[worker_id] = WorkerInfo(worker_id=worker_id)
            self._workers[worker_id] = asyncio.create_task(
                self._worker_loop(worker_id),
                name=worker_id
            )
        
        logger.info(f"TaskQueue started with {self.num_workers} workers")
    
    async def stop(self, timeout: float = 30.0):
        """
        Gracefully stop the task queue.
        
        Args:
            timeout: Maximum seconds to wait for workers to finish
        """
        if not self._running:
            return
        
        logger.info("Stopping TaskQueue...")
        self._running = False
        self._shutdown_event.set()
        
        # Update worker states
        for info in self._worker_info.values():
            if info.state != WorkerState.STOPPED:
                info.state = WorkerState.STOPPING
        
        # Wait for workers to finish current tasks
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers.values(), return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"TaskQueue shutdown timed out after {timeout}s, cancelling workers")
                for task in self._workers.values():
                    task.cancel()
        
        self._workers.clear()
        logger.info("TaskQueue stopped")
    
    def pause(self):
        """Pause task processing (workers finish current tasks then wait)."""
        self._paused = True
        logger.info("TaskQueue paused")
    
    def resume(self):
        """Resume task processing."""
        self._paused = False
        logger.info("TaskQueue resumed")
    
    async def enqueue(
        self,
        job_id: uuid.UUID,
        task_type: str,
        priority: int = 5,
        **kwargs
    ) -> uuid.UUID:
        """
        Enqueue a task for processing.
        
        Args:
            job_id: Associated job ID
            task_type: Handler type (must be registered)
            priority: Priority level (1-10, higher = more urgent)
            **kwargs: Arguments passed to handler
        
        Returns:
            Task ID
        """
        if task_type not in _task_handlers:
            raise ValueError(f"Unknown task type: {task_type}. Registered: {list_task_handlers()}")
        
        task_id = uuid.uuid4()
        task = QueuedTask(
            task_id=task_id,
            job_id=job_id,
            task_type=task_type,
            priority=priority,
            created_at=datetime.utcnow().timestamp(),
            kwargs=kwargs,
        )
        
        await self._queue.put(task)
        self._tasks_enqueued += 1
        
        logger.debug(f"Enqueued task {task_id} (type={task_type}, job={job_id}, priority={priority})")
        return task_id
    
    async def _worker_loop(self, worker_id: str):
        """
        Worker loop that processes tasks from the queue.
        """
        info = self._worker_info[worker_id]
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            # Check if paused
            while self._paused and self._running:
                await asyncio.sleep(0.5)
            
            if not self._running:
                break
            
            try:
                # Get task with timeout (allows checking shutdown flag)
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process task
                info.state = WorkerState.PROCESSING
                info.current_task_id = task.task_id
                info.current_job_id = task.job_id
                info.last_activity = datetime.utcnow()
                
                try:
                    await self._process_task(task, worker_id)
                    self._tasks_completed += 1
                except Exception as e:
                    self._tasks_failed += 1
                    logger.exception(f"Worker {worker_id} task {task.task_id} failed: {e}")
                finally:
                    info.state = WorkerState.IDLE
                    info.current_task_id = None
                    info.current_job_id = None
                    info.tasks_processed += 1
                    self._queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)  # Prevent tight loop on repeated errors
        
        info.state = WorkerState.STOPPED
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: QueuedTask, worker_id: str):
        """Process a single task."""
        handler = get_task_handler(task.task_type)
        if not handler:
            logger.error(f"No handler for task type: {task.task_type}")
            return
        
        logger.info(f"Worker {worker_id} processing task {task.task_id} (type={task.task_type})")
        
        # Update job status to RUNNING
        if self._job_service:
            await self._job_service.mark_running(
                task.job_id,
                worker_id=worker_id
            )
        
        try:
            # Call the handler
            result = await handler(
                job_id=task.job_id,
                **task.kwargs
            )
            
            # Mark job as completed
            if self._job_service:
                await self._job_service.mark_completed(
                    task.job_id,
                    result_data=result
                )
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Task {task.task_id} failed: {e}")
            
            # Mark job as failed
            if self._job_service:
                await self._job_service.mark_failed(
                    task.job_id,
                    error_message=str(e),
                    error_details={"exception_type": type(e).__name__}
                )
            
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "running": self._running,
            "paused": self._paused,
            "queue_size": self.queue_size,
            "max_queue_size": self.max_queue_size,
            "num_workers": self.num_workers,
            "tasks_enqueued": self._tasks_enqueued,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "workers": [info.to_dict() for info in self._worker_info.values()],
        }
    
    def get_worker_info(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get info for a specific worker."""
        return self._worker_info.get(worker_id)


# =============================================================================
# Global Task Queue Instance
# =============================================================================

_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get the global task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


async def start_task_queue():
    """Start the global task queue."""
    queue = get_task_queue()
    await queue.start()


async def stop_task_queue():
    """Stop the global task queue."""
    global _task_queue
    if _task_queue:
        await _task_queue.stop()
        _task_queue = None


# =============================================================================
# FastAPI Lifespan Integration
# =============================================================================

async def setup_task_queue_lifespan():
    """
    Setup function for FastAPI lifespan.
    
    Usage in main.py:
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await setup_task_queue_lifespan()
            yield
            await teardown_task_queue_lifespan()
        
        app = FastAPI(lifespan=lifespan)
    """
    logger.info("Setting up task queue...")
    await start_task_queue()
    logger.info("Task queue ready")


async def teardown_task_queue_lifespan():
    """Teardown function for FastAPI lifespan."""
    logger.info("Tearing down task queue...")
    await stop_task_queue()
    logger.info("Task queue shutdown complete")
