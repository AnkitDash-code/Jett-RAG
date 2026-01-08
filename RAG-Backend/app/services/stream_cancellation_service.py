"""
Stream Cancellation Service for Phase 5.

Provides ability to cancel in-progress streaming responses:
- Track active streams by session/user
- Cancel streams via API or on client disconnect
- Clean up resources when streams are cancelled

This improves UX by allowing users to stop long-running generations.
"""
import asyncio
import logging
import uuid
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ActiveStream:
    """Metadata for an active streaming session."""
    stream_id: str
    user_id: uuid.UUID
    conversation_id: Optional[str]
    started_at: datetime
    cancel_event: asyncio.Event
    tokens_generated: int = 0
    cancelled: bool = False
    cancel_reason: Optional[str] = None


class StreamCancellationService:
    """
    Service for managing and cancelling active streaming sessions.
    
    Usage:
    1. Register stream when starting: stream_id = register(user_id)
    2. Check cancellation in generator: if is_cancelled(stream_id): break
    3. Cancel via API: cancel(stream_id) or cancel_user_streams(user_id)
    4. Cleanup when done: unregister(stream_id)
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._active_streams: Dict[str, ActiveStream] = {}
        self._user_streams: Dict[str, Set[str]] = {}  # user_id -> set of stream_ids
        self._lock = asyncio.Lock()
        self._timeout = getattr(settings, 'STREAM_TIMEOUT_SECONDS', 120)
        self._initialized = True
        
        logger.info(f"StreamCancellationService initialized (timeout: {self._timeout}s)")
    
    async def register(
        self,
        user_id: uuid.UUID,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        Register a new streaming session.
        
        Args:
            user_id: User starting the stream
            conversation_id: Optional conversation context
            
        Returns:
            Unique stream ID for tracking
        """
        stream_id = str(uuid.uuid4())
        user_key = str(user_id)
        
        async with self._lock:
            # Create stream record
            stream = ActiveStream(
                stream_id=stream_id,
                user_id=user_id,
                conversation_id=conversation_id,
                started_at=datetime.utcnow(),
                cancel_event=asyncio.Event(),
            )
            
            self._active_streams[stream_id] = stream
            
            # Track by user
            if user_key not in self._user_streams:
                self._user_streams[user_key] = set()
            self._user_streams[user_key].add(stream_id)
        
        logger.debug(f"Registered stream {stream_id} for user {user_id}")
        return stream_id
    
    async def unregister(self, stream_id: str) -> None:
        """
        Unregister a completed or cancelled stream.
        
        Args:
            stream_id: Stream ID to unregister
        """
        async with self._lock:
            stream = self._active_streams.pop(stream_id, None)
            
            if stream:
                user_key = str(stream.user_id)
                if user_key in self._user_streams:
                    self._user_streams[user_key].discard(stream_id)
                    if not self._user_streams[user_key]:
                        del self._user_streams[user_key]
        
        logger.debug(f"Unregistered stream {stream_id}")
    
    async def cancel(
        self,
        stream_id: str,
        reason: str = "User requested cancellation",
    ) -> bool:
        """
        Cancel an active stream.
        
        Args:
            stream_id: Stream ID to cancel
            reason: Reason for cancellation
            
        Returns:
            True if stream was found and cancelled
        """
        async with self._lock:
            stream = self._active_streams.get(stream_id)
            
            if stream and not stream.cancelled:
                stream.cancelled = True
                stream.cancel_reason = reason
                stream.cancel_event.set()  # Signal the generator to stop
                logger.info(f"Cancelled stream {stream_id}: {reason}")
                return True
        
        return False
    
    async def cancel_user_streams(
        self,
        user_id: uuid.UUID,
        reason: str = "User cancelled all streams",
    ) -> int:
        """
        Cancel all active streams for a user.
        
        Args:
            user_id: User ID
            reason: Cancellation reason
            
        Returns:
            Number of streams cancelled
        """
        user_key = str(user_id)
        cancelled_count = 0
        
        async with self._lock:
            stream_ids = list(self._user_streams.get(user_key, set()))
        
        for stream_id in stream_ids:
            if await self.cancel(stream_id, reason):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} streams for user {user_id}")
        return cancelled_count
    
    def is_cancelled(self, stream_id: str) -> bool:
        """
        Check if a stream has been cancelled.
        
        This should be called in the generator loop to check for cancellation.
        
        Args:
            stream_id: Stream ID to check
            
        Returns:
            True if stream is cancelled
        """
        stream = self._active_streams.get(stream_id)
        return stream.cancelled if stream else True
    
    async def wait_for_cancel_or_timeout(
        self,
        stream_id: str,
        timeout: float = 0.1,
    ) -> bool:
        """
        Wait for cancellation event with timeout.
        
        Use this in generators for periodic cancellation checks.
        
        Args:
            stream_id: Stream ID
            timeout: How long to wait
            
        Returns:
            True if cancelled, False if timeout
        """
        stream = self._active_streams.get(stream_id)
        if not stream:
            return True
        
        try:
            await asyncio.wait_for(
                stream.cancel_event.wait(),
                timeout=timeout,
            )
            return True  # Event was set (cancelled)
        except asyncio.TimeoutError:
            return False  # Timeout, not cancelled
    
    def update_progress(self, stream_id: str, tokens: int = 1) -> None:
        """
        Update token count for a stream.
        
        Args:
            stream_id: Stream ID
            tokens: Number of tokens to add
        """
        stream = self._active_streams.get(stream_id)
        if stream:
            stream.tokens_generated += tokens
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an active stream.
        
        Args:
            stream_id: Stream ID
            
        Returns:
            Stream info dict or None
        """
        stream = self._active_streams.get(stream_id)
        if not stream:
            return None
        
        elapsed = (datetime.utcnow() - stream.started_at).total_seconds()
        
        return {
            "stream_id": stream.stream_id,
            "user_id": str(stream.user_id),
            "conversation_id": stream.conversation_id,
            "started_at": stream.started_at.isoformat(),
            "elapsed_seconds": elapsed,
            "tokens_generated": stream.tokens_generated,
            "cancelled": stream.cancelled,
            "cancel_reason": stream.cancel_reason,
        }
    
    def get_active_streams(
        self,
        user_id: Optional[uuid.UUID] = None,
    ) -> list[Dict[str, Any]]:
        """
        Get list of active streams.
        
        Args:
            user_id: Optional filter by user
            
        Returns:
            List of stream info dicts
        """
        streams = []
        
        if user_id:
            user_key = str(user_id)
            stream_ids = self._user_streams.get(user_key, set())
        else:
            stream_ids = self._active_streams.keys()
        
        for stream_id in stream_ids:
            info = self.get_stream_info(stream_id)
            if info:
                streams.append(info)
        
        return streams
    
    async def cleanup_stale_streams(self) -> int:
        """
        Clean up streams that exceeded timeout.
        
        Returns:
            Number of streams cleaned up
        """
        now = datetime.utcnow()
        stale_ids = []
        
        async with self._lock:
            for stream_id, stream in self._active_streams.items():
                elapsed = (now - stream.started_at).total_seconds()
                if elapsed > self._timeout:
                    stale_ids.append(stream_id)
        
        for stream_id in stale_ids:
            await self.cancel(stream_id, "Stream timeout")
            await self.unregister(stream_id)
        
        if stale_ids:
            logger.info(f"Cleaned up {len(stale_ids)} stale streams")
        
        return len(stale_ids)


# Singleton getter
_stream_service: Optional[StreamCancellationService] = None


def get_stream_cancellation_service() -> StreamCancellationService:
    """Get stream cancellation service instance."""
    global _stream_service
    if _stream_service is None:
        _stream_service = StreamCancellationService()
    return _stream_service


# Helper context manager for stream management
class StreamContext:
    """
    Context manager for stream lifecycle management.
    
    Usage:
        async with StreamContext(user_id) as stream_id:
            async for token in generate():
                if stream_ctx.is_cancelled():
                    break
                yield token
    """
    
    def __init__(
        self,
        user_id: uuid.UUID,
        conversation_id: Optional[str] = None,
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.stream_id: Optional[str] = None
        self._service = get_stream_cancellation_service()
    
    async def __aenter__(self) -> str:
        self.stream_id = await self._service.register(
            self.user_id,
            self.conversation_id,
        )
        return self.stream_id
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.stream_id:
            await self._service.unregister(self.stream_id)
    
    def is_cancelled(self) -> bool:
        if not self.stream_id:
            return True
        return self._service.is_cancelled(self.stream_id)
    
    def update_progress(self, tokens: int = 1) -> None:
        if self.stream_id:
            self._service.update_progress(self.stream_id, tokens)
