"""
Prometheus metrics for monitoring.
Provides metrics for chat, retrieval, and system performance.
"""
import logging
import time
from typing import Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client not installed. Metrics disabled.")


# ============================================================================
# Metric Definitions (only created if prometheus available)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Chat metrics
    CHAT_REQUESTS = Counter(
        "rag_chat_requests_total",
        "Total chat requests",
        ["status", "role"]
    )
    
    CHAT_LATENCY = Histogram(
        "rag_chat_latency_seconds",
        "Chat request latency in seconds",
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    )
    
    CHAT_TOKENS = Counter(
        "rag_chat_tokens_total",
        "Total tokens processed",
        ["type"]  # prompt, completion
    )
    
    # Retrieval metrics
    RETRIEVAL_REQUESTS = Counter(
        "rag_retrieval_requests_total",
        "Total retrieval requests",
        ["status"]
    )
    
    RETRIEVAL_LATENCY = Histogram(
        "rag_retrieval_latency_seconds",
        "Retrieval latency in seconds",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    )
    
    RETRIEVAL_CHUNKS = Histogram(
        "rag_retrieval_chunks_count",
        "Number of chunks retrieved per request",
        buckets=[0, 1, 3, 5, 10, 20, 50]
    )
    
    # Document metrics
    DOCUMENTS_UPLOADED = Counter(
        "rag_documents_uploaded_total",
        "Total documents uploaded",
        ["status"]
    )
    
    DOCUMENTS_INDEXED = Gauge(
        "rag_documents_indexed_count",
        "Current number of indexed documents"
    )
    
    CHUNKS_INDEXED = Gauge(
        "rag_chunks_indexed_count",
        "Current number of indexed chunks"
    )
    
    # LLM metrics
    LLM_REQUESTS = Counter(
        "rag_llm_requests_total",
        "Total LLM API requests",
        ["llm_type", "status"]  # main/utility, success/error
    )
    
    LLM_LATENCY = Histogram(
        "rag_llm_latency_seconds",
        "LLM request latency in seconds",
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
    )
    
    # System metrics
    ACTIVE_SESSIONS = Gauge(
        "rag_active_sessions_count",
        "Current number of active chat sessions"
    )


class MetricsService:
    """
    Service for recording and exposing Prometheus metrics.
    """
    
    def __init__(self):
        self.available = PROMETHEUS_AVAILABLE
        if not self.available:
            logger.warning("MetricsService: prometheus-client not available")
    
    # ========================================================================
    # Chat Metrics
    # ========================================================================
    
    def record_chat_request(
        self,
        status: str = "success",
        role: str = "user",
        latency: Optional[float] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ):
        """Record a chat request."""
        if not self.available:
            return
        
        CHAT_REQUESTS.labels(status=status, role=role).inc()
        
        if latency is not None:
            CHAT_LATENCY.observe(latency)
        
        if prompt_tokens > 0:
            CHAT_TOKENS.labels(type="prompt").inc(prompt_tokens)
        if completion_tokens > 0:
            CHAT_TOKENS.labels(type="completion").inc(completion_tokens)
    
    # ========================================================================
    # Retrieval Metrics
    # ========================================================================
    
    def record_retrieval(
        self,
        status: str = "success",
        latency: Optional[float] = None,
        chunk_count: int = 0
    ):
        """Record a retrieval operation."""
        if not self.available:
            return
        
        RETRIEVAL_REQUESTS.labels(status=status).inc()
        
        if latency is not None:
            RETRIEVAL_LATENCY.observe(latency)
        
        RETRIEVAL_CHUNKS.observe(chunk_count)
    
    # ========================================================================
    # Document Metrics
    # ========================================================================
    
    def record_document_upload(self, status: str = "success"):
        """Record a document upload."""
        if not self.available:
            return
        DOCUMENTS_UPLOADED.labels(status=status).inc()
    
    def set_document_count(self, count: int):
        """Set the current document count."""
        if not self.available:
            return
        DOCUMENTS_INDEXED.set(count)
    
    def set_chunk_count(self, count: int):
        """Set the current chunk count."""
        if not self.available:
            return
        CHUNKS_INDEXED.set(count)
    
    # ========================================================================
    # LLM Metrics
    # ========================================================================
    
    def record_llm_request(
        self,
        llm_type: str = "main",
        status: str = "success",
        latency: Optional[float] = None
    ):
        """Record an LLM request."""
        if not self.available:
            return
        
        LLM_REQUESTS.labels(llm_type=llm_type, status=status).inc()
        
        if latency is not None:
            LLM_LATENCY.observe(latency)
    
    # ========================================================================
    # Session Metrics
    # ========================================================================
    
    def set_active_sessions(self, count: int):
        """Set the current active session count."""
        if not self.available:
            return
        ACTIVE_SESSIONS.set(count)
    
    # ========================================================================
    # Export
    # ========================================================================
    
    def get_metrics(self) -> tuple[bytes, str]:
        """
        Get metrics in Prometheus format.
        Returns (content, content_type).
        """
        if not self.available:
            return b"# Metrics disabled\n", "text/plain"
        
        return generate_latest(), CONTENT_TYPE_LATEST


# Global metrics service instance
_metrics: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    """Get or create metrics service singleton."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsService()
    return _metrics


def timed_operation(metric_type: str = "retrieval"):
    """
    Decorator for timing operations.
    
    Usage:
        @timed_operation("retrieval")
        async def retrieve(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = get_metrics_service()
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                latency = time.time() - start_time
                
                if metric_type == "retrieval":
                    metrics.record_retrieval(status=status, latency=latency)
                elif metric_type == "llm":
                    metrics.record_llm_request(status=status, latency=latency)
        
        return wrapper
    return decorator
