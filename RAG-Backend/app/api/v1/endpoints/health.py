"""
Health check and monitoring endpoints.
"""
import logging
from datetime import datetime

from fastapi import APIRouter, Response

from app.config import settings
from app.services.metrics_service import get_metrics_service
from app.services.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Returns service status and configuration info.
    """
    vector_store = FAISSVectorStore()
    
    # Check component health
    components = {
        "api": True,
        "vector_store": vector_store.is_initialized(),
        "llm": True,  # Will be checked below
    }
    
    # Try to check LLM connectivity
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.LLM_API_URL.rstrip('/v1/chat/completions')}")
            components["llm"] = resp.status_code < 500
    except Exception as e:
        components["llm"] = False
        logger.warning(f"LLM health check failed: {e}")
    
    # Overall health
    is_healthy = all(components.values())
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "components": components,
        "config": {
            "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
            "cross_encoder": settings.ENABLE_CROSS_ENCODER,
            "query_expansion": settings.ENABLE_QUERY_EXPANSION,
            "rate_limiting": settings.RATE_LIMIT_ENABLED,
        }
    }


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check for load balancers.
    Returns 200 if service is ready to accept requests.
    """
    vector_store = FAISSVectorStore()
    
    if not vector_store.is_initialized():
        return Response(
            content='{"status": "not_ready", "reason": "vector_store_not_initialized"}',
            status_code=503,
            media_type="application/json"
        )
    
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check for orchestration systems.
    Returns 200 if the service process is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    metrics_service = get_metrics_service()
    content, content_type = metrics_service.get_metrics()
    
    return Response(content=content, media_type=content_type)


@router.get("/stats")
async def get_stats():
    """
    Get system statistics.
    Returns document/chunk counts and usage stats.
    """
    vector_store = FAISSVectorStore()
    
    # Get document stats
    try:
        doc_count = len(set(
            m.get("document_id") 
            for m in vector_store._metadata.values() 
            if m.get("document_id")
        ))
        chunk_count = len(vector_store._metadata)
    except Exception:
        doc_count = 0
        chunk_count = 0
    
    # Update metrics
    metrics = get_metrics_service()
    metrics.set_document_count(doc_count)
    metrics.set_chunk_count(chunk_count)
    
    return {
        "documents": {
            "indexed": doc_count,
        },
        "chunks": {
            "indexed": chunk_count,
        },
        "features": {
            "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
            "cross_encoder": settings.ENABLE_CROSS_ENCODER,
            "query_expansion": settings.ENABLE_QUERY_EXPANSION,
            "utility_llm": settings.UTILITY_LLM_ENABLED,
        }
    }
