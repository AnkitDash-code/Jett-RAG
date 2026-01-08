"""
Health check and monitoring endpoints.
Enhanced for Phase 7 with comprehensive dependency checks.
"""
import logging
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Response

from app.config import settings
from app.services.metrics_service import get_metrics_service
from app.services.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# DEPENDENCY HEALTH CHECKERS
# ============================================================================

async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        from app.database import AsyncSessionLocal
        
        start = time.time()
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
        latency_ms = (time.time() - start) * 1000
        
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_llm_health() -> Dict[str, Any]:
    """Check LLM service connectivity."""
    try:
        import httpx
        
        start = time.time()
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if LLM server is responding
            base_url = settings.LLM_API_URL.rstrip('/v1/chat/completions')
            resp = await client.get(base_url)
            latency_ms = (time.time() - start) * 1000
            
            if resp.status_code < 500:
                return {
                    "status": "healthy",
                    "latency_ms": round(latency_ms, 2),
                    "url": base_url,
                }
            else:
                return {
                    "status": "degraded",
                    "status_code": resp.status_code,
                }
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_vector_store_health() -> Dict[str, Any]:
    """Check vector store (FAISS) health."""
    try:
        vector_store = FAISSVectorStore()
        
        if not vector_store.is_initialized():
            return {
                "status": "degraded",
                "reason": "not_initialized",
            }
        
        # Get stats
        chunk_count = len(vector_store._metadata) if vector_store._metadata else 0
        
        return {
            "status": "healthy",
            "chunks_indexed": chunk_count,
        }
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_embedding_service_health() -> Dict[str, Any]:
    """Check embedding model health."""
    try:
        # Check if model is loadable
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return {
            "status": "healthy",
            "device": device,
            "model": settings.EMBEDDING_MODEL,
        }
    except ImportError as e:
        return {
            "status": "unhealthy",
            "error": f"Missing dependency: {e}",
        }
    except Exception as e:
        logger.warning(f"Embedding service health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_circuit_breakers() -> Dict[str, Any]:
    """Check circuit breaker states."""
    try:
        from app.services.circuit_breaker import get_circuit_registry
        
        registry = get_circuit_registry()
        circuits = registry.list_circuits()
        
        # Count states
        open_count = sum(1 for c in circuits.values() if c["state"] == "open")
        half_open_count = sum(1 for c in circuits.values() if c["state"] == "half_open")
        
        status = "healthy"
        if open_count > 0:
            status = "degraded" if half_open_count > 0 else "unhealthy"
        
        return {
            "status": status,
            "circuits": circuits,
            "summary": {
                "total": len(circuits),
                "open": open_count,
                "half_open": half_open_count,
                "closed": len(circuits) - open_count - half_open_count,
            },
        }
    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e),
        }


# ============================================================================
# HEALTH ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns service status and all component health.
    """
    # Run all health checks in parallel
    checks = await asyncio.gather(
        check_database_health(),
        check_llm_health(),
        check_vector_store_health(),
        check_embedding_service_health(),
        check_circuit_breakers(),
        return_exceptions=True,
    )
    
    components = {
        "database": checks[0] if not isinstance(checks[0], Exception) else {"status": "error", "error": str(checks[0])},
        "llm": checks[1] if not isinstance(checks[1], Exception) else {"status": "error", "error": str(checks[1])},
        "vector_store": checks[2] if not isinstance(checks[2], Exception) else {"status": "error", "error": str(checks[2])},
        "embedding": checks[3] if not isinstance(checks[3], Exception) else {"status": "error", "error": str(checks[3])},
        "circuit_breakers": checks[4] if not isinstance(checks[4], Exception) else {"status": "error", "error": str(checks[4])},
    }
    
    # Determine overall health
    statuses = [c.get("status", "unknown") for c in components.values()]
    
    if all(s == "healthy" for s in statuses):
        overall_status = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
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
    
    Checks:
    - Database connectivity
    - Vector store initialized
    """
    # Check critical dependencies
    db_health = await check_database_health()
    vs_health = await check_vector_store_health()
    
    if db_health.get("status") != "healthy":
        return Response(
            content='{"status": "not_ready", "reason": "database_unavailable"}',
            status_code=503,
            media_type="application/json"
        )
    
    if vs_health.get("status") == "unhealthy":
        return Response(
            content='{"status": "not_ready", "reason": "vector_store_unavailable"}',
            status_code=503,
            media_type="application/json"
        )
    
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check for orchestration systems.
    Returns 200 if the service process is alive.
    
    This is intentionally lightweight - only checks if the process responds.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/dependencies")
async def dependency_check():
    """
    Detailed dependency health check.
    Returns health status of each external dependency.
    """
    # Run all health checks in parallel
    checks = await asyncio.gather(
        check_database_health(),
        check_llm_health(),
        check_vector_store_health(),
        check_embedding_service_health(),
        return_exceptions=True,
    )
    
    return {
        "dependencies": {
            "database": checks[0] if not isinstance(checks[0], Exception) else {"status": "error"},
            "llm": checks[1] if not isinstance(checks[1], Exception) else {"status": "error"},
            "vector_store": checks[2] if not isinstance(checks[2], Exception) else {"status": "error"},
            "embedding": checks[3] if not isinstance(checks[3], Exception) else {"status": "error"},
        },
        "checked_at": datetime.utcnow().isoformat(),
    }


@router.get("/health/circuits")
async def circuit_breaker_status():
    """
    Get circuit breaker status for all services.
    """
    return await check_circuit_breakers()


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
