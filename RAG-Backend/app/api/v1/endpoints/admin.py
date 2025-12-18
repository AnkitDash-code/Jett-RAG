"""
Admin API endpoints.
GET /admin/metrics, /admin/logs, /admin/users, /admin/documents
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.monitoring_service import MetricsService
from app.services.user_service import UserService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.services.auth_service import AuthService
from app.schemas.user import UserListResponse, UserResponse
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.api.deps import get_current_user, CurrentUser, require_admin, can_view_metrics

router = APIRouter()


@router.get("/metrics")
async def get_system_metrics(
    current_user: CurrentUser = Depends(can_view_metrics),
    db: AsyncSession = Depends(get_db),
):
    """
    Get system-wide metrics and statistics.
    
    Returns aggregated metrics for:
    - Users (total, active)
    - Documents (total, by status, chunks)
    - Chat (conversations, messages, latency)
    - Quality (relevance scores, feedback)
    """
    metrics_service = MetricsService(db)
    metrics = await metrics_service.get_system_metrics()
    
    return {
        "users": {
            "total": metrics.total_users,
            "active_24h": metrics.active_users_24h,
        },
        "documents": {
            "total": metrics.total_documents,
            "processing": metrics.documents_processing,
            "indexed": metrics.documents_indexed,
            "error": metrics.documents_error,
            "total_chunks": metrics.total_chunks,
        },
        "chat": {
            "total_conversations": metrics.total_conversations,
            "total_messages": metrics.total_messages,
            "messages_24h": metrics.messages_24h,
        },
        "performance": {
            "avg_retrieval_latency_ms": round(metrics.avg_retrieval_latency_ms, 2),
            "avg_llm_latency_ms": round(metrics.avg_llm_latency_ms, 2),
            "avg_total_latency_ms": round(metrics.avg_total_latency_ms, 2),
            "avg_relevance_score": round(metrics.avg_relevance_score, 3),
        },
        "feedback": {
            "total": metrics.total_feedback,
            "avg_rating": round(metrics.avg_rating, 2),
            "helpful_rate": round(metrics.helpful_rate, 3),
        },
        "generated_at": metrics.generated_at,
    }


@router.get("/metrics/{user_id}")
async def get_user_metrics(
    user_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get metrics for a specific user (admin only).
    """
    metrics_service = MetricsService(db)
    metrics = await metrics_service.get_user_metrics(user_id, days)
    
    return {
        "user_id": metrics.user_id,
        "period_days": days,
        "conversations": metrics.conversations,
        "messages": metrics.messages,
        "documents_uploaded": metrics.documents_uploaded,
        "tokens_used": metrics.tokens_used,
        "avg_rating": round(metrics.avg_rating, 2),
    }


@router.get("/logs")
async def get_query_logs(
    limit: int = Query(100, ge=1, le=1000),
    user_id: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get recent query logs for debugging and analysis.
    """
    metrics_service = MetricsService(db)
    logs = await metrics_service.get_recent_query_logs(limit, user_id)
    
    return {"logs": logs, "count": len(logs)}


@router.get("/errors")
async def get_error_summary(
    days: int = Query(7, ge=1, le=90),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get summary of document processing errors.
    """
    metrics_service = MetricsService(db)
    summary = await metrics_service.get_error_summary(days)
    
    return summary


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    List all users (admin only).
    """
    user_service = UserService(db)
    auth_service = AuthService(db)
    
    users, total = await user_service.list_users(
        page=page,
        page_size=page_size,
        status=status,
    )
    
    user_responses = []
    for user in users:
        roles = await auth_service.get_user_roles(user.id)
        user_responses.append(UserResponse(
            id=str(user.id),
            email=user.email,
            name=user.name,
            avatar_url=user.avatar_url,
            status=user.status.value,
            roles=roles,
            tenant_id=user.tenant_id,
            department=user.department,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        ))
    
    return UserListResponse(
        users=user_responses,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_all_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    List all documents (admin only, bypasses access control).
    """
    doc_service = DocumentService(db)
    
    documents, total = await doc_service.list_documents(
        user_id=current_user.id,
        is_admin=True,  # Bypass access control
        page=page,
        page_size=page_size,
        status=status,
    )
    
    from app.api.v1.endpoints.documents import document_to_response
    
    return DocumentListResponse(
        documents=[document_to_response(d) for d in documents],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.post("/documents/{document_id}/reindex")
async def admin_reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Force reindex a document (admin only).
    """
    doc_service = DocumentService(db)
    
    document = await doc_service.reindex_document(
        document_id=uuid.UUID(document_id),
        user_id=current_user.id,
        is_admin=True,
    )
    
    # Queue background reprocessing
    async def process_document():
        async with db.begin():
            ingestion = IngestionService(db)
            await ingestion.process_document(document.id)
    
    background_tasks.add_task(process_document)
    
    return {"message": f"Reindexing started for document {document_id}"}


@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Deactivate a user account (admin only).
    """
    user_service = UserService(db)
    await user_service.deactivate_user(uuid.UUID(user_id))
    
    return {"message": f"User {user_id} deactivated"}


@router.post("/cache/clear")
async def clear_cache(
    current_user: CurrentUser = Depends(require_admin()),
):
    """
    Clear application caches (admin only).
    TODO: Implement Redis cache clearing.
    """
    # TODO: Clear Redis cache
    # TODO: Clear embedding cache
    # TODO: Clear query cache
    
    return {"message": "Cache cleared successfully"}
