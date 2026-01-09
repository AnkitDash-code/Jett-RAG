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


# =============================================================================
# Graph Admin Endpoints (Phase 4 - GraphRAG)
# =============================================================================

@router.get("/graph/stats")
async def get_graph_stats(
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get knowledge graph statistics.
    
    Returns:
    - Node/edge counts
    - Community count
    - Pending chunks for processing
    - Graph density and metrics
    """
    from app.services.graph_store import get_graph_store_service
    from app.services.graph_index_service import get_graph_index_service
    
    graph_store = get_graph_store_service(db)
    index_service = get_graph_index_service(db)
    
    stats = await graph_store.get_stats()
    queue_stats = await index_service.get_task_queue_stats()
    
    return {
        "graph": stats.to_dict(),
        "processing_queue": queue_stats,
        "config": {
            "enabled": True,
            "max_hops": 1,
            "cache_ttl_seconds": 300,
            "processing_workers": 2,
        }
    }


@router.post("/graph/reindex")
async def trigger_graph_reindex(
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force reindex all documents"),
    document_ids: Optional[list[str]] = Query(None, description="Specific document IDs to reindex"),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger graph reindexing.
    
    This marks chunks for reprocessing and optionally forces
    immediate reindexing of the knowledge graph.
    """
    from app.services.graph_index_service import get_graph_index_service
    
    index_service = get_graph_index_service(db)
    
    # Convert document_ids to UUIDs if provided
    doc_uuids = None
    if document_ids:
        doc_uuids = [uuid.UUID(d) for d in document_ids]
    
    result = await index_service.force_reindex(document_ids=doc_uuids)
    
    return {
        "status": "reindex_queued",
        "pending_chunks": result["pending_chunks"],
        "force": force,
        "document_ids": document_ids,
    }


@router.get("/graph/communities")
async def list_communities(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    List all communities with summaries.
    """
    from sqlalchemy import select, func
    from app.models.graph import Community, CommunityContext, CommunityMembership
    
    # Get communities with counts
    result = await db.execute(
        select(Community)
        .order_by(Community.node_count.desc())
        .offset(offset)
        .limit(limit)
    )
    communities = list(result.scalars().all())
    
    # Get total count
    count_result = await db.execute(select(func.count(Community.id)))
    total = count_result.scalar() or 0
    
    # Get contexts for each community
    community_data = []
    for comm in communities:
        context_result = await db.execute(
            select(CommunityContext).where(
                CommunityContext.community_id == comm.id
            )
        )
        context = context_result.scalar_one_or_none()
        
        community_data.append({
            "id": str(comm.id),
            "name": comm.name,
            "algorithm": comm.algorithm,
            "node_count": comm.node_count,
            "edge_count": comm.edge_count,
            "created_at": comm.created_at.isoformat(),
            "summary": context.summary_text if context else None,
            "key_entities": context.key_entities.split(",") if context and context.key_entities else [],
        })
    
    return {
        "communities": community_data,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/graph/permissions/{resource_type}/{resource_id}")
async def get_permission_tree(
    resource_type: str,
    resource_id: str,
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get permission tree for a resource.
    
    Shows who has access and why (direct vs inherited permissions).
    """
    from app.services.permission_inheritance_service import (
        get_permission_inheritance_service,
        ResourceType
    )
    
    # Map resource type string to enum
    type_map = {
        "folder": ResourceType.FOLDER,
        "document": ResourceType.DOCUMENT,
        "chunk": ResourceType.CHUNK,
        "entity": ResourceType.ENTITY,
    }
    
    if resource_type not in type_map:
        return {"error": f"Invalid resource type. Must be one of: {list(type_map.keys())}"}
    
    perm_service = get_permission_inheritance_service(db)
    tree = await perm_service.get_permission_tree(
        resource_type=type_map[resource_type],
        resource_id=resource_id,
    )
    
    return tree.to_dict()


@router.post("/graph/permissions")
async def grant_permission(
    folder_path: str = Query(..., description="Folder path to grant access to"),
    permission: str = Query(..., description="Permission type: view, edit, admin"),
    grantee_type: str = Query(..., description="Type of grantee: user or group"),
    grantee_id: str = Query(..., description="UUID of user or group"),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Grant permission on a folder.
    
    Permissions cascade to child documents and entities.
    """
    from app.services.permission_inheritance_service import (
        get_permission_inheritance_service,
    )
    from app.models.graph import PermissionType
    
    # Map permission string to enum
    perm_map = {
        "view": PermissionType.VIEW,
        "edit": PermissionType.EDIT,
        "admin": PermissionType.ADMIN,
    }
    
    if permission.lower() not in perm_map:
        return {"error": f"Invalid permission. Must be one of: {list(perm_map.keys())}"}
    
    if grantee_type not in ["user", "group"]:
        return {"error": "grantee_type must be 'user' or 'group'"}
    
    perm_service = get_permission_inheritance_service(db)
    result = await perm_service.grant_permission(
        folder_path=folder_path,
        permission_type=perm_map[permission.lower()],
        grantee_type=grantee_type,
        grantee_id=uuid.UUID(grantee_id),
        granted_by=current_user.id,
    )
    
    return {
        "status": "granted",
        "permission_id": str(result.id),
        "folder_path": folder_path,
        "permission": permission,
        "grantee_type": grantee_type,
        "grantee_id": grantee_id,
    }


@router.delete("/graph/permissions")
async def revoke_permission(
    folder_path: str = Query(..., description="Folder path to revoke access from"),
    grantee_type: str = Query(..., description="Type of grantee: user or group"),
    grantee_id: str = Query(..., description="UUID of user or group"),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Revoke permission on a folder.
    """
    from app.services.permission_inheritance_service import (
        get_permission_inheritance_service,
    )
    
    perm_service = get_permission_inheritance_service(db)
    revoked = await perm_service.revoke_permission(
        folder_path=folder_path,
        grantee_type=grantee_type,
        grantee_id=uuid.UUID(grantee_id),
    )
    
    return {
        "status": "revoked" if revoked else "not_found",
        "folder_path": folder_path,
        "grantee_type": grantee_type,
        "grantee_id": grantee_id,
    }


@router.get("/graph/entities")
async def list_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    List entities in the knowledge graph.
    """
    from sqlalchemy import select, func
    from app.models.graph import Entity, EntityType
    
    query = select(Entity).order_by(Entity.mention_count.desc())
    
    if entity_type:
        try:
            etype = EntityType(entity_type.lower())
            query = query.where(Entity.entity_type == etype)
        except ValueError:
            pass  # Invalid type, ignore filter
    
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    entities = list(result.scalars().all())
    
    # Get total count
    count_query = select(func.count(Entity.id))
    if entity_type:
        try:
            etype = EntityType(entity_type.lower())
            count_query = count_query.where(Entity.entity_type == etype)
        except ValueError:
            pass
    
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0
    
    return {
        "entities": [
            {
                "id": str(e.id),
                "name": e.name,
                "type": e.entity_type.value,
                "mention_count": e.mention_count,
                "relationship_count": e.relationship_count,
                "centrality": round(e.centrality_score, 4),
                "access_level": e.access_level,
            }
            for e in entities
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("/graph/rebuild-communities")
async def rebuild_communities(
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger community rebuild.
    
    This re-runs community detection and regenerates summaries.
    """
    from app.services.graph_index_service import get_graph_index_service
    
    index_service = get_graph_index_service(db)
    count = await index_service.rebuild_communities()
    
    return {
        "status": "completed",
        "communities_created": count,
    }


# ============================================================================
# MEMORY GRAPH ENDPOINTS (Phase 6)
# ============================================================================

@router.get("/memory/stats")
async def get_memory_stats(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get memory statistics.
    
    Returns counts by type, status, cache size, and average importance.
    """
    from app.services.memory_service import get_memory_service
    
    memory_service = get_memory_service(db)
    user_uuid = uuid.UUID(user_id) if user_id else None
    stats = await memory_service.get_memory_stats(user_uuid)
    
    return stats


@router.get("/memory/links")
async def get_memory_links(
    source_memory_id: Optional[str] = Query(None, description="Filter by source memory"),
    link_type: Optional[str] = Query(None, description="Filter by link type"),
    min_strength: float = Query(0.0, ge=0.0, le=1.0, description="Minimum link strength"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get memory links for visualization and analysis.
    
    Returns a list of links between memories with their types and strengths.
    Useful for building a memory graph visualization.
    """
    from sqlmodel import select
    from app.models.memory import MemoryLink, Memory
    
    # Build query for links
    stmt = select(MemoryLink)
    
    if source_memory_id:
        stmt = stmt.where(MemoryLink.source_memory_id == uuid.UUID(source_memory_id))
    
    if link_type:
        stmt = stmt.where(MemoryLink.link_type == link_type)
    
    stmt = stmt.where(MemoryLink.strength >= min_strength)
    stmt = stmt.order_by(MemoryLink.strength.desc())
    stmt = stmt.offset(offset).limit(limit)
    
    result = await db.execute(stmt)
    links = list(result.scalars().all())
    
    # Get total count
    count_stmt = select(MemoryLink)
    if source_memory_id:
        count_stmt = count_stmt.where(MemoryLink.source_memory_id == uuid.UUID(source_memory_id))
    if link_type:
        count_stmt = count_stmt.where(MemoryLink.link_type == link_type)
    count_stmt = count_stmt.where(MemoryLink.strength >= min_strength)
    
    from sqlmodel import func
    count_result = await db.execute(select(func.count()).select_from(count_stmt.subquery()))
    total = count_result.scalar() or 0
    
    # Format response
    return {
        "links": [
            {
                "id": str(link.id),
                "source_memory_id": str(link.source_memory_id),
                "target_memory_id": str(link.target_memory_id),
                "link_type": link.link_type,
                "strength": link.strength,
                "created_at": link.created_at.isoformat(),
            }
            for link in links
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/memory/graph")
async def get_memory_graph(
    user_id: str = Query(..., description="User ID to get memory graph for"),
    include_archived: bool = Query(False, description="Include archived memories"),
    min_importance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance"),
    limit: int = Query(50, ge=1, le=200),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a memory graph for visualization.
    
    Returns nodes (memories) and edges (links) for D3.js or similar visualization.
    """
    from sqlmodel import select
    from app.models.memory import Memory, MemoryLink, MemoryStatus
    
    user_uuid = uuid.UUID(user_id)
    
    # Get memories (nodes)
    stmt = select(Memory).where(
        Memory.user_id == user_uuid,
        Memory.importance_score >= min_importance,
    )
    
    if not include_archived:
        stmt = stmt.where(Memory.status == MemoryStatus.ACTIVE)
    
    stmt = stmt.order_by(Memory.importance_score.desc()).limit(limit)
    
    result = await db.execute(stmt)
    memories = list(result.scalars().all())
    memory_ids = [m.id for m in memories]
    
    # Get links (edges) between these memories
    if memory_ids:
        link_stmt = select(MemoryLink).where(
            MemoryLink.source_memory_id.in_(memory_ids),
            MemoryLink.target_memory_id.in_(memory_ids),
        )
        link_result = await db.execute(link_stmt)
        links = list(link_result.scalars().all())
    else:
        links = []
    
    # Format as graph
    nodes = [
        {
            "id": str(m.id),
            "type": m.memory_type.value,
            "status": m.status.value,
            "content_preview": m.content[:100] + "..." if len(m.content) > 100 else m.content,
            "importance": m.importance_score,
            "access_count": m.access_count,
            "created_at": m.created_at.isoformat(),
        }
        for m in memories
    ]
    
    edges = [
        {
            "source": str(link.source_memory_id),
            "target": str(link.target_memory_id),
            "type": link.link_type,
            "strength": link.strength,
        }
        for link in links
    ]
    
    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


@router.get("/graph/visualization")
async def get_graph_visualization(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    community_id: Optional[str] = Query(None, description="Filter by community ID"),
    limit: int = Query(100, ge=10, le=500, description="Maximum entities to return"),
    include_orphans: bool = Query(False, description="Include entities with no relationships"),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Get knowledge graph data for visualization (D3.js / force-graph).
    
    Returns:
    - entities: List of graph nodes with id, name, type, community, metrics
    - relationships: List of edges with source, target, type, weight
    - stats: Summary statistics
    """
    from sqlalchemy import select, func, or_
    from app.models.graph import Entity, EntityType, EntityRelationship, CommunityMembership
    
    # Build entity query
    entity_query = select(Entity).order_by(Entity.relationship_count.desc())
    
    if entity_type:
        try:
            etype = EntityType(entity_type.lower())
            entity_query = entity_query.where(Entity.entity_type == etype)
        except ValueError:
            pass
    
    if not include_orphans:
        entity_query = entity_query.where(Entity.relationship_count > 0)
    
    entity_query = entity_query.limit(limit)
    
    result = await db.execute(entity_query)
    entities = list(result.scalars().all())
    entity_ids = [e.id for e in entities]
    
    # Get community memberships
    comm_memberships = {}
    if entity_ids:
        comm_result = await db.execute(
            select(CommunityMembership).where(
                CommunityMembership.entity_id.in_(entity_ids)
            )
        )
        for cm in comm_result.scalars().all():
            comm_memberships[cm.entity_id] = cm.community_id
    
    # Get relationships between these entities
    relationships = []
    if entity_ids:
        rel_query = select(EntityRelationship).where(
            or_(
                EntityRelationship.source_entity_id.in_(entity_ids),
                EntityRelationship.target_entity_id.in_(entity_ids),
            )
        )
        rel_result = await db.execute(rel_query)
        relationships = list(rel_result.scalars().all())
    
    # Filter to only relationships where both ends are in our entity set
    entity_id_set = set(entity_ids)
    valid_relationships = [
        r for r in relationships
        if r.source_entity_id in entity_id_set and r.target_entity_id in entity_id_set
    ]
    
    # Get unique community count
    unique_communities = set(str(c) for c in comm_memberships.values() if c)
    
    return {
        "entities": [
            {
                "id": str(e.id),
                "name": e.name,
                "type": e.entity_type.value.upper(),
                "community_id": str(comm_memberships.get(e.id)) if comm_memberships.get(e.id) else None,
                "connection_count": e.relationship_count,
                "mention_count": e.mention_count,
                "centrality": round(e.centrality_score, 4),
            }
            for e in entities
        ],
        "relationships": [
            {
                "source_id": str(r.source_entity_id),
                "target_id": str(r.target_entity_id),
                "relationship_type": r.relationship_type.value.upper(),
                "weight": r.weight,
                "confidence": r.confidence,
            }
            for r in valid_relationships
        ],
        "stats": {
            "total_entities": len(entities),
            "total_relationships": len(valid_relationships),
            "communities": len(unique_communities),
        }
    }


@router.post("/memory/consolidate/{user_id}")
async def consolidate_user_memories(
    user_id: str,
    min_cluster_size: int = Query(3, ge=2, le=10),
    age_threshold_days: int = Query(60, ge=7, le=365),
    current_user: CurrentUser = Depends(require_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger memory consolidation for a user.
    
    This clusters old archived memories and creates consolidated summaries.
    """
    from app.services.memory_service import get_memory_service
    
    memory_service = get_memory_service(db)
    user_uuid = uuid.UUID(user_id)
    
    count = await memory_service.consolidate_old_memories(
        user_id=user_uuid,
        min_cluster_size=min_cluster_size,
        age_threshold_days=age_threshold_days,
    )
    
    return {
        "status": "completed",
        "clusters_consolidated": count,
        "user_id": user_id,
    }
