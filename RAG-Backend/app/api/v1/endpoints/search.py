"""
Search API endpoints for autocomplete and suggestions.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, col

from app.database import get_db
from app.api.deps import get_current_user, CurrentUser

router = APIRouter()


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=1, max_length=100, description="Search query prefix"),
    limit: int = Query(10, ge=1, le=50),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get search suggestions based on query prefix.
    
    Returns:
        - entities: Matching entity names from knowledge graph
        - documents: Matching document titles
        - recent_queries: User's recent similar queries
        - popular_topics: Popular search topics
    """
    suggestions = {
        "entities": [],
        "documents": [],
        "recent_queries": [],
        "popular_topics": [],
    }
    
    query_lower = q.lower()
    
    # 1. Get matching entities from graph
    try:
        from app.models.graph import Entity
        result = await db.execute(
            select(Entity.name, Entity.entity_type)
            .where(col(Entity.name).ilike(f"%{query_lower}%"))
            .limit(limit)
        )
        entities = result.all()
        suggestions["entities"] = [
            {"name": name, "type": etype}
            for name, etype in entities
        ]
    except Exception:
        pass  # Entity table might not exist
    
    # 2. Get matching document titles
    try:
        from app.models.document import Document
        result = await db.execute(
            select(Document.id, Document.title, Document.original_filename)
            .where(
                (col(Document.title).ilike(f"%{query_lower}%")) |
                (col(Document.original_filename).ilike(f"%{query_lower}%"))
            )
            .where(Document.owner_id == current_user.id)
            .limit(limit)
        )
        docs = result.all()
        suggestions["documents"] = [
            {"id": str(doc_id), "title": title or filename}
            for doc_id, title, filename in docs
        ]
    except Exception:
        pass
    
    # 3. Get user's recent similar queries
    try:
        from app.models.chat import Message, Conversation
        result = await db.execute(
            select(Message.content)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.user_id == current_user.id)
            .where(Message.role == "user")
            .where(col(Message.content).ilike(f"%{query_lower}%"))
            .order_by(Message.created_at.desc())
            .limit(5)
        )
        recent = result.scalars().all()
        suggestions["recent_queries"] = list(set(recent))[:5]
    except Exception:
        pass
    
    # 4. Popular topics (static for now, could be computed from query logs)
    popular = [
        "How to", "What is", "Explain", "Compare",
        "Summary of", "Key points", "Analysis",
    ]
    suggestions["popular_topics"] = [
        topic for topic in popular
        if query_lower in topic.lower() or topic.lower().startswith(query_lower)
    ][:3]
    
    return suggestions


@router.get("/autocomplete")
async def autocomplete(
    q: str = Query(..., min_length=1, max_length=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Simple autocomplete endpoint returning just text suggestions.
    Faster than /suggestions - returns flat list of strings.
    """
    results = []
    query_lower = q.lower()
    
    # Get entity names
    try:
        from app.models.graph import Entity
        result = await db.execute(
            select(Entity.name)
            .where(col(Entity.name).ilike(f"{query_lower}%"))
            .limit(10)
        )
        results.extend(result.scalars().all())
    except Exception:
        pass
    
    # Get document titles
    try:
        from app.models.document import Document
        result = await db.execute(
            select(Document.title)
            .where(col(Document.title).ilike(f"{query_lower}%"))
            .where(Document.owner_id == current_user.id)
            .limit(5)
        )
        results.extend([t for t in result.scalars().all() if t])
    except Exception:
        pass
    
    # Dedupe and limit
    seen = set()
    unique = []
    for r in results:
        if r.lower() not in seen:
            seen.add(r.lower())
            unique.append(r)
    
    return {"suggestions": unique[:15]}
