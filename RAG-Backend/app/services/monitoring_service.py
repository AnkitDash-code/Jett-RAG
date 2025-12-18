"""
Monitoring service for metrics, logs, and system health.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from app.models.user import User
from app.models.document import Document, DocumentStatusEnum
from app.models.chat import Conversation, Message, QueryLog, Feedback

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Aggregated system metrics."""
    # Users
    total_users: int = 0
    active_users_24h: int = 0
    
    # Documents
    total_documents: int = 0
    documents_processing: int = 0
    documents_indexed: int = 0
    documents_error: int = 0
    total_chunks: int = 0
    
    # Chat
    total_conversations: int = 0
    total_messages: int = 0
    messages_24h: int = 0
    
    # Query quality
    avg_retrieval_latency_ms: float = 0.0
    avg_llm_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    avg_relevance_score: float = 0.0
    
    # Feedback
    total_feedback: int = 0
    avg_rating: float = 0.0
    helpful_rate: float = 0.0
    
    # Timestamp
    generated_at: str = ""


@dataclass
class UserMetrics:
    """Per-user metrics."""
    user_id: str
    conversations: int = 0
    messages: int = 0
    documents_uploaded: int = 0
    tokens_used: int = 0
    avg_rating: float = 0.0


class MetricsService:
    """Service for collecting and aggregating metrics."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get aggregated system metrics."""
        metrics = SystemMetrics()
        metrics.generated_at = datetime.utcnow().isoformat()
        
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        
        # User metrics
        total_users = await self.db.execute(select(func.count(User.id)))
        metrics.total_users = total_users.scalar_one()
        
        active_users = await self.db.execute(
            select(func.count(User.id)).where(User.last_login_at >= day_ago)
        )
        metrics.active_users_24h = active_users.scalar_one()
        
        # Document metrics
        total_docs = await self.db.execute(select(func.count(Document.id)))
        metrics.total_documents = total_docs.scalar_one()
        
        processing = await self.db.execute(
            select(func.count(Document.id)).where(
                Document.status == DocumentStatusEnum.PROCESSING
            )
        )
        metrics.documents_processing = processing.scalar_one()
        
        indexed = await self.db.execute(
            select(func.count(Document.id)).where(
                Document.status.in_([
                    DocumentStatusEnum.INDEXED_LIGHT,
                    DocumentStatusEnum.INDEXED_FULL,
                ])
            )
        )
        metrics.documents_indexed = indexed.scalar_one()
        
        error = await self.db.execute(
            select(func.count(Document.id)).where(
                Document.status == DocumentStatusEnum.ERROR
            )
        )
        metrics.documents_error = error.scalar_one()
        
        total_chunks = await self.db.execute(select(func.sum(Document.chunk_count)))
        metrics.total_chunks = total_chunks.scalar_one() or 0
        
        # Chat metrics
        total_convs = await self.db.execute(select(func.count(Conversation.id)))
        metrics.total_conversations = total_convs.scalar_one()
        
        total_msgs = await self.db.execute(select(func.count(Message.id)))
        metrics.total_messages = total_msgs.scalar_one()
        
        msgs_24h = await self.db.execute(
            select(func.count(Message.id)).where(Message.created_at >= day_ago)
        )
        metrics.messages_24h = msgs_24h.scalar_one()
        
        # Latency metrics (from query logs)
        latency_stats = await self.db.execute(
            select(
                func.avg(QueryLog.vector_search_ms + QueryLog.reranking_ms),
                func.avg(Message.llm_latency_ms),
                func.avg(Message.total_latency_ms),
            )
            .select_from(QueryLog)
            .join(Message, Message.id == QueryLog.message_id)
        )
        row = latency_stats.one()
        metrics.avg_retrieval_latency_ms = float(row[0] or 0)
        metrics.avg_llm_latency_ms = float(row[1] or 0)
        metrics.avg_total_latency_ms = float(row[2] or 0)
        
        # Relevance metrics
        relevance = await self.db.execute(
            select(func.avg(QueryLog.top_chunk_score))
        )
        metrics.avg_relevance_score = float(relevance.scalar_one() or 0)
        
        # Feedback metrics
        total_feedback = await self.db.execute(select(func.count(Feedback.id)))
        metrics.total_feedback = total_feedback.scalar_one()
        
        if metrics.total_feedback > 0:
            avg_rating = await self.db.execute(select(func.avg(Feedback.rating)))
            metrics.avg_rating = float(avg_rating.scalar_one() or 0)
            
            helpful = await self.db.execute(
                select(func.count(Feedback.id)).where(Feedback.is_helpful == True)
            )
            metrics.helpful_rate = helpful.scalar_one() / metrics.total_feedback
        
        return metrics
    
    async def get_user_metrics(
        self,
        user_id: str,
        days: int = 30,
    ) -> UserMetrics:
        """Get metrics for a specific user."""
        from uuid import UUID
        uid = UUID(user_id)
        
        metrics = UserMetrics(user_id=user_id)
        
        since = datetime.utcnow() - timedelta(days=days)
        
        # Conversations
        convs = await self.db.execute(
            select(func.count(Conversation.id)).where(
                Conversation.user_id == uid,
                Conversation.created_at >= since,
            )
        )
        metrics.conversations = convs.scalar_one()
        
        # Messages
        msgs = await self.db.execute(
            select(func.count(Message.id))
            .select_from(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Conversation.user_id == uid,
                Message.created_at >= since,
            )
        )
        metrics.messages = msgs.scalar_one()
        
        # Documents
        docs = await self.db.execute(
            select(func.count(Document.id)).where(
                Document.owner_id == uid,
                Document.created_at >= since,
            )
        )
        metrics.documents_uploaded = docs.scalar_one()
        
        # Tokens (sum of output tokens from assistant messages)
        tokens = await self.db.execute(
            select(func.sum(Message.output_tokens))
            .select_from(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Conversation.user_id == uid,
                Message.created_at >= since,
            )
        )
        metrics.tokens_used = tokens.scalar_one() or 0
        
        return metrics
    
    async def get_recent_query_logs(
        self,
        limit: int = 100,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent query logs for debugging."""
        from uuid import UUID
        
        query = select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit)
        
        if user_id:
            query = query.where(QueryLog.user_id == UUID(user_id))
        
        result = await self.db.execute(query)
        logs = result.scalars().all()
        
        return [
            {
                "id": str(log.id),
                "message_id": str(log.message_id),
                "user_id": str(log.user_id),
                "raw_query": log.raw_query,
                "query_type": log.query_type.value,
                "retrieval_mode": log.retrieval_mode,
                "chunks_retrieved": log.chunks_retrieved,
                "top_chunk_score": log.top_chunk_score,
                "relevance_grade": log.relevance_grade,
                "vector_search_ms": log.vector_search_ms,
                "reranking_ms": log.reranking_ms,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ]
    
    async def get_error_summary(
        self,
        days: int = 7,
    ) -> Dict[str, Any]:
        """Get summary of document processing errors."""
        since = datetime.utcnow() - timedelta(days=days)
        
        result = await self.db.execute(
            select(Document)
            .where(
                Document.status == DocumentStatusEnum.ERROR,
                Document.updated_at >= since,
            )
            .order_by(Document.updated_at.desc())
            .limit(50)
        )
        docs = result.scalars().all()
        
        return {
            "total_errors": len(docs),
            "errors": [
                {
                    "document_id": str(d.id),
                    "filename": d.original_filename,
                    "error": d.error_message,
                    "updated_at": d.updated_at.isoformat(),
                }
                for d in docs
            ]
        }
