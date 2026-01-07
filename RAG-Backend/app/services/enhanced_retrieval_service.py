"""
Enhanced Retrieval Pipeline with Query Intelligence.

Integrates Phase 3 services:
- Query expansion (multi-query RAG)
- Query classification & routing
- Relevance grading (self-reflection)
- Conversation context

This module provides an enhanced retrieve function that wraps the base RetrievalService.
"""
import logging
import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.retrieval_service import RetrievalService, RetrievalResult, RetrievedChunk
from app.services.query_expansion_service import get_query_expansion_service
from app.services.query_classification_service import get_query_classification_service, QueryType
from app.services.relevance_grader import get_relevance_grader
from app.services.conversation_context_service import get_conversation_context_service

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRetrievalResult:
    """Extended retrieval result with intelligence metadata."""
    base_result: RetrievalResult
    
    # Query intelligence
    query_classification: Dict[str, Any] = field(default_factory=dict)
    expanded_queries: List[str] = field(default_factory=list)
    resolved_query: str = ""
    is_followup: bool = False
    
    # Relevance grading
    relevance_grading: Dict[str, Any] = field(default_factory=dict)
    chunks_filtered: int = 0
    
    # Conversation context
    conversation_context: str = ""
    
    # Timing
    classification_ms: int = 0
    expansion_ms: int = 0
    grading_ms: int = 0
    context_ms: int = 0
    
    @property
    def chunks(self) -> List[RetrievedChunk]:
        """Get chunks from base result."""
        return self.base_result.chunks
    
    @property
    def total_ms(self) -> int:
        """Total time including intelligence overhead."""
        return (
            self.base_result.total_ms +
            self.classification_ms +
            self.expansion_ms +
            self.grading_ms +
            self.context_ms
        )


class EnhancedRetrievalService:
    """
    Enhanced retrieval with query intelligence.
    
    Pipeline:
    1. Conversation context resolution (follow-up detection)
    2. Query classification (determine routing)
    3. Query expansion (multi-query RAG)
    4. Base retrieval (vector + BM25)
    5. Relevance grading (self-reflection)
    6. Result filtering
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._base_service = RetrievalService(db)
        
        # Phase 3 services
        self._expansion_service = get_query_expansion_service()
        self._classification_service = get_query_classification_service()
        self._grader = get_relevance_grader()
        self._context_service = get_conversation_context_service()
    
    async def retrieve(
        self,
        query: str,
        user_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        access_levels: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = None,
        conversation_id: Optional[uuid.UUID] = None,
        session_hint: Optional[str] = None,
        enable_expansion: bool = True,
        enable_grading: bool = True,
        enable_context: bool = True,
    ) -> EnhancedRetrievalResult:
        """
        Execute enhanced retrieval pipeline with query intelligence.
        
        Args:
            query: User query
            user_id: User identifier
            tenant_id: Tenant for multi-tenancy
            access_levels: Allowed access levels
            document_ids: Filter by specific documents
            top_k: Number of results to return
            conversation_id: Conversation UUID for context
            session_hint: Session identifier for context
            enable_expansion: Whether to expand queries
            enable_grading: Whether to grade relevance
            enable_context: Whether to use conversation context
            
        Returns:
            EnhancedRetrievalResult with chunks and intelligence metadata
        """
        result = EnhancedRetrievalResult(
            base_result=None,  # Will be set later
            resolved_query=query,
        )
        
        user_id_str = str(user_id)
        
        # ====================================================================
        # Step 1: Conversation Context Resolution
        # ====================================================================
        t0 = time.perf_counter()
        
        if enable_context and (conversation_id or session_hint):
            context = await self._context_service.get_context_for_query(
                query=query,
                user_id=user_id_str,
                session_hint=session_hint or str(conversation_id),
            )
            
            result.resolved_query = context.resolved_query
            result.is_followup = context.is_followup
            result.conversation_context = self._context_service.get_formatted_context(context)
            
            # Use resolved query for subsequent steps
            query = context.resolved_query
        
        result.context_ms = int((time.perf_counter() - t0) * 1000)
        
        # ====================================================================
        # Step 2: Query Classification
        # ====================================================================
        t0 = time.perf_counter()
        
        classification = await self._classification_service.classify(query)
        routing = classification.routing
        
        result.query_classification = {
            "type": classification.query_type.value,
            "complexity": classification.complexity.value,
            "needs_graph": classification.needs_graph,
            "confidence": classification.confidence,
            "routing": routing,
        }
        
        result.classification_ms = int((time.perf_counter() - t0) * 1000)
        
        # Override top_k if routing suggests different
        if top_k is None:
            top_k = routing.get("num_chunks", settings.RETRIEVAL_TOP_K)
        
        # ====================================================================
        # Step 3: Query Expansion
        # ====================================================================
        t0 = time.perf_counter()
        
        queries_to_use = [query]
        
        if enable_expansion and routing.get("use_expansion", False):
            expansion = await self._expansion_service.expand(
                query=query,
                num_variants=2,
                extract_keywords=True,
            )
            queries_to_use = expansion.variants
            result.expanded_queries = expansion.variants
        
        result.expansion_ms = int((time.perf_counter() - t0) * 1000)
        
        # ====================================================================
        # Step 4: Base Retrieval
        # ====================================================================
        # For multi-query, we do multiple retrievals and merge
        if len(queries_to_use) > 1:
            base_result = await self._multi_query_retrieve(
                queries=queries_to_use,
                user_id=user_id,
                tenant_id=tenant_id,
                access_levels=access_levels,
                document_ids=document_ids,
                top_k=top_k,
                conversation_id=conversation_id,
            )
        else:
            base_result = await self._base_service.retrieve(
                query=queries_to_use[0],
                user_id=user_id,
                tenant_id=tenant_id,
                access_levels=access_levels,
                document_ids=document_ids,
                top_k=top_k,
                conversation_id=conversation_id,
            )
        
        result.base_result = base_result
        
        # ====================================================================
        # Step 5: Relevance Grading
        # ====================================================================
        t0 = time.perf_counter()
        
        if enable_grading and routing.get("use_reranking", True) and base_result.chunks:
            chunk_texts = [c.chunk.text for c in base_result.chunks]
            grading = await self._grader.grade(
                query=query,
                chunks=chunk_texts,
                threshold=0.4,
            )
            
            result.relevance_grading = {
                "is_relevant": grading.is_relevant,
                "confidence": grading.confidence,
                "should_retry": grading.should_retry,
                "suggestion": grading.suggestion,
            }
            
            # Filter chunks if grading says some are irrelevant
            if grading.relevant_indices and len(grading.relevant_indices) < len(base_result.chunks):
                original_count = len(base_result.chunks)
                base_result.chunks = [
                    base_result.chunks[i] 
                    for i in grading.relevant_indices 
                    if i < len(base_result.chunks)
                ]
                result.chunks_filtered = original_count - len(base_result.chunks)
        
        result.grading_ms = int((time.perf_counter() - t0) * 1000)
        
        # ====================================================================
        # Step 6: Add to Conversation History
        # ====================================================================
        if enable_context and (conversation_id or session_hint):
            # Add user query to history
            self._context_service.add_message(
                user_id=user_id_str,
                role="user",
                content=query,
                session_hint=session_hint or str(conversation_id),
            )
        
        return result
    
    async def _multi_query_retrieve(
        self,
        queries: List[str],
        user_id: uuid.UUID,
        tenant_id: Optional[str],
        access_levels: Optional[List[str]],
        document_ids: Optional[List[str]],
        top_k: int,
        conversation_id: Optional[uuid.UUID],
    ) -> RetrievalResult:
        """
        Execute multi-query retrieval and merge results.
        
        Uses reciprocal rank fusion to combine results from multiple queries.
        """
        all_chunks = {}  # chunk_id -> (chunk, best_score, query_count)
        base_result = None
        
        for query in queries:
            result = await self._base_service.retrieve(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                access_levels=access_levels,
                document_ids=document_ids,
                top_k=top_k,
                conversation_id=conversation_id,
            )
            
            if base_result is None:
                base_result = result
            
            # Merge chunks using reciprocal rank fusion
            for rank, chunk in enumerate(result.chunks):
                chunk_id = str(chunk.chunk.id)
                rrf_score = 1.0 / (60 + rank)  # k=60 is common default
                
                if chunk_id in all_chunks:
                    existing = all_chunks[chunk_id]
                    # Combine scores
                    all_chunks[chunk_id] = (
                        chunk,
                        existing[1] + rrf_score,
                        existing[2] + 1,
                    )
                else:
                    all_chunks[chunk_id] = (chunk, rrf_score, 1)
        
        # Sort by combined RRF score
        sorted_chunks = sorted(
            all_chunks.values(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Update base result with merged chunks
        if base_result:
            merged_chunks = []
            for chunk, rrf_score, query_count in sorted_chunks[:top_k]:
                chunk.final_score = rrf_score
                merged_chunks.append(chunk)
            base_result.chunks = merged_chunks
        
        return base_result
    
    def add_assistant_response(
        self,
        user_id: uuid.UUID,
        response: str,
        session_hint: Optional[str] = None,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> None:
        """
        Add assistant response to conversation history.
        
        Call this after generating a response.
        """
        self._context_service.add_message(
            user_id=str(user_id),
            role="assistant",
            content=response,
            session_hint=session_hint or (str(conversation_id) if conversation_id else None),
        )


def get_enhanced_retrieval_service(db: AsyncSession) -> EnhancedRetrievalService:
    """Factory function to create enhanced retrieval service."""
    return EnhancedRetrievalService(db)
