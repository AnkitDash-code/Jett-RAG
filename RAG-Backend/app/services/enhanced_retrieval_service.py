"""
Enhanced Retrieval Pipeline with Query Intelligence.

Integrates Phase 3 services:
- Query expansion (multi-query RAG)
- Query classification & routing
- Relevance grading (self-reflection)
- Conversation context

Phase 4 additions:
- Graph traversal for entity-aware retrieval
- Community context injection
- Hybrid scoring (vector + rerank + graph)

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

# Phase 4: GraphRAG integration (conditional import)
_graph_traversal_service = None
def _get_graph_traversal_service():
    """Lazy import for graph traversal service."""
    global _graph_traversal_service
    if _graph_traversal_service is None and getattr(settings, 'ENABLE_GRAPH_RAG', False):
        try:
            from app.services.graph_traversal_service import get_graph_traversal_service
            _graph_traversal_service = get_graph_traversal_service()
        except ImportError:
            logger.warning("GraphRAG not available - graph_traversal_service not found")
            _graph_traversal_service = False
    return _graph_traversal_service if _graph_traversal_service else None

# Phase 5: Hierarchical retrieval integration (conditional import)
_hierarchical_service = None
def _get_hierarchical_service(db: AsyncSession):
    """Lazy import for hierarchical retrieval service."""
    if not getattr(settings, 'ENABLE_HIERARCHICAL_CHUNKING', False):
        return None
    try:
        from app.services.hierarchical_retrieval_service import get_hierarchical_retrieval_service
        return get_hierarchical_retrieval_service(db)
    except ImportError:
        logger.warning("Hierarchical retrieval not available")
        return None

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
    
    # Phase 6: Memory context
    memory_context: str = ""
    memory_count: int = 0
    memory_retrieval_ms: int = 0
    
    # Graph retrieval (Phase 4)
    graph_context: str = ""
    graph_entities: List[str] = field(default_factory=list)
    graph_communities: List[Dict[str, Any]] = field(default_factory=list)
    graph_traversal_ms: int = 0
    
    # Hierarchical retrieval (Phase 5)
    hierarchy_context: List[Dict[str, Any]] = field(default_factory=list)
    hierarchy_expansion_ms: int = 0
    
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
            self.context_ms +
            self.graph_traversal_ms +
            self.hierarchy_expansion_ms +
            self.memory_retrieval_ms
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
        enable_memory: bool = True,
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
            enable_memory: Whether to retrieve long-term memories (Phase 6)
            
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
        # Step 1.5: Long-Term Memory Retrieval (Phase 6)
        # ====================================================================
        t0 = time.perf_counter()
        
        if enable_memory:
            try:
                from app.services.memory_service import get_memory_service
                memory_service = get_memory_service(self.db)
                
                memory_result = await memory_service.retrieve_memories(
                    query=query,
                    user_id=user_id,
                    top_k=5,
                    relevance_threshold=0.3,
                )
                
                if memory_result.memories:
                    # Format memories as context
                    memory_lines = []
                    for i, memory in enumerate(memory_result.memories, 1):
                        score = memory_result.hybrid_scores.get(str(memory.id), 0)
                        memory_lines.append(
                            f"[Memory {i} ({memory.memory_type.value}, relevance: {score:.2f})]: "
                            f"{memory.content[:300]}..."
                            if len(memory.content) > 300 else
                            f"[Memory {i} ({memory.memory_type.value}, relevance: {score:.2f})]: "
                            f"{memory.content}"
                        )
                    
                    result.memory_context = "\n".join(memory_lines)
                    result.memory_count = len(memory_result.memories)
                    
                    logger.debug(
                        f"Retrieved {result.memory_count} memories for query "
                        f"(retrieval: {memory_result.retrieval_ms}ms)"
                    )
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
        
        result.memory_retrieval_ms = int((time.perf_counter() - t0) * 1000)
        
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
        # Step 4.5: Graph Traversal (Phase 4 - GraphRAG)
        # ====================================================================
        # Enhance retrieval with graph-based entity connections and community context
        t0 = time.perf_counter()
        
        graph_service = _get_graph_traversal_service()
        if graph_service and routing.get("use_graph", False):
            try:
                graph_result = await graph_service.retrieve(
                    query=query,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    max_hops=getattr(settings, 'GRAPH_MAX_HOPS', 1),
                    top_k=top_k,
                )
                
                # Merge graph chunks with vector results
                if graph_result.chunks:
                    merged_result = graph_service.merge_with_vector_results(
                        vector_result=base_result,
                        graph_result=graph_result,
                        vector_weight=1.0 - getattr(settings, 'GRAPH_SCORE_WEIGHT', 0.3),
                        graph_weight=getattr(settings, 'GRAPH_SCORE_WEIGHT', 0.3),
                    )
                    result.base_result = merged_result
                    
                    # Store graph metadata
                    result.graph_entities = graph_result.entities
                    result.graph_communities = [
                        {"name": c.community_name, "summary": c.summary}
                        for c in graph_result.community_contexts
                    ]
                    
                # Build context string with community summaries
                if graph_result.community_contexts:
                    result.graph_context = graph_service.build_context_with_communities(
                        graph_result
                    )
                    
                logger.debug(
                    f"GraphRAG: {len(graph_result.chunks)} graph chunks, "
                    f"{len(graph_result.entities)} entities, "
                    f"{len(graph_result.community_contexts)} communities"
                )
            except Exception as e:
                logger.warning(f"Graph traversal failed, continuing with vector-only: {e}")
        
        result.graph_traversal_ms = int((time.perf_counter() - t0) * 1000)
        
        # ====================================================================
        # Step 4.6: Hierarchical Expansion (Phase 5)
        # ====================================================================
        # Expand retrieval with parent summaries for broader context
        t0 = time.perf_counter()
        
        hierarchical_service = _get_hierarchical_service(self.db)
        if hierarchical_service and result.base_result and result.base_result.chunks:
            try:
                chunk_ids = [c.chunk.id for c in result.base_result.chunks]
                
                # Expand to parents for broader context
                expanded_chunks = await hierarchical_service.expand_retrieval_results(
                    chunk_ids=chunk_ids,
                    strategy="parent",  # Add parent summaries
                    max_chunks=top_k + 5,  # Allow a few extra for context
                )
                
                # Add parent context to result metadata
                parent_chunks = [c for c in expanded_chunks if c.hierarchy_level > 0]
                if parent_chunks:
                    result.hierarchy_context = [
                        {"level": c.hierarchy_level, "text": c.text[:200]}
                        for c in parent_chunks
                    ]
                    logger.debug(f"Hierarchical: added {len(parent_chunks)} parent summaries")
                    
            except Exception as e:
                logger.warning(f"Hierarchical expansion failed: {e}")
        
        result.hierarchy_expansion_ms = int((time.perf_counter() - t0) * 1000)
        
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
    
    # ========================================================================
    # GRACEFUL DEGRADATION (Phase 7)
    # ========================================================================
    
    async def retrieve_with_fallback(
        self,
        query: str,
        user_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        access_levels: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = None,
        conversation_id: Optional[uuid.UUID] = None,
        session_hint: Optional[str] = None,
    ) -> EnhancedRetrievalResult:
        """
        Retrieve with graceful degradation on failures.
        
        Fallback chain:
        1. Full enhanced retrieval (expansion + grading + memory + graph)
        2. Basic enhanced retrieval (expansion + grading only)
        3. Simple vector retrieval (no intelligence)
        4. Cached results (if available)
        
        Returns best available result.
        """
        degradation_level = 0
        
        # Try full enhanced retrieval
        try:
            return await self.retrieve(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                access_levels=access_levels,
                document_ids=document_ids,
                top_k=top_k,
                conversation_id=conversation_id,
                session_hint=session_hint,
                enable_expansion=True,
                enable_grading=True,
                enable_context=True,
                enable_memory=True,
            )
        except Exception as e:
            logger.warning(f"Full enhanced retrieval failed, trying degraded mode: {e}")
            degradation_level = 1
        
        # Try basic enhanced retrieval (no memory/graph)
        try:
            return await self.retrieve(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                access_levels=access_levels,
                document_ids=document_ids,
                top_k=top_k,
                conversation_id=conversation_id,
                session_hint=session_hint,
                enable_expansion=True,
                enable_grading=False,  # Skip grading
                enable_context=True,
                enable_memory=False,  # Skip memory
            )
        except Exception as e:
            logger.warning(f"Basic enhanced retrieval failed, trying simple mode: {e}")
            degradation_level = 2
        
        # Try simple vector retrieval
        try:
            base_result = await self._base_service.retrieve(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                access_levels=access_levels,
                document_ids=document_ids,
                top_k=top_k or settings.RETRIEVAL_TOP_K,
                conversation_id=conversation_id,
            )
            
            return EnhancedRetrievalResult(
                base_result=base_result,
                resolved_query=query,
                query_classification={"degraded": True, "level": degradation_level},
            )
        except Exception as e:
            logger.error(f"Simple retrieval failed: {e}")
            degradation_level = 3
        
        # Return empty result
        logger.error(f"All retrieval methods failed for query: {query[:100]}")
        
        from app.services.retrieval_service import RetrievalResult
        empty_result = RetrievalResult(
            chunks=[],
            query=query,
            total_ms=0,
            embedding_ms=0,
            search_ms=0,
            rerank_ms=0,
        )
        
        return EnhancedRetrievalResult(
            base_result=empty_result,
            resolved_query=query,
            query_classification={"degraded": True, "level": degradation_level, "failed": True},
        )
    
    async def get_circuit_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers used by this service."""
        try:
            from app.services.circuit_breaker import get_circuit_registry
            registry = get_circuit_registry()
            return registry.list_circuits()
        except ImportError:
            return {}


def get_enhanced_retrieval_service(db: AsyncSession) -> EnhancedRetrievalService:
    """Factory function to create enhanced retrieval service."""
    return EnhancedRetrievalService(db)
