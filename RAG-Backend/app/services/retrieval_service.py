"""
Retrieval service for RAG pipeline.
Implements hybrid search: vector + BM25, with optional graph traversal.
"""
import uuid
import json
import logging
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.models.document import Document, Chunk, ChunkStatusEnum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Classification of query type for routing."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    ENTITY = "entity"
    DOCUMENT = "document"
    CONVERSATIONAL = "conversational"


class RetrievalMode(str, Enum):
    """Retrieval strategy modes."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    GRAPH = "graph"


@dataclass
class QueryContext:
    """Context object for retrieval queries."""
    raw_query: str
    user_id: uuid.UUID
    tenant_id: Optional[str] = None
    
    # Filters
    document_ids: Optional[List[uuid.UUID]] = None
    tags: Optional[List[str]] = None
    access_levels: List[str] = field(default_factory=lambda: ["public"])
    
    # Classification
    query_type: QueryType = QueryType.SIMPLE
    
    # Conversation context
    conversation_id: Optional[uuid.UUID] = None
    previous_queries: List[str] = field(default_factory=list)
    
    # Processing state
    normalized_query: str = ""
    expanded_queries: List[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A chunk with retrieval scores."""
    chunk: Chunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    
    # Source info for citation
    document_name: str = ""
    page_number: Optional[int] = None
    section_path: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result of retrieval pipeline."""
    chunks: List[RetrievedChunk]
    query_context: QueryContext
    
    # Timing (ms)
    normalization_ms: int = 0
    embedding_ms: int = 0
    vector_search_ms: int = 0
    bm25_search_ms: int = 0
    graph_search_ms: int = 0
    reranking_ms: int = 0
    total_ms: int = 0
    
    # Quality signals
    top_score: float = 0.0
    mean_score: float = 0.0
    relevance_grade: str = "medium"


class QueryPreprocessor:
    """Handles query normalization, classification, and expansion."""
    
    def normalize(self, query: str) -> str:
        """
        Normalize query text.
        - Lowercasing
        - Trimming whitespace
        - Basic cleaning
        """
        normalized = query.strip().lower()
        # Remove multiple spaces
        normalized = " ".join(normalized.split())
        return normalized
    
    def classify(self, query: str) -> QueryType:
        """
        Classify query type for routing.
        Simple heuristic-based for now.
        TODO: Use ML classifier or LLM.
        """
        query_lower = query.lower()
        
        # Entity-focused queries
        entity_keywords = ["who", "what is", "define", "meaning of", "explain"]
        for kw in entity_keywords:
            if kw in query_lower:
                return QueryType.ENTITY
        
        # Document-specific queries
        doc_keywords = ["in the document", "according to", "from the file", "in section"]
        for kw in doc_keywords:
            if kw in query_lower:
                return QueryType.DOCUMENT
        
        # Complex queries (multiple parts, comparisons)
        complex_indicators = [" and ", " versus ", " compare ", " difference between"]
        for ind in complex_indicators:
            if ind in query_lower:
                return QueryType.COMPLEX
        
        # Conversational follow-ups
        followup_patterns = ["what about", "how about", "also", "tell me more"]
        for pat in followup_patterns:
            if query_lower.startswith(pat):
                return QueryType.CONVERSATIONAL
        
        return QueryType.SIMPLE
    
    def expand_query(self, query: str, num_variants: int = 2) -> List[str]:
        """
        Generate query variants for multi-query RAG.
        TODO: Use LLM for better expansions.
        """
        variants = [query]  # Original always included
        
        # Simple keyword extraction and rephrasing
        # This is a placeholder - in production, use LLM
        if len(query.split()) > 3:
            # Create a shortened version
            words = query.split()
            variants.append(" ".join(words[:len(words)//2 + 1]))
        
        return variants[:num_variants + 1]


class EmbeddingService:
    """Service for generating text embeddings."""
    
    # Class-level model cache for reuse across instances
    _cached_model = None
    _cached_model_name = None
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load the embedding model with GPU support."""
        # Use class-level cache to avoid reloading the model
        if EmbeddingService._cached_model is not None and EmbeddingService._cached_model_name == self.model_name:
            self._model = EmbeddingService._cached_model
            return self._model
            
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading embedding model on device: {device}")
                
                self._model = SentenceTransformer(self.model_name, device=device)
                
                # Cache at class level
                EmbeddingService._cached_model = self._model
                EmbeddingService._cached_model_name = self.model_name
                
                logger.info(f"Loaded embedding model: {self.model_name} on {device}")
            except ImportError as e:
                logger.warning(f"sentence-transformers not installed: {e}")
                self._model = None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}", exc_info=True)
                self._model = None
        return self._model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        model = self._load_model()
        
        if model is None:
            # Return dummy embeddings if model not available
            logger.warning("Returning dummy embeddings")
            return [[0.0] * settings.EMBEDDING_DIMENSION for _ in texts]
        
        # Use larger batch size for GPU
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed([query])[0]


class VectorSearchService:
    """Service for vector similarity search using Weaviate."""
    
    def __init__(self):
        self._weaviate = None
    
    def _get_weaviate(self):
        """Get Weaviate service (lazy initialization)."""
        if self._weaviate is None:
            from app.services.vector_store import weaviate_service
            self._weaviate = weaviate_service
        return self._weaviate
    
    async def search(
        self,
        embedding: List[float],
        query_text: Optional[str] = None,
        tenant_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        access_levels: List[str] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search vector store for similar chunks using Weaviate hybrid search.
        Returns list of {chunk_id, score, text, metadata}.
        """
        weaviate = self._get_weaviate()
        
        if not weaviate.is_available():
            logger.warning("Weaviate not available, returning empty results")
            return []
        
        try:
            results = await weaviate.search(
                query_embedding=embedding,
                query_text=query_text,
                tenant_id=tenant_id,
                owner_id=owner_id,
                access_levels=access_levels,
                document_ids=document_ids,
                top_k=top_k,
                alpha=0.5,  # Hybrid: 50% vector, 50% BM25
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []


class RetrievalService:
    """
    Main retrieval orchestrator.
    Implements multi-phase retrieval pipeline from RAG.txt.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.preprocessor = QueryPreprocessor()
        self.embedding_service = EmbeddingService()
        self.vector_search = VectorSearchService()
    
    async def retrieve(
        self,
        query: str,
        user_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        access_levels: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = settings.RETRIEVAL_TOP_K,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> RetrievalResult:
        """
        Execute full retrieval pipeline.
        """
        start_time = time.perf_counter()
        
        # Initialize context
        context = QueryContext(
            raw_query=query,
            user_id=user_id,
            tenant_id=tenant_id,
            access_levels=access_levels or ["public", "internal"],
            document_ids=[uuid.UUID(d) for d in document_ids] if document_ids else None,
            conversation_id=conversation_id,
        )
        
        # Phase 0: Query preprocessing
        t0 = time.perf_counter()
        context.normalized_query = self.preprocessor.normalize(query)
        context.query_type = self.preprocessor.classify(query)
        context.expanded_queries = self.preprocessor.expand_query(query)
        normalization_ms = int((time.perf_counter() - t0) * 1000)
        
        # Phase 1: Generate query embedding
        t0 = time.perf_counter()
        query_embedding = self.embedding_service.embed_query(context.normalized_query)
        embedding_ms = int((time.perf_counter() - t0) * 1000)
        
        # Phase 2: Vector search (hybrid: vector + BM25)
        t0 = time.perf_counter()
        vector_results = await self.vector_search.search(
            embedding=query_embedding,
            query_text=context.normalized_query,  # For hybrid BM25 search
            tenant_id=tenant_id,
            owner_id=None,  # Don't filter by owner in dev mode - allow all private docs
            access_levels=context.access_levels,
            document_ids=[str(d) for d in context.document_ids] if context.document_ids else None,
            top_k=top_k * 2,  # Get more for reranking
        )
        vector_search_ms = int((time.perf_counter() - t0) * 1000)
        
        # Phase 3: Database fallback (if vector search unavailable or empty)
        if not vector_results:
            logger.info("Vector search returned no results, falling back to DB search")
            vector_results = await self._db_fallback_search(
                query=context.normalized_query,
                context=context,
                limit=top_k * 2,
            )
        
        # Phase 4: Build retrieved chunks from results
        # Weaviate returns text directly, but we still fetch from DB for full chunk objects
        chunk_ids = [r["chunk_id"] for r in vector_results if r.get("chunk_id")]
        chunks = await self._fetch_chunks(chunk_ids)
        
        # Build retrieved chunks with scores - use Weaviate metadata when available
        retrieved_chunks = []
        result_map = {str(r["chunk_id"]): r for r in vector_results}
        
        for chunk in chunks:
            result = result_map.get(str(chunk.id), {})
            doc_name = result.get("document_name") or await self._get_document_name(chunk.document_id)
            retrieved_chunks.append(RetrievedChunk(
                chunk=chunk,
                vector_score=result.get("score", 0.0),
                final_score=result.get("score", 0.0),
                document_name=doc_name,
                page_number=result.get("page_number") or chunk.page_number,
                section_path=result.get("section_path") or chunk.section_path,
            ))
        
        # Phase 5: Reranking (placeholder - implement cross-encoder)
        t0 = time.perf_counter()
        retrieved_chunks = self._rerank(context.normalized_query, retrieved_chunks)
        reranking_ms = int((time.perf_counter() - t0) * 1000)
        
        # Take top_k
        retrieved_chunks = retrieved_chunks[:top_k]
        
        # Calculate quality metrics
        top_score = retrieved_chunks[0].final_score if retrieved_chunks else 0.0
        mean_score = (
            sum(c.final_score for c in retrieved_chunks) / len(retrieved_chunks)
            if retrieved_chunks else 0.0
        )
        relevance_grade = self._grade_relevance(top_score)
        
        total_ms = int((time.perf_counter() - start_time) * 1000)
        
        return RetrievalResult(
            chunks=retrieved_chunks,
            query_context=context,
            normalization_ms=normalization_ms,
            embedding_ms=embedding_ms,
            vector_search_ms=vector_search_ms,
            reranking_ms=reranking_ms,
            total_ms=total_ms,
            top_score=top_score,
            mean_score=mean_score,
            relevance_grade=relevance_grade,
        )
    
    async def _db_fallback_search(
        self,
        query: str,
        context: QueryContext,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback to database text search when vector search unavailable.
        """
        query_filter = select(Chunk).where(
            Chunk.status == ChunkStatusEnum.EMBEDDED
        )
        
        # Apply access control
        if context.tenant_id:
            query_filter = query_filter.where(
                (Chunk.tenant_id == context.tenant_id) |
                (Chunk.access_level.in_(context.access_levels))
            )
        
        if context.document_ids:
            query_filter = query_filter.where(
                Chunk.document_id.in_(context.document_ids)
            )
        
        # Simple text matching (not ideal, but works as fallback)
        query_filter = query_filter.where(
            Chunk.text.ilike(f"%{query}%")
        ).limit(limit)
        
        result = await self.db.execute(query_filter)
        chunks = result.scalars().all()
        
        return [
            {"chunk_id": str(chunk.id), "score": 0.5, "metadata": {}}
            for chunk in chunks
        ]
    
    async def _fetch_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        """Fetch chunks from database by IDs."""
        if not chunk_ids:
            return []
        
        # Convert string IDs to UUIDs
        uuids = []
        for cid in chunk_ids:
            try:
                uuids.append(uuid.UUID(cid))
            except ValueError:
                continue
        
        result = await self.db.execute(
            select(Chunk).where(Chunk.id.in_(uuids))
        )
        return list(result.scalars().all())
    
    async def _get_document_name(self, document_id: uuid.UUID) -> str:
        """Get document filename for citation."""
        result = await self.db.execute(
            select(Document.original_filename).where(Document.id == document_id)
        )
        name = result.scalar_one_or_none()
        return name or "Unknown Document"
    
    def _rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks using cross-encoder.
        """
        if not settings.ENABLE_CROSS_ENCODER or not chunks:
            # Just sort by existing score if cross-encoder disabled
            return sorted(chunks, key=lambda c: c.final_score, reverse=True)
        
        try:
            from app.services.reranker_service import get_reranker
            
            reranker = get_reranker()
            texts = [c.chunk.text for c in chunks]
            
            reranked = reranker.rerank(
                query=query,
                texts=texts,
                top_k=len(chunks)  # Keep all, just reorder
            )
            
            # Update scores based on reranking
            for i, (original_idx, score) in enumerate(reranked):
                chunks[original_idx].rerank_score = score
                chunks[original_idx].final_score = score
            
            return sorted(chunks, key=lambda c: c.final_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original scores: {e}")
            return sorted(chunks, key=lambda c: c.final_score, reverse=True)
    
    def _grade_relevance(self, top_score: float) -> str:
        """Grade the relevance of retrieval results."""
        if top_score >= 0.8:
            return "high"
        elif top_score >= 0.5:
            return "medium"
        else:
            return "low"
