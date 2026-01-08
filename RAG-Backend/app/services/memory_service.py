"""
Memory Service for the Supermemory System (Phase 6).
Manages episodic and semantic memory with importance-based retention.

Features:
- Dual memory types: episodic (events) and semantic (concepts)
- Importance scoring: (1/days_old) × interaction_count × feedback_score
- Hybrid retrieval: 60% vector similarity + 40% entity overlap
- Smart caching with LRU eviction (DB archival, not deletion)
"""
import uuid
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from app.config import settings
from app.models.memory import Memory, MemoryAccess, MemoryLink, MemoryType, MemoryStatus

logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES FOR RESULTS
# ============================================================================

@dataclass
class MemoryRetrievalResult:
    """Result from memory retrieval."""
    memories: List[Memory]
    hybrid_scores: Dict[str, float]  # memory_id -> hybrid score
    retrieval_ms: int
    method_used: str
    total_candidates: int = 0
    

@dataclass 
class MemoryCreationResult:
    """Result from memory creation."""
    memory: Memory
    embedding_ms: int
    total_ms: int
    was_duplicate: bool = False


@dataclass
class ImportanceUpdateResult:
    """Result from importance recalculation."""
    updated_count: int
    archived_count: int
    update_ms: int


# ============================================================================
# MEMORY VECTOR STORE
# ============================================================================

class MemoryVectorStore:
    """
    FAISS-based vector store specifically for memory embeddings.
    Separate from document chunks to allow independent scaling.
    """
    
    _instance = None
    _index = None
    _id_to_idx: Dict[str, int] = {}
    _idx_to_id: Dict[int, str] = {}
    _embeddings: Dict[str, List[float]] = {}
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._load()
            MemoryVectorStore._initialized = True
    
    @property
    def _persist_dir(self):
        """Get persistence directory for memory vectors."""
        from pathlib import Path
        base_dir = Path(settings.UPLOAD_DIR).parent
        persist_dir = base_dir / "memory_vector_store"
        persist_dir.mkdir(parents=True, exist_ok=True)
        return persist_dir
    
    def _load(self):
        """Load index from disk."""
        try:
            import faiss
            import pickle
            
            index_path = self._persist_dir / "memory_faiss.index"
            mappings_path = self._persist_dir / "memory_mappings.pkl"
            
            if index_path.exists():
                self._index = faiss.read_index(str(index_path))
                logger.info(f"Loaded memory FAISS index with {self._index.ntotal} vectors")
            else:
                self._index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
                logger.info("Created new memory FAISS index")
            
            if mappings_path.exists():
                with open(mappings_path, 'rb') as f:
                    data = pickle.load(f)
                    self._id_to_idx = data.get("id_to_idx", {})
                    self._idx_to_id = data.get("idx_to_id", {})
                    self._embeddings = data.get("embeddings", {})
                    
        except ImportError:
            logger.error("faiss-cpu not installed")
            self._index = None
        except Exception as e:
            logger.error(f"Error loading memory index: {e}")
            import faiss
            self._index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
            self._id_to_idx = {}
            self._idx_to_id = {}
            self._embeddings = {}
    
    def _save(self):
        """Persist index to disk."""
        try:
            import faiss
            import pickle
            
            index_path = self._persist_dir / "memory_faiss.index"
            mappings_path = self._persist_dir / "memory_mappings.pkl"
            
            faiss.write_index(self._index, str(index_path))
            
            with open(mappings_path, 'wb') as f:
                pickle.dump({
                    "id_to_idx": self._id_to_idx,
                    "idx_to_id": self._idx_to_id,
                    "embeddings": self._embeddings,
                }, f)
                
        except Exception as e:
            logger.error(f"Error saving memory index: {e}")
    
    def _normalize(self, vectors):
        """L2 normalize vectors for cosine similarity."""
        import numpy as np
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
    
    def is_available(self) -> bool:
        """Check if index is available."""
        return self._index is not None
    
    def add_memory(self, memory_id: str, embedding: List[float]) -> str:
        """Add a memory embedding to the index."""
        import numpy as np
        
        if not self.is_available():
            return ""
        
        if memory_id in self._id_to_idx:
            return memory_id  # Already indexed
        
        vector_id = f"mem_{memory_id}"
        
        # Store embedding
        self._embeddings[vector_id] = embedding
        
        # Add to index
        vector = np.array([embedding], dtype=np.float32)
        vector = self._normalize(vector)
        
        idx = self._index.ntotal
        self._index.add(vector)
        
        self._id_to_idx[vector_id] = idx
        self._idx_to_id[idx] = vector_id
        
        self._save()
        return vector_id
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[tuple]:
        """
        Search for similar memories.
        Returns: List of (vector_id, score) tuples.
        """
        import numpy as np
        
        if not self.is_available() or self._index.ntotal == 0:
            return []
        
        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        query = self._normalize(query)
        
        # Search
        k = min(top_k + len(exclude_ids or []), self._index.ntotal)
        scores, indices = self._index.search(query, k)
        
        results = []
        exclude_set = set(exclude_ids or [])
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            vector_id = self._idx_to_id.get(idx)
            if vector_id and vector_id not in exclude_set:
                results.append((vector_id, float(score)))
                if len(results) >= top_k:
                    break
        
        return results
    
    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the index.
        Note: FAISS doesn't support true deletion, so we just remove from mappings.
        The vector remains but is never returned.
        """
        vector_id = f"mem_{memory_id}"
        
        if vector_id in self._id_to_idx:
            idx = self._id_to_idx.pop(vector_id)
            self._idx_to_id.pop(idx, None)
            self._embeddings.pop(vector_id, None)
            self._save()
            return True
        
        return False


# ============================================================================
# EMBEDDING SERVICE (reuse from retrieval)
# ============================================================================

class MemoryEmbeddingService:
    """
    Embedding service for memory content.
    Reuses the same model as document embeddings for consistency.
    """
    
    _cached_model = None
    _cached_model_name: Optional[str] = None
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load embedding model."""
        if MemoryEmbeddingService._cached_model is not None:
            if MemoryEmbeddingService._cached_model_name == self.model_name:
                self._model = MemoryEmbeddingService._cached_model
                return self._model
        
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = SentenceTransformer(self.model_name, device=device)
                
                MemoryEmbeddingService._cached_model = self._model
                MemoryEmbeddingService._cached_model_name = self.model_name
                
                logger.info(f"Loaded memory embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._model = None
        
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        model = self._load_model()
        
        if model is None:
            return [0.0] * settings.EMBEDDING_DIMENSION
        
        embedding = model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        model = self._load_model()
        
        if model is None:
            return [[0.0] * settings.EMBEDDING_DIMENSION for _ in texts]
        
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


# ============================================================================
# LRU CACHE FOR MEMORIES
# ============================================================================

class MemoryCache:
    """
    LRU cache for frequently accessed memories.
    Eviction is cache-only; memories remain in database.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, Memory] = {}
        self._access_order: List[str] = []
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Get memory from cache."""
        if memory_id in self._cache:
            # Move to end (most recent)
            self._access_order.remove(memory_id)
            self._access_order.append(memory_id)
            return self._cache[memory_id]
        return None
    
    def put(self, memory: Memory) -> None:
        """Add memory to cache."""
        memory_id = str(memory.id)
        
        # Evict if full
        while len(self._cache) >= self.max_size and memory_id not in self._cache:
            if self._access_order:
                lru_id = self._access_order.pop(0)
                self._cache.pop(lru_id, None)
            else:
                break
        
        # Add to cache
        self._cache[memory_id] = memory
        if memory_id in self._access_order:
            self._access_order.remove(memory_id)
        self._access_order.append(memory_id)
    
    def remove(self, memory_id: str) -> None:
        """Remove memory from cache."""
        self._cache.pop(memory_id, None)
        if memory_id in self._access_order:
            self._access_order.remove(memory_id)
    
    def get_all_ids(self) -> List[str]:
        """Get all cached memory IDs."""
        return list(self._cache.keys())
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# ============================================================================
# MAIN MEMORY SERVICE
# ============================================================================

class MemoryService:
    """
    Core service for managing long-term memory.
    
    Responsibilities:
    - Create/store episodic and semantic memories
    - Retrieve memories using hybrid scoring (vector + entity overlap)
    - Calculate and update importance scores
    - Manage cache eviction (smart forgetting)
    """
    
    # Module-level cache shared across instances
    _shared_cache: Optional[MemoryCache] = None
    
    def __init__(
        self,
        db: AsyncSession,
        cache_size: int = 100,
    ):
        self.db = db
        
        # Initialize shared cache if needed
        if MemoryService._shared_cache is None:
            MemoryService._shared_cache = MemoryCache(max_size=cache_size)
        self._cache = MemoryService._shared_cache
        
        # Lazy-load services
        self._embedding_service: Optional[MemoryEmbeddingService] = None
        self._vector_store: Optional[MemoryVectorStore] = None
    
    def _get_embedding_service(self) -> MemoryEmbeddingService:
        """Lazy load embedding service."""
        if self._embedding_service is None:
            self._embedding_service = MemoryEmbeddingService()
        return self._embedding_service
    
    def _get_vector_store(self) -> MemoryVectorStore:
        """Lazy load vector store."""
        if self._vector_store is None:
            self._vector_store = MemoryVectorStore()
        return self._vector_store
    
    # ========================================================================
    # CREATE MEMORIES
    # ========================================================================
    
    async def create_memory(
        self,
        user_id: uuid.UUID,
        content: str,
        memory_type: MemoryType,
        conversation_id: Optional[uuid.UUID] = None,
        summary: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        related_chunk_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> MemoryCreationResult:
        """
        Create a new memory with vector embedding.
        
        Args:
            user_id: Owner of the memory
            content: The memory content text
            memory_type: EPISODIC or SEMANTIC
            conversation_id: Associated conversation (if any)
            summary: Optional condensed version
            entity_ids: Related entity UUIDs for hybrid search
            related_chunk_ids: Source document chunks
            metadata: Additional metadata dict
            tenant_id: Multi-tenant isolation
            
        Returns:
            MemoryCreationResult with created memory and timing
        """
        t_start = time.perf_counter()
        
        # Generate embedding
        t_embed_start = time.perf_counter()
        embedding = self._get_embedding_service().embed(content)
        embedding_ms = int((time.perf_counter() - t_embed_start) * 1000)
        
        # Create memory record
        memory = Memory(
            user_id=user_id,
            conversation_id=conversation_id,
            memory_type=memory_type,
            content=content,
            summary=summary,
            entity_ids_json=json.dumps(entity_ids or []),
            related_chunk_ids_json=json.dumps(related_chunk_ids or []),
            metadata_json=json.dumps(metadata or {}),
            tenant_id=tenant_id,
            embedding_model=self._get_embedding_service().model_name,
        )
        
        self.db.add(memory)
        await self.db.flush()  # Get ID
        
        # Add to vector store
        vector_id = self._get_vector_store().add_memory(
            str(memory.id),
            embedding
        )
        memory.vector_id = vector_id
        
        # Add to cache
        self._cache.put(memory)
        
        total_ms = int((time.perf_counter() - t_start) * 1000)
        
        logger.info(
            f"Created {memory_type.value} memory {memory.id} "
            f"(embedding: {embedding_ms}ms, total: {total_ms}ms)"
        )
        
        return MemoryCreationResult(
            memory=memory,
            embedding_ms=embedding_ms,
            total_ms=total_ms,
            was_duplicate=False,
        )
    
    async def create_episodic_memory(
        self,
        user_id: uuid.UUID,
        event_description: str,
        conversation_id: Optional[uuid.UUID] = None,
        entity_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryCreationResult:
        """Convenience method for creating episodic memories."""
        return await self.create_memory(
            user_id=user_id,
            content=event_description,
            memory_type=MemoryType.EPISODIC,
            conversation_id=conversation_id,
            entity_ids=entity_ids,
            metadata=metadata,
        )
    
    async def create_semantic_memory(
        self,
        user_id: uuid.UUID,
        concept: str,
        summary: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        related_chunk_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryCreationResult:
        """Convenience method for creating semantic memories."""
        return await self.create_memory(
            user_id=user_id,
            content=concept,
            memory_type=MemoryType.SEMANTIC,
            summary=summary,
            entity_ids=entity_ids,
            related_chunk_ids=related_chunk_ids,
            metadata=metadata,
        )
    
    # ========================================================================
    # RETRIEVE MEMORIES
    # ========================================================================
    
    async def retrieve_memories(
        self,
        query: str,
        user_id: uuid.UUID,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        relevance_threshold: float = 0.3,
        include_archived: bool = False,
    ) -> MemoryRetrievalResult:
        """
        Retrieve memories using hybrid scoring.
        
        Hybrid Score = 0.6 × vector_similarity + 0.4 × entity_overlap
        
        Args:
            query: Search query
            user_id: User to search memories for
            top_k: Maximum memories to return
            memory_type: Filter by EPISODIC or SEMANTIC
            relevance_threshold: Minimum hybrid score (default 0.3)
            include_archived: Include archived memories
            
        Returns:
            MemoryRetrievalResult with scored memories
        """
        t_start = time.perf_counter()
        
        # Generate query embedding
        query_embedding = self._get_embedding_service().embed(query)
        
        # Vector search for candidates
        vector_results = self._get_vector_store().search(
            query_embedding,
            top_k=top_k * 3,  # Get more candidates for filtering
        )
        
        # Build mapping of vector_id to score
        vector_scores: Dict[str, float] = {}
        for vector_id, score in vector_results:
            # Convert mem_xxx to memory UUID
            memory_id = vector_id.replace("mem_", "")
            vector_scores[memory_id] = score
        
        # Get memory records from DB
        memory_ids = [uuid.UUID(mid) for mid in vector_scores.keys()]
        
        if not memory_ids:
            return MemoryRetrievalResult(
                memories=[],
                hybrid_scores={},
                retrieval_ms=int((time.perf_counter() - t_start) * 1000),
                method_used="hybrid",
                total_candidates=0,
            )
        
        # Build query
        stmt = select(Memory).where(
            Memory.id.in_(memory_ids),
            Memory.user_id == user_id,
        )
        
        if not include_archived:
            stmt = stmt.where(Memory.status == MemoryStatus.ACTIVE)
        
        if memory_type:
            stmt = stmt.where(Memory.memory_type == memory_type)
        
        result = await self.db.execute(stmt)
        memories = list(result.scalars().all())
        
        # Extract query entities for hybrid scoring
        query_entities = await self._extract_query_entities(query)
        query_entity_set = set(query_entities)
        
        # Calculate hybrid scores
        hybrid_scores: Dict[str, float] = {}
        
        for memory in memories:
            memory_id = str(memory.id)
            
            # Vector score (already normalized 0-1)
            vector_score = vector_scores.get(memory_id, 0.0)
            
            # Entity overlap score
            memory_entities = set(memory.entity_ids)
            if query_entity_set and memory_entities:
                entity_overlap = len(memory_entities & query_entity_set) / len(query_entity_set)
            else:
                entity_overlap = 0.0
            
            # Hybrid formula: 60% vector + 40% entity
            hybrid_score = (0.6 * vector_score) + (0.4 * entity_overlap)
            
            if hybrid_score >= relevance_threshold:
                hybrid_scores[memory_id] = hybrid_score
        
        # Filter and sort by hybrid score
        filtered_memories = [
            m for m in memories
            if str(m.id) in hybrid_scores
        ]
        filtered_memories.sort(
            key=lambda m: hybrid_scores[str(m.id)],
            reverse=True
        )
        final_memories = filtered_memories[:top_k]
        
        # Record access for retrieved memories
        for memory in final_memories:
            await self._record_access(
                memory,
                query,
                hybrid_scores[str(memory.id)],
            )
        
        retrieval_ms = int((time.perf_counter() - t_start) * 1000)
        
        logger.info(
            f"Retrieved {len(final_memories)}/{len(memories)} memories "
            f"for user {user_id} (threshold: {relevance_threshold}, time: {retrieval_ms}ms)"
        )
        
        return MemoryRetrievalResult(
            memories=final_memories,
            hybrid_scores={str(m.id): hybrid_scores[str(m.id)] for m in final_memories},
            retrieval_ms=retrieval_ms,
            method_used="hybrid",
            total_candidates=len(memories),
        )
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract entity references from query for hybrid scoring.
        TODO: Integrate with entity_extraction_service for full NER.
        """
        # Simple keyword extraction for now
        # In production, use entity_extraction_service
        words = query.lower().split()
        # Filter common stop words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "when", "where"}
        entities = [w for w in words if len(w) > 3 and w not in stopwords]
        return entities[:10]  # Limit
    
    async def _record_access(
        self,
        memory: Memory,
        query: str,
        relevance_score: float,
    ) -> None:
        """Record memory access for importance tracking."""
        # Update memory stats
        memory.last_accessed_at = datetime.utcnow()
        memory.access_count += 1
        memory.interaction_count += 1
        
        # Recalculate importance
        memory.update_days_old()
        memory.importance_score = memory.calculate_importance()
        
        # Create access record
        access = MemoryAccess(
            memory_id=memory.id,
            query=query[:500],  # Truncate long queries
            relevance_score=relevance_score,
            retrieval_method="hybrid",
        )
        self.db.add(access)
        
        # Update cache
        self._cache.put(memory)
    
    # ========================================================================
    # IMPORTANCE & ARCHIVAL
    # ========================================================================
    
    async def update_importance_scores(
        self,
        user_id: Optional[uuid.UUID] = None,
    ) -> ImportanceUpdateResult:
        """
        Recalculate importance scores for all active memories.
        
        Formula: importance = (1 / max(days_old, 0.1)) × interaction_count × feedback_score
        
        Called by maintenance job (daily).
        """
        t_start = time.perf_counter()
        
        # Get active memories
        stmt = select(Memory).where(Memory.status == MemoryStatus.ACTIVE)
        if user_id:
            stmt = stmt.where(Memory.user_id == user_id)
        
        result = await self.db.execute(stmt)
        memories = list(result.scalars().all())
        
        updated_count = 0
        now = datetime.utcnow()
        
        for memory in memories:
            # Update days_old
            delta = now - memory.created_at
            memory.days_old = delta.total_seconds() / 86400
            
            # Recalculate importance
            old_importance = memory.importance_score
            memory.importance_score = memory.calculate_importance()
            memory.updated_at = now
            
            if memory.importance_score != old_importance:
                updated_count += 1
        
        await self.db.commit()
        
        update_ms = int((time.perf_counter() - t_start) * 1000)
        
        logger.info(
            f"Updated importance for {updated_count}/{len(memories)} memories "
            f"in {update_ms}ms"
        )
        
        return ImportanceUpdateResult(
            updated_count=updated_count,
            archived_count=0,
            update_ms=update_ms,
        )
    
    async def evict_low_importance_memories(
        self,
        user_id: Optional[uuid.UUID] = None,
        importance_threshold: float = 0.3,
        age_threshold_days: int = 30,
    ) -> ImportanceUpdateResult:
        """
        Archive memories with low importance that are older than threshold.
        
        Criteria: importance < 0.3 AND days_old > 30
        
        This is CACHE EVICTION only - memories remain in DB with status=ARCHIVED.
        Called by maintenance job (weekly).
        """
        t_start = time.perf_counter()
        
        # Get candidates for archival
        stmt = select(Memory).where(
            Memory.status == MemoryStatus.ACTIVE,
            Memory.importance_score < importance_threshold,
            Memory.days_old > age_threshold_days,
        )
        if user_id:
            stmt = stmt.where(Memory.user_id == user_id)
        
        result = await self.db.execute(stmt)
        memories = list(result.scalars().all())
        
        archived_count = 0
        now = datetime.utcnow()
        
        for memory in memories:
            # Update status
            memory.status = MemoryStatus.ARCHIVED
            memory.is_cached = False
            memory.archived_at = now
            memory.updated_at = now
            
            # Remove from cache
            self._cache.remove(str(memory.id))
            
            # Note: We DON'T remove from vector store
            # Archived memories can still be found if explicitly requested
            
            archived_count += 1
        
        await self.db.commit()
        
        update_ms = int((time.perf_counter() - t_start) * 1000)
        
        logger.info(
            f"Archived {archived_count} low-importance memories "
            f"(threshold: {importance_threshold}, age: {age_threshold_days} days, time: {update_ms}ms)"
        )
        
        return ImportanceUpdateResult(
            updated_count=0,
            archived_count=archived_count,
            update_ms=update_ms,
        )
    
    async def restore_memory(self, memory_id: uuid.UUID) -> Optional[Memory]:
        """Restore an archived memory to active status."""
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.db.execute(stmt)
        memory = result.scalar_one_or_none()
        
        if not memory:
            return None
        
        if memory.status == MemoryStatus.ARCHIVED:
            memory.status = MemoryStatus.ACTIVE
            memory.is_cached = True
            memory.archived_at = None
            memory.updated_at = datetime.utcnow()
            
            # Add back to cache
            self._cache.put(memory)
            
            await self.db.commit()
        
        return memory
    
    # ========================================================================
    # MEMORY FEEDBACK
    # ========================================================================
    
    async def update_feedback(
        self,
        memory_id: uuid.UUID,
        feedback_delta: float,
    ) -> Optional[Memory]:
        """
        Update memory feedback score.
        
        Args:
            memory_id: Memory to update
            feedback_delta: Value to add to feedback_score (-1 to +1)
        """
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.db.execute(stmt)
        memory = result.scalar_one_or_none()
        
        if not memory:
            return None
        
        # Update feedback (clamp to 0-1)
        new_score = memory.feedback_score + feedback_delta
        memory.feedback_score = max(0.0, min(1.0, new_score))
        memory.updated_at = datetime.utcnow()
        
        # Recalculate importance
        memory.importance_score = memory.calculate_importance()
        
        # Update cache
        self._cache.put(memory)
        
        await self.db.commit()
        
        return memory
    
    # ========================================================================
    # MEMORY LINKS (Associative Recall)
    # ========================================================================
    
    async def create_link(
        self,
        source_memory_id: uuid.UUID,
        target_memory_id: uuid.UUID,
        link_type: str = "related",
        strength: float = 0.5,
    ) -> MemoryLink:
        """Create a link between two memories for associative recall."""
        link = MemoryLink(
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            link_type=link_type,
            strength=strength,
        )
        self.db.add(link)
        await self.db.flush()
        return link
    
    async def get_linked_memories(
        self,
        memory_id: uuid.UUID,
        link_type: Optional[str] = None,
        min_strength: float = 0.0,
    ) -> List[Memory]:
        """Get memories linked to a given memory."""
        # Get link records
        stmt = select(MemoryLink).where(
            MemoryLink.source_memory_id == memory_id,
            MemoryLink.strength >= min_strength,
        )
        if link_type:
            stmt = stmt.where(MemoryLink.link_type == link_type)
        
        result = await self.db.execute(stmt)
        links = list(result.scalars().all())
        
        if not links:
            return []
        
        # Get linked memories
        target_ids = [link.target_memory_id for link in links]
        stmt = select(Memory).where(Memory.id.in_(target_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    # ========================================================================
    # CLUSTER SUMMARIZATION (Before Forgetting)
    # ========================================================================
    
    async def _summarize_archived_cluster(
        self,
        memories: List[Memory],
        user_id: uuid.UUID,
    ) -> Optional[MemoryCreationResult]:
        """
        Summarize a cluster of archived memories into a single semantic memory.
        
        Called before forgetting to preserve key concepts.
        Uses LLM to generate summary if available.
        
        Args:
            memories: List of memories to summarize
            user_id: Owner of the memories
            
        Returns:
            MemoryCreationResult with the consolidated memory, or None if too few
        """
        if len(memories) < 2:
            return None
        
        # Collect content from memories
        contents = [m.content[:500] for m in memories]  # Truncate for prompt
        combined_text = "\n---\n".join(contents)
        
        # Collect all entity IDs from the cluster
        all_entities = set()
        all_chunk_ids = set()
        for m in memories:
            all_entities.update(m.entity_ids)
            all_chunk_ids.update(m.related_chunk_ids)
        
        # Generate summary
        summary = await self._generate_cluster_summary(combined_text)
        
        # Create consolidated semantic memory
        result = await self.create_memory(
            user_id=user_id,
            content=summary,
            memory_type=MemoryType.SEMANTIC,
            summary=f"Consolidated from {len(memories)} memories",
            entity_ids=list(all_entities)[:50],  # Limit entity count
            related_chunk_ids=list(all_chunk_ids)[:20],
            metadata={
                "consolidated": True,
                "source_count": len(memories),
                "source_ids": [str(m.id) for m in memories[:10]],
            },
        )
        
        logger.info(
            f"Consolidated {len(memories)} memories into {result.memory.id} "
            f"for user {user_id}"
        )
        
        return result
    
    async def _generate_cluster_summary(self, combined_text: str) -> str:
        """
        Generate a summary of the combined memory texts.
        Falls back to simple truncation if LLM unavailable.
        """
        try:
            from app.services.llm_service import LLMService
            
            llm = LLMService()
            prompt = (
                "Summarize the following memory fragments into a single coherent "
                "concept or fact. Keep only the essential information:\n\n"
                f"{combined_text[:3000]}\n\n"
                "Summary:"
            )
            
            response = await llm.generate(prompt, max_tokens=200)
            return response.strip() if response else combined_text[:500]
            
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            # Fallback: first memory content
            return combined_text[:500]
    
    async def consolidate_old_memories(
        self,
        user_id: uuid.UUID,
        min_cluster_size: int = 3,
        age_threshold_days: int = 60,
    ) -> int:
        """
        Find clusters of old memories and consolidate them.
        
        Uses entity overlap to identify related memories.
        Called by maintenance job.
        
        Returns:
            Number of consolidated clusters
        """
        # Get old, low-importance memories
        stmt = select(Memory).where(
            Memory.user_id == user_id,
            Memory.status == MemoryStatus.ARCHIVED,
            Memory.days_old > age_threshold_days,
        ).order_by(Memory.created_at)
        
        result = await self.db.execute(stmt)
        memories = list(result.scalars().all())
        
        if len(memories) < min_cluster_size:
            return 0
        
        # Simple clustering by entity overlap
        clusters = self._cluster_by_entities(memories, min_cluster_size)
        
        consolidated_count = 0
        for cluster in clusters:
            result = await self._summarize_archived_cluster(cluster, user_id)
            if result:
                # Mark source memories as consolidated
                for m in cluster:
                    m.status = MemoryStatus.CONSOLIDATED
                    m.updated_at = datetime.utcnow()
                consolidated_count += 1
        
        if consolidated_count > 0:
            await self.db.commit()
        
        return consolidated_count
    
    def _cluster_by_entities(
        self,
        memories: List[Memory],
        min_size: int = 3,
    ) -> List[List[Memory]]:
        """Simple entity-based clustering."""
        clusters = []
        used = set()
        
        for i, m1 in enumerate(memories):
            if i in used:
                continue
            
            cluster = [m1]
            m1_entities = set(m1.entity_ids)
            
            for j, m2 in enumerate(memories[i+1:], start=i+1):
                if j in used:
                    continue
                
                m2_entities = set(m2.entity_ids)
                if m1_entities and m2_entities:
                    overlap = len(m1_entities & m2_entities)
                    if overlap >= 1:  # At least 1 shared entity
                        cluster.append(m2)
                        used.add(j)
            
            if len(cluster) >= min_size:
                clusters.append(cluster)
                used.add(i)
        
        return clusters
    
    # ========================================================================
    # SEMANTIC CONCEPT STRENGTHENING
    # ========================================================================
    
    async def strengthen_concept(
        self,
        memory_id: uuid.UUID,
        reinforcement_text: str,
        boost_factor: float = 0.1,
    ) -> Optional[Memory]:
        """
        Strengthen a semantic memory when the concept is re-encountered.
        
        This implements the "learning by repetition" pattern:
        - Updates interaction count
        - Boosts importance score
        - Optionally appends reinforcement text
        
        Args:
            memory_id: The semantic memory to strengthen
            reinforcement_text: New context or example of the concept
            boost_factor: Amount to boost feedback (default 0.1)
            
        Returns:
            Updated memory or None if not found
        """
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.db.execute(stmt)
        memory = result.scalar_one_or_none()
        
        if not memory:
            return None
        
        # Only strengthen semantic memories
        if memory.memory_type != MemoryType.SEMANTIC:
            logger.warning(f"Cannot strengthen non-semantic memory {memory_id}")
            return memory
        
        # Boost interaction count
        memory.interaction_count += 1
        memory.last_accessed_at = datetime.utcnow()
        
        # Boost feedback score (capped at 1.0)
        memory.feedback_score = min(1.0, memory.feedback_score + boost_factor)
        
        # Recalculate importance
        memory.importance_score = memory.calculate_importance()
        
        # Optionally update metadata with reinforcement
        metadata = memory.extra_metadata or {}
        reinforcements = metadata.get("reinforcements", [])
        reinforcements.append({
            "text": reinforcement_text[:200],
            "timestamp": datetime.utcnow().isoformat(),
        })
        metadata["reinforcements"] = reinforcements[-5:]  # Keep last 5
        memory.metadata_json = json.dumps(metadata)
        
        memory.updated_at = datetime.utcnow()
        
        # Update cache
        self._cache.put(memory)
        
        await self.db.commit()
        
        logger.info(
            f"Strengthened semantic memory {memory_id} "
            f"(interactions: {memory.interaction_count}, importance: {memory.importance_score:.3f})"
        )
        
        return memory
    
    async def find_and_strengthen(
        self,
        user_id: uuid.UUID,
        concept_query: str,
        reinforcement_text: str,
        similarity_threshold: float = 0.7,
    ) -> Optional[Memory]:
        """
        Find a matching semantic memory and strengthen it.
        
        If a matching concept exists, strengthen it.
        Otherwise, create a new semantic memory.
        
        Args:
            user_id: User to search for
            concept_query: The concept to find
            reinforcement_text: New context for the concept
            similarity_threshold: Minimum similarity to match
            
        Returns:
            The strengthened or newly created memory
        """
        # Search for matching semantic memories
        result = await self.retrieve_memories(
            query=concept_query,
            user_id=user_id,
            top_k=1,
            memory_type=MemoryType.SEMANTIC,
            relevance_threshold=similarity_threshold,
        )
        
        if result.memories:
            # Found existing - strengthen it
            return await self.strengthen_concept(
                result.memories[0].id,
                reinforcement_text,
            )
        else:
            # No match - create new semantic memory
            creation_result = await self.create_semantic_memory(
                user_id=user_id,
                concept=concept_query,
                summary=reinforcement_text[:200],
                metadata={"created_via": "strengthen"},
            )
            return creation_result.memory
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    async def get_memory_stats(
        self,
        user_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Get memory statistics for a user or all users."""
        # Count by type and status
        base_stmt = select(
            Memory.memory_type,
            Memory.status,
            func.count(Memory.id).label("count"),
            func.avg(Memory.importance_score).label("avg_importance"),
        ).group_by(Memory.memory_type, Memory.status)
        
        if user_id:
            base_stmt = base_stmt.where(Memory.user_id == user_id)
        
        result = await self.db.execute(base_stmt)
        rows = result.all()
        
        stats = {
            "by_type": {},
            "by_status": {},
            "cache_size": self._cache.size,
            "total_active": 0,
            "total_archived": 0,
            "avg_importance": 0.0,
        }
        
        importance_sum = 0.0
        importance_count = 0
        
        for row in rows:
            memory_type = row.memory_type.value if row.memory_type else "unknown"
            status = row.status.value if row.status else "unknown"
            count = row.count
            avg_imp = row.avg_importance or 0.0
            
            # By type
            if memory_type not in stats["by_type"]:
                stats["by_type"][memory_type] = 0
            stats["by_type"][memory_type] += count
            
            # By status
            if status not in stats["by_status"]:
                stats["by_status"][status] = 0
            stats["by_status"][status] += count
            
            # Totals
            if status == "active":
                stats["total_active"] += count
            elif status == "archived":
                stats["total_archived"] += count
            
            importance_sum += avg_imp * count
            importance_count += count
        
        if importance_count > 0:
            stats["avg_importance"] = round(importance_sum / importance_count, 4)
        
        return stats


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_memory_service(db: AsyncSession) -> MemoryService:
    """Factory function to create MemoryService instance."""
    return MemoryService(db)
