"""
Graph Index Service for background graph building.

Implements iRAG-style lazy indexing:
- Immediate: Documents get embeddings, marked needs_graph_processing=True
- Background: ThreadPoolExecutor processes flagged chunks
- On-demand: Synchronous processing if needed during retrieval
"""
import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, update, and_, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.document import Chunk, ChunkStatusEnum
from app.models.graph import (
    GraphProcessingTask, Entity, Community, CommunityMembership,
    CommunityContext
)
from app.services.entity_extraction_service import (
    EntityExtractionService, get_entity_extraction_service
)
from app.services.graph_store import GraphStoreService, get_graph_store_service
from app.services.utility_llm_client import get_utility_llm

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IndexingStats:
    """Statistics from an indexing run."""
    chunks_processed: int = 0
    entities_created: int = 0
    entities_linked: int = 0
    relationships_created: int = 0
    errors: int = 0
    processing_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks_processed": self.chunks_processed,
            "entities_created": self.entities_created,
            "entities_linked": self.entities_linked,
            "relationships_created": self.relationships_created,
            "errors": self.errors,
            "processing_ms": self.processing_ms,
        }


class GraphIndexService:
    """
    Service for managing graph indexing operations.
    
    Uses ThreadPoolExecutor for background processing of chunks
    that need graph indexing (entity extraction, relationship building).
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for background worker."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db: AsyncSession = None):
        if self._initialized and db is None:
            return
        
        self.db = db
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._batch_size = settings.GRAPH_BATCH_SIZE
        self._num_workers = settings.GRAPH_PROCESSING_WORKERS
        self._poll_interval = 30  # seconds
        self._community_rebuild_threshold = settings.COMMUNITY_REBUILD_THRESHOLD
        self._entities_since_rebuild = 0
        
        GraphIndexService._initialized = True
        logger.info(
            f"GraphIndexService initialized: workers={self._num_workers}, "
            f"batch_size={self._batch_size}"
        )
    
    async def start_background_worker(self) -> None:
        """Start the background indexing worker."""
        if self._running:
            logger.warning("Background worker already running")
            return
        
        if not settings.ENABLE_GRAPH_RAG:
            logger.info("GraphRAG disabled, not starting background worker")
            return
        
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self._num_workers)
        self._worker_task = asyncio.create_task(self._background_loop())
        logger.info("Graph indexing background worker started")
    
    async def stop_background_worker(self) -> None:
        """Stop the background indexing worker."""
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        logger.info("Graph indexing background worker stopped")
    
    async def _background_loop(self) -> None:
        """Main background processing loop."""
        logger.info("Starting background indexing loop")
        
        while self._running:
            try:
                # Process a batch of pending chunks
                stats = await self.process_pending_batch()
                
                if stats.chunks_processed > 0:
                    logger.info(
                        f"Background indexing: processed {stats.chunks_processed} chunks, "
                        f"{stats.entities_created} new entities"
                    )
                    
                    # Check if we need to rebuild communities
                    self._entities_since_rebuild += stats.entities_created
                    if self._entities_since_rebuild >= self._community_rebuild_threshold:
                        await self.rebuild_communities()
                        self._entities_since_rebuild = 0
                
                # Wait before next poll
                await asyncio.sleep(self._poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background indexing error: {e}")
                await asyncio.sleep(self._poll_interval)
    
    async def process_pending_batch(self) -> IndexingStats:
        """Process a batch of chunks that need graph indexing."""
        if self.db is None:
            logger.warning("No database session available")
            return IndexingStats()
        
        t0 = time.perf_counter()
        stats = IndexingStats()
        
        # Get pending chunks
        result = await self.db.execute(
            select(Chunk)
            .where(
                and_(
                    Chunk.needs_graph_processing == True,
                    Chunk.status == ChunkStatusEnum.EMBEDDED,
                )
            )
            .limit(self._batch_size)
        )
        chunks = list(result.scalars().all())
        
        if not chunks:
            return stats
        
        # Process each chunk
        extraction_service = get_entity_extraction_service(self.db)
        
        for chunk in chunks:
            try:
                # Extract entities and relationships
                extraction_result = await extraction_service.extract_from_chunk(
                    chunk=chunk,
                    tenant_id=chunk.tenant_id,
                    access_level=chunk.access_level,
                )
                
                # Store in database
                process_stats = await extraction_service.process_extraction_result(
                    result=extraction_result,
                    tenant_id=chunk.tenant_id,
                    access_level=chunk.access_level,
                    department=chunk.department,
                )
                
                # Update chunk status
                chunk.needs_graph_processing = False
                chunk.status = ChunkStatusEnum.GRAPH_PROCESSED
                
                # Accumulate stats
                stats.chunks_processed += 1
                stats.entities_created += process_stats["entities_created"]
                stats.entities_linked += process_stats["entities_linked"]
                stats.relationships_created += process_stats["relationships_created"]
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {e}")
                stats.errors += 1
        
        await self.db.commit()
        
        stats.processing_ms = int((time.perf_counter() - t0) * 1000)
        return stats
    
    async def queue_chunk_for_processing(
        self,
        chunk_id: uuid.UUID,
        document_id: uuid.UUID,
        priority: int = 0,
    ) -> GraphProcessingTask:
        """Add a chunk to the processing queue."""
        task = GraphProcessingTask(
            chunk_id=chunk_id,
            document_id=document_id,
            priority=priority,
            status=TaskStatus.PENDING.value,
        )
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        
        logger.debug(f"Queued chunk {chunk_id} for graph processing")
        return task
    
    async def queue_document_chunks(
        self,
        document_id: uuid.UUID,
        priority: int = 0,
    ) -> int:
        """Queue all chunks from a document for processing."""
        result = await self.db.execute(
            select(Chunk.id).where(
                and_(
                    Chunk.document_id == document_id,
                    Chunk.needs_graph_processing == True,
                )
            )
        )
        chunk_ids = list(result.scalars().all())
        
        for chunk_id in chunk_ids:
            task = GraphProcessingTask(
                chunk_id=chunk_id,
                document_id=document_id,
                priority=priority,
                status=TaskStatus.PENDING.value,
            )
            self.db.add(task)
        
        await self.db.commit()
        
        logger.info(f"Queued {len(chunk_ids)} chunks from document {document_id}")
        return len(chunk_ids)
    
    async def process_chunk_sync(
        self,
        chunk_id: uuid.UUID,
    ) -> IndexingStats:
        """
        Process a single chunk synchronously (on-demand).
        Used when a query hits an unprocessed chunk.
        """
        t0 = time.perf_counter()
        stats = IndexingStats()
        
        result = await self.db.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        chunk = result.scalar_one_or_none()
        
        if not chunk or not chunk.needs_graph_processing:
            return stats
        
        try:
            extraction_service = get_entity_extraction_service(self.db)
            
            extraction_result = await extraction_service.extract_from_chunk(
                chunk=chunk,
                tenant_id=chunk.tenant_id,
                access_level=chunk.access_level,
            )
            
            process_stats = await extraction_service.process_extraction_result(
                result=extraction_result,
                tenant_id=chunk.tenant_id,
                access_level=chunk.access_level,
                department=chunk.department,
            )
            
            chunk.needs_graph_processing = False
            chunk.status = ChunkStatusEnum.GRAPH_PROCESSED
            await self.db.commit()
            
            stats.chunks_processed = 1
            stats.entities_created = process_stats["entities_created"]
            stats.entities_linked = process_stats["entities_linked"]
            stats.relationships_created = process_stats["relationships_created"]
            
        except Exception as e:
            logger.error(f"Sync processing error for chunk {chunk_id}: {e}")
            stats.errors = 1
        
        stats.processing_ms = int((time.perf_counter() - t0) * 1000)
        return stats
    
    async def rebuild_communities(self) -> int:
        """
        Rebuild community assignments using connected components.
        
        Returns:
            Number of communities created
        """
        if not settings.ENABLE_COMMUNITY_SUMMARIES:
            return 0
        
        logger.info("Rebuilding communities...")
        t0 = time.perf_counter()
        
        # Load graph
        graph_store = get_graph_store_service(self.db)
        await graph_store.load_graph()
        
        if not graph_store._graph or graph_store._graph.num_nodes() == 0:
            logger.info("No nodes in graph, skipping community detection")
            return 0
        
        # Clear existing communities
        await self.db.execute(
            update(CommunityMembership).values(community_id=None)
        )
        
        # Use rustworkx connected components
        try:
            import rustworkx as rx
            
            # Get weakly connected components (for directed graph)
            components = rx.weakly_connected_components(graph_store._graph)
            
            communities_created = 0
            llm = get_utility_llm()
            
            for component_nodes in components:
                if len(component_nodes) < 2:
                    continue  # Skip single-node components
                
                # Create community
                community = Community(
                    algorithm="connected_components",
                    node_count=len(component_nodes),
                )
                self.db.add(community)
                await self.db.flush()
                
                # Add memberships
                entity_names = []
                entity_types = []
                
                for node_idx in component_nodes:
                    entity_id = graph_store._node_to_entity.get(node_idx)
                    if entity_id:
                        node_data = graph_store._graph[node_idx]
                        entity_names.append(node_data.name)
                        entity_types.append(node_data.entity_type.value)
                        
                        membership = CommunityMembership(
                            community_id=community.id,
                            entity_id=entity_id,
                            is_central=len(entity_names) <= 3,  # First 3 are "central"
                        )
                        self.db.add(membership)
                
                communities_created += 1
                
                # Generate community summary if enabled
                if settings.ENABLE_COMMUNITY_SUMMARIES and entity_names:
                    try:
                        summary_result = await llm.generate_community_summary(
                            entity_names=entity_names[:10],
                            entity_types=entity_types[:10],
                            relationship_descriptions=[],
                        )
                        
                        context = CommunityContext(
                            community_id=community.id,
                            summary_text=summary_result.get("summary", ""),
                            key_entities=",".join(entity_names[:5]),
                            key_topics=",".join(summary_result.get("key_topics", [])),
                            entity_count_at_generation=len(entity_names),
                        )
                        self.db.add(context)
                    except Exception as e:
                        logger.warning(f"Failed to generate community summary: {e}")
            
            await self.db.commit()
            
            processing_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                f"Rebuilt {communities_created} communities in {processing_ms}ms"
            )
            
            return communities_created
            
        except ImportError:
            logger.warning("rustworkx not available for community detection")
            return 0
        except Exception as e:
            logger.error(f"Community detection error: {e}")
            return 0
    
    async def get_pending_count(self) -> int:
        """Get count of chunks pending graph processing."""
        result = await self.db.execute(
            select(func.count(Chunk.id)).where(
                Chunk.needs_graph_processing == True
            )
        )
        return result.scalar() or 0
    
    async def get_task_queue_stats(self) -> Dict[str, int]:
        """Get statistics about the task queue."""
        stats = {}
        
        for status in TaskStatus:
            result = await self.db.execute(
                select(func.count(GraphProcessingTask.id)).where(
                    GraphProcessingTask.status == status.value
                )
            )
            stats[status.value] = result.scalar() or 0
        
        return stats
    
    async def force_reindex(
        self,
        document_ids: Optional[List[uuid.UUID]] = None,
    ) -> Dict[str, Any]:
        """
        Force reindexing of all or specific documents.
        
        Args:
            document_ids: Optional list of document IDs to reindex.
                         If None, reindex all documents.
        
        Returns:
            Dict with reindexing status
        """
        t0 = time.perf_counter()
        
        # Mark chunks as needing reprocessing
        if document_ids:
            await self.db.execute(
                update(Chunk)
                .where(Chunk.document_id.in_(document_ids))
                .values(
                    needs_graph_processing=True,
                    status=ChunkStatusEnum.EMBEDDED,
                )
            )
        else:
            await self.db.execute(
                update(Chunk)
                .values(
                    needs_graph_processing=True,
                    status=ChunkStatusEnum.EMBEDDED,
                )
            )
        
        await self.db.commit()
        
        # Get pending count
        pending = await self.get_pending_count()
        
        # Invalidate graph cache
        graph_store = get_graph_store_service(self.db)
        await graph_store.invalidate_cache()
        
        return {
            "status": "queued",
            "pending_chunks": pending,
            "started_at": datetime.utcnow().isoformat(),
            "processing_ms": int((time.perf_counter() - t0) * 1000),
        }


# =============================================================================
# Service Factory
# =============================================================================

_graph_index_service: Optional[GraphIndexService] = None


def get_graph_index_service(db: AsyncSession = None) -> GraphIndexService:
    """Get the GraphIndexService singleton."""
    global _graph_index_service
    if _graph_index_service is None:
        _graph_index_service = GraphIndexService(db)
    elif db is not None:
        _graph_index_service.db = db
    return _graph_index_service
