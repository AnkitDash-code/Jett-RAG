"""
Graph Traversal Service for GraphRAG retrieval.

Provides graph-aware retrieval:
- Entity extraction from queries
- K-hop neighbor traversal
- Community context injection
- Hybrid scoring with vector results
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.graph import (
    Entity, EntityMention, EntityRelationship,
    Community, CommunityMembership, CommunityContext,
    EntityType
)
from app.models.document import Chunk
from app.services.graph_store import (
    GraphStoreService, get_graph_store_service,
    RBACParams, TraversalResult, GraphNode
)
from app.services.entity_extraction_service import get_entity_extraction_service
from app.services.utility_llm_client import get_utility_llm

logger = logging.getLogger(__name__)


@dataclass
class GraphChunk:
    """A chunk retrieved via graph traversal."""
    chunk_id: uuid.UUID
    text: str
    document_id: uuid.UUID
    graph_score: float  # Score based on graph metrics
    hop_distance: int  # How many hops from query entity
    matched_entities: List[str]  # Entities that led to this chunk
    centrality_score: float = 0.0
    community_id: Optional[uuid.UUID] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "document_id": str(self.document_id),
            "graph_score": round(self.graph_score, 4),
            "hop_distance": self.hop_distance,
            "matched_entities": self.matched_entities,
            "centrality_score": round(self.centrality_score, 4),
            "community_id": str(self.community_id) if self.community_id else None,
        }


@dataclass
class CommunityContextInfo:
    """Context from a related community."""
    community_id: uuid.UUID
    summary: str
    key_entities: List[str]
    relevance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "community_id": str(self.community_id),
            "summary": self.summary,
            "key_entities": self.key_entities,
            "relevance_score": round(self.relevance_score, 4),
        }


@dataclass
class GraphRetrievalResult:
    """Result from graph-based retrieval."""
    chunks: List[GraphChunk]
    communities: List[CommunityContextInfo]
    query_entities: List[str]  # Entities extracted from query
    entities_found: int
    traversal_ms: int
    method: str  # "graph", "hybrid", "fallback"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "communities": [c.to_dict() for c in self.communities],
            "query_entities": self.query_entities,
            "entities_found": self.entities_found,
            "traversal_ms": self.traversal_ms,
            "method": self.method,
        }


class GraphTraversalService:
    """
    Service for graph-aware retrieval operations.
    
    Integrates with the enhanced retrieval pipeline to:
    1. Extract entities from queries
    2. Traverse knowledge graph to find related chunks
    3. Score chunks based on graph metrics
    4. Provide community context for better answers
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._graph_store = get_graph_store_service(db)
        self._llm = get_utility_llm()
        self._max_hops = settings.GRAPH_MAX_HOPS
        self._score_weight = settings.GRAPH_SCORE_WEIGHT
    
    async def retrieve(
        self,
        query: str,
        user_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        access_levels: Optional[List[str]] = None,
        top_k: int = 10,
        include_communities: bool = True,
    ) -> GraphRetrievalResult:
        """
        Retrieve relevant chunks using graph traversal.
        
        Args:
            query: User query
            user_id: User ID for RBAC
            tenant_id: Tenant ID for filtering
            access_levels: Allowed access levels
            top_k: Maximum chunks to return
            include_communities: Whether to include community context
        
        Returns:
            GraphRetrievalResult with chunks and community context
        """
        t0 = time.perf_counter()
        
        # Build RBAC params
        rbac_params = RBACParams(
            user_id=user_id,
            tenant_ids={tenant_id} if tenant_id else set(),
            access_levels=set(access_levels) if access_levels else {"public", "internal"},
        )
        
        # Step 1: Extract entities from query
        extraction_service = get_entity_extraction_service(self.db)
        query_entities = await extraction_service.extract_entities_from_query(query)
        
        if not query_entities:
            logger.debug("No entities found in query, returning empty result")
            return GraphRetrievalResult(
                chunks=[],
                communities=[],
                query_entities=[],
                entities_found=0,
                traversal_ms=int((time.perf_counter() - t0) * 1000),
                method="no_entities",
            )
        
        logger.debug(f"Extracted query entities: {query_entities}")
        
        # Step 2: Find matching entities in graph
        matched_entities = await self._graph_store.find_entities_by_names(
            names=query_entities,
            rbac_params=rbac_params,
        )
        
        if not matched_entities:
            logger.debug("No matching entities found in graph")
            return GraphRetrievalResult(
                chunks=[],
                communities=[],
                query_entities=query_entities,
                entities_found=0,
                traversal_ms=int((time.perf_counter() - t0) * 1000),
                method="no_matches",
            )
        
        # Step 3: Traverse graph for each matched entity
        all_chunk_ids: Set[uuid.UUID] = set()
        chunk_to_entities: Dict[uuid.UUID, List[str]] = {}
        chunk_to_hop: Dict[uuid.UUID, int] = {}
        entity_centralities: Dict[uuid.UUID, float] = {}
        
        for entity in matched_entities:
            # Get neighbors (k-hop traversal)
            traversal = await self._graph_store.get_neighbors(
                entity_id=entity.id,
                max_hops=self._max_hops,
                rbac_params=rbac_params,
            )
            
            # Collect chunk IDs from traversal
            for chunk_id in traversal.chunk_ids:
                all_chunk_ids.add(chunk_id)
                
                if chunk_id not in chunk_to_entities:
                    chunk_to_entities[chunk_id] = []
                chunk_to_entities[chunk_id].append(entity.name)
                
                # Track minimum hop distance
                current_hop = chunk_to_hop.get(chunk_id, self._max_hops + 1)
                if 0 < current_hop:  # Entity's own chunks are hop 0
                    chunk_to_hop[chunk_id] = min(current_hop, 1)
            
            # Also get direct mentions (hop 0)
            direct_mentions = await self._get_entity_chunks(entity.id)
            for chunk_id in direct_mentions:
                all_chunk_ids.add(chunk_id)
                if chunk_id not in chunk_to_entities:
                    chunk_to_entities[chunk_id] = []
                chunk_to_entities[chunk_id].append(entity.name)
                chunk_to_hop[chunk_id] = 0
            
            entity_centralities[entity.id] = entity.centrality_score
        
        # Step 4: Fetch and score chunks
        graph_chunks = await self._fetch_and_score_chunks(
            chunk_ids=list(all_chunk_ids),
            chunk_to_entities=chunk_to_entities,
            chunk_to_hop=chunk_to_hop,
            entity_centralities={
                e.name: e.centrality_score for e in matched_entities
            },
            rbac_params=rbac_params,
        )
        
        # Sort by score and limit
        graph_chunks.sort(key=lambda c: c.graph_score, reverse=True)
        graph_chunks = graph_chunks[:top_k]
        
        # Step 5: Get community context if enabled
        communities = []
        if include_communities and settings.ENABLE_COMMUNITY_SUMMARIES:
            communities = await self._get_community_context(
                entities=matched_entities,
                rbac_params=rbac_params,
            )
        
        traversal_ms = int((time.perf_counter() - t0) * 1000)
        
        return GraphRetrievalResult(
            chunks=graph_chunks,
            communities=communities,
            query_entities=query_entities,
            entities_found=len(matched_entities),
            traversal_ms=traversal_ms,
            method="graph",
        )
    
    async def _get_entity_chunks(self, entity_id: uuid.UUID) -> List[uuid.UUID]:
        """Get chunks where an entity is mentioned."""
        result = await self.db.execute(
            select(EntityMention.chunk_id).where(
                EntityMention.entity_id == entity_id
            )
        )
        return list(result.scalars().all())
    
    async def _fetch_and_score_chunks(
        self,
        chunk_ids: List[uuid.UUID],
        chunk_to_entities: Dict[uuid.UUID, List[str]],
        chunk_to_hop: Dict[uuid.UUID, int],
        entity_centralities: Dict[str, float],
        rbac_params: RBACParams,
    ) -> List[GraphChunk]:
        """Fetch chunks from database and compute graph scores."""
        if not chunk_ids:
            return []
        
        # Build RBAC filter
        conditions = [Chunk.id.in_(chunk_ids)]
        
        if rbac_params.tenant_ids:
            conditions.append(
                or_(
                    Chunk.tenant_id.in_(rbac_params.tenant_ids),
                    Chunk.tenant_id.is_(None)
                )
            )
        
        if rbac_params.access_levels:
            conditions.append(Chunk.access_level.in_(rbac_params.access_levels))
        
        result = await self.db.execute(
            select(Chunk).where(and_(*conditions))
        )
        chunks = list(result.scalars().all())
        
        graph_chunks = []
        for chunk in chunks:
            matched_entities = chunk_to_entities.get(chunk.id, [])
            hop_distance = chunk_to_hop.get(chunk.id, self._max_hops)
            
            # Compute graph score
            # Score components:
            # 1. Entity match: 1.0 if chunk contains query entity
            # 2. Hop distance: Decay with distance
            # 3. Centrality: Higher for central entities
            # 4. Entity count: More matching entities = higher score
            
            entity_match_score = 1.0 if matched_entities else 0.0
            hop_decay = 1.0 / (1.0 + hop_distance)  # 1.0 for hop 0, 0.5 for hop 1, etc.
            
            avg_centrality = 0.0
            if matched_entities:
                centralities = [
                    entity_centralities.get(e, 0.0) for e in matched_entities
                ]
                avg_centrality = sum(centralities) / len(centralities)
            
            entity_count_bonus = min(len(matched_entities) * 0.1, 0.3)
            
            # Combined score
            graph_score = (
                entity_match_score * 0.4 +
                hop_decay * 0.3 +
                avg_centrality * 0.2 +
                entity_count_bonus * 0.1
            )
            
            # Get community for this chunk's entities
            community_id = None
            if matched_entities:
                community_id = await self._get_entity_community(matched_entities[0])
            
            graph_chunks.append(GraphChunk(
                chunk_id=chunk.id,
                text=chunk.text,
                document_id=chunk.document_id,
                graph_score=graph_score,
                hop_distance=hop_distance,
                matched_entities=matched_entities,
                centrality_score=avg_centrality,
                community_id=community_id,
            ))
        
        return graph_chunks
    
    async def _get_entity_community(self, entity_name: str) -> Optional[uuid.UUID]:
        """Get the primary community for an entity."""
        # Find entity
        result = await self.db.execute(
            select(Entity).where(
                Entity.normalized_name == entity_name.lower().strip()
            )
        )
        entity = result.scalar_one_or_none()
        
        if not entity:
            return None
        
        # Get membership
        membership_result = await self.db.execute(
            select(CommunityMembership).where(
                CommunityMembership.entity_id == entity.id
            ).limit(1)
        )
        membership = membership_result.scalar_one_or_none()
        
        return membership.community_id if membership else None
    
    async def _get_community_context(
        self,
        entities: List[Entity],
        rbac_params: RBACParams,
    ) -> List[CommunityContextInfo]:
        """Get community summaries for relevant entities."""
        if not entities:
            return []
        
        # Get unique communities for these entities
        entity_ids = [e.id for e in entities]
        
        result = await self.db.execute(
            select(CommunityMembership.community_id)
            .where(CommunityMembership.entity_id.in_(entity_ids))
            .distinct()
        )
        community_ids = list(result.scalars().all())
        
        if not community_ids:
            return []
        
        # Get community contexts
        contexts = []
        for community_id in community_ids[:3]:  # Limit to 3 communities
            context_result = await self.db.execute(
                select(CommunityContext).where(
                    CommunityContext.community_id == community_id
                )
            )
            context = context_result.scalar_one_or_none()
            
            if context and context.summary_text:
                contexts.append(CommunityContextInfo(
                    community_id=community_id,
                    summary=context.summary_text,
                    key_entities=context.key_entities.split(",") if context.key_entities else [],
                    relevance_score=0.8,  # Base relevance for matched communities
                ))
        
        return contexts
    
    async def merge_with_vector_results(
        self,
        vector_chunks: List[Dict[str, Any]],
        graph_result: GraphRetrievalResult,
        vector_weight: float = 0.4,
        rerank_weight: float = 0.3,
        graph_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Merge graph retrieval results with vector search results.
        
        Creates a unified ranked list using hybrid scoring.
        """
        # Create a map of graph chunks by ID
        graph_chunk_map = {gc.chunk_id: gc for gc in graph_result.chunks}
        
        merged_results = []
        seen_ids: Set[uuid.UUID] = set()
        
        # Process vector chunks, augmenting with graph data
        for vec_chunk in vector_chunks:
            chunk_id = uuid.UUID(vec_chunk.get("chunk_id", vec_chunk.get("id", "")))
            
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            
            # Get scores
            vector_score = vec_chunk.get("score", 0.0)
            rerank_score = vec_chunk.get("rerank_score", vector_score)
            
            # Check if this chunk was also found via graph
            graph_chunk = graph_chunk_map.get(chunk_id)
            graph_score = graph_chunk.graph_score if graph_chunk else 0.0
            
            # Compute hybrid score
            final_score = (
                vector_score * vector_weight +
                rerank_score * rerank_weight +
                graph_score * graph_weight
            )
            
            result = {
                **vec_chunk,
                "final_score": final_score,
                "vector_score": vector_score,
                "rerank_score": rerank_score,
                "graph_score": graph_score,
                "matched_entities": graph_chunk.matched_entities if graph_chunk else [],
                "hop_distance": graph_chunk.hop_distance if graph_chunk else None,
            }
            merged_results.append(result)
        
        # Add graph-only chunks (not in vector results)
        for graph_chunk in graph_result.chunks:
            if graph_chunk.chunk_id not in seen_ids:
                seen_ids.add(graph_chunk.chunk_id)
                
                # Graph-only chunk gets 0 for vector/rerank
                final_score = graph_chunk.graph_score * graph_weight
                
                merged_results.append({
                    "chunk_id": str(graph_chunk.chunk_id),
                    "text": graph_chunk.text,
                    "document_id": str(graph_chunk.document_id),
                    "final_score": final_score,
                    "vector_score": 0.0,
                    "rerank_score": 0.0,
                    "graph_score": graph_chunk.graph_score,
                    "matched_entities": graph_chunk.matched_entities,
                    "hop_distance": graph_chunk.hop_distance,
                    "source": "graph_only",
                })
        
        # Sort by final score
        merged_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return merged_results
    
    async def build_context_with_communities(
        self,
        graph_result: GraphRetrievalResult,
    ) -> str:
        """
        Build context string including community summaries.
        
        Returns a formatted string for injection into LLM prompt.
        """
        parts = []
        
        # Add community context first (high-level)
        if graph_result.communities:
            parts.append("=== Related Context ===")
            for comm in graph_result.communities:
                parts.append(f"Topic area: {comm.summary}")
            parts.append("")
        
        # Add entity relationship info
        if graph_result.query_entities:
            parts.append(f"Query mentions: {', '.join(graph_result.query_entities)}")
            parts.append("")
        
        return "\n".join(parts)


# =============================================================================
# Service Factory
# =============================================================================

def get_graph_traversal_service(db: AsyncSession) -> GraphTraversalService:
    """Get a GraphTraversalService instance."""
    return GraphTraversalService(db)
