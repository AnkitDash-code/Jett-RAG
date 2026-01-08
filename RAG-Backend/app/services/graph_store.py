"""
Graph Store Service using rustworkx + SQLite.

Provides graph operations for the knowledge graph:
- Load graph from SQLite into rustworkx
- Get entity neighbors (k-hop traversal)
- Compute centrality scores
- Apply RBAC filtering
- Cache management
"""
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

try:
    import rustworkx as rx
    RUSTWORKX_AVAILABLE = True
except ImportError:
    rx = None
    RUSTWORKX_AVAILABLE = False

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.graph import (
    Entity, EntityMention, EntityRelationship,
    Community, CommunityMembership, CommunityContext,
    EntityType, RelationshipType, PermissionType
)

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the graph."""
    entity_id: uuid.UUID
    name: str
    normalized_name: str
    entity_type: EntityType
    tenant_id: Optional[str]
    access_level: str
    centrality: float = 0.0
    mention_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": str(self.entity_id),
            "name": self.name,
            "normalized_name": self.normalized_name,
            "entity_type": self.entity_type.value,
            "tenant_id": self.tenant_id,
            "access_level": self.access_level,
            "centrality": self.centrality,
            "mention_count": self.mention_count,
        }


@dataclass
class GraphEdge:
    """Represents an edge in the graph."""
    relationship_id: uuid.UUID
    source_id: uuid.UUID
    target_id: uuid.UUID
    relationship_type: RelationshipType
    weight: float = 1.0
    confidence: float = 1.0


@dataclass
class TraversalResult:
    """Result from a graph traversal."""
    entities: List[GraphNode]
    relationships: List[GraphEdge]
    chunk_ids: Set[uuid.UUID]
    hops: int
    traversal_ms: int


@dataclass
class GraphStats:
    """Statistics about the graph."""
    node_count: int
    edge_count: int
    community_count: int
    pending_chunks: int
    avg_degree: float
    density: float
    last_updated: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "community_count": self.community_count,
            "pending_chunks": self.pending_chunks,
            "avg_degree": round(self.avg_degree, 2),
            "density": round(self.density, 4),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass
class RBACParams:
    """RBAC parameters for graph filtering."""
    user_id: uuid.UUID
    tenant_ids: Set[str] = field(default_factory=set)
    access_levels: Set[str] = field(default_factory=set)
    departments: Set[str] = field(default_factory=set)
    group_ids: Set[uuid.UUID] = field(default_factory=set)
    is_admin: bool = False


class GraphStoreService:
    """
    Service for managing the knowledge graph using rustworkx + SQLite.
    
    The graph structure is persisted in SQLite tables and loaded into
    rustworkx for fast in-memory operations. Changes are written back
    to SQLite and the in-memory graph is refreshed periodically.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._graph: Optional[rx.PyDiGraph] = None if not RUSTWORKX_AVAILABLE else None
        self._entity_to_node: Dict[uuid.UUID, int] = {}  # entity_id -> node index
        self._node_to_entity: Dict[int, uuid.UUID] = {}  # node index -> entity_id
        self._last_load: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=settings.GRAPH_CACHE_TTL_SECONDS)
    
    @property
    def is_available(self) -> bool:
        """Check if rustworkx is available."""
        return RUSTWORKX_AVAILABLE and settings.ENABLE_GRAPH_RAG
    
    async def ensure_loaded(self) -> bool:
        """Ensure the graph is loaded and fresh."""
        if not self.is_available:
            return False
        
        now = datetime.utcnow()
        if self._graph is None or (
            self._last_load and now - self._last_load > self._cache_ttl
        ):
            await self.load_graph()
        
        return self._graph is not None
    
    async def load_graph(self) -> None:
        """Load the entire graph from SQLite into rustworkx."""
        if not RUSTWORKX_AVAILABLE:
            logger.warning("rustworkx not available, graph operations disabled")
            return
        
        t0 = time.perf_counter()
        logger.info("Loading graph from database...")
        
        # Create new graph
        self._graph = rx.PyDiGraph()
        self._entity_to_node = {}
        self._node_to_entity = {}
        
        # Load all entities as nodes
        result = await self.db.execute(select(Entity))
        entities = result.scalars().all()
        
        for entity in entities:
            node_data = GraphNode(
                entity_id=entity.id,
                name=entity.name,
                normalized_name=entity.normalized_name,
                entity_type=entity.entity_type,
                tenant_id=entity.tenant_id,
                access_level=entity.access_level,
                centrality=entity.centrality_score,
                mention_count=entity.mention_count,
            )
            node_idx = self._graph.add_node(node_data)
            self._entity_to_node[entity.id] = node_idx
            self._node_to_entity[node_idx] = entity.id
        
        # Load all relationships as edges
        result = await self.db.execute(select(EntityRelationship))
        relationships = result.scalars().all()
        
        edges_added = 0
        for rel in relationships:
            source_idx = self._entity_to_node.get(rel.source_entity_id)
            target_idx = self._entity_to_node.get(rel.target_entity_id)
            
            if source_idx is not None and target_idx is not None:
                edge_data = GraphEdge(
                    relationship_id=rel.id,
                    source_id=rel.source_entity_id,
                    target_id=rel.target_entity_id,
                    relationship_type=rel.relationship_type,
                    weight=rel.weight,
                    confidence=rel.confidence,
                )
                self._graph.add_edge(source_idx, target_idx, edge_data)
                edges_added += 1
        
        self._last_load = datetime.utcnow()
        load_ms = int((time.perf_counter() - t0) * 1000)
        
        logger.info(
            f"Graph loaded: {len(entities)} nodes, {edges_added} edges in {load_ms}ms"
        )
    
    async def get_neighbors(
        self,
        entity_id: uuid.UUID,
        max_hops: int = None,
        rbac_params: Optional[RBACParams] = None,
    ) -> TraversalResult:
        """
        Get neighbors of an entity up to max_hops away.
        
        Args:
            entity_id: The starting entity
            max_hops: Maximum traversal depth (default from config)
            rbac_params: RBAC filtering parameters
        
        Returns:
            TraversalResult with entities, relationships, and chunk IDs
        """
        t0 = time.perf_counter()
        max_hops = max_hops or settings.GRAPH_MAX_HOPS
        
        if not await self.ensure_loaded():
            return TraversalResult(
                entities=[], relationships=[], chunk_ids=set(),
                hops=0, traversal_ms=0
            )
        
        start_idx = self._entity_to_node.get(entity_id)
        if start_idx is None:
            return TraversalResult(
                entities=[], relationships=[], chunk_ids=set(),
                hops=0, traversal_ms=int((time.perf_counter() - t0) * 1000)
            )
        
        # BFS traversal
        visited_nodes: Set[int] = {start_idx}
        visited_edges: List[GraphEdge] = []
        current_frontier = {start_idx}
        
        for hop in range(max_hops):
            next_frontier = set()
            for node_idx in current_frontier:
                # Get outgoing edges
                for edge_idx in self._graph.out_edges(node_idx):
                    target_idx = edge_idx[1]
                    edge_data = self._graph.get_edge_data(node_idx, target_idx)
                    
                    if target_idx not in visited_nodes:
                        # Check RBAC if provided
                        target_node = self._graph[target_idx]
                        if self._check_node_access(target_node, rbac_params):
                            visited_nodes.add(target_idx)
                            next_frontier.add(target_idx)
                            visited_edges.append(edge_data)
                
                # Get incoming edges
                for edge_idx in self._graph.in_edges(node_idx):
                    source_idx = edge_idx[0]
                    edge_data = self._graph.get_edge_data(source_idx, node_idx)
                    
                    if source_idx not in visited_nodes:
                        source_node = self._graph[source_idx]
                        if self._check_node_access(source_node, rbac_params):
                            visited_nodes.add(source_idx)
                            next_frontier.add(source_idx)
                            visited_edges.append(edge_data)
            
            current_frontier = next_frontier
            if not current_frontier:
                break
        
        # Collect results
        entities = [self._graph[idx] for idx in visited_nodes]
        
        # Get chunk IDs for these entities
        entity_ids = [self._node_to_entity[idx] for idx in visited_nodes]
        chunk_ids = await self._get_chunk_ids_for_entities(entity_ids)
        
        traversal_ms = int((time.perf_counter() - t0) * 1000)
        
        return TraversalResult(
            entities=entities,
            relationships=visited_edges,
            chunk_ids=chunk_ids,
            hops=max_hops,
            traversal_ms=traversal_ms,
        )
    
    def _check_node_access(
        self, node: GraphNode, rbac_params: Optional[RBACParams]
    ) -> bool:
        """Check if a node is accessible given RBAC params."""
        if rbac_params is None or rbac_params.is_admin:
            return True
        
        # Check tenant
        if rbac_params.tenant_ids and node.tenant_id:
            if node.tenant_id not in rbac_params.tenant_ids:
                return False
        
        # Check access level
        if rbac_params.access_levels:
            if node.access_level not in rbac_params.access_levels:
                return False
        
        return True
    
    async def _get_chunk_ids_for_entities(
        self, entity_ids: List[uuid.UUID]
    ) -> Set[uuid.UUID]:
        """Get all chunk IDs where these entities are mentioned."""
        if not entity_ids:
            return set()
        
        result = await self.db.execute(
            select(EntityMention.chunk_id).where(
                EntityMention.entity_id.in_(entity_ids)
            )
        )
        return set(result.scalars().all())
    
    async def compute_centrality(self, algorithm: str = "pagerank") -> Dict[uuid.UUID, float]:
        """
        Compute centrality scores for all nodes.
        
        Args:
            algorithm: 'pagerank', 'degree', or 'betweenness'
        
        Returns:
            Dict mapping entity_id to centrality score
        """
        if not await self.ensure_loaded() or self._graph.num_nodes() == 0:
            return {}
        
        t0 = time.perf_counter()
        
        if algorithm == "pagerank":
            # PageRank
            scores = rx.pagerank(self._graph)
        elif algorithm == "degree":
            # Degree centrality
            scores = {
                idx: self._graph.out_degree(idx) + self._graph.in_degree(idx)
                for idx in range(self._graph.num_nodes())
            }
            # Normalize
            max_score = max(scores.values()) if scores else 1
            scores = {k: v / max_score for k, v in scores.items()}
        else:
            # Default to out-degree
            scores = {
                idx: self._graph.out_degree(idx)
                for idx in range(self._graph.num_nodes())
            }
        
        # Map back to entity IDs
        result = {}
        for node_idx, score in scores.items():
            entity_id = self._node_to_entity.get(node_idx)
            if entity_id:
                result[entity_id] = score
        
        logger.debug(
            f"Computed {algorithm} centrality for {len(result)} nodes "
            f"in {int((time.perf_counter() - t0) * 1000)}ms"
        )
        
        return result
    
    async def update_centrality_scores(self) -> int:
        """Compute and persist centrality scores to database."""
        scores = await self.compute_centrality("pagerank")
        
        updated = 0
        for entity_id, score in scores.items():
            result = await self.db.execute(
                select(Entity).where(Entity.id == entity_id)
            )
            entity = result.scalar_one_or_none()
            if entity:
                entity.centrality_score = score
                entity.updated_at = datetime.utcnow()
                updated += 1
        
        await self.db.commit()
        logger.info(f"Updated centrality scores for {updated} entities")
        return updated
    
    async def get_entity_by_name(
        self,
        name: str,
        fuzzy: bool = True,
        rbac_params: Optional[RBACParams] = None,
    ) -> Optional[Entity]:
        """
        Find an entity by name, optionally with fuzzy matching.
        
        Args:
            name: Entity name to search
            fuzzy: If True, use normalized name matching
            rbac_params: RBAC filtering
        
        Returns:
            Entity if found, None otherwise
        """
        normalized = name.lower().strip()
        
        query = select(Entity)
        
        if fuzzy:
            query = query.where(Entity.normalized_name == normalized)
        else:
            query = query.where(Entity.name == name)
        
        # Apply RBAC
        if rbac_params and not rbac_params.is_admin:
            conditions = []
            if rbac_params.tenant_ids:
                conditions.append(
                    or_(
                        Entity.tenant_id.in_(rbac_params.tenant_ids),
                        Entity.tenant_id.is_(None)
                    )
                )
            if rbac_params.access_levels:
                conditions.append(Entity.access_level.in_(rbac_params.access_levels))
            
            if conditions:
                query = query.where(and_(*conditions))
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def find_entities_by_names(
        self,
        names: List[str],
        rbac_params: Optional[RBACParams] = None,
    ) -> List[Entity]:
        """Find multiple entities by their names."""
        normalized_names = [n.lower().strip() for n in names]
        
        query = select(Entity).where(
            Entity.normalized_name.in_(normalized_names)
        )
        
        # Apply RBAC
        if rbac_params and not rbac_params.is_admin:
            if rbac_params.tenant_ids:
                query = query.where(
                    or_(
                        Entity.tenant_id.in_(rbac_params.tenant_ids),
                        Entity.tenant_id.is_(None)
                    )
                )
            if rbac_params.access_levels:
                query = query.where(Entity.access_level.in_(rbac_params.access_levels))
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_stats(self) -> GraphStats:
        """Get statistics about the graph."""
        # Count entities
        entity_result = await self.db.execute(
            select(Entity.id)
        )
        node_count = len(entity_result.all())
        
        # Count relationships
        rel_result = await self.db.execute(
            select(EntityRelationship.id)
        )
        edge_count = len(rel_result.all())
        
        # Count communities
        comm_result = await self.db.execute(
            select(Community.id)
        )
        community_count = len(comm_result.all())
        
        # Count pending chunks
        from app.models.document import Chunk
        pending_result = await self.db.execute(
            select(Chunk.id).where(Chunk.needs_graph_processing == True)
        )
        pending_chunks = len(pending_result.all())
        
        # Compute density and avg degree
        avg_degree = 0.0
        density = 0.0
        if node_count > 0:
            avg_degree = (2 * edge_count) / node_count
            if node_count > 1:
                density = edge_count / (node_count * (node_count - 1))
        
        return GraphStats(
            node_count=node_count,
            edge_count=edge_count,
            community_count=community_count,
            pending_chunks=pending_chunks,
            avg_degree=avg_degree,
            density=density,
            last_updated=self._last_load,
        )
    
    async def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        tenant_id: Optional[str] = None,
        access_level: str = "internal",
        department: Optional[str] = None,
        owner_id: Optional[uuid.UUID] = None,
        description: Optional[str] = None,
    ) -> Entity:
        """Add a new entity to the graph."""
        normalized = name.lower().strip()
        
        # Check for existing entity
        existing = await self.get_entity_by_name(name, fuzzy=True)
        if existing:
            return existing
        
        entity = Entity(
            name=name,
            normalized_name=normalized,
            entity_type=entity_type,
            tenant_id=tenant_id,
            access_level=access_level,
            department=department,
            owner_id=owner_id,
            description=description,
        )
        
        self.db.add(entity)
        await self.db.commit()
        await self.db.refresh(entity)
        
        # Invalidate cache
        self._last_load = None
        
        logger.debug(f"Added entity: {name} ({entity_type.value})")
        return entity
    
    async def add_relationship(
        self,
        source_entity_id: uuid.UUID,
        target_entity_id: uuid.UUID,
        relationship_type: RelationshipType,
        weight: float = 1.0,
        confidence: float = 1.0,
        evidence_chunk_id: Optional[uuid.UUID] = None,
        tenant_id: Optional[str] = None,
        access_level: str = "internal",
    ) -> EntityRelationship:
        """Add a relationship between two entities."""
        relationship = EntityRelationship(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            weight=weight,
            confidence=confidence,
            evidence_chunk_id=evidence_chunk_id,
            tenant_id=tenant_id,
            access_level=access_level,
        )
        
        self.db.add(relationship)
        await self.db.commit()
        await self.db.refresh(relationship)
        
        # Update relationship counts
        await self._update_relationship_count(source_entity_id)
        await self._update_relationship_count(target_entity_id)
        
        # Invalidate cache
        self._last_load = None
        
        logger.debug(
            f"Added relationship: {source_entity_id} "
            f"--[{relationship_type.value}]--> {target_entity_id}"
        )
        return relationship
    
    async def _update_relationship_count(self, entity_id: uuid.UUID) -> None:
        """Update the relationship count for an entity."""
        result = await self.db.execute(
            select(Entity).where(Entity.id == entity_id)
        )
        entity = result.scalar_one_or_none()
        if entity:
            # Count both source and target relationships
            source_count = await self.db.execute(
                select(EntityRelationship.id).where(
                    EntityRelationship.source_entity_id == entity_id
                )
            )
            target_count = await self.db.execute(
                select(EntityRelationship.id).where(
                    EntityRelationship.target_entity_id == entity_id
                )
            )
            entity.relationship_count = len(source_count.all()) + len(target_count.all())
            entity.updated_at = datetime.utcnow()
    
    async def add_mention(
        self,
        entity_id: uuid.UUID,
        chunk_id: uuid.UUID,
        mention_text: str,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
        confidence: float = 1.0,
        extraction_method: str = "llm",
    ) -> EntityMention:
        """Record an entity mention in a chunk."""
        mention = EntityMention(
            entity_id=entity_id,
            chunk_id=chunk_id,
            mention_text=mention_text,
            start_position=start_position,
            end_position=end_position,
            confidence=confidence,
            extraction_method=extraction_method,
        )
        
        self.db.add(mention)
        await self.db.commit()
        await self.db.refresh(mention)
        
        # Update mention count
        result = await self.db.execute(
            select(Entity).where(Entity.id == entity_id)
        )
        entity = result.scalar_one_or_none()
        if entity:
            entity.mention_count += 1
            entity.updated_at = datetime.utcnow()
        
        return mention
    
    async def invalidate_cache(self) -> None:
        """Force cache invalidation for next operation."""
        self._last_load = None
        self._graph = None
        self._entity_to_node = {}
        self._node_to_entity = {}
        logger.info("Graph cache invalidated")


# =============================================================================
# Service Factory
# =============================================================================

_graph_store_instance: Optional[GraphStoreService] = None


def get_graph_store_service(db: AsyncSession) -> GraphStoreService:
    """Get or create the GraphStoreService instance."""
    return GraphStoreService(db)
