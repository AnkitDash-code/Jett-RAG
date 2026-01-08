"""
Community Detection Service for Phase 5.

Provides multiple community detection algorithms for GraphRAG:
- Louvain (modularity-based, best for larger graphs)
- Label Propagation (fast, good for large graphs)
- Connected Components (fallback for small graphs)

Integrates with the knowledge graph to find clusters of
related entities for better context retrieval.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, update

from app.config import settings
from app.models.graph import Entity, Community, CommunityMembership, CommunityContext
from app.services.utility_llm_client import get_utility_llm

logger = logging.getLogger(__name__)


@dataclass
class CommunityResult:
    """Result of community detection."""
    community_count: int
    algorithm: str
    processing_ms: int
    communities: List[Dict[str, Any]] = field(default_factory=list)
    modularity: Optional[float] = None


class CommunityDetectionService:
    """
    Service for detecting communities in the knowledge graph.
    
    Supports multiple algorithms:
    - louvain: Best quality, uses Louvain modularity optimization
    - label_propagation: Fast, good for very large graphs
    - connected_components: Simple fallback
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.algorithm = getattr(settings, 'COMMUNITY_DETECTION_ALGORITHM', 'louvain')
        self.louvain_resolution = getattr(settings, 'LOUVAIN_RESOLUTION', 1.0)
        self.min_community_size = getattr(settings, 'MIN_COMMUNITY_SIZE', 2)
        self._utility_llm = get_utility_llm()
    
    async def detect_communities(
        self,
        algorithm: Optional[str] = None,
        resolution: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> CommunityResult:
        """
        Detect communities in the knowledge graph.
        
        Args:
            algorithm: Algorithm to use (louvain, label_propagation, connected_components)
            resolution: Resolution parameter for Louvain (higher = more communities)
            tenant_id: Optional filter for tenant-specific graphs
            
        Returns:
            CommunityResult with detected communities
        """
        algo = algorithm or self.algorithm
        res = resolution or self.louvain_resolution
        
        logger.info(f"Starting community detection: algorithm={algo}, resolution={res}")
        t0 = time.perf_counter()
        
        # Load graph data
        graph, node_to_entity = await self._load_graph(tenant_id)
        
        if not graph or len(graph) == 0:
            return CommunityResult(
                community_count=0,
                algorithm=algo,
                processing_ms=0,
            )
        
        # Clear existing community assignments
        await self._clear_communities()
        
        # Run detection algorithm
        if algo == "louvain":
            communities, modularity = self._louvain_detect(graph, res)
        elif algo == "label_propagation":
            communities = self._label_propagation_detect(graph)
            modularity = None
        else:
            communities = self._connected_components_detect(graph)
            modularity = None
        
        # Filter small communities
        communities = [c for c in communities if len(c) >= self.min_community_size]
        
        # Store communities in database
        community_data = await self._store_communities(
            communities, node_to_entity, algo
        )
        
        processing_ms = int((time.perf_counter() - t0) * 1000)
        
        logger.info(
            f"Community detection complete: {len(communities)} communities "
            f"in {processing_ms}ms (modularity: {modularity})"
        )
        
        return CommunityResult(
            community_count=len(communities),
            algorithm=algo,
            processing_ms=processing_ms,
            communities=community_data,
            modularity=modularity,
        )
    
    async def _load_graph(
        self,
        tenant_id: Optional[str] = None,
    ) -> Tuple[Dict[int, Set[int]], Dict[int, uuid.UUID]]:
        """
        Load graph as adjacency list from database.
        
        Returns:
            (adjacency_dict, node_to_entity_mapping)
        """
        # Get all entities
        query = select(Entity)
        if tenant_id:
            query = query.where(Entity.tenant_id == tenant_id)
        
        result = await self.db.execute(query)
        entities = list(result.scalars().all())
        
        if not entities:
            return {}, {}
        
        # Build entity lookup and graph
        entity_to_node: Dict[uuid.UUID, int] = {}
        node_to_entity: Dict[int, uuid.UUID] = {}
        graph: Dict[int, Set[int]] = {}
        
        for i, entity in enumerate(entities):
            entity_to_node[entity.id] = i
            node_to_entity[i] = entity.id
            graph[i] = set()
        
        # Add edges from relationships
        # We'll use the relationships table to build edges
        from app.models.graph import Relationship
        
        rel_query = select(Relationship)
        result = await self.db.execute(rel_query)
        relationships = list(result.scalars().all())
        
        for rel in relationships:
            source_node = entity_to_node.get(rel.source_entity_id)
            target_node = entity_to_node.get(rel.target_entity_id)
            
            if source_node is not None and target_node is not None:
                graph[source_node].add(target_node)
                graph[target_node].add(source_node)  # Undirected for community detection
        
        return graph, node_to_entity
    
    def _louvain_detect(
        self,
        graph: Dict[int, Set[int]],
        resolution: float = 1.0,
    ) -> Tuple[List[Set[int]], float]:
        """
        Louvain community detection algorithm.
        
        Uses modularity optimization to find community structure.
        
        Args:
            graph: Adjacency list representation
            resolution: Higher values = more communities
            
        Returns:
            (list of node sets per community, modularity score)
        """
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities
            
            # Convert to NetworkX graph
            G = nx.Graph()
            G.add_nodes_from(graph.keys())
            
            for node, neighbors in graph.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            
            # Run Louvain
            communities = louvain_communities(G, resolution=resolution)
            
            # Calculate modularity
            from networkx.algorithms.community import modularity
            mod = modularity(G, communities)
            
            return list(communities), mod
            
        except ImportError:
            logger.warning("NetworkX not available, falling back to custom Louvain")
            return self._custom_louvain(graph, resolution)
    
    def _custom_louvain(
        self,
        graph: Dict[int, Set[int]],
        resolution: float = 1.0,
        max_iterations: int = 100,
    ) -> Tuple[List[Set[int]], float]:
        """
        Custom Louvain implementation for when NetworkX is unavailable.
        
        Simplified version that still provides good community detection.
        """
        # Total number of edges (counting each once)
        m = sum(len(neighbors) for neighbors in graph.values()) / 2
        if m == 0:
            return [set(graph.keys())], 0.0
        
        # Initialize each node in its own community
        node_to_community: Dict[int, int] = {node: node for node in graph}
        communities: Dict[int, Set[int]] = {node: {node} for node in graph}
        
        # Degree of each node
        degrees: Dict[int, int] = {node: len(neighbors) for node, neighbors in graph.items()}
        
        # Iterate until no improvement
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for node in graph:
                current_comm = node_to_community[node]
                
                # Count edges to each neighboring community
                comm_edges: Dict[int, int] = {}
                for neighbor in graph[node]:
                    neighbor_comm = node_to_community[neighbor]
                    comm_edges[neighbor_comm] = comm_edges.get(neighbor_comm, 0) + 1
                
                # Find best community to move to
                best_comm = current_comm
                best_gain = 0.0
                
                for target_comm, edges_to_comm in comm_edges.items():
                    if target_comm == current_comm:
                        continue
                    
                    # Calculate modularity gain (simplified)
                    comm_size = len(communities[target_comm])
                    comm_degree = sum(degrees[n] for n in communities[target_comm])
                    
                    gain = (
                        edges_to_comm / m -
                        resolution * degrees[node] * comm_degree / (2 * m * m)
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = target_comm
                
                # Move node if beneficial
                if best_comm != current_comm:
                    communities[current_comm].remove(node)
                    if not communities[current_comm]:
                        del communities[current_comm]
                    
                    if best_comm not in communities:
                        communities[best_comm] = set()
                    communities[best_comm].add(node)
                    node_to_community[node] = best_comm
                    improved = True
        
        # Calculate final modularity
        modularity = self._calculate_modularity(graph, communities, m)
        
        return list(communities.values()), modularity
    
    def _calculate_modularity(
        self,
        graph: Dict[int, Set[int]],
        communities: Dict[int, Set[int]],
        m: float,
    ) -> float:
        """Calculate modularity score for community assignment."""
        if m == 0:
            return 0.0
        
        Q = 0.0
        degrees = {node: len(neighbors) for node, neighbors in graph.items()}
        
        for community_nodes in communities.values():
            for i in community_nodes:
                for j in community_nodes:
                    if i == j:
                        continue
                    
                    # Edge exists
                    a_ij = 1 if j in graph.get(i, set()) else 0
                    
                    # Expected edges
                    expected = degrees[i] * degrees[j] / (2 * m)
                    
                    Q += a_ij - expected
        
        return Q / (2 * m)
    
    def _label_propagation_detect(
        self,
        graph: Dict[int, Set[int]],
        max_iterations: int = 100,
    ) -> List[Set[int]]:
        """
        Label Propagation community detection.
        
        Fast algorithm that assigns nodes to communities based on
        the most common label among neighbors.
        """
        import random
        
        # Initialize each node with unique label
        labels = {node: node for node in graph}
        
        nodes = list(graph.keys())
        
        for _ in range(max_iterations):
            random.shuffle(nodes)
            changed = False
            
            for node in nodes:
                if not graph[node]:
                    continue
                
                # Count neighbor labels
                label_counts: Dict[int, int] = {}
                for neighbor in graph[node]:
                    label = labels[neighbor]
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                # Find most common label
                if label_counts:
                    max_count = max(label_counts.values())
                    best_labels = [l for l, c in label_counts.items() if c == max_count]
                    new_label = random.choice(best_labels)
                    
                    if new_label != labels[node]:
                        labels[node] = new_label
                        changed = True
            
            if not changed:
                break
        
        # Group nodes by label
        communities_dict: Dict[int, Set[int]] = {}
        for node, label in labels.items():
            if label not in communities_dict:
                communities_dict[label] = set()
            communities_dict[label].add(node)
        
        return list(communities_dict.values())
    
    def _connected_components_detect(
        self,
        graph: Dict[int, Set[int]],
    ) -> List[Set[int]]:
        """
        Simple connected components detection.
        
        Fallback algorithm that finds disconnected subgraphs.
        """
        visited: Set[int] = set()
        components: List[Set[int]] = []
        
        for start_node in graph:
            if start_node in visited:
                continue
            
            # BFS to find connected component
            component: Set[int] = set()
            queue = [start_node]
            
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                
                visited.add(node)
                component.add(node)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            components.append(component)
        
        return components
    
    async def _clear_communities(self) -> None:
        """Clear existing community assignments."""
        await self.db.execute(
            update(CommunityMembership).values(community_id=None)
        )
        await self.db.commit()
    
    async def _store_communities(
        self,
        communities: List[Set[int]],
        node_to_entity: Dict[int, uuid.UUID],
        algorithm: str,
    ) -> List[Dict[str, Any]]:
        """
        Store detected communities in database.
        
        Returns:
            List of community info dicts
        """
        community_data = []
        
        for i, community_nodes in enumerate(communities):
            # Get entity IDs for this community
            entity_ids = [
                node_to_entity[node]
                for node in community_nodes
                if node in node_to_entity
            ]
            
            if not entity_ids:
                continue
            
            # Get entity names for summary
            result = await self.db.execute(
                select(Entity).where(Entity.id.in_(entity_ids))
            )
            entities = list(result.scalars().all())
            entity_names = [e.name for e in entities]
            entity_types = [e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type) for e in entities]
            
            # Create community record
            community = Community(
                algorithm=algorithm,
                node_count=len(entity_ids),
            )
            self.db.add(community)
            await self.db.flush()
            
            # Add memberships
            for j, entity_id in enumerate(entity_ids):
                membership = CommunityMembership(
                    community_id=community.id,
                    entity_id=entity_id,
                    is_central=j < 3,  # First 3 are "central"
                )
                self.db.add(membership)
            
            # Generate summary
            if settings.ENABLE_COMMUNITY_SUMMARIES and entity_names:
                try:
                    summary_result = await self._utility_llm.generate_community_summary(
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
            
            community_data.append({
                "id": str(community.id),
                "node_count": len(entity_ids),
                "central_entities": entity_names[:3],
            })
        
        await self.db.commit()
        return community_data


# Singleton getter
def get_community_detection_service(db: AsyncSession) -> CommunityDetectionService:
    """Get community detection service instance."""
    return CommunityDetectionService(db)
