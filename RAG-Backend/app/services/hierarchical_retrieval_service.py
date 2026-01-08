"""
Hierarchical Retrieval Service for Phase 5.

Provides context expansion via hierarchical chunk relationships:
- expand_to_parents: Get parent summaries for broader context
- expand_to_children: Get child chunks for detailed context
- expand_siblings: Get adjacent chunks within same parent

This enables the retrieval pipeline to provide both summary-level
context (parents) and fine-grained detail (children).
"""
import logging
import uuid
from typing import List, Optional, Dict, Any, Set

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.models.document import Chunk

logger = logging.getLogger(__name__)


class HierarchicalRetrievalService:
    """
    Service for hierarchical chunk expansion in RAG retrieval.
    
    Hierarchy levels:
    - Level 0: Leaf chunks (original text chunks)
    - Level 1: Section/paragraph summaries
    - Level 2: Document-level summaries (future)
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.enabled = getattr(settings, 'ENABLE_HIERARCHICAL_CHUNKING', False)
    
    async def expand_to_parents(
        self,
        chunk_ids: List[uuid.UUID],
        include_original: bool = True,
    ) -> List[Chunk]:
        """
        Expand a set of chunks to include their parent summaries.
        
        This provides broader context by retrieving the summary chunks
        that contain the retrieved leaf chunks.
        
        Args:
            chunk_ids: List of chunk UUIDs to expand
            include_original: Whether to include the original chunks in result
            
        Returns:
            List of chunks including parents (and optionally originals)
        """
        if not self.enabled or not chunk_ids:
            return []
        
        # Get original chunks
        result = await self.db.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        )
        original_chunks = list(result.scalars().all())
        
        if not original_chunks:
            return []
        
        # Collect parent IDs
        parent_ids: Set[uuid.UUID] = set()
        for chunk in original_chunks:
            if chunk.parent_chunk_id:
                parent_ids.add(chunk.parent_chunk_id)
        
        # Fetch parent chunks
        parent_chunks = []
        if parent_ids:
            result = await self.db.execute(
                select(Chunk).where(Chunk.id.in_(list(parent_ids)))
            )
            parent_chunks = list(result.scalars().all())
        
        # Combine results
        if include_original:
            # Parents first (broader context), then originals
            return parent_chunks + original_chunks
        else:
            return parent_chunks
    
    async def expand_to_children(
        self,
        chunk_ids: List[uuid.UUID],
        include_original: bool = True,
    ) -> List[Chunk]:
        """
        Expand parent/summary chunks to include their children.
        
        This provides detailed context when a summary chunk is retrieved,
        by including the actual text chunks it summarizes.
        
        Args:
            chunk_ids: List of chunk UUIDs to expand
            include_original: Whether to include the original chunks in result
            
        Returns:
            List of chunks including children (and optionally originals)
        """
        if not self.enabled or not chunk_ids:
            return []
        
        # Get original chunks
        result = await self.db.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        )
        original_chunks = list(result.scalars().all())
        
        if not original_chunks:
            return []
        
        # Collect child IDs from children_ids field
        child_ids: Set[uuid.UUID] = set()
        for chunk in original_chunks:
            if chunk.children_ids:
                for child_id_str in chunk.children_ids.split(","):
                    child_id_str = child_id_str.strip()
                    if child_id_str:
                        try:
                            child_ids.add(uuid.UUID(child_id_str))
                        except ValueError:
                            logger.warning(f"Invalid child UUID: {child_id_str}")
        
        # Fetch child chunks
        child_chunks = []
        if child_ids:
            result = await self.db.execute(
                select(Chunk).where(Chunk.id.in_(list(child_ids)))
            )
            child_chunks = list(result.scalars().all())
            # Sort by chunk_index for proper order
            child_chunks.sort(key=lambda c: c.chunk_index)
        
        # Combine results
        if include_original:
            # Originals (summaries) first, then children (details)
            return original_chunks + child_chunks
        else:
            return child_chunks
    
    async def expand_siblings(
        self,
        chunk_ids: List[uuid.UUID],
        window: int = 1,
        include_original: bool = True,
    ) -> List[Chunk]:
        """
        Expand chunks to include adjacent siblings within same parent.
        
        This provides local context by retrieving nearby chunks that
        share the same parent (i.e., are in the same section).
        
        Args:
            chunk_ids: List of chunk UUIDs to expand
            window: Number of siblings before/after to include
            include_original: Whether to include the original chunks in result
            
        Returns:
            List of chunks including siblings
        """
        if not chunk_ids:
            return []
        
        # Get original chunks
        result = await self.db.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        )
        original_chunks = list(result.scalars().all())
        
        if not original_chunks:
            return []
        
        # Group by document and parent to find siblings
        all_sibling_ids: Set[uuid.UUID] = set()
        
        for chunk in original_chunks:
            # Find siblings by same document and parent
            sibling_query = select(Chunk).where(
                (Chunk.document_id == chunk.document_id) &
                (Chunk.hierarchy_level == chunk.hierarchy_level)
            )
            
            # If has parent, filter by same parent
            if chunk.parent_chunk_id:
                sibling_query = sibling_query.where(
                    Chunk.parent_chunk_id == chunk.parent_chunk_id
                )
            
            # Filter by chunk_index range
            min_idx = max(0, chunk.chunk_index - window)
            max_idx = chunk.chunk_index + window
            sibling_query = sibling_query.where(
                (Chunk.chunk_index >= min_idx) &
                (Chunk.chunk_index <= max_idx)
            )
            
            result = await self.db.execute(sibling_query)
            for sibling in result.scalars().all():
                if sibling.id not in chunk_ids:  # Don't duplicate originals
                    all_sibling_ids.add(sibling.id)
        
        # Fetch all siblings
        sibling_chunks = []
        if all_sibling_ids:
            result = await self.db.execute(
                select(Chunk).where(Chunk.id.in_(list(all_sibling_ids)))
            )
            sibling_chunks = list(result.scalars().all())
        
        # Combine and sort by document then index
        all_chunks = original_chunks + sibling_chunks if include_original else sibling_chunks
        all_chunks.sort(key=lambda c: (str(c.document_id), c.chunk_index))
        
        return all_chunks
    
    async def get_chunk_hierarchy(
        self,
        chunk_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """
        Get the full hierarchy context for a single chunk.
        
        Returns:
            Dict with 'chunk', 'parent', 'children', 'siblings'
        """
        # Get the chunk
        result = await self.db.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        )
        chunk = result.scalar_one_or_none()
        
        if not chunk:
            return {"chunk": None, "parent": None, "children": [], "siblings": []}
        
        hierarchy = {
            "chunk": chunk,
            "parent": None,
            "children": [],
            "siblings": [],
        }
        
        # Get parent
        if chunk.parent_chunk_id:
            parents = await self.expand_to_parents([chunk_id], include_original=False)
            hierarchy["parent"] = parents[0] if parents else None
        
        # Get children
        children = await self.expand_to_children([chunk_id], include_original=False)
        hierarchy["children"] = children
        
        # Get siblings
        siblings = await self.expand_siblings([chunk_id], window=1, include_original=False)
        hierarchy["siblings"] = siblings
        
        return hierarchy
    
    async def expand_retrieval_results(
        self,
        chunk_ids: List[uuid.UUID],
        strategy: str = "parent",
        max_chunks: int = 20,
    ) -> List[Chunk]:
        """
        Expand retrieval results using specified strategy.
        
        Strategies:
        - "parent": Add parent summaries for broader context
        - "children": Add child chunks for detail
        - "siblings": Add adjacent chunks for local context
        - "full": Add parents, children, and siblings
        
        Args:
            chunk_ids: Retrieved chunk IDs to expand
            strategy: Expansion strategy
            max_chunks: Maximum total chunks to return
            
        Returns:
            Expanded list of chunks
        """
        if not self.enabled:
            # Just return original chunks
            result = await self.db.execute(
                select(Chunk).where(Chunk.id.in_(chunk_ids))
            )
            return list(result.scalars().all())
        
        expanded_chunks: List[Chunk] = []
        seen_ids: Set[uuid.UUID] = set()
        
        if strategy in ("parent", "full"):
            parents = await self.expand_to_parents(chunk_ids, include_original=True)
            for c in parents:
                if c.id not in seen_ids:
                    expanded_chunks.append(c)
                    seen_ids.add(c.id)
        
        if strategy in ("children", "full"):
            children = await self.expand_to_children(chunk_ids, include_original=(strategy != "full"))
            for c in children:
                if c.id not in seen_ids:
                    expanded_chunks.append(c)
                    seen_ids.add(c.id)
        
        if strategy in ("siblings", "full"):
            siblings = await self.expand_siblings(chunk_ids, window=1, include_original=(strategy not in ("parent", "children", "full")))
            for c in siblings:
                if c.id not in seen_ids:
                    expanded_chunks.append(c)
                    seen_ids.add(c.id)
        
        # If no strategy matched, just return originals
        if not expanded_chunks:
            result = await self.db.execute(
                select(Chunk).where(Chunk.id.in_(chunk_ids))
            )
            expanded_chunks = list(result.scalars().all())
        
        # Sort by hierarchy level (parents first) then by chunk index
        expanded_chunks.sort(key=lambda c: (c.hierarchy_level, str(c.document_id), c.chunk_index))
        
        # Limit total chunks
        return expanded_chunks[:max_chunks]


# Singleton getter
_hierarchical_service: Optional[HierarchicalRetrievalService] = None


def get_hierarchical_retrieval_service(db: AsyncSession) -> HierarchicalRetrievalService:
    """Get hierarchical retrieval service instance."""
    return HierarchicalRetrievalService(db)
