"""
Entity Extraction Service for GraphRAG.

Handles:
- Entity extraction from chunk text using UtilityLLM
- Entity normalization and deduplication
- Entity linking to existing graph entities
- Relationship extraction
"""
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.graph import (
    Entity, EntityMention, EntityRelationship,
    EntityType, RelationshipType
)
from app.models.document import Chunk
from app.services.utility_llm_client import get_utility_llm

logger = logging.getLogger(__name__)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


@dataclass
class ExtractedEntity:
    """Entity extracted from text, before database storage."""
    name: str
    normalized_name: str
    entity_type: EntityType
    confidence: float
    mention_text: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


@dataclass
class ExtractedRelationship:
    """Relationship extracted from text."""
    source_name: str
    target_name: str
    relationship_type: RelationshipType
    confidence: float


@dataclass
class ExtractionResult:
    """Result of entity/relationship extraction from a chunk."""
    chunk_id: uuid.UUID
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    processing_ms: int
    extraction_method: str


class EntityExtractionService:
    """
    Service for extracting and managing entities from document chunks.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._llm = get_utility_llm()
        self._fuzzy_threshold = settings.GRAPH_ENTITY_FUZZY_THRESHOLD
    
    async def extract_from_chunk(
        self,
        chunk: Chunk,
        tenant_id: Optional[str] = None,
        access_level: str = "internal",
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a chunk.
        
        Args:
            chunk: The chunk to process
            tenant_id: Tenant for RBAC
            access_level: Access level for extracted entities
        
        Returns:
            ExtractionResult with extracted entities and relationships
        """
        import time
        t0 = time.perf_counter()
        
        # Use combined extraction for efficiency
        result = await self._llm.extract_entities_and_relationships(chunk.text)
        
        entities = []
        for ent_data in result.get("entities", []):
            entity_type = self._map_entity_type(ent_data.get("type", "OTHER"))
            name = ent_data.get("name", "").strip()
            
            if name:
                entities.append(ExtractedEntity(
                    name=name,
                    normalized_name=self._normalize_name(name),
                    entity_type=entity_type,
                    confidence=ent_data.get("confidence", 0.8),
                    mention_text=name,
                ))
        
        relationships = []
        for rel_data in result.get("relationships", []):
            rel_type = self._map_relationship_type(rel_data.get("type", "RELATED_TO"))
            source = rel_data.get("source", "").strip()
            target = rel_data.get("target", "").strip()
            
            if source and target:
                relationships.append(ExtractedRelationship(
                    source_name=source,
                    target_name=target,
                    relationship_type=rel_type,
                    confidence=rel_data.get("confidence", 0.7),
                ))
        
        processing_ms = int((time.perf_counter() - t0) * 1000)
        
        return ExtractionResult(
            chunk_id=chunk.id,
            entities=entities,
            relationships=relationships,
            processing_ms=processing_ms,
            extraction_method=result.get("extraction_method", "llm"),
        )
    
    async def process_extraction_result(
        self,
        result: ExtractionResult,
        tenant_id: Optional[str] = None,
        access_level: str = "internal",
        department: Optional[str] = None,
        owner_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """
        Process extraction result: link entities, create mentions, store relationships.
        
        Returns:
            Dict with counts of created/linked items
        """
        stats = {
            "entities_created": 0,
            "entities_linked": 0,
            "mentions_created": 0,
            "relationships_created": 0,
        }
        
        # Map entity names to database entity IDs
        entity_name_to_id: Dict[str, uuid.UUID] = {}
        
        # Process entities
        for extracted in result.entities:
            entity, is_new = await self._get_or_create_entity(
                extracted=extracted,
                tenant_id=tenant_id,
                access_level=access_level,
                department=department,
                owner_id=owner_id,
            )
            
            if is_new:
                stats["entities_created"] += 1
            else:
                stats["entities_linked"] += 1
            
            entity_name_to_id[extracted.normalized_name] = entity.id
            
            # Create mention
            mention = EntityMention(
                entity_id=entity.id,
                chunk_id=result.chunk_id,
                mention_text=extracted.mention_text,
                start_position=extracted.start_pos,
                end_position=extracted.end_pos,
                confidence=extracted.confidence,
                extraction_method=result.extraction_method,
            )
            self.db.add(mention)
            stats["mentions_created"] += 1
        
        await self.db.flush()
        
        # Process relationships
        for rel in result.relationships:
            source_norm = self._normalize_name(rel.source_name)
            target_norm = self._normalize_name(rel.target_name)
            
            source_id = entity_name_to_id.get(source_norm)
            target_id = entity_name_to_id.get(target_norm)
            
            # Try fuzzy match if exact match not found
            if not source_id:
                source_id = await self._fuzzy_find_entity_id(source_norm, entity_name_to_id)
            if not target_id:
                target_id = await self._fuzzy_find_entity_id(target_norm, entity_name_to_id)
            
            if source_id and target_id and source_id != target_id:
                relationship = EntityRelationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel.relationship_type,
                    weight=rel.confidence,
                    confidence=rel.confidence,
                    evidence_chunk_id=result.chunk_id,
                    tenant_id=tenant_id,
                    access_level=access_level,
                )
                self.db.add(relationship)
                stats["relationships_created"] += 1
        
        await self.db.commit()
        
        logger.debug(
            f"Processed chunk {result.chunk_id}: "
            f"{stats['entities_created']} new entities, "
            f"{stats['entities_linked']} linked, "
            f"{stats['relationships_created']} relationships"
        )
        
        return stats
    
    async def _get_or_create_entity(
        self,
        extracted: ExtractedEntity,
        tenant_id: Optional[str],
        access_level: str,
        department: Optional[str],
        owner_id: Optional[uuid.UUID],
    ) -> Tuple[Entity, bool]:
        """
        Get existing entity or create new one.
        Uses fuzzy matching to link similar entities.
        
        Returns:
            Tuple of (Entity, is_new)
        """
        # Try exact match first
        result = await self.db.execute(
            select(Entity).where(Entity.normalized_name == extracted.normalized_name)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update mention count
            existing.mention_count += 1
            existing.updated_at = datetime.utcnow()
            return existing, False
        
        # Try fuzzy match
        fuzzy_match = await self._find_fuzzy_match(extracted.normalized_name)
        if fuzzy_match:
            fuzzy_match.mention_count += 1
            fuzzy_match.updated_at = datetime.utcnow()
            return fuzzy_match, False
        
        # Create new entity
        entity = Entity(
            name=extracted.name,
            normalized_name=extracted.normalized_name,
            entity_type=extracted.entity_type,
            tenant_id=tenant_id,
            access_level=access_level,
            department=department,
            owner_id=owner_id,
            mention_count=1,
        )
        self.db.add(entity)
        await self.db.flush()
        
        return entity, True
    
    async def _find_fuzzy_match(self, normalized_name: str) -> Optional[Entity]:
        """Find an existing entity with similar name using Levenshtein distance."""
        # Get candidates (entities starting with same letter for efficiency)
        if not normalized_name:
            return None
        
        first_char = normalized_name[0]
        result = await self.db.execute(
            select(Entity).where(Entity.normalized_name.startswith(first_char))
        )
        candidates = result.scalars().all()
        
        best_match: Optional[Entity] = None
        best_distance = self._fuzzy_threshold + 1
        
        for entity in candidates:
            distance = _levenshtein_distance(normalized_name, entity.normalized_name)
            if distance <= self._fuzzy_threshold and distance < best_distance:
                best_match = entity
                best_distance = distance
        
        return best_match
    
    async def _fuzzy_find_entity_id(
        self,
        normalized_name: str,
        local_cache: Dict[str, uuid.UUID],
    ) -> Optional[uuid.UUID]:
        """Find entity ID using fuzzy matching, checking local cache first."""
        # Check local cache with fuzzy matching
        for cached_name, entity_id in local_cache.items():
            if _levenshtein_distance(normalized_name, cached_name) <= self._fuzzy_threshold:
                return entity_id
        
        # Check database
        match = await self._find_fuzzy_match(normalized_name)
        if match:
            return match.id
        
        return None
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        # Lowercase
        normalized = name.lower().strip()
        # Collapse multiple spaces
        normalized = " ".join(normalized.split())
        return normalized
    
    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map LLM entity type string to EntityType enum."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "LOCATION": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.PRODUCT,
            "TECHNOLOGY": EntityType.TECHNOLOGY,
            "CONCEPT": EntityType.CONCEPT,
        }
        return mapping.get(type_str.upper(), EntityType.OTHER)
    
    def _map_relationship_type(self, type_str: str) -> RelationshipType:
        """Map LLM relationship type string to RelationshipType enum."""
        mapping = {
            "WORKS_FOR": RelationshipType.WORKS_FOR,
            "LOCATED_IN": RelationshipType.LOCATED_IN,
            "RELATED_TO": RelationshipType.RELATED_TO,
            "PART_OF": RelationshipType.PART_OF,
            "CREATED_BY": RelationshipType.CREATED_BY,
            "OWNS": RelationshipType.OWNS,
            "MANAGES": RelationshipType.MANAGES,
            "MENTIONED_WITH": RelationshipType.MENTIONED_WITH,
        }
        return mapping.get(type_str.upper(), RelationshipType.RELATED_TO)
    
    async def extract_entities_from_query(
        self,
        query: str,
    ) -> List[str]:
        """
        Extract entity names from a query for graph traversal.
        
        Returns:
            List of entity names found in the query
        """
        result = await self._llm.extract_entities(query)
        
        entities = []
        for ent_data in result.get("entities", []):
            name = ent_data.get("name", "").strip()
            if name:
                entities.append(name)
        
        return entities
    
    async def get_entity_count(self) -> int:
        """Get total number of entities in the graph."""
        result = await self.db.execute(select(func.count(Entity.id)))
        return result.scalar() or 0
    
    async def get_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100,
    ) -> List[Entity]:
        """Get entities of a specific type."""
        result = await self.db.execute(
            select(Entity)
            .where(Entity.entity_type == entity_type)
            .order_by(Entity.mention_count.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# =============================================================================
# Service Factory
# =============================================================================

def get_entity_extraction_service(db: AsyncSession) -> EntityExtractionService:
    """Get an EntityExtractionService instance."""
    return EntityExtractionService(db)
