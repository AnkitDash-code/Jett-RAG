"""
Graph-related models for Phase 4 GraphRAG.
Includes: Entity, EntityMention, EntityRelationship, Community, CommunityMembership,
          Group, GroupMembership, FolderPermission, CommunityContext
"""
import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlmodel import SQLModel, Field, Relationship


# =============================================================================
# Enums
# =============================================================================

class EntityType(str, Enum):
    """Types of entities extracted from documents."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    MENTIONED_WITH = "mentioned_with"  # Co-occurrence in same chunk
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CREATED_BY = "created_by"
    OWNS = "owns"
    MANAGES = "manages"
    OTHER = "other"


class PermissionType(str, Enum):
    """Permission levels for graph RBAC."""
    VIEW = "view"
    EDIT = "edit"
    ADMIN = "admin"


class GroupRole(str, Enum):
    """Roles within a group."""
    MEMBER = "member"
    ADMIN = "admin"
    OWNER = "owner"


# =============================================================================
# Entity Tables
# =============================================================================

class Entity(SQLModel, table=True):
    """
    Represents a named entity extracted from documents.
    Core node in the knowledge graph.
    """
    __tablename__ = "entities"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Entity identification
    name: str = Field(max_length=255, index=True)
    normalized_name: str = Field(max_length=255, index=True)  # Lowercase, trimmed
    entity_type: EntityType = Field(default=EntityType.OTHER, index=True)
    description: Optional[str] = Field(default=None)
    
    # RBAC fields (denormalized for efficient filtering)
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    access_level: str = Field(default="internal", max_length=50)
    department: Optional[str] = Field(default=None, max_length=100)
    owner_id: Optional[uuid.UUID] = Field(default=None, index=True)
    
    # Graph metrics (cached)
    mention_count: int = Field(default=0)
    relationship_count: int = Field(default=0)
    centrality_score: float = Field(default=0.0)
    
    # Metadata
    metadata_json: str = Field(default="{}")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    mentions: List["EntityMention"] = Relationship(back_populates="entity")
    source_relationships: List["EntityRelationship"] = Relationship(
        back_populates="source_entity",
        sa_relationship_kwargs={"foreign_keys": "[EntityRelationship.source_entity_id]"}
    )
    target_relationships: List["EntityRelationship"] = Relationship(
        back_populates="target_entity",
        sa_relationship_kwargs={"foreign_keys": "[EntityRelationship.target_entity_id]"}
    )
    community_memberships: List["CommunityMembership"] = Relationship(back_populates="entity")


class EntityMention(SQLModel, table=True):
    """
    Tracks where an entity is mentioned in a chunk.
    Edge: Entity -> Chunk
    """
    __tablename__ = "entity_mentions"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Foreign keys
    entity_id: uuid.UUID = Field(foreign_key="entities.id", index=True)
    chunk_id: uuid.UUID = Field(foreign_key="chunks.id", index=True)
    
    # Position in chunk text
    start_position: Optional[int] = Field(default=None)
    end_position: Optional[int] = Field(default=None)
    mention_text: str = Field(max_length=255)  # The actual text that matched
    
    # Confidence
    confidence: float = Field(default=1.0)  # 0.0 to 1.0
    extraction_method: str = Field(default="llm", max_length=50)  # llm, ner, regex
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    entity: Optional[Entity] = Relationship(back_populates="mentions")


class EntityRelationship(SQLModel, table=True):
    """
    Represents a relationship between two entities.
    Edge: Entity -> Entity
    """
    __tablename__ = "entity_relationships"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Relationship endpoints
    source_entity_id: uuid.UUID = Field(foreign_key="entities.id", index=True)
    target_entity_id: uuid.UUID = Field(foreign_key="entities.id", index=True)
    
    # Relationship details
    relationship_type: RelationshipType = Field(default=RelationshipType.RELATED_TO)
    description: Optional[str] = Field(default=None, max_length=500)
    weight: float = Field(default=1.0)  # Strength of relationship
    
    # Evidence
    evidence_chunk_id: Optional[uuid.UUID] = Field(default=None, index=True)
    confidence: float = Field(default=1.0)
    
    # RBAC (inherited from entities, but denormalized for query efficiency)
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    access_level: str = Field(default="internal", max_length=50)
    
    # Metadata
    metadata_json: str = Field(default="{}")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    source_entity: Optional[Entity] = Relationship(
        back_populates="source_relationships",
        sa_relationship_kwargs={"foreign_keys": "[EntityRelationship.source_entity_id]"}
    )
    target_entity: Optional[Entity] = Relationship(
        back_populates="target_relationships",
        sa_relationship_kwargs={"foreign_keys": "[EntityRelationship.target_entity_id]"}
    )


# =============================================================================
# Community Tables
# =============================================================================

class Community(SQLModel, table=True):
    """
    Represents a community/cluster of related entities.
    Detected via connected_components or modularity algorithms.
    """
    __tablename__ = "communities"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Community info
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None)
    algorithm: str = Field(default="connected_components", max_length=50)
    
    # Statistics
    node_count: int = Field(default=0)
    edge_count: int = Field(default=0)
    density: float = Field(default=0.0)
    
    # RBAC (community inherits most restrictive access from members)
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    min_access_level: str = Field(default="public", max_length=50)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    memberships: List["CommunityMembership"] = Relationship(back_populates="community")
    context: Optional["CommunityContext"] = Relationship(back_populates="community")


class CommunityMembership(SQLModel, table=True):
    """
    Links entities to their communities.
    An entity can belong to multiple communities with different scores.
    """
    __tablename__ = "community_memberships"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Foreign keys
    community_id: uuid.UUID = Field(foreign_key="communities.id", index=True)
    entity_id: uuid.UUID = Field(foreign_key="entities.id", index=True)
    
    # Membership details
    membership_score: float = Field(default=1.0)  # How strongly entity belongs
    is_central: bool = Field(default=False)  # Is this entity central to community?
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    community: Optional[Community] = Relationship(back_populates="memberships")
    entity: Optional[Entity] = Relationship(back_populates="community_memberships")


class CommunityContext(SQLModel, table=True):
    """
    Pre-computed summary/context for a community.
    Used to inject high-level context into RAG prompts.
    """
    __tablename__ = "community_contexts"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Foreign key
    community_id: uuid.UUID = Field(foreign_key="communities.id", index=True, unique=True)
    
    # Summary content
    summary_text: str  # LLM-generated summary of the community
    key_entities: str = Field(default="")  # Comma-separated top entity names
    key_topics: str = Field(default="")  # Comma-separated topic keywords
    
    # Generation metadata
    token_count: int = Field(default=0)
    generation_model: str = Field(default="utility_llm", max_length=100)
    
    # Validity
    is_stale: bool = Field(default=False)  # Needs regeneration
    entity_count_at_generation: int = Field(default=0)
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    community: Optional[Community] = Relationship(back_populates="context")


# =============================================================================
# Permission Tables (Group-based RBAC)
# =============================================================================

class Group(SQLModel, table=True):
    """
    User groups for permission management.
    Groups can have access to folders, documents, or entities.
    """
    __tablename__ = "groups"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Group info
    name: str = Field(max_length=100, index=True)
    description: Optional[str] = Field(default=None)
    
    # Tenant scoping
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    
    # Metadata
    is_system: bool = Field(default=False)  # System-created group
    metadata_json: str = Field(default="{}")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[uuid.UUID] = Field(default=None)
    
    # Relationships
    memberships: List["GroupMembership"] = Relationship(back_populates="group")
    folder_permissions: List["FolderPermission"] = Relationship(
        back_populates="group",
        sa_relationship_kwargs={"foreign_keys": "[FolderPermission.group_id]"}
    )


class GroupMembership(SQLModel, table=True):
    """
    Links users to groups with roles.
    """
    __tablename__ = "group_memberships"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Foreign keys
    group_id: uuid.UUID = Field(foreign_key="groups.id", index=True)
    user_id: uuid.UUID = Field(index=True)  # References users table
    
    # Role in group
    role: GroupRole = Field(default=GroupRole.MEMBER)
    
    # Timestamps
    added_at: datetime = Field(default_factory=datetime.utcnow)
    added_by: Optional[uuid.UUID] = Field(default=None)
    
    # Relationships
    group: Optional[Group] = Relationship(back_populates="memberships")


class FolderPermission(SQLModel, table=True):
    """
    Permissions on folders that cascade to documents and entities.
    Can be assigned to users or groups.
    """
    __tablename__ = "folder_permissions"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Resource
    folder_path: str = Field(max_length=512, index=True)  # e.g., "/legal/contracts"
    
    # Grantee (one of group_id or user_id should be set)
    group_id: Optional[uuid.UUID] = Field(default=None, foreign_key="groups.id", index=True)
    user_id: Optional[uuid.UUID] = Field(default=None, index=True)
    
    # Permission level
    permission_type: PermissionType = Field(default=PermissionType.VIEW)
    
    # Inheritance tracking
    inherited_from: Optional[str] = Field(default=None, max_length=512)  # Parent folder path
    is_inherited: bool = Field(default=False)  # Was this inherited or directly assigned?
    
    # Tenant scoping
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[uuid.UUID] = Field(default=None)
    expires_at: Optional[datetime] = Field(default=None)  # Optional expiration
    
    # Relationships
    group: Optional[Group] = Relationship(
        back_populates="folder_permissions",
        sa_relationship_kwargs={"foreign_keys": "[FolderPermission.group_id]"}
    )


# =============================================================================
# Graph Processing Queue
# =============================================================================

class GraphProcessingTask(SQLModel, table=True):
    """
    Queue for background graph processing tasks.
    Tracks chunks that need entity extraction and graph indexing.
    """
    __tablename__ = "graph_processing_tasks"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Task target
    chunk_id: uuid.UUID = Field(foreign_key="chunks.id", index=True)
    document_id: uuid.UUID = Field(index=True)
    
    # Task status
    status: str = Field(default="pending", max_length=50, index=True)  # pending, processing, completed, failed
    priority: int = Field(default=0)  # Higher = more urgent
    
    # Processing info
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)
    error_message: Optional[str] = Field(default=None)
    worker_id: Optional[str] = Field(default=None, max_length=100)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
