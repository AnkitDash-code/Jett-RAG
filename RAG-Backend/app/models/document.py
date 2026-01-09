"""
Document and indexing related models.
Includes: Document, DocumentVersion, Chunk
"""
import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlmodel import SQLModel, Field, Relationship


class DocumentStatusEnum(str, Enum):
    """Document processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    INDEXED_LIGHT = "indexed_light"  # iRAG style - embeddings done
    INDEXED_FULL = "indexed_full"    # Graph processing complete
    ERROR = "error"
    FAILED = "failed"
    ARCHIVED = "archived"


class ChunkStatusEnum(str, Enum):
    """Chunk processing status."""
    PENDING = "pending"
    EMBEDDED = "embedded"
    GRAPH_PROCESSED = "graph_processed"
    SUPERSEDED = "superseded"  # Old version, kept for audit


class Document(SQLModel, table=True):
    """Main document model representing uploaded files."""
    __tablename__ = "documents"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # File information
    filename: str = Field(max_length=255, index=True)
    original_filename: str = Field(max_length=255)
    file_path: str = Field(max_length=512)  # Path in object storage
    file_size: int = Field(default=0)  # Size in bytes
    mime_type: str = Field(max_length=100)
    file_hash: str = Field(max_length=64, index=True)  # SHA-256 for deduplication
    
    # Ownership and access control
    owner_id: uuid.UUID = Field(index=True)
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    access_level: str = Field(default="private", max_length=50)  # public, internal, confidential, etc.
    department: Optional[str] = Field(default=None, max_length=100)
    
    # Metadata
    title: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None)
    tags: str = Field(default="")  # Comma-separated tags
    metadata_json: str = Field(default="{}")  # Additional metadata as JSON
    
    # Processing status
    status: DocumentStatusEnum = Field(default=DocumentStatusEnum.UPLOADING)
    error_message: Optional[str] = Field(default=None)
    processing_job_id: Optional[uuid.UUID] = Field(default=None, index=True)  # Phase 5: linked job
    
    # Version tracking
    current_version: int = Field(default=1)
    
    # Statistics
    page_count: Optional[int] = Field(default=None)
    chunk_count: int = Field(default=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    versions: List["DocumentVersion"] = Relationship(back_populates="document")
    chunks: List["Chunk"] = Relationship(back_populates="document")


class DocumentVersion(SQLModel, table=True):
    """Track document versions for updates and audit."""
    __tablename__ = "document_versions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="documents.id", index=True)
    version_number: int = Field(index=True)
    
    # File info for this version
    file_path: str = Field(max_length=512)
    file_size: int = Field(default=0)
    file_hash: str = Field(max_length=64)
    
    # Status
    status: DocumentStatusEnum = Field(default=DocumentStatusEnum.PROCESSING)
    
    # Change tracking
    change_summary: Optional[str] = Field(default=None)
    created_by: uuid.UUID
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    document: Optional[Document] = Relationship(back_populates="versions")


class Chunk(SQLModel, table=True):
    """Text chunks for RAG retrieval."""
    __tablename__ = "chunks"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="documents.id", index=True)
    version_number: int = Field(default=1)
    
    # Content
    text: str  # The actual chunk text
    content_hash: str = Field(max_length=64, index=True)  # For deduplication
    
    # Position in document
    chunk_index: int = Field(index=True)  # Order within document
    page_number: Optional[int] = Field(default=None)
    section_path: Optional[str] = Field(default=None, max_length=255)  # e.g., "Chapter 1 > Section 2"
    
    # Hierarchical chunking
    parent_chunk_id: Optional[uuid.UUID] = Field(default=None, index=True)
    children_ids: str = Field(default="")  # Comma-separated UUIDs
    hierarchy_level: int = Field(default=0)  # 0=leaf, 1=section, 2=chapter, etc.
    
    # Access control (denormalized for vector DB filtering)
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    access_level: str = Field(default="private", max_length=50)
    department: Optional[str] = Field(default=None, max_length=100)
    
    # Processing status
    status: ChunkStatusEnum = Field(default=ChunkStatusEnum.PENDING)
    needs_graph_processing: bool = Field(default=True)  # iRAG flag
    
    # Vector DB reference
    vector_id: Optional[str] = Field(default=None, max_length=100)  # ID in vector store
    
    # Metadata
    metadata_json: str = Field(default="{}")  # Additional metadata
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    embedded_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    document: Optional[Document] = Relationship(back_populates="chunks")
