"""
Document schemas for request/response validation.
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    id: str
    filename: str
    status: str
    message: str = "Document uploaded successfully. Processing started."


class DocumentResponse(BaseModel):
    """Full document details response."""
    id: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    title: Optional[str]
    description: Optional[str]
    tags: List[str]
    status: str
    access_level: str
    page_count: Optional[int]
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    indexed_at: Optional[datetime]
    
    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated document list response."""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    pages: int


class ChunkResponse(BaseModel):
    """Chunk information for debugging/admin."""
    id: str
    document_id: str
    text: str
    chunk_index: int
    page_number: Optional[int]
    section_path: Optional[str]
    status: str
    relevance_score: Optional[float] = None
    
    model_config = {"from_attributes": True}
