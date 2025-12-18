"""
Document service for upload, ingestion, and indexing.
"""
import os
import uuid
import hashlib
import json
import logging
from datetime import datetime
from typing import Optional, List

import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from app.config import settings
from app.models.document import Document, Chunk, DocumentStatusEnum, ChunkStatusEnum
from app.middleware.error_handler import NotFoundError, ValidationAppError

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        owner_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        access_level: str = "private",
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Document:
        """
        Upload and store a document, queue for processing.
        """
        # Validate file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValidationAppError(
                f"File type {ext} not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Validate file size
        file_size = len(file_content)
        max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size > max_size:
            raise ValidationAppError(
                f"File size {file_size} exceeds maximum {max_size} bytes"
            )
        
        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Generate unique filename
        doc_id = uuid.uuid4()
        stored_filename = f"{doc_id}{ext}"
        
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, stored_filename)
        
        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)
        
        # Create document record
        document = Document(
            id=doc_id,
            filename=stored_filename,
            original_filename=filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=content_type,
            file_hash=file_hash,
            owner_id=owner_id,
            tenant_id=tenant_id,
            access_level=access_level,
            title=title or filename,
            description=description,
            tags=",".join(tags) if tags else "",
            status=DocumentStatusEnum.PROCESSING,
        )
        
        self.db.add(document)
        await self.db.flush()
        
        # TODO: Queue background job for processing
        # In production: celery_app.send_task("process_document", args=[str(doc_id)])
        
        return document
    
    async def get_document_by_id(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> Optional[Document]:
        """Get a document by ID with access control."""
        query = select(Document).where(Document.id == document_id)
        
        # Apply access control (unless admin)
        if not is_admin:
            query = query.where(
                (Document.owner_id == user_id) | 
                (Document.access_level == "public") |
                ((Document.tenant_id == tenant_id) & (Document.access_level == "internal"))
            )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def list_documents(
        self,
        user_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        is_admin: bool = False,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> tuple[List[Document], int]:
        """List documents with pagination and access control."""
        query = select(Document)
        count_query = select(func.count(Document.id))
        
        # Apply access control
        if not is_admin:
            access_filter = (
                (Document.owner_id == user_id) | 
                (Document.access_level == "public") |
                ((Document.tenant_id == tenant_id) & (Document.access_level == "internal"))
            )
            query = query.where(access_filter)
            count_query = count_query.where(access_filter)
        
        # Apply status filter
        if status:
            query = query.where(Document.status == status)
            count_query = count_query.where(Document.status == status)
        
        # Apply search
        if search:
            search_filter = (
                Document.title.ilike(f"%{search}%") |
                Document.original_filename.ilike(f"%{search}%") |
                Document.tags.ilike(f"%{search}%")
            )
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)
        
        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar_one()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size).order_by(Document.created_at.desc())
        
        result = await self.db.execute(query)
        documents = result.scalars().all()
        
        return list(documents), total
    
    async def delete_document(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> bool:
        """Delete a document and its chunks."""
        document = await self.get_document_by_id(
            document_id, user_id, is_admin=is_admin
        )
        
        if not document:
            raise NotFoundError("Document not found")
        
        # Check ownership (unless admin)
        if not is_admin and document.owner_id != user_id:
            raise NotFoundError("Document not found")
        
        # Delete file from storage
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete from vector store first
        try:
            from app.services.vector_store import faiss_service
            deleted_count = await faiss_service.delete_by_document(str(document_id))
            logger.info(f"Deleted {deleted_count} chunks from vector store for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete from vector store: {e}")
        
        # Delete chunks from database (use delete statement, not select)
        from sqlmodel import delete
        await self.db.execute(
            delete(Chunk).where(Chunk.document_id == document_id)
        )
        
        # Delete document record
        await self.db.delete(document)
        await self.db.commit()
        
        return True
    
    async def reindex_document(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> Document:
        """Trigger reindexing of a document."""
        document = await self.get_document_by_id(
            document_id, user_id, is_admin=is_admin
        )
        
        if not document:
            raise NotFoundError("Document not found")
        
        # Reset status
        document.status = DocumentStatusEnum.PROCESSING
        document.updated_at = datetime.utcnow()
        
        # Mark existing chunks as superseded
        result = await self.db.execute(
            select(Chunk).where(Chunk.document_id == document_id)
        )
        chunks = result.scalars().all()
        for chunk in chunks:
            chunk.status = ChunkStatusEnum.SUPERSEDED
        
        await self.db.flush()
        
        # TODO: Queue background job for reprocessing
        
        return document
    
    async def get_chunks_for_document(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        is_admin: bool = False,
    ) -> List[Chunk]:
        """Get all active chunks for a document."""
        # Verify access to document first
        document = await self.get_document_by_id(
            document_id, user_id, is_admin=is_admin
        )
        
        if not document:
            raise NotFoundError("Document not found")
        
        result = await self.db.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .where(Chunk.status != ChunkStatusEnum.SUPERSEDED)
            .order_by(Chunk.chunk_index)
        )
        
        return list(result.scalars().all())
