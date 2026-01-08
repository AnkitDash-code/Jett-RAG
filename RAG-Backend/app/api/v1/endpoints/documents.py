"""
Document API endpoints.
POST /documents/upload, GET /documents, GET /documents/{id}, etc.
"""
import uuid
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, async_session_maker
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
)
from app.api.deps import get_current_user, CurrentUser, can_upload_documents
from app.middleware.error_handler import NotFoundError

router = APIRouter()


def document_to_response(doc) -> DocumentResponse:
    """Convert Document model to response schema."""
    return DocumentResponse(
        id=str(doc.id),
        filename=doc.filename,
        original_filename=doc.original_filename,
        file_size=doc.file_size,
        mime_type=doc.mime_type,
        title=doc.title,
        description=doc.description,
        tags=doc.tags.split(",") if doc.tags else [],
        status=doc.status.value,
        access_level=doc.access_level,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        indexed_at=doc.indexed_at,
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # Comma-separated
    access_level: str = Form("private"),
    current_user: CurrentUser = Depends(can_upload_documents),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document for processing and indexing.
    
    - Accepts PDF, DOCX, TXT, MD files
    - Maximum size: 50MB (configurable)
    - Queues background processing for parsing and embedding
    """
    # Read file content
    content = await file.read()
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    
    doc_service = DocumentService(db)
    
    document = await doc_service.upload_document(
        file_content=content,
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        owner_id=current_user.id,
        tenant_id=current_user.tenant_id,
        access_level=access_level,
        title=title,
        description=description,
        tags=tag_list,
    )
    
    # Commit the document BEFORE starting background task
    await db.commit()
    
    # Queue background processing with its own session
    document_id = document.id
    doc_filename = document.original_filename
    
    print(f">>> QUEUING BACKGROUND TASK for {doc_filename}")

    async def process_document():
        print(f">>> BACKGROUND TASK STARTED for {doc_filename}")
        import asyncio
        import logging
        log = logging.getLogger(__name__)
        
        log.info(f"[BACKGROUND] Starting background task for document: {doc_filename}")
        
        # Small delay to ensure transaction is fully visible
        await asyncio.sleep(0.1)
        
        async with async_session_maker() as bg_session:
            try:
                print(f">>> CREATED DB SESSION for {doc_filename}")
                log.info(f"[BACKGROUND] Created DB session, starting ingestion...")
                ingestion = IngestionService(bg_session)
                await ingestion.process_document(document_id)
                await bg_session.commit()
                print(f">>> PROCESSING COMPLETE for {doc_filename}")
                log.info(f"[BACKGROUND] Committed successfully for: {doc_filename}")
            except Exception as e:
                print(f">>> ERROR: {e}")
                await bg_session.rollback()
                log.error(f"[BACKGROUND] Processing failed for {doc_filename}: {e}", exc_info=True)
    
    # Use asyncio.create_task instead of BackgroundTasks for reliable async execution
    asyncio.create_task(process_document())
    
    return DocumentUploadResponse(
        id=str(document.id),
        filename=document.original_filename,
        status=document.status.value,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List documents with pagination and access control.
    
    - Returns documents owned by user or accessible via tenant/public access
    - Supports filtering by status and search
    """
    doc_service = DocumentService(db)
    
    documents, total = await doc_service.list_documents(
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
        is_admin=current_user.is_admin,
        page=page,
        page_size=page_size,
        status=status,
        search=search,
    )
    
    return DocumentListResponse(
        documents=[document_to_response(d) for d in documents],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a document by ID.
    """
    doc_service = DocumentService(db)
    
    document = await doc_service.get_document_by_id(
        document_id=uuid.UUID(document_id),
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
        is_admin=current_user.is_admin,
    )
    
    if not document:
        raise NotFoundError("Document not found")
    
    return document_to_response(document)


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    document_id: str,
    current_user: CurrentUser = Depends(can_upload_documents),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger reindexing of a document (admin/owner only).
    """
    doc_service = DocumentService(db)
    
    document = await doc_service.reindex_document(
        document_id=uuid.UUID(document_id),
        user_id=current_user.id,
        is_admin=current_user.is_admin,
    )
    
    # Queue background reprocessing
    doc_id = document.id
    
    async def process_document():
        async with async_session_maker() as bg_session:
            try:
                ingestion = IngestionService(bg_session)
                await ingestion.process_document(doc_id)
                await bg_session.commit()
            except Exception as e:
                await bg_session.rollback()
                import logging
                logging.getLogger(__name__).error(f"Reindex failed: {e}")
    
    asyncio.create_task(process_document())
    
    return document_to_response(document)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a document and all its chunks.
    """
    doc_service = DocumentService(db)
    
    await doc_service.delete_document(
        document_id=uuid.UUID(document_id),
        user_id=current_user.id,
        is_admin=current_user.is_admin,
    )
    
    return {"message": "Document deleted successfully"}


@router.get("/chunks/{chunk_id}/preview")
async def get_chunk_preview(
    chunk_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a chunk preview with context for citation display.
    
    Returns:
        - text: The chunk text
        - metadata: Chunk metadata (page, section, etc.)
        - prev_chunk_id: ID of previous chunk (if exists)
        - next_chunk_id: ID of next chunk (if exists)
        - document: Parent document info
        - related_entities: Entities mentioned in chunk
    """
    from sqlmodel import select
    from app.models.document import Chunk, Document
    
    # Get the chunk
    result = await db.execute(
        select(Chunk).where(Chunk.id == uuid.UUID(chunk_id))
    )
    chunk = result.scalar_one_or_none()
    
    if not chunk:
        raise NotFoundError("Chunk", chunk_id)
    
    # Get parent document
    result = await db.execute(
        select(Document).where(Document.id == chunk.document_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise NotFoundError("Document", str(chunk.document_id))
    
    # Check access (owner or admin)
    if document.owner_id != current_user.id and not current_user.is_admin:
        raise NotFoundError("Chunk", chunk_id)  # Don't reveal existence
    
    # Get prev/next chunks
    prev_chunk_id = None
    next_chunk_id = None
    
    result = await db.execute(
        select(Chunk.id)
        .where(Chunk.document_id == chunk.document_id)
        .where(Chunk.chunk_index == chunk.chunk_index - 1)
    )
    prev = result.scalar_one_or_none()
    if prev:
        prev_chunk_id = str(prev)
    
    result = await db.execute(
        select(Chunk.id)
        .where(Chunk.document_id == chunk.document_id)
        .where(Chunk.chunk_index == chunk.chunk_index + 1)
    )
    next_c = result.scalar_one_or_none()
    if next_c:
        next_chunk_id = str(next_c)
    
    # Get related entities (if graph exists)
    related_entities = []
    try:
        from app.services.graph_service import get_graph_service
        graph_service = get_graph_service()
        # Simple entity extraction from chunk text
        entities = await graph_service.get_entities_in_text(chunk.text[:500])
        related_entities = [{"name": e.name, "type": e.entity_type} for e in entities[:10]]
    except Exception:
        pass  # Graph not available
    
    return {
        "id": str(chunk.id),
        "text": chunk.text,
        "metadata": chunk.metadata or {},
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number,
        "section_path": chunk.section_path,
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,
        "document": {
            "id": str(document.id),
            "filename": document.original_filename,
            "title": document.title,
        },
        "related_entities": related_entities,
    }
