"""
Document Ingestion Task Handlers for Phase 5.

Provides background processing for:
- Document parsing and chunking
- Embedding generation
- Vector indexing
- Graph extraction
"""
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import StaleDataError
from sqlalchemy.exc import PendingRollbackError

from app.config import settings
from app.database import async_session_maker
from app.models.document import Document, DocumentStatusEnum
from app.models.job import JobType, JobStatus, IngestionStep
from app.services.task_queue import task_handler, get_task_queue
from app.services.job_service import JobService

logger = logging.getLogger(__name__)


@task_handler("document_ingestion")
async def handle_document_ingestion(
    job_id: uuid.UUID,
    document_id: uuid.UUID,
    resume_from: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a document through the full ingestion pipeline.
    
    Steps:
    1. Parsing - Extract text from document
    2. Chunking - Split into chunks
    3. Hierarchy - Generate parent summaries (if enabled)
    4. Embedding - Generate vector embeddings
    5. Vector Index - Add to FAISS/vector store
    6. Graph Extract - Extract entities (if enabled)
    7. Graph Index - Build graph relationships
    8. Finalize - Update document status
    
    Each step saves a checkpoint for resumability.
    """
    from app.services.ingestion_service import IngestionService
    
    logger.info(f"Starting document ingestion: job={job_id}, document={document_id}")
    
    stats = {
        "chunks_created": 0,
        "embeddings_generated": 0,
        "entities_extracted": 0,
        "processing_time_ms": 0,
    }
    
    start_time = datetime.utcnow()
    
    async with async_session_maker() as db:
        job_service = JobService(db)
        ingestion_service = IngestionService(db)
        
        # Get document
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        # Check for resume point
        last_checkpoint = None
        if resume_from:
            last_checkpoint = await job_service.get_last_checkpoint(job_id)
            if last_checkpoint:
                logger.info(f"Resuming from checkpoint: {last_checkpoint.step_name}")
        
        # Determine starting step
        start_step_order = 0
        if last_checkpoint:
            start_step_order = last_checkpoint.step_order + 1
        
        try:
            # Step 1: Parsing
            if start_step_order <= IngestionStep.ORDER[IngestionStep.PARSING]:
                await job_service.update_progress(
                    job_id,
                    percent=IngestionStep.PROGRESS[IngestionStep.PARSING],
                    message=IngestionStep.MESSAGES[IngestionStep.PARSING],
                    current_step=IngestionStep.PARSING,
                    current_step_number=1,
                )
                
                # Parse document
                parsed_result = await ingestion_service.parse_document(document)
                
                await job_service.save_checkpoint(
                    job_id=job_id,
                    document_id=document_id,
                    step_name=IngestionStep.PARSING,
                    step_data={"text_length": len(parsed_result.get("text", ""))},
                )
            
            # Step 2: Chunking
            if start_step_order <= IngestionStep.ORDER[IngestionStep.CHUNKING]:
                await job_service.update_progress(
                    job_id,
                    percent=IngestionStep.PROGRESS[IngestionStep.CHUNKING],
                    message=IngestionStep.MESSAGES[IngestionStep.CHUNKING],
                    current_step=IngestionStep.CHUNKING,
                    current_step_number=2,
                )
                
                # Chunk document (this also saves chunks to DB)
                chunks = await ingestion_service.chunk_and_save(document)
                stats["chunks_created"] = len(chunks)
                
                await job_service.save_checkpoint(
                    job_id=job_id,
                    document_id=document_id,
                    step_name=IngestionStep.CHUNKING,
                    step_data={"chunks_count": len(chunks)},
                    items_processed=len(chunks),
                    items_total=len(chunks),
                )
            
            # Step 3: Hierarchy (optional)
            if (start_step_order <= IngestionStep.ORDER[IngestionStep.HIERARCHY] and 
                getattr(settings, 'ENABLE_HIERARCHICAL_CHUNKING', False)):
                await job_service.update_progress(
                    job_id,
                    percent=IngestionStep.PROGRESS[IngestionStep.HIERARCHY],
                    message=IngestionStep.MESSAGES[IngestionStep.HIERARCHY],
                    current_step=IngestionStep.HIERARCHY,
                    current_step_number=3,
                )
                
                # Generate parent summaries
                parent_count = await ingestion_service.generate_hierarchy(document)
                
                await job_service.save_checkpoint(
                    job_id=job_id,
                    document_id=document_id,
                    step_name=IngestionStep.HIERARCHY,
                    step_data={"parent_chunks": parent_count},
                )
            
            # Step 4: Embedding
            if start_step_order <= IngestionStep.ORDER[IngestionStep.EMBEDDING]:
                await job_service.update_progress(
                    job_id,
                    percent=IngestionStep.PROGRESS[IngestionStep.EMBEDDING],
                    message=IngestionStep.MESSAGES[IngestionStep.EMBEDDING],
                    current_step=IngestionStep.EMBEDDING,
                    current_step_number=4,
                )
                
                # Generate embeddings
                embeddings_count = await ingestion_service.generate_embeddings(document)
                stats["embeddings_generated"] = embeddings_count
                
                await job_service.save_checkpoint(
                    job_id=job_id,
                    document_id=document_id,
                    step_name=IngestionStep.EMBEDDING,
                    step_data={"embeddings_count": embeddings_count},
                )
            
            # Step 5: Vector Index
            if start_step_order <= IngestionStep.ORDER[IngestionStep.VECTOR_INDEX]:
                await job_service.update_progress(
                    job_id,
                    percent=IngestionStep.PROGRESS[IngestionStep.VECTOR_INDEX],
                    message=IngestionStep.MESSAGES[IngestionStep.VECTOR_INDEX],
                    current_step=IngestionStep.VECTOR_INDEX,
                    current_step_number=5,
                )
                
                # Add to vector store
                await ingestion_service.index_in_vector_store(document)
                
                # Update document status to INDEXED_LIGHT immediately after vector indexing
                # This makes the document searchable while GraphRAG continues in background
                document.status = DocumentStatusEnum.INDEXED_LIGHT
                document.indexed_at = datetime.utcnow()
                await db.commit()
                await db.refresh(document)
                logger.info(f"Document {document_id} status updated to INDEXED_LIGHT (Vector Ready)")
                
                await job_service.save_checkpoint(
                    job_id=job_id,
                    document_id=document_id,
                    step_name=IngestionStep.VECTOR_INDEX,
                )
            
            # Step 6: Graph Extract (optional)
            if (start_step_order <= IngestionStep.ORDER[IngestionStep.GRAPH_EXTRACT] and 
                getattr(settings, 'ENABLE_GRAPH_RAG', False)):
                await job_service.update_progress(
                    job_id,
                    percent=IngestionStep.PROGRESS[IngestionStep.GRAPH_EXTRACT],
                    message=IngestionStep.MESSAGES[IngestionStep.GRAPH_EXTRACT],
                    current_step=IngestionStep.GRAPH_EXTRACT,
                    current_step_number=6,
                )
                
                # Extract entities
                entities_count = await ingestion_service.extract_entities(document)
                stats["entities_extracted"] = entities_count
                
                await job_service.save_checkpoint(
                    job_id=job_id,
                    document_id=document_id,
                    step_name=IngestionStep.GRAPH_EXTRACT,
                    step_data={"entities_count": entities_count},
                )
            
            # Step 7: Graph Index (optional)
            if (start_step_order <= IngestionStep.ORDER[IngestionStep.GRAPH_INDEX] and 
                getattr(settings, 'ENABLE_GRAPH_RAG', False)):
                await job_service.update_progress(
                    job_id,
                    percent=IngestionStep.PROGRESS[IngestionStep.GRAPH_INDEX],
                    message=IngestionStep.MESSAGES[IngestionStep.GRAPH_INDEX],
                    current_step=IngestionStep.GRAPH_INDEX,
                    current_step_number=7,
                )
                
                # Build graph relationships
                await ingestion_service.index_in_graph(document)
                
                await job_service.save_checkpoint(
                    job_id=job_id,
                    document_id=document_id,
                    step_name=IngestionStep.GRAPH_INDEX,
                )
            
            # Step 8: Finalize
            await job_service.update_progress(
                job_id,
                percent=IngestionStep.PROGRESS[IngestionStep.FINALIZE],
                message=IngestionStep.MESSAGES[IngestionStep.FINALIZE],
                current_step=IngestionStep.FINALIZE,
                current_step_number=8,
            )
            
            # Final touch - status is already INDEXED_LIGHT, but we ensure persistence
            # If we had a "INDEXED_FULL" status, we would set it here.
            # For now, just ensuring everything is committed.
            await db.commit()
            
            await job_service.save_checkpoint(
                job_id=job_id,
                document_id=document_id,
                step_name=IngestionStep.FINALIZE,
            )
            
            logger.info(f"Document ingestion (Full) complete: {document_id}")
            
            # Calculate processing time
            end_time = datetime.utcnow()
            stats["processing_time_ms"] = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(f"Document ingestion complete: {document_id}, stats={stats}")
            
            return stats
            
        except (StaleDataError, PendingRollbackError) as e:
            # Document was deleted during ingestion - this is not really an error
            logger.warning(f"Document {document_id} was deleted during ingestion, aborting gracefully")
            await db.rollback()
            return {"status": "aborted", "reason": "document_deleted"}
            
        except Exception as e:
            logger.exception(f"Document ingestion failed: {document_id}")
            
            # Rollback first to clear any pending state
            await db.rollback()
            
            # Check if document still exists before updating status
            try:
                result = await db.execute(
                    select(Document).where(Document.id == document_id)
                )
                doc = result.scalar_one_or_none()
                
                if doc:
                    # Document still exists, update status to failed
                    doc.status = DocumentStatusEnum.FAILED
                    await db.commit()
                else:
                    logger.warning(f"Document {document_id} was deleted during ingestion, skipping status update")
                    return {"status": "aborted", "reason": "document_deleted"}
            except Exception as update_error:
                logger.warning(f"Could not update document status: {update_error}")
                await db.rollback()
            
            raise


@task_handler("graph_rebuild")
async def handle_graph_rebuild(
    job_id: uuid.UUID,
    force: bool = False,
    document_ids: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Rebuild the knowledge graph.
    """
    from app.services.graph_index_service import get_graph_index_service
    
    logger.info(f"Starting graph rebuild: job={job_id}, force={force}")
    
    async with async_session_maker() as db:
        job_service = JobService(db)
        graph_service = get_graph_index_service(db)
        
        await job_service.update_progress(
            job_id,
            percent=10,
            message="Loading graph data...",
        )
        
        # Perform rebuild
        result = await graph_service.force_reindex(
            document_ids=[uuid.UUID(d) for d in document_ids] if document_ids else None
        )
        
        await job_service.update_progress(
            job_id,
            percent=100,
            message="Graph rebuild complete",
        )
        
        return result


@task_handler("community_detection")
async def handle_community_detection(
    job_id: uuid.UUID,
    algorithm: str = "louvain",
    **kwargs
) -> Dict[str, Any]:
    """
    Run community detection on the knowledge graph.
    """
    from app.services.graph_index_service import get_graph_index_service
    
    logger.info(f"Starting community detection: job={job_id}, algorithm={algorithm}")
    
    async with async_session_maker() as db:
        job_service = JobService(db)
        graph_service = get_graph_index_service(db)
        
        await job_service.update_progress(
            job_id,
            percent=20,
            message="Detecting communities...",
        )
        
        # Run detection
        result = await graph_service.detect_communities(algorithm=algorithm)
        
        await job_service.update_progress(
            job_id,
            percent=100,
            message=f"Found {result.get('community_count', 0)} communities",
        )
        
        return result


@task_handler("ocr_processing")
async def handle_ocr_processing(
    job_id: uuid.UUID,
    document_id: uuid.UUID,
    **kwargs
) -> Dict[str, Any]:
    """
    Process OCR for a document using Vision LLM.
    
    Uses the Vision LLM API for text extraction from images/scanned documents.
    """
    logger.info(f"OCR processing via Vision LLM: job={job_id}, document={document_id}")
    
    async with async_session_maker() as db:
        job_service = JobService(db)
        
        await job_service.update_progress(
            job_id,
            percent=0,
            message="OCR processing via Vision LLM",
        )
    
    # Vision LLM is integrated into the document parser (docling_parser.py)
    # This task is a placeholder - actual OCR happens during document parsing
    return {"status": "completed", "message": "OCR handled by Vision LLM during parsing"}


# =============================================================================
# Helper Functions
# =============================================================================

async def enqueue_document_ingestion(
    document_id: uuid.UUID,
    user_id: uuid.UUID,
    tenant_id: Optional[str] = None,
    priority: int = 5,
) -> uuid.UUID:
    """
    Create a job and enqueue document for ingestion.
    
    Returns:
        Job ID
    """
    async with async_session_maker() as db:
        job_service = JobService(db)
        
        # Create job
        job = await job_service.create_job(
            job_type=JobType.DOCUMENT_INGESTION,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type="document",
            resource_id=document_id,
            priority=priority,
            total_steps=8,  # Number of ingestion steps
            estimated_duration_seconds=60,  # Default estimate
        )
        
        await db.commit()
        
        # Enqueue task
        task_queue = get_task_queue()
        await task_queue.enqueue(
            job_id=job.id,
            task_type="document_ingestion",
            priority=priority,
            document_id=document_id,
        )
        
        logger.info(f"Enqueued document ingestion: job={job.id}, document={document_id}")
        
        return job.id
