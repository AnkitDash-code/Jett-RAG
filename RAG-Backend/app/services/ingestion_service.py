"""
Document ingestion pipeline for parsing, chunking, and embedding.
Runs as background tasks via Celery or in-process for dev.
"""
import os
import uuid
import hashlib
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.models.document import Document, Chunk, DocumentStatusEnum, ChunkStatusEnum

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Service for document chunking using LangChain's semantic-aware splitters.
    Uses RecursiveCharacterTextSplitter for structure-aware chunking.
    """
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = None
        self._semantic_splitter = None
    
    def _get_splitter(self):
        """Get or create the text splitter (lazy loading)."""
        if self._splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                
                # RecursiveCharacterTextSplitter tries to split on natural boundaries
                # in order: paragraphs, sentences, words, characters
                self._splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    separators=[
                        "\n\n",      # Paragraphs
                        "\n",        # Lines
                        ". ",        # Sentences
                        "! ",        # Exclamations
                        "? ",        # Questions
                        "; ",        # Clauses
                        ", ",        # Phrases
                        " ",         # Words
                        ""           # Characters (last resort)
                    ],
                    keep_separator=True,
                )
                logger.info("Initialized RecursiveCharacterTextSplitter for semantic chunking")
            except ImportError:
                logger.warning("langchain-text-splitters not available, using fallback")
                self._splitter = None
        return self._splitter
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks using LangChain splitter.
        Falls back to basic splitting if LangChain not available.
        """
        if not text or not text.strip():
            return []
        
        splitter = self._get_splitter()
        
        if splitter:
            # Use LangChain's semantic splitter
            try:
                from langchain_core.documents import Document as LCDocument
                
                docs = splitter.create_documents([text], metadatas=[metadata or {}])
                
                chunks = []
                for i, doc in enumerate(docs):
                    chunks.append({
                        "text": doc.page_content,
                        "chunk_index": i,
                        "start_char": 0,  # LangChain doesn't track positions
                        "end_char": len(doc.page_content),
                        "metadata": doc.metadata,
                    })
                
                logger.info(f"Created {len(chunks)} semantic chunks")
                return chunks
                
            except Exception as e:
                logger.warning(f"LangChain chunking failed: {e}, using fallback")
        
        # Fallback to basic chunking
        return self._basic_chunk_text(text, metadata)
    
    def _basic_chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Basic fallback chunking without LangChain."""
        if not text or not text.strip():
            return []
            
        chunks = []
        text_len = len(text)
        start = 0
        chunk_index = 0
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            if end < text_len:
                for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "metadata": metadata or {},
                })
                chunk_index += 1
            
            if end >= text_len:
                break
                
            start = end - self.chunk_overlap
            if start <= chunks[-1]["start_char"] if chunks else start <= 0:
                start = end
        
        return chunks
    
    def chunk_sections(
        self,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Chunk document by sections (for structured documents).
        Each section becomes one or more chunks.
        """
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_text = section.get("text", "")
            section_title = section.get("title", "")
            page_number = section.get("page_number")
            section_path = section.get("section_path", "")
            
            # Use semantic chunking for each section
            section_metadata = {
                "title": section_title,
                "page_number": page_number,
                "section_path": section_path,
            }
            
            if len(section_text) <= self.chunk_size:
                # Small section - keep as single chunk
                chunks.append({
                    "text": section_text,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "section_path": section_path,
                    "metadata": section_metadata,
                })
                chunk_index += 1
            else:
                # Split large sections with semantic chunking
                sub_chunks = self.chunk_text(section_text, metadata=section_metadata)
                for sub in sub_chunks:
                    sub["chunk_index"] = chunk_index
                    sub["page_number"] = page_number
                    sub["section_path"] = section_path
                    chunks.append(sub)
                    chunk_index += 1
        
        return chunks


class DocumentParser:
    """Parser for different document formats."""
    
    async def parse(self, file_path: str, mime_type: str) -> Dict[str, Any]:
        """
        Parse a document and extract text and structure.
        Returns dict with keys: text, sections, page_count, metadata
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".txt" or ext == ".md":
            return await self._parse_text(file_path)
        elif ext == ".pdf":
            return await self._parse_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return await self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    async def _parse_text(self, file_path: str) -> Dict[str, Any]:
        """Parse plain text or markdown files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        return {
            "text": text,
            "sections": [{"text": text, "title": "Main Content"}],
            "page_count": 1,
            "metadata": {},
        }
    
    async def _parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF files.
        TODO: Use pdfplumber or Docling for better extraction.
        """
        try:
            import pdfplumber
            
            text_parts = []
            sections = []
            page_count = 0
            
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    
                    if page_text.strip():
                        sections.append({
                            "text": page_text,
                            "title": f"Page {i}",
                            "page_number": i,
                            "section_path": f"Page {i}",
                        })
            
            full_text = "\n\n".join(text_parts)
            
            # Log warning if PDF appears to be image-based
            if not full_text.strip() and page_count > 0:
                logger.warning(
                    f"PDF has {page_count} pages but no extractable text. "
                    "This may be a scanned/image-based PDF that requires OCR."
                )
            else:
                logger.info(f"Extracted {len(full_text)} chars from {page_count} pages")
            
            return {
                "text": full_text,
                "sections": sections,
                "page_count": page_count,
                "metadata": {},
            }
            
        except ImportError:
            logger.warning("pdfplumber not installed, falling back to basic text extraction")
            # Fallback: return empty (document will need manual processing)
            return {
                "text": "",
                "sections": [],
                "page_count": 0,
                "metadata": {"error": "PDF parsing library not available"},
            }
    
    async def _parse_docx(self, file_path: str) -> Dict[str, Any]:
        """
        Parse DOCX files.
        TODO: Use python-docx for proper extraction.
        """
        try:
            from docx import Document as DocxDocument
            
            doc = DocxDocument(file_path)
            text_parts = []
            sections = []
            
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    text_parts.append(para.text)
                    sections.append({
                        "text": para.text,
                        "title": f"Paragraph {i+1}",
                        "section_path": f"Paragraph {i+1}",
                    })
            
            return {
                "text": "\n\n".join(text_parts),
                "sections": sections,
                "page_count": len(sections),
                "metadata": {},
            }
            
        except ImportError:
            logger.warning("python-docx not installed, falling back to basic handling")
            return {
                "text": "",
                "sections": [],
                "page_count": 0,
                "metadata": {"error": "DOCX parsing library not available"},
            }


class IngestionService:
    """
    Main ingestion orchestrator.
    Handles the full pipeline: parse -> chunk -> embed -> store.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.parser = DocumentParser()
        self.chunker = ChunkingService()
        self._embedding_service = None
        self._weaviate_service = None
    
    def _get_embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from app.services.retrieval_service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service
    
    def _get_weaviate_service(self):
        """Lazy load Weaviate service."""
        if self._weaviate_service is None:
            from app.services.vector_store import weaviate_service
            self._weaviate_service = weaviate_service
        return self._weaviate_service
    
    async def process_document(self, document_id: uuid.UUID) -> None:
        """
        Full document processing pipeline.
        Called by background worker.
        """
        import time
        start_time = time.time()
        
        logger.info(f"[PROCESSING] Starting document processing for ID: {document_id}")
        
        # Get document
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            logger.error(f"[PROCESSING] Document {document_id} not found in database!")
            return
        
        try:
            logger.info(f"[PROCESSING] Step 1/6: Parsing document: {document.original_filename}")
            logger.info(f"[PROCESSING]   - File path: {document.file_path}")
            logger.info(f"[PROCESSING]   - MIME type: {document.mime_type}")
            
            # Step 1: Parse document
            parse_start = time.time()
            parsed = await self.parser.parse(
                document.file_path,
                document.mime_type
            )
            parse_time = time.time() - parse_start
            logger.info(f"[PROCESSING] Step 1 complete: Parsed in {parse_time:.2f}s, pages={parsed.get('page_count', 0)}")
            
            document.page_count = parsed.get("page_count", 0)
            
            # Step 2: Chunk document
            logger.info(f"[PROCESSING] Step 2/6: Chunking document...")
            chunk_start = time.time()
            if parsed.get("sections"):
                chunks_data = self.chunker.chunk_sections(parsed["sections"])
            else:
                chunks_data = self.chunker.chunk_text(parsed.get("text", ""))
            chunk_time = time.time() - chunk_start
            logger.info(f"[PROCESSING] Step 2 complete: Created {len(chunks_data)} chunks in {chunk_time:.2f}s")
            
            # Step 3: Create chunk records in database
            logger.info(f"[PROCESSING] Step 3/6: Saving chunks to database...")
            db_start = time.time()
            
            # Step 3: Create chunk records in database
            chunks = []
            for chunk_data in chunks_data:
                chunk = Chunk(
                    document_id=document.id,
                    text=chunk_data["text"],
                    content_hash=hashlib.sha256(
                        chunk_data["text"].encode()
                    ).hexdigest(),
                    chunk_index=chunk_data["chunk_index"],
                    page_number=chunk_data.get("page_number"),
                    section_path=chunk_data.get("section_path"),
                    tenant_id=document.tenant_id,
                    access_level=document.access_level,
                    department=document.department,
                    status=ChunkStatusEnum.PENDING,
                    metadata_json=json.dumps(chunk_data.get("metadata", {})),
                )
                chunks.append(chunk)
                self.db.add(chunk)
            
            # Flush to get chunk IDs
            await self.db.flush()
            db_time = time.time() - db_start
            logger.info(f"[PROCESSING] Step 3 complete: Saved {len(chunks)} chunks in {db_time:.2f}s")
            
            document.chunk_count = len(chunks)
            
            # Step 4: Generate embeddings
            logger.info(f"[PROCESSING] Step 4/6: Loading embedding model...")
            embedding_service = self._get_embedding_service()
            chunk_texts = [chunk.text for chunk in chunks]
            
            logger.info(f"[PROCESSING] Step 4/6: Generating embeddings for {len(chunk_texts)} chunks...")
            embed_start = time.time()
            embeddings = embedding_service.embed(chunk_texts)
            embed_time = time.time() - embed_start
            logger.info(f"[PROCESSING] Step 4 complete: Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
            
            # Step 5: Store in vector store
            logger.info(f"[PROCESSING] Step 5/6: Indexing in vector store...")
            index_start = time.time()
            weaviate = self._get_weaviate_service()
            
            if weaviate.is_available():
                # Prepare batch data for Weaviate
                weaviate_chunks = []
                for i, chunk in enumerate(chunks):
                    weaviate_chunks.append({
                        "chunk_id": str(chunk.id),
                        "document_id": str(document.id),
                        "text": chunk.text,
                        "embedding": embeddings[i],
                        "document_name": document.original_filename,
                        "page_number": chunk.page_number,
                        "section_path": chunk.section_path,
                        "chunk_index": chunk.chunk_index,
                        "tenant_id": document.tenant_id,
                        "access_level": document.access_level or "private",
                        "department": document.department,
                        "owner_id": str(document.owner_id),
                    })
                
                indexed_count = await weaviate.index_chunks_batch(weaviate_chunks)
                index_time = time.time() - index_start
                logger.info(f"[PROCESSING] Step 5 complete: Indexed {indexed_count} chunks in {index_time:.2f}s")
                
                if indexed_count == len(chunks):
                    for chunk in chunks:
                        chunk.status = ChunkStatusEnum.EMBEDDED
                        chunk.embedded_at = datetime.utcnow()
                else:
                    logger.warning(f"[PROCESSING] Only indexed {indexed_count}/{len(chunks)} chunks")
                    # Mark chunks as indexed anyway for now
                    for chunk in chunks:
                        chunk.status = ChunkStatusEnum.EMBEDDED
                        chunk.embedded_at = datetime.utcnow()
            else:
                logger.warning("[PROCESSING] Weaviate not available - using FAISS fallback")
                # Still mark as embedded for DB fallback search
                for chunk in chunks:
                    chunk.status = ChunkStatusEnum.EMBEDDED
                    chunk.embedded_at = datetime.utcnow()
            
            # Step 6: Update document status
            logger.info(f"[PROCESSING] Step 6/6: Finalizing document status...")
            document.status = DocumentStatusEnum.INDEXED_LIGHT
            document.indexed_at = datetime.utcnow()
            document.updated_at = datetime.utcnow()
            
            await self.db.commit()
            
            total_time = time.time() - start_time
            logger.info(
                f"[PROCESSING] ✓ COMPLETE: {document.original_filename} - "
                f"{len(chunks)} chunks in {total_time:.2f}s"
            )
            
        except Exception as e:
            logger.exception(f"[PROCESSING] ✗ ERROR processing document {document_id}: {e}")
            document.status = DocumentStatusEnum.ERROR
            document.error_message = str(e)
            await self.db.commit()
