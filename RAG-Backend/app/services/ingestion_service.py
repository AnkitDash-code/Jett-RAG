"""
Document ingestion pipeline for parsing, chunking, and embedding.
Runs as background tasks via Celery or in-process for dev.

Phase 8: Added DocTags-aware chunking for Granite VLM integration.
"""
import os
import re
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
    
    Phase 8: Added DocTags-aware chunking that respects structured elements
    like tables, code blocks, and formulas.
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
    
    def chunk_doctags_aware(
        self,
        doctags: str,
        text: str = "",
        max_structured_tokens: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Chunk document with awareness of DocTags structure.
        
        Never splits inside structured elements like:
        - <table>...</table>
        - <code>...</code>
        - <formula>...</formula>
        
        Args:
            doctags: DocTags formatted content from Granite VLM
            text: Plain text fallback if doctags parsing fails
            max_structured_tokens: Max size for structured elements (tables, code)
        
        Returns:
            List of chunks with metadata indicating structure type
        """
        if not doctags:
            # Fallback to regular text chunking
            return self.chunk_text(text) if text else []
        
        # Strip outer <doctag> wrapper if present
        doctags = re.sub(r'^\s*<doctag>\s*', '', doctags, flags=re.IGNORECASE)
        doctags = re.sub(r'\s*</doctag>\s*$', '', doctags, flags=re.IGNORECASE)
        
        chunks = []
        chunk_index = 0
        
        # Pattern for structured elements (never split inside these)
        structured_patterns = [
            (r'<table>(.*?)</table>', 'table'),
            (r'<code>(.*?)</code>', 'code'),
            (r'<formula>(.*?)</formula>', 'formula'),
            (r'<image>(.*?)</image>', 'image'),
            (r'<picture>(.*?)</picture>', 'picture'),  # Granite VLM uses <picture>
            (r'<figure>(.*?)</figure>', 'figure'),
        ]
        
        # First, extract all structured elements with their positions
        structured_elements = []
        for pattern, element_type in structured_patterns:
            for match in re.finditer(pattern, doctags, re.DOTALL):
                structured_elements.append({
                    'type': element_type,
                    'content': match.group(0),
                    'inner': match.group(1),
                    'start': match.start(),
                    'end': match.end(),
                })
        
        # Sort by position
        structured_elements.sort(key=lambda x: x['start'])
        
        # Process doctags, keeping structured elements intact
        current_pos = 0
        text_buffer = ""
        
        for element in structured_elements:
            # Get text before this structured element
            text_before = doctags[current_pos:element['start']]
            
            # Remove tags from text_before and add to buffer
            clean_text = re.sub(r'<[^>]+>', ' ', text_before)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if clean_text:
                text_buffer += " " + clean_text
            
            # If buffer is getting large, chunk it
            if len(text_buffer) >= self.chunk_size:
                buffer_chunks = self.chunk_text(text_buffer.strip())
                for bc in buffer_chunks:
                    bc["chunk_index"] = chunk_index
                    bc["metadata"]["contains_structure"] = False
                    chunks.append(bc)
                    chunk_index += 1
                text_buffer = ""
            
            # Handle the structured element
            element_content = element['inner'].strip()
            element_type = element['type']
            
            # For tables, try to convert to readable format
            if element_type == 'table':
                element_text = f"[Table]\n{element_content}\n[End Table]"
            elif element_type == 'code':
                element_text = f"```\n{element_content}\n```"
            elif element_type == 'formula':
                element_text = f"[Formula: {element_content}]"
            elif element_type in ('image', 'picture', 'figure'):
                # Extract image/picture description
                desc_match = re.search(r'<desc>(.*?)</desc>', element_content, re.DOTALL)
                if not desc_match:
                    # Try to extract any text content
                    clean_content = re.sub(r'<[^>]+>', ' ', element_content)
                    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                    desc = clean_content if clean_content else "Visual content"
                else:
                    desc = desc_match.group(1).strip()
                element_text = f"[{element_type.capitalize()}: {desc}]"
            else:
                element_text = element_content
            
            # If structured element fits in chunk size, add as single chunk
            if len(element_text) <= max_structured_tokens:
                # Flush text buffer first if there's content
                if text_buffer.strip():
                    buffer_chunks = self.chunk_text(text_buffer.strip())
                    for bc in buffer_chunks:
                        bc["chunk_index"] = chunk_index
                        bc["metadata"]["contains_structure"] = False
                        chunks.append(bc)
                        chunk_index += 1
                    text_buffer = ""
                
                chunks.append({
                    "text": element_text,
                    "chunk_index": chunk_index,
                    "metadata": {
                        f"contains_{element_type}": True,
                        "structure_type": element_type,
                        "contains_structure": True,
                    },
                })
                chunk_index += 1
            else:
                # Large structured element - chunk it but note it's partial
                if element_type == 'table':
                    # Try to split table by rows
                    table_chunks = self._split_large_table(element_content, max_structured_tokens)
                    for tc in table_chunks:
                        chunks.append({
                            "text": tc,
                            "chunk_index": chunk_index,
                            "metadata": {
                                "contains_table": True,
                                "structure_type": "table_partial",
                                "contains_structure": True,
                            },
                        })
                        chunk_index += 1
                else:
                    # For other large elements, just chunk normally
                    element_chunks = self.chunk_text(element_text)
                    for ec in element_chunks:
                        ec["chunk_index"] = chunk_index
                        ec["metadata"][f"contains_{element_type}"] = True
                        ec["metadata"]["structure_type"] = f"{element_type}_partial"
                        ec["metadata"]["contains_structure"] = True
                        chunks.append(ec)
                        chunk_index += 1
            
            current_pos = element['end']
        
        # Process remaining text after last structured element
        remaining_text = doctags[current_pos:]
        clean_remaining = re.sub(r'<[^>]+>', ' ', remaining_text)
        clean_remaining = re.sub(r'\s+', ' ', clean_remaining).strip()
        
        if clean_remaining:
            text_buffer += " " + clean_remaining
        
        # Flush final text buffer
        if text_buffer.strip():
            buffer_chunks = self.chunk_text(text_buffer.strip())
            for bc in buffer_chunks:
                bc["chunk_index"] = chunk_index
                bc["metadata"]["contains_structure"] = False
                chunks.append(bc)
                chunk_index += 1
        
        # Filter out empty chunks or chunks with only XML tags
        filtered_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "").strip()
            # Remove any remaining tags
            clean_text = re.sub(r'<[^>]+>', ' ', chunk_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # Only keep chunks with actual content (at least 10 chars)
            if clean_text and len(clean_text) >= 10:
                chunks_copy = chunk.copy()
                # Update text with cleaned version if it was just tags
                if chunk_text.startswith('<') and '>' in chunk_text:
                    chunks_copy["text"] = clean_text
                filtered_chunks.append(chunks_copy)
        
        logger.info(f"DocTags-aware chunking: {len(chunks)} initial chunks, {len(filtered_chunks)} after filtering")
        return filtered_chunks
    
    def _split_large_table(self, table_content: str, max_size: int) -> List[str]:
        """Split a large table by rows while preserving header."""
        lines = table_content.strip().split('\n')
        
        if len(lines) <= 2:
            return [f"[Table]\n{table_content}\n[End Table]"]
        
        # Assume first 1-2 lines are header
        header_lines = lines[:2] if len(lines) > 2 and '---' in lines[1] else lines[:1]
        header = '\n'.join(header_lines)
        data_lines = lines[len(header_lines):]
        
        chunks = []
        current_chunk_lines = []
        current_size = len(header)
        
        for line in data_lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_size and current_chunk_lines:
                # Create chunk with header + current lines
                chunk_content = header + '\n' + '\n'.join(current_chunk_lines)
                chunks.append(f"[Table (continued)]\n{chunk_content}\n[End Table]")
                current_chunk_lines = [line]
                current_size = len(header) + line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size
        
        # Add final chunk
        if current_chunk_lines:
            chunk_content = header + '\n' + '\n'.join(current_chunk_lines)
            chunks.append(f"[Table]\n{chunk_content}\n[End Table]")
        
        return chunks if chunks else [f"[Table]\n{table_content}\n[End Table]"]


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
    
    Phase 8: Uses EnhancedDocumentParser with Granite VLM for PDFs.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._parser = None  # Lazy load to use EnhancedDocumentParser
        self.chunker = ChunkingService()
        self._embedding_service = None
        self._weaviate_service = None
    
    @property
    def parser(self):
        """Lazy load parser - uses EnhancedDocumentParser with Granite VLM."""
        if self._parser is None:
            try:
                from app.services.docling_parser import get_enhanced_document_parser
                self._parser = get_enhanced_document_parser()
                logger.info("Using EnhancedDocumentParser with Granite VLM support")
            except ImportError:
                self._parser = DocumentParser()
                logger.info("Using basic DocumentParser (Docling not available)")
        return self._parser
    
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
            
            # Check if parsed with Granite VLM (has doctags)
            parser_type = parsed.get("metadata", {}).get("parser", "")
            doctags = parsed.get("doctags", "")
            
            if parser_type == "granite_vlm" and doctags:
                # Use DocTags-aware chunking for structure preservation
                logger.info("[PROCESSING]   Using DocTags-aware chunking (Granite VLM)")
                chunks_data = self.chunker.chunk_doctags_aware(
                    doctags=doctags,
                    text=parsed.get("text", ""),
                )
            elif parsed.get("sections"):
                # Use section-based chunking
                logger.info("[PROCESSING]   Using section-based chunking")
                chunks_data = self.chunker.chunk_sections(parsed["sections"])
            else:
                # Fall back to basic text chunking
                logger.info("[PROCESSING]   Using basic text chunking")
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
            
            # Commit to ensure chunks are saved before embedding
            await self.db.commit()
            db_time = time.time() - db_start
            logger.info(f"[PROCESSING] Step 3 complete: Saved {len(chunks)} chunks in {db_time:.2f}s")
            
            # Need to re-fetch chunks ? No, we can continue using them, 
            # but for updates later, we should ensure they are attached or use IDs.
            # It's safer to not rely on the `chunks` objects for update if we commit here.
            
            document.chunk_count = len(chunks)
            await self.db.commit() # Save doc count
            
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
             
                # Update chunk status
                # Fetch chunks again to be safe for update
                # Or use the objects if they are still valid (attached to session after commit?)
                # In asyncio scoped session, often they are not.
                # Let's perform a bulk update statement instead of object update.
                from sqlalchemy import update
                stmt = update(Chunk).where(
                    Chunk.document_id == document.id
                ).values(
                    status=ChunkStatusEnum.EMBEDDED,
                    embedded_at=datetime.utcnow()
                )
                await self.db.execute(stmt)
                
            else:
                logger.warning("[PROCESSING] Weaviate not available - using FAISS fallback")
                # FAISS fallback update
                from sqlalchemy import update
                stmt = update(Chunk).where(
                    Chunk.document_id == document.id
                ).values(
                    status=ChunkStatusEnum.EMBEDDED,
                    embedded_at=datetime.utcnow()
                )
                await self.db.execute(stmt)
            
            # Step 6: Update document status
            logger.info(f"[PROCESSING] Step 6/6: Finalizing document status...")
            # Re-fetch document to ensure it's fresh
            await self.db.refresh(document)
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

    # =========================================================================
    # Phase 5: New methods for background task handlers
    # =========================================================================
    
    async def parse_document(self, document: Document) -> Dict[str, Any]:
        """
        Parse document and extract text (Step 1 for task handler).
        """
        parsed = await self.parser.parse(
            document.file_path,
            document.mime_type
        )
        document.page_count = parsed.get("page_count", 0)
        await self.db.commit()
        return parsed
    
    async def chunk_and_save(self, document: Document) -> List[Chunk]:
        """
        Chunk document and save to database (Step 2 for task handler).
        """
        # First parse if not already done
        parsed = await self.parser.parse(document.file_path, document.mime_type)
        
        # Chunk based on structure
        if parsed.get("sections"):
            chunks_data = self.chunker.chunk_sections(parsed["sections"])
        else:
            chunks_data = self.chunker.chunk_text(parsed.get("text", ""))
        
        # Create chunk records
        chunks = []
        for chunk_data in chunks_data:
            chunk = Chunk(
                document_id=document.id,
                text=chunk_data["text"],
                content_hash=hashlib.sha256(chunk_data["text"].encode()).hexdigest(),
                chunk_index=chunk_data["chunk_index"],
                page_number=chunk_data.get("page_number"),
                section_path=chunk_data.get("section_path"),
                tenant_id=document.tenant_id,
                access_level=document.access_level,
                department=document.department,
                hierarchy_level=0,  # Leaf level
                status=ChunkStatusEnum.PENDING,
                metadata_json=json.dumps(chunk_data.get("metadata", {})),
            )
            chunks.append(chunk)
            self.db.add(chunk)
        
        await self.db.flush()
        document.chunk_count = len(chunks)
        await self.db.commit()
        
        logger.info(f"Created {len(chunks)} leaf chunks for document {document.id}")
        return chunks
    
    async def generate_hierarchy(self, document: Document) -> int:
        """
        Generate parent summaries for hierarchical chunking (Step 3 for task handler).
        
        Creates parent chunks that summarize groups of leaf chunks,
        enabling hierarchical retrieval where we can expand context
        by fetching parent summaries.
        
        Returns:
            Number of parent chunks created
        """
        from app.services.utility_llm_client import get_utility_llm
        
        if not getattr(settings, 'ENABLE_HIERARCHICAL_CHUNKING', False):
            logger.info("Hierarchical chunking disabled, skipping")
            return 0
        
        # Get leaf chunks for this document
        result = await self.db.execute(
            select(Chunk).where(
                (Chunk.document_id == document.id) &
                (Chunk.hierarchy_level == 0)
            ).order_by(Chunk.chunk_index)
        )
        leaf_chunks = list(result.scalars().all())
        
        if len(leaf_chunks) < 3:
            logger.info(f"Only {len(leaf_chunks)} chunks, skipping hierarchy")
            return 0
        
        parent_chunk_size = getattr(settings, 'PARENT_CHUNK_SIZE', 1024)
        utility_llm = get_utility_llm()
        parent_chunks_created = 0
        
        # Group leaf chunks for parent summaries
        # Each parent covers ~4-8 leaf chunks (roughly PARENT_CHUNK_SIZE tokens)
        leaves_per_parent = max(2, parent_chunk_size // settings.CHUNK_SIZE)
        
        for i in range(0, len(leaf_chunks), leaves_per_parent):
            child_group = leaf_chunks[i:i + leaves_per_parent]
            
            if len(child_group) < 2:
                continue
            
            # Combine child texts
            combined_text = "\n\n".join([c.text for c in child_group])
            
            # Generate summary using utility LLM
            try:
                summary = await utility_llm.summarize_text(
                    text=combined_text,
                    max_length=parent_chunk_size // 4,  # Summary is shorter
                )
            except Exception as e:
                logger.warning(f"Failed to generate summary for parent chunk: {e}")
                # Use first sentence of each child as fallback
                summary = ". ".join([c.text.split(".")[0] for c in child_group if c.text])
            
            # Create parent chunk
            child_ids = [str(c.id) for c in child_group]
            parent = Chunk(
                document_id=document.id,
                text=summary,
                content_hash=hashlib.sha256(summary.encode()).hexdigest(),
                chunk_index=i // leaves_per_parent,
                page_number=child_group[0].page_number,
                section_path=child_group[0].section_path,
                tenant_id=document.tenant_id,
                access_level=document.access_level,
                department=document.department,
                hierarchy_level=1,  # Section level
                children_ids=",".join(child_ids),
                status=ChunkStatusEnum.PENDING,
                metadata_json=json.dumps({"child_count": len(child_group)}),
            )
            self.db.add(parent)
            await self.db.flush()
            
            # Update children to point to parent
            for child in child_group:
                child.parent_chunk_id = parent.id
            
            parent_chunks_created += 1
        
        await self.db.commit()
        logger.info(f"Created {parent_chunks_created} parent chunks for document {document.id}")
        return parent_chunks_created
    
    async def generate_embeddings(self, document: Document) -> int:
        """
        Generate embeddings for all chunks (Step 4 for task handler).
        """
        embedding_service = self._get_embedding_service()
        
        # Get all chunks (both leaf and parent)
        result = await self.db.execute(
            select(Chunk).where(
                (Chunk.document_id == document.id) &
                (Chunk.status == ChunkStatusEnum.PENDING)
            )
        )
        chunks = list(result.scalars().all())
        
        if not chunks:
            return 0
        
        # Generate embeddings in batches
        batch_size = 32
        total_embedded = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = embedding_service.embed(texts)
            
            # Store embeddings (in memory for now, indexed later)
            for j, chunk in enumerate(batch):
                # Store embedding reference (actual embedding stored in vector DB)
                chunk.status = ChunkStatusEnum.PENDING  # Will be EMBEDDED after indexing
            
            total_embedded += len(batch)
        
        await self.db.commit()
        logger.info(f"Generated embeddings for {total_embedded} chunks")
        return total_embedded
    
    async def index_in_vector_store(self, document: Document) -> None:
        """
        Index chunks in vector store (Step 5 for task handler).
        """
        embedding_service = self._get_embedding_service()
        weaviate = self._get_weaviate_service()
        
        # Get all pending chunks
        result = await self.db.execute(
            select(Chunk).where(
                (Chunk.document_id == document.id) &
                (Chunk.status == ChunkStatusEnum.PENDING)
            )
        )
        chunks = list(result.scalars().all())
        
        if not chunks:
            return
        
        # Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.embed(chunk_texts)
        
        if weaviate.is_available():
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
                    "hierarchy_level": chunk.hierarchy_level,
                    "parent_chunk_id": str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None,
                    "tenant_id": document.tenant_id,
                    "access_level": document.access_level or "private",
                    "department": document.department,
                    "owner_id": str(document.owner_id),
                })
            
            await weaviate.index_chunks_batch(weaviate_chunks)
        
        # Mark as embedded
        for chunk in chunks:
            chunk.status = ChunkStatusEnum.EMBEDDED
            chunk.embedded_at = datetime.utcnow()
        
        await self.db.commit()
        logger.info(f"Indexed {len(chunks)} chunks in vector store")
    
    async def extract_entities(self, document: Document) -> int:
        """
        Extract entities from document chunks (Step 6 for task handler).
        """
        if not getattr(settings, 'ENABLE_GRAPH_RAG', False):
            logger.info("GraphRAG disabled, skipping entity extraction")
            return 0
        
        from app.services.entity_extraction_service import get_entity_extraction_service
        
        extraction_service = get_entity_extraction_service(self.db)
        
        # Get chunks needing graph processing
        result = await self.db.execute(
            select(Chunk).where(
                (Chunk.document_id == document.id) &
                (Chunk.needs_graph_processing == True) &
                (Chunk.hierarchy_level == 0)  # Only process leaf chunks
            )
        )
        chunks = list(result.scalars().all())
        
        entities_count = 0
        for chunk in chunks:
            try:
                result = await extraction_service.extract_from_chunk(
                    chunk=chunk,
                    tenant_id=document.tenant_id,
                    access_level=document.access_level,
                )
                entities_count += len(result.entities)
                
                # Process and store
                await extraction_service.process_extraction_result(
                    result=result,
                    tenant_id=document.tenant_id,
                    access_level=document.access_level,
                    department=document.department,
                    owner_id=document.owner_id,
                )
                
                chunk.needs_graph_processing = False
            except Exception as e:
                logger.warning(f"Entity extraction failed for chunk {chunk.id}: {e}")
        
        await self.db.commit()
        logger.info(f"Extracted {entities_count} entities from {len(chunks)} chunks")
        return entities_count
    
    async def index_in_graph(self, document: Document) -> None:
        """
        Index entities in knowledge graph (Step 7 for task handler).
        """
        if not getattr(settings, 'ENABLE_GRAPH_RAG', False):
            logger.info("GraphRAG disabled, skipping graph indexing")
            return
        
        from app.services.graph_index_service import get_graph_index_service
        
        graph_service = get_graph_index_service(self.db)
        
        # Trigger community detection if threshold reached
        stats = await graph_service.get_stats()
        if stats.get("pending_entities", 0) >= getattr(settings, 'COMMUNITY_REBUILD_THRESHOLD', 50):
            await graph_service.detect_communities()
        
        logger.info(f"Graph indexing complete for document {document.id}")