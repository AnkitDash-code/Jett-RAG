"""
Docling Parser Integration for Phase 8.

Uses Docling for advanced document parsing with:
- Table extraction
- Image extraction with VLM-generated descriptions (via Granite Docling 258M)
- Document structure recognition with DocTags
- Enhanced PDF handling

Pipeline:
1. Try Granite VLM (remote) for PDFs - provides DocTags, image descriptions
2. Fallback to local Docling library
3. Fallback raises error

Docling: https://github.com/docling-project/docling
Granite VLM: ibm/granite-docling via Ollama API
"""
import asyncio
import logging
import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config import settings
from app.services.vision_llm_client import get_vision_llm_client

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Advanced document parser using Docling library + optional Granite VLM.
    
    Provides enhanced parsing for:
    - Complex PDF documents with tables/images
    - Multi-column layouts
    - Document structure (headings, sections)
    - Embedded images with VLM-generated descriptions
    
    Falls back to basic pdfplumber if Docling/VLM is unavailable.
    """
    
    def __init__(self):
        self.enabled = True
        self.timeout = 120
        self._vision_client = None
    
    def _check_docling(self) -> bool:
        """Check if Docling is installed and available."""
        if self._docling_available is not None:
            return self._docling_available
        
        try:
            from docling.document_converter import DocumentConverter
            self._docling_available = True
            logger.info("Docling parser available")
        except ImportError:
            self._docling_available = False
            logger.warning("Docling not installed. Install with: pip install docling")
        
        return self._docling_available
    
    def _get_granite_client(self):
        """Get Granite VLM client (lazy loading)."""
        if self._granite_client is None and self.vlm_enabled:
            try:
                from app.services.granite_client import get_granite_client
                self._granite_client = get_granite_client()
                logger.info("Granite VLM client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Granite VLM client: {e}")
                self._granite_client = None
        return self._granite_client
    
    def _get_vision_client(self):
        """Get Vision LLM client (lazy loading)."""
        if self._vision_client is None:
            try:
                self._vision_client = get_vision_llm_client()
                logger.info("Vision LLM client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Vision LLM client: {e}")
                self._vision_client = None
        return self._vision_client
    
    def _needs_vision_fallback(self, extracted_text: str, page_count: int) -> bool:
        """Check if document needs Vision LLM fallback (insufficient text)."""
        if not extracted_text:
            return True
        
        text_per_page = len(extracted_text) / max(page_count, 1)
        return text_per_page < self._min_text_threshold
    
    def is_available(self) -> bool:
        """Always available (Vision LLM only)."""
        return self.enabled
    
    async def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse document using Vision LLM only.
        Returns:
            Dict with keys:
            - text: Full extracted text
            - sections: List of sections (if available)
            - images: List of image metadata (if available)
            - page_count: Number of pages
            - metadata: Document metadata
        """
        ext = os.path.splitext(file_path)[1].lower()
        vision_client = self._get_vision_client()
        if not vision_client or not vision_client.is_available():
            raise RuntimeError("Vision LLM client not available")
        if ext == ".pdf":
            result = await vision_client.extract_from_pdf_pages(file_path)
        else:
            result = await vision_client.extract_from_file(file_path)
        # Ensure standard keys
        return {
            "text": result.get("text", ""),
            "sections": result.get("sections", []),
            "images": result.get("images", []),
            "page_count": result.get("page_count", 1),
            "metadata": result.get("metadata", {}),
        }
    
    async def _parse_with_granite_vlm(self, file_path: str) -> Dict[str, Any]:
        """Parse document using remote Granite VLM via /v1/convert."""
        from app.services.circuit_breaker import CircuitOpenError
        
        granite_client = self._get_granite_client()
        
        if granite_client is None:
            raise RuntimeError("Granite VLM client not available")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
        max_size = getattr(settings, 'GRANITE_MAX_FILE_SIZE_MB', 100)
        
        if file_size_mb > max_size:
            raise ValueError(
                f"File ({file_size_mb:.1f}MB) exceeds Granite max size ({max_size}MB)"
            )
        
        try:
            # Call Granite VLM
            vlm_result = await granite_client.convert_document(
                file_path,
                extract_images=True,
                extract_tables=True,
            )
            
            logger.info(
                f"Granite VLM result: {len(vlm_result.get('tables', []))} tables, "
                f"{len(vlm_result.get('images', []))} images, "
                f"{vlm_result.get('page_count', 0)} pages"
            )
            
            # Convert to standard format with sections
            sections = self._convert_doctags_to_sections(
                vlm_result.get("doctags", ""),
                vlm_result.get("text", ""),
            )
            
            result = {
                "text": vlm_result.get("text", ""),
                "sections": sections,
                "tables": vlm_result.get("tables", []),
                "images": vlm_result.get("images", []),
                "page_count": vlm_result.get("page_count", 1),
                "metadata": vlm_result.get("metadata", {}),
                "doctags": vlm_result.get("doctags", ""),  # Keep for DocTags-aware chunking
                "markdown": vlm_result.get("markdown", ""),
            }
            
            # Check if we need Vision LLM fallback (empty or insufficient text)
            extracted_text = result["text"].strip()
            page_count = result["page_count"]
            
            vision_client = self._get_vision_client()
            if vision_client and vision_client.is_available():
                if self._needs_vision_fallback(extracted_text, page_count):
                    logger.info(
                        f"Granite VLM returned insufficient text ({len(extracted_text)} chars), "
                        f"triggering Vision LLM fallback"
                    )
                    try:
                        vision_result = await vision_client.extract_from_pdf_pages(file_path)
                        vision_text = vision_result.get("text", "").strip()
                        
                        if vision_text and len(vision_text) > len(extracted_text):
                            logger.info(
                                f"Vision LLM extracted {len(vision_text)} chars vs "
                                f"Granite {len(extracted_text)} chars, using Vision LLM text"
                            )
                            result["text"] = vision_text
                            result["sections"] = vision_result.get("sections", sections)
                            result["metadata"]["vision_fallback"] = True
                            result["metadata"]["vision_confidence"] = vision_result.get("metadata", {}).get("confidence", 0.0)
                    except Exception as vision_error:
                        logger.warning(f"Vision LLM fallback failed: {vision_error}")
            
            return result
            
        except CircuitOpenError as e:
            logger.warning(f"Granite VLM circuit breaker open: {e}")
            raise
        except Exception as e:
            logger.error(f"Granite VLM parsing error: {e}")
            raise
    
    def _convert_doctags_to_sections(
        self,
        doctags: str,
        full_text: str,
    ) -> List[Dict[str, Any]]:
        """Convert DocTags to sections format for chunking."""
        sections = []
        
        if doctags:
            # Extract headings and content from DocTags
            heading_pattern = r'<heading level="(\d+)">(.*?)</heading>'
            
            current_section = {"text": "", "title": "Introduction", "level": 0}
            
            # Split by headings
            parts = re.split(r'(<heading level="\d+">.*?</heading>)', doctags, flags=re.DOTALL)
            
            for part in parts:
                heading_match = re.match(heading_pattern, part, re.DOTALL)
                if heading_match:
                    # Save previous section
                    if current_section["text"].strip():
                        sections.append(current_section.copy())
                    
                    # Start new section
                    level = int(heading_match.group(1))
                    title = heading_match.group(2).strip()
                    current_section = {
                        "text": title,
                        "title": title,
                        "level": level,
                        "section_path": title,
                    }
                else:
                    # Add content to current section
                    # Remove tags and add text
                    text = re.sub(r'<[^>]+>', ' ', part)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text:
                        current_section["text"] += " " + text
            
            # Add final section
            if current_section["text"].strip():
                sections.append(current_section)
        
        # If no sections found from DocTags, create from full text
        if not sections and full_text:
            # Split by paragraphs
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            for i, para in enumerate(paragraphs):
                sections.append({
                    "text": para,
                    "title": f"Section {i + 1}",
                    "level": 0,
                    "section_path": f"Section {i + 1}",
                })
        
        return sections
    
    def _parse_sync(self, file_path: str) -> Dict[str, Any]:
        """Synchronous Docling parsing."""
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling_core.types.doc import DoclingDocument
        
        try:
            # Configure pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True  # Extract tables
            pipeline_options.do_ocr = getattr(settings, 'ENABLE_OCR_DETECTION', False)
            
            # Create converter
            converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.HTML,
                    InputFormat.IMAGE,
                ],
            )
            
            # Convert document
            result = converter.convert(file_path)
            doc: DoclingDocument = result.document
            
            # Extract text
            full_text = doc.export_to_text()
            
            # Extract sections with hierarchy
            sections = self._extract_sections(doc)
            
            # Extract tables
            tables = self._extract_tables(doc)
            
            # Extract images
            images = self._extract_images(doc)
            
            # Get page count
            page_count = len(doc.pages) if hasattr(doc, 'pages') else 1
            
            # Document metadata
            metadata = {
                "parser": "docling",
                "tables_count": len(tables),
                "images_count": len(images),
            }
            
            logger.info(
                f"Docling parsed: {len(full_text)} chars, "
                f"{len(sections)} sections, "
                f"{len(tables)} tables, "
                f"{len(images)} images"
            )
            
            return {
                "text": full_text,
                "sections": sections,
                "tables": tables,
                "images": images,
                "page_count": page_count,
                "metadata": metadata,
            }
            
        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            raise
    
    def _extract_sections(self, doc) -> List[Dict[str, Any]]:
        """Extract sections from Docling document."""
        sections = []
        
        try:
            # Iterate through document structure
            for i, item in enumerate(doc.iterate_items()):
                if hasattr(item, 'text') and item.text:
                    section = {
                        "text": item.text,
                        "title": getattr(item, 'title', f"Section {i+1}"),
                        "section_path": getattr(item, 'path', f"Section {i+1}"),
                        "level": getattr(item, 'level', 0),
                    }
                    
                    # Page info if available
                    if hasattr(item, 'page_no'):
                        section["page_number"] = item.page_no
                    
                    sections.append(section)
        except Exception as e:
            logger.warning(f"Section extraction error: {e}")
        
        return sections
    
    def _extract_tables(self, doc) -> List[Dict[str, Any]]:
        """Extract tables from Docling document."""
        tables = []
        
        try:
            for i, table in enumerate(doc.tables):
                table_data = {
                    "index": i,
                    "text": table.export_to_text() if hasattr(table, 'export_to_text') else str(table),
                    "markdown": table.export_to_markdown() if hasattr(table, 'export_to_markdown') else "",
                    "rows": getattr(table, 'num_rows', 0),
                    "cols": getattr(table, 'num_cols', 0),
                }
                
                if hasattr(table, 'page_no'):
                    table_data["page_number"] = table.page_no
                
                tables.append(table_data)
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")
        
        return tables
    
    def _extract_images(self, doc) -> List[Dict[str, Any]]:
        """Extract image metadata from Docling document."""
        images = []
        
        try:
            for i, image in enumerate(doc.pictures):
                image_data = {
                    "index": i,
                    "caption": getattr(image, 'caption', None),
                    "alt_text": getattr(image, 'alt_text', None),
                }
                
                if hasattr(image, 'page_no'):
                    image_data["page_number"] = image.page_no
                
                if hasattr(image, 'bbox'):
                    image_data["bbox"] = {
                        "x": image.bbox.x,
                        "y": image.bbox.y,
                        "width": image.bbox.width,
                        "height": image.bbox.height,
                    }
                
                images.append(image_data)
        except Exception as e:
            logger.warning(f"Image extraction error: {e}")
        
        return images


class EnhancedDocumentParser:
    """
    Document parser that uses Docling for complex documents
    and falls back to basic parsers for simple ones.
    """
    
    def __init__(self):
        self.docling = DoclingParser()
        self._basic_parser = None
    
    def _get_basic_parser(self):
        """Lazy import basic parser to avoid circular imports."""
        if self._basic_parser is None:
            from app.services.ingestion_service import DocumentParser
            self._basic_parser = DocumentParser()
        return self._basic_parser
    
    async def parse(self, file_path: str, mime_type: str) -> Dict[str, Any]:
        """
        Parse document with best available parser.
        
        Strategy:
        1. Use Docling for PDFs if enabled and available
        2. Fall back to basic parser otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # Use Docling for complex document types
        if ext == ".pdf" and self.docling.is_available():
            try:
                return await self.docling.parse(file_path)
            except Exception as e:
                logger.warning(f"Docling failed, falling back to basic parser: {e}")
        
        # Fall back to basic parser
        basic = self._get_basic_parser()
        return await basic.parse(file_path, mime_type)
    
    def should_use_docling(self, file_path: str, text_length: int) -> bool:
        """
        Determine if we should use Docling for a document.
        
        Returns True for:
        - PDFs with extracted text < 100 chars per page (likely image-based)
        - Documents with suspected tables
        - Explicitly flagged for enhanced parsing
        """
        if not self.docling.is_available():
            return False
        
        ext = os.path.splitext(file_path)[1].lower()
        
        # Only process supported types
        if ext not in [".pdf", ".docx", ".pptx"]:
            return False
        
        # Low text density suggests image-based or complex layout
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        if file_size > 50000 and text_length < 500:  # 50KB file with < 500 chars
            return True
        
        return False


# Singleton getters
_docling_parser: Optional[DoclingParser] = None
_enhanced_parser: Optional[EnhancedDocumentParser] = None


def get_docling_parser() -> DoclingParser:
    """Get Docling parser instance."""
    global _docling_parser
    if _docling_parser is None:
        _docling_parser = DoclingParser()
    return _docling_parser


def get_enhanced_document_parser() -> EnhancedDocumentParser:
    """Get enhanced document parser instance."""
    global _enhanced_parser
    if _enhanced_parser is None:
        _enhanced_parser = EnhancedDocumentParser()
    return _enhanced_parser
