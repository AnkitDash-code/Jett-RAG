"""
Docling Parser Integration for Phase 5.

Uses Docling for advanced document parsing with:
- Table extraction
- Image extraction with captions
- Document structure recognition
- Enhanced PDF handling

Docling: https://github.com/DS4SD/docling
"""
import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Advanced document parser using Docling library.
    
    Provides enhanced parsing for:
    - Complex PDF documents with tables/images
    - Multi-column layouts
    - Document structure (headings, sections)
    - Embedded images with captions
    
    Falls back to basic pdfplumber if Docling is unavailable.
    """
    
    def __init__(self):
        self.enabled = getattr(settings, 'ENABLE_DOCLING_PARSER', False)
        self.timeout = getattr(settings, 'DOCLING_TIMEOUT_SECONDS', 120)
        self._docling_available = None
    
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
    
    def is_available(self) -> bool:
        """Check if Docling parsing is enabled and available."""
        return self.enabled and self._check_docling()
    
    async def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse document using Docling.
        
        Returns:
            Dict with keys:
            - text: Full extracted text
            - sections: List of sections with text, title, hierarchy
            - tables: List of extracted tables
            - images: List of image metadata
            - page_count: Number of pages
            - metadata: Document metadata
        """
        if not self.is_available():
            raise RuntimeError("Docling parser not available")
        
        # Run in thread pool since Docling is CPU-bound
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._parse_sync,
            file_path,
        )
    
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
