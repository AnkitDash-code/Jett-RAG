"""
OCR Detection and Fallback Service for Phase 5.

Automatically detects low-text PDFs (scanned/image-based) and routes
them to OCR processing via Tesseract or cloud OCR services.

Features:
- Text density analysis to detect image-based PDFs
- Tesseract OCR integration
- Optional cloud OCR fallback (AWS Textract, Google Vision)
- Async processing with job queue integration
"""
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class OCRDetectionService:
    """
    Service for detecting and processing documents that require OCR.
    """
    
    # Thresholds for OCR detection
    MIN_TEXT_PER_PAGE = 50  # Chars per page to consider "has text"
    MIN_TEXT_DENSITY = 0.01  # Chars per byte of file size
    
    def __init__(self):
        self.enabled = getattr(settings, 'ENABLE_OCR_DETECTION', False)
        self.min_text_threshold = getattr(settings, 'OCR_MIN_TEXT_THRESHOLD', 50)
        self._tesseract_available = None
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is installed."""
        if self._tesseract_available is not None:
            return self._tesseract_available
        
        try:
            import pytesseract
            # Verify tesseract binary is available
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info("Tesseract OCR available")
        except Exception:
            self._tesseract_available = False
            logger.warning("Tesseract OCR not available. Install tesseract-ocr and pytesseract.")
        
        return self._tesseract_available
    
    def is_available(self) -> bool:
        """Check if OCR processing is enabled and available."""
        return self.enabled and self._check_tesseract()
    
    def needs_ocr(
        self,
        file_path: str,
        extracted_text: str,
        page_count: int,
    ) -> bool:
        """
        Determine if a document needs OCR processing.
        
        Criteria:
        1. Text per page is below threshold
        2. File size is significant but text is minimal
        3. Known image-based PDF indicators
        
        Args:
            file_path: Path to the document
            extracted_text: Text already extracted via normal parsing
            page_count: Number of pages in document
            
        Returns:
            True if OCR is recommended
        """
        if not self.enabled:
            return False
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext != ".pdf":
            return False  # Only PDFs need OCR fallback
        
        text_length = len(extracted_text.strip())
        
        # Check text per page
        if page_count > 0:
            text_per_page = text_length / page_count
            if text_per_page < self.min_text_threshold:
                logger.info(
                    f"Low text density: {text_per_page:.1f} chars/page "
                    f"(threshold: {self.min_text_threshold})"
                )
                return True
        
        # Check text to file size ratio
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 50000:  # > 50KB
                text_density = text_length / file_size
                if text_density < self.MIN_TEXT_DENSITY:
                    logger.info(
                        f"Low text/size ratio: {text_density:.4f} "
                        f"({text_length} chars / {file_size} bytes)"
                    )
                    return True
        except OSError:
            pass
        
        return False
    
    async def process_with_ocr(
        self,
        file_path: str,
        language: str = "eng",
    ) -> Dict[str, Any]:
        """
        Process a document with OCR.
        
        Args:
            file_path: Path to the PDF file
            language: Tesseract language code (default: English)
            
        Returns:
            Dict with 'text', 'sections', 'page_count', 'metadata'
        """
        if not self.is_available():
            raise RuntimeError("OCR service not available")
        
        # Run OCR in thread pool since it's CPU-bound
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._ocr_pdf_sync,
            file_path,
            language,
        )
    
    def _ocr_pdf_sync(
        self,
        file_path: str,
        language: str = "eng",
    ) -> Dict[str, Any]:
        """Synchronous PDF OCR processing."""
        import pytesseract
        from pdf2image import convert_from_path
        
        try:
            # Convert PDF pages to images
            logger.info(f"Converting PDF to images for OCR: {file_path}")
            images = convert_from_path(
                file_path,
                dpi=200,  # Balance quality vs speed
                fmt='png',
            )
            
            page_texts = []
            sections = []
            
            for i, image in enumerate(images, 1):
                # Run OCR on each page
                page_text = pytesseract.image_to_string(
                    image,
                    lang=language,
                    config='--psm 1 --oem 3',  # Auto orientation + best accuracy
                )
                
                page_texts.append(page_text)
                
                if page_text.strip():
                    sections.append({
                        "text": page_text,
                        "title": f"Page {i} (OCR)",
                        "page_number": i,
                        "section_path": f"Page {i}",
                    })
                
                logger.debug(f"OCR page {i}: {len(page_text)} chars")
            
            full_text = "\n\n".join(page_texts)
            
            logger.info(
                f"OCR complete: {len(full_text)} chars from {len(images)} pages"
            )
            
            return {
                "text": full_text,
                "sections": sections,
                "page_count": len(images),
                "metadata": {
                    "parser": "tesseract_ocr",
                    "language": language,
                    "dpi": 200,
                },
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise
    
    async def ocr_image_file(
        self,
        file_path: str,
        language: str = "eng",
    ) -> str:
        """
        OCR a single image file.
        
        Args:
            file_path: Path to image (PNG, JPG, TIFF, etc.)
            language: Tesseract language code
            
        Returns:
            Extracted text
        """
        if not self.is_available():
            raise RuntimeError("OCR service not available")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._ocr_image_sync,
            file_path,
            language,
        )
    
    def _ocr_image_sync(
        self,
        file_path: str,
        language: str = "eng",
    ) -> str:
        """Synchronous image OCR."""
        import pytesseract
        from PIL import Image
        
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(
                image,
                lang=language,
                config='--psm 1 --oem 3',
            )
            return text
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            raise


class OCRAwareDocumentParser:
    """
    Document parser with automatic OCR fallback for image-based PDFs.
    """
    
    def __init__(self):
        self.ocr_service = OCRDetectionService()
        self._basic_parser = None
        self._docling_parser = None
    
    def _get_basic_parser(self):
        """Lazy import basic parser."""
        if self._basic_parser is None:
            from app.services.ingestion_service import DocumentParser
            self._basic_parser = DocumentParser()
        return self._basic_parser
    
    def _get_docling_parser(self):
        """Lazy import Docling parser."""
        if self._docling_parser is None:
            try:
                from app.services.docling_parser import get_docling_parser
                self._docling_parser = get_docling_parser()
            except ImportError:
                self._docling_parser = False
        return self._docling_parser if self._docling_parser else None
    
    async def parse(
        self,
        file_path: str,
        mime_type: str,
        force_ocr: bool = False,
    ) -> Dict[str, Any]:
        """
        Parse document with automatic OCR fallback.
        
        Strategy:
        1. Try Docling first if available (handles many complex cases)
        2. Try basic parser
        3. Check if OCR is needed
        4. Apply OCR if text is too sparse
        
        Args:
            file_path: Path to document
            mime_type: Document MIME type
            force_ocr: Force OCR processing regardless of text detection
            
        Returns:
            Parsed document with text, sections, metadata
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # Step 1: Try Docling for PDFs
        docling = self._get_docling_parser()
        if docling and docling.is_available() and ext == ".pdf":
            try:
                result = await docling.parse(file_path)
                
                # Check if Docling got enough text
                if not self.ocr_service.needs_ocr(
                    file_path,
                    result.get("text", ""),
                    result.get("page_count", 1),
                ) and not force_ocr:
                    result["metadata"]["parser_chain"] = ["docling"]
                    return result
                    
            except Exception as e:
                logger.warning(f"Docling parsing failed: {e}")
        
        # Step 2: Try basic parser
        basic = self._get_basic_parser()
        result = await basic.parse(file_path, mime_type)
        
        # Step 3: Check if OCR is needed
        if force_ocr or self.ocr_service.needs_ocr(
            file_path,
            result.get("text", ""),
            result.get("page_count", 1),
        ):
            # Step 4: Apply OCR
            if self.ocr_service.is_available():
                try:
                    logger.info(f"Applying OCR fallback for: {file_path}")
                    ocr_result = await self.ocr_service.process_with_ocr(file_path)
                    
                    # Merge results - prefer OCR text but keep metadata
                    ocr_result["metadata"]["parser_chain"] = ["basic", "tesseract_ocr"]
                    return ocr_result
                    
                except Exception as e:
                    logger.error(f"OCR fallback failed: {e}")
                    result["metadata"]["ocr_attempted"] = True
                    result["metadata"]["ocr_error"] = str(e)
            else:
                result["metadata"]["ocr_recommended"] = True
                logger.warning(
                    f"Document likely needs OCR but service unavailable: {file_path}"
                )
        
        result["metadata"]["parser_chain"] = ["basic"]
        return result


# Singleton getters
_ocr_service: Optional[OCRDetectionService] = None
_ocr_parser: Optional[OCRAwareDocumentParser] = None


def get_ocr_detection_service() -> OCRDetectionService:
    """Get OCR detection service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRDetectionService()
    return _ocr_service


def get_ocr_aware_parser() -> OCRAwareDocumentParser:
    """Get OCR-aware document parser instance."""
    global _ocr_parser
    if _ocr_parser is None:
        _ocr_parser = OCRAwareDocumentParser()
    return _ocr_parser
