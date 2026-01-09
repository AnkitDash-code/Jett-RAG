"""
OCR Detection and Processing Service for Phase 5.

Uses EasyOCR for GPU-accelerated text recognition from images and PDFs.
EasyOCR leverages the existing PyTorch installation for GPU acceleration.

Features:
- GPU-accelerated OCR via EasyOCR (uses existing PyTorch)
- Text density analysis to detect image-based PDFs
- Graceful degradation when EasyOCR not installed
- Async processing with thread pool for CPU-bound work
"""
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_boxes": self.bounding_boxes,
            "language": self.language,
        }


@dataclass
class OCRDetectionResult:
    """Result from OCR need detection."""
    needs_ocr: bool
    reason: str
    text_density: float
    text_per_page: float


class OCRService:
    """
    GPU-accelerated OCR service using EasyOCR.
    
    EasyOCR uses PyTorch under the hood, so it automatically uses
    the GPU if torch.cuda.is_available() returns True.
    """
    
    # Thresholds for OCR detection
    MIN_TEXT_PER_PAGE = 50  # Chars per page to consider "has text"
    MIN_TEXT_DENSITY = 0.01  # Chars per byte of file size
    
    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize OCR service.
        
        Args:
            languages: List of language codes for EasyOCR (default: ['en'])
                      See: https://www.jaided.ai/easyocr/
        """
        self.languages = languages or ['en']
        self.enabled = getattr(settings, 'ENABLE_OCR_DETECTION', False)
        self.min_text_threshold = getattr(settings, 'OCR_MIN_TEXT_THRESHOLD', 50)
        self._reader = None
        self._available = None
        self._tesseract_available = None
    
    def _get_reader(self):
        """Lazy-load EasyOCR reader (downloads models on first use)."""
        if self._reader is None:
            try:
                import easyocr
                # GPU is automatically used if available via PyTorch
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=True,  # Will fallback to CPU if no GPU
                    verbose=False,
                )
                logger.info(f"EasyOCR initialized with languages: {self.languages}")
            except ImportError:
                logger.warning(
                    "EasyOCR not installed. Install with: pip install easyocr"
                )
                raise RuntimeError("EasyOCR not available")
        return self._reader
    
    def _check_available(self) -> bool:
        """Check if EasyOCR is installed and working."""
        if self._available is not None:
            return self._available
        
        try:
            import easyocr
            self._available = True
            logger.info("EasyOCR available")
        except ImportError:
            self._available = False
            logger.warning(
                "EasyOCR not installed. OCR features disabled. "
                "Install with: pip install easyocr"
            )
        
        return self._available
    
    def _check_tesseract(self) -> bool:
        """Check if pytesseract is installed and Tesseract binary is available."""
        if self._tesseract_available is not None:
            return self._tesseract_available
        
        try:
            import pytesseract
            # Try to get version to verify it's working
            version = pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info(f"Tesseract OCR available: {version}")
            return True
        except ImportError:
            self._tesseract_available = False
            logger.debug("pytesseract not installed")
            return False
        except Exception as e:
            self._tesseract_available = False
            logger.debug(f"Tesseract binary not found: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if OCR processing is enabled and available (either EasyOCR or Tesseract)."""
        if not self.enabled:
            return False
        return self._check_available() or self._check_tesseract()
    
    def detect_ocr_need(
        self,
        file_path: str,
        extracted_text: str,
        page_count: int,
    ) -> OCRDetectionResult:
        """
        Determine if a document needs OCR processing.
        
        Args:
            file_path: Path to the document
            extracted_text: Text already extracted via normal parsing
            page_count: Number of pages in document
            
        Returns:
            OCRDetectionResult with recommendation
        """
        ext = os.path.splitext(file_path)[1].lower()
        text_length = len(extracted_text.strip())
        
        # Calculate metrics
        text_per_page = text_length / max(page_count, 1)
        text_density = 0.0
        
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                text_density = text_length / file_size
        except OSError:
            pass
        
        # Only PDFs typically need OCR fallback
        if ext != ".pdf":
            return OCRDetectionResult(
                needs_ocr=False,
                reason="Not a PDF file",
                text_density=text_density,
                text_per_page=text_per_page,
            )
        
        if not self.enabled:
            return OCRDetectionResult(
                needs_ocr=False,
                reason="OCR detection disabled",
                text_density=text_density,
                text_per_page=text_per_page,
            )
        
        # Check text per page
        if text_per_page < self.min_text_threshold:
            return OCRDetectionResult(
                needs_ocr=True,
                reason=f"Low text density: {text_per_page:.1f} chars/page",
                text_density=text_density,
                text_per_page=text_per_page,
            )
        
        # Check text to file size ratio for large files
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 50000 and text_density < self.MIN_TEXT_DENSITY:
                return OCRDetectionResult(
                    needs_ocr=True,
                    reason=f"Low text/size ratio: {text_density:.4f}",
                    text_density=text_density,
                    text_per_page=text_per_page,
                )
        except OSError:
            pass
        
        return OCRDetectionResult(
            needs_ocr=False,
            reason="Sufficient text content",
            text_density=text_density,
            text_per_page=text_per_page,
        )
    
    def needs_ocr(
        self,
        file_path: str,
        extracted_text: str,
        page_count: int,
    ) -> bool:
        """
        Simple check if document needs OCR.
        
        For backward compatibility with OCRDetectionService interface.
        """
        result = self.detect_ocr_need(file_path, extracted_text, page_count)
        if result.needs_ocr:
            logger.info(f"OCR recommended: {result.reason}")
        return result.needs_ocr
    
    async def process_image(
        self,
        image_input: Any,
        detail_level: int = 0,
    ) -> OCRResult:
        """
        Process a single image with OCR.
        
        Args:
            image_input: Can be:
                - Path to image file (str or Path)
                - PIL Image
                - numpy array
            detail_level: 0 = paragraph, 1 = line-by-line
            
        Returns:
            OCRResult with extracted text and metadata
        """
        if not self.is_available():
            raise RuntimeError("OCR service not available")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._process_image_sync,
            image_input,
            detail_level,
        )
    
    def _process_image_sync(
        self,
        image_input: Any,
        detail_level: int = 0,
    ) -> OCRResult:
        """Synchronous image OCR processing."""
        # Try pytesseract first (faster, lighter)
        if self._check_tesseract():
            try:
                return self._process_with_tesseract(image_input)
            except Exception as e:
                logger.warning(f"Tesseract OCR failed, trying EasyOCR: {e}")
        
        # Fallback to EasyOCR (more accurate)
        reader = self._get_reader()
        
        # Handle different input types
        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
        else:
            image_path = image_input  # PIL Image or numpy array
        
        try:
            # EasyOCR with paragraph=False returns list of (bbox, text, confidence)
            # With paragraph=True it may return different structure
            results = reader.readtext(
                image_path,
                detail=1,
                paragraph=False,  # Always use detail mode for consistent output
            )
            
            if not results:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    bounding_boxes=[],
                    language=self.languages[0],
                )
            
            # Extract text and calculate average confidence
            texts = []
            bboxes = []
            confidences = []
            
            for result in results:
                # Handle both formats: (bbox, text, conf) or (text, conf)
                if len(result) == 3:
                    bbox, text, conf = result
                    bboxes.append({
                        "coordinates": bbox,
                        "text": text,
                        "confidence": conf,
                    })
                else:
                    text, conf = result
                    bbox = None
                texts.append(text)
                confidences.append(conf)
            
            # Join text based on detail level preference
            full_text = " ".join(texts) if detail_level == 0 else "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                bounding_boxes=bboxes,
                language=self.languages[0],
            )
            
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            raise
    
    def _process_with_tesseract(self, image_input: Any) -> OCRResult:
        """Process image using pytesseract (lightweight, fast)."""
        import pytesseract
        from PIL import Image
        
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        else:
            image = image_input
        
        # Extract text with confidence data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng')
        
        # Filter out low-confidence results and build text
        texts = []
        confidences = []
        bboxes = []
        
        for i, conf in enumerate(data['conf']):
            if conf > 0:  # Valid detection
                text = data['text'][i].strip()
                if text:
                    texts.append(text)
                    confidences.append(conf / 100.0)  # Normalize to 0-1
                    bboxes.append({
                        "coordinates": [
                            [data['left'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                            [data['left'][i], data['top'][i] + data['height'][i]]
                        ],
                        "text": text,
                        "confidence": conf / 100.0
                    })
        
        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        logger.info(f"Tesseract extracted {len(texts)} text blocks, avg confidence: {avg_confidence:.2f}")
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bounding_boxes=bboxes,
            language='en',
        )
    
    async def process_pdf_page(
        self,
        pdf_path: str,
        page_number: int,
        dpi: int = 200,
    ) -> OCRResult:
        """
        OCR a single page from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            dpi: Resolution for PDF to image conversion
            
        Returns:
            OCRResult for the page
        """
        if not self.is_available():
            raise RuntimeError("OCR service not available")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._process_pdf_page_sync,
            pdf_path,
            page_number,
            dpi,
        )
    
    def _process_pdf_page_sync(
        self,
        pdf_path: str,
        page_number: int,
        dpi: int = 200,
    ) -> OCRResult:
        """Synchronous PDF page OCR."""
        try:
            from pdf2image import convert_from_path
            from PIL import Image
        except ImportError:
            raise RuntimeError(
                "pdf2image not installed. Install with: pip install pdf2image"
            )
        
        try:
            # Convert single page to image
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_number,
                last_page=page_number,
                fmt='png',
            )
            
            if not images:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    bounding_boxes=[],
                )
            
            # OCR the page image
            image = images[0]
            image_array = np.array(image)
            
            return self._process_image_sync(image_array, detail_level=1)
            
        except Exception as e:
            logger.error(f"PDF page OCR failed: {e}")
            raise
    
    async def process_with_ocr(
        self,
        file_path: str,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Process a full PDF document with OCR.
        
        Args:
            file_path: Path to the PDF file
            language: Language code (default: English)
            
        Returns:
            Dict with 'text', 'sections', 'page_count', 'metadata'
        """
        if not self.is_available():
            raise RuntimeError("OCR service not available")
        
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
        language: str = "en",
    ) -> Dict[str, Any]:
        """Synchronous PDF OCR processing."""
        # Try PyMuPDF first (no external dependencies)
        try:
            import fitz  # PyMuPDF
            return self._ocr_pdf_with_pymupdf(file_path, language)
        except ImportError:
            logger.warning("PyMuPDF not available. Install with: pip install pymupdf")
        except Exception as e:
            logger.error(f"PyMuPDF OCR failed: {e}")
        
        # Fallback to pdf2image (requires poppler)
        try:
            from pdf2image import convert_from_path
            return self._ocr_pdf_with_pdf2image(file_path, language)
        except ImportError:
            error_msg = (
                "No PDF converter available. OCR requires PyMuPDF (recommended) or pdf2image.\n"
                "Install with: pip install pymupdf\n"
                "OR install pdf2image + poppler: pip install pdf2image"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"PDF to image conversion failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _ocr_pdf_with_pymupdf(
        self,
        file_path: str,
        language: str = "en",
    ) -> Dict[str, Any]:
        """OCR PDF using PyMuPDF for image extraction (no poppler required)."""
        import fitz
        from PIL import Image
        import io
        
        doc = None
        try:
            logger.info(f"Converting PDF to images with PyMuPDF: {file_path}")
            doc = fitz.open(file_path)
            page_count = len(doc)
            
            # First, extract all page images to memory before closing doc
            page_images = []
            for page_num in range(page_count):
                page = doc[page_num]
                # Render page to pixmap at 200 DPI
                pix = page.get_pixmap(dpi=200)
                # Convert pixmap to PNG bytes
                img_data = pix.tobytes("png")
                page_images.append(img_data)
            
            # Close document before OCR (frees memory, allows file access)
            doc.close()
            doc = None
            
            # Now OCR each page image
            page_texts = []
            sections = []
            total_confidence = 0.0
            
            for page_num, img_data in enumerate(page_images):
                image = Image.open(io.BytesIO(img_data))
                image_array = np.array(image)
                
                # OCR the page
                result = self._process_image_sync(image_array, detail_level=1)
                
                page_texts.append(result.text)
                total_confidence += result.confidence
                
                if result.text.strip():
                    sections.append({
                        "text": result.text,
                        "title": f"Page {page_num + 1} (OCR)",
                        "page_number": page_num + 1,
                        "section_path": f"Page {page_num + 1}",
                        "confidence": result.confidence,
                    })
                
                logger.debug(
                    f"OCR page {page_num + 1}: {len(result.text)} chars, "
                    f"confidence: {result.confidence:.2f}"
                )
            
            full_text = "\n\n".join(page_texts)
            avg_confidence = total_confidence / page_count if page_count > 0 else 0.0
            
            logger.info(
                f"OCR complete (PyMuPDF): {len(full_text)} chars from {page_count} pages, "
                f"avg confidence: {avg_confidence:.2f}"
            )
            
            return {
                "text": full_text,
                "sections": sections,
                "page_count": page_count,
                "metadata": {
                    "parser": "pymupdf+ocr",
                    "language": language,
                    "dpi": 200,
                    "confidence": avg_confidence,
                },
            }
        except Exception as e:
            logger.error(f"PyMuPDF OCR failed: {e}")
            raise
        finally:
            # Ensure document is closed even on error
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
    
    def _ocr_pdf_with_pdf2image(
        self,
        file_path: str,
        language: str = "en",
    ) -> Dict[str, Any]:
        """OCR PDF using pdf2image (requires poppler)."""
        from pdf2image import convert_from_path
        
        try:
            # Convert PDF pages to images
            logger.info(f"Converting PDF to images with pdf2image: {file_path}")
            images = convert_from_path(
                file_path,
                dpi=200,
                fmt='png',
            )
            
            page_texts = []
            sections = []
            total_confidence = 0.0
            
            for i, image in enumerate(images, 1):
                # Convert PIL to numpy for EasyOCR
                image_array = np.array(image)
                result = self._process_image_sync(image_array, detail_level=1)
                
                page_texts.append(result.text)
                total_confidence += result.confidence
                
                if result.text.strip():
                    sections.append({
                        "text": result.text,
                        "title": f"Page {i} (OCR)",
                        "page_number": i,
                        "section_path": f"Page {i}",
                        "confidence": result.confidence,
                    })
                
                logger.debug(
                    f"OCR page {i}: {len(result.text)} chars, "
                    f"confidence: {result.confidence:.2f}"
                )
            
            full_text = "\n\n".join(page_texts)
            avg_confidence = total_confidence / len(images) if images else 0.0
            
            logger.info(
                f"OCR complete (pdf2image): {len(full_text)} chars from {len(images)} pages, "
                f"avg confidence: {avg_confidence:.2f}"
            )
            
            return {
                "text": full_text,
                "sections": sections,
                "page_count": len(images),
                "metadata": {
                    "parser": "pdf2image+ocr",
                    "language": language,
                    "dpi": 200,
                    "confidence": avg_confidence,
                },
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise
    
    async def ocr_image_file(
        self,
        file_path: str,
        language: str = "en",
    ) -> str:
        """
        OCR a single image file.
        
        Backward-compatible method for OCRDetectionService interface.
        
        Args:
            file_path: Path to image (PNG, JPG, TIFF, etc.)
            language: Language code
            
        Returns:
            Extracted text
        """
        result = await self.process_image(file_path, detail_level=1)
        return result.text


# Alias for backward compatibility
OCRDetectionService = OCRService


class OCRAwareDocumentParser:
    """
    Document parser with automatic OCR fallback for image-based PDFs.
    """
    
    def __init__(self, languages: Optional[List[str]] = None):
        self.ocr_service = OCRService(languages=languages)
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
                    ocr_result["metadata"]["parser_chain"] = ["basic", "easyocr"]
                    return ocr_result
                    
                except Exception as e:
                    logger.error(f"OCR fallback failed: {e}")
                    result["metadata"]["ocr_attempted"] = True
                    result["metadata"]["ocr_error"] = str(e)
            else:
                result["metadata"]["ocr_recommended"] = True
                logger.warning(
                    f"Document likely needs OCR but EasyOCR not installed: {file_path}"
                )
        
        result["metadata"]["parser_chain"] = ["basic"]
        return result


# Singleton getters
_ocr_service: Optional[OCRService] = None
_ocr_parser: Optional[OCRAwareDocumentParser] = None


def get_ocr_service(languages: Optional[List[str]] = None) -> OCRService:
    """Get OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService(languages=languages)
    return _ocr_service


# Alias for backward compatibility
def get_ocr_detection_service() -> OCRService:
    """Get OCR detection service instance (backward compatible)."""
    return get_ocr_service()


def get_ocr_aware_parser(
    languages: Optional[List[str]] = None
) -> OCRAwareDocumentParser:
    """Get OCR-aware document parser instance."""
    global _ocr_parser
    if _ocr_parser is None:
        _ocr_parser = OCRAwareDocumentParser(languages=languages)
    return _ocr_parser
