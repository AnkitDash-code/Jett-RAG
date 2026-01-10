"""
Vision LLM Client for Image Text Extraction.

Uses the Vision LLM API at /vision/extract endpoint to extract text
and descriptions from images. This replaces Tesseract/EasyOCR with
a more powerful LLM-based approach.

Endpoint: POST {VISION_LLM_URL}/vision/extract
- Accepts: multipart/form-data with PNG/JPEG file
- Returns: JSON with extracted text and description
"""
import asyncio
import logging
import os
import io
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import httpx
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result from Vision LLM processing."""
    text: str
    description: str
    confidence: float
    qr_codes: List[Dict[str, str]] = None  # List of detected QR/barcodes
    
    def __post_init__(self):
        if self.qr_codes is None:
            self.qr_codes = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "description": self.description,
            "confidence": self.confidence,
            "qr_codes": self.qr_codes,
        }


class VisionLLMClient:
    """
    Client for Vision LLM API that extracts text from images.
    
    This replaces traditional OCR (Tesseract/EasyOCR) with LLM-based
    image understanding that can handle:
    - Scanned documents
    - Screenshots
    - Complex layouts
    - Handwritten text
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Use same base URL as Granite VLM (friend's server)
        self.api_url = getattr(settings, 'GRANITE_API_URL', 'http://192.168.1.10:8088')
        self.endpoint = "/vision/extract"
        self.full_url = f"{self.api_url.rstrip('/')}{self.endpoint}"
        self.timeout = getattr(settings, 'GRANITE_SERVER_TIMEOUT', 300)
        self.enabled = getattr(settings, 'ENABLE_OCR_DETECTION', True)
        
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = True
        
        logger.info(f"VisionLLMClient initialized: {self.full_url}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def is_available(self) -> bool:
        """Check if Vision LLM is enabled."""
        return self.enabled
    
    def _scan_barcodes(self, image_data: bytes) -> List[Dict[str, str]]:
        """
        Scan QR codes and barcodes from image bytes using pyzbar.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            List of dicts with 'type' and 'data' for each detected code
        """
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            from PIL import Image
            
            img = Image.open(io.BytesIO(image_data))
            decoded = pyzbar_decode(img)
            
            results = []
            for d in decoded:
                results.append({
                    "type": d.type,  # QRCODE, EAN13, CODE128, etc.
                    "data": d.data.decode("utf-8", errors="replace")
                })
            
            if results:
                logger.info(f"Detected {len(results)} QR/barcodes: {[r['type'] for r in results]}")
            
            return results
        except ImportError:
            logger.warning("pyzbar not installed, skipping barcode scan")
            return []
        except Exception as e:
            logger.warning(f"Barcode scan failed: {e}")
            return []
    
    def _should_scan_for_codes(self, description: str) -> bool:
        """
        Check if VLM description suggests QR/barcode presence.
        """
        keywords = ["qr", "barcode", "bar code", "scan code", "qr code", "2d code"]
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in keywords)
    
    async def extract_from_image(
        self,
        image_data: bytes,
        filename: str = "image.png",
    ) -> VisionResult:
        """
        Extract text from image using Vision LLM.
        
        Args:
            image_data: Raw image bytes (PNG or JPEG)
            filename: Filename hint for the API
            
        Returns:
            VisionResult with extracted text and description
        """
        if not self.enabled:
            return VisionResult(text="", description="Vision LLM disabled", confidence=0.0)
        
        client = await self._get_client()
        
        try:
            # Send image to Vision LLM
            files = {"file": (filename, image_data, "image/png")}
            
            logger.info(f"Sending image to Vision LLM: {filename} ({len(image_data)} bytes)")
            
            response = await client.post(
                self.full_url,
                files=files,
            )
            
            if response.status_code != 200:
                logger.error(f"Vision LLM returned status {response.status_code}: {response.text}")
                raise RuntimeError(f"Vision LLM API error: {response.status_code}")
            
            result = response.json()
            
            # Extract text from response - API returns {"full_text": "..."}
            text = result.get("full_text", "") or result.get("text", "") or result.get("content", "") or result.get("extracted_text", "")
            description = result.get("description", "") or result.get("explanation", "") or ""
            confidence = result.get("confidence", 0.9)  # Default high confidence for LLM
            
            # If response is a simple string, use it directly
            if isinstance(result, str):
                text = result
                description = ""
            
            logger.info(f"Vision LLM extracted {len(text)} chars from {filename}")
            
            # Check if VLM detected QR/barcode, then scan it
            qr_codes = []
            if self._should_scan_for_codes(description) or self._should_scan_for_codes(text):
                logger.info(f"VLM detected code reference, scanning image...")
                qr_codes = self._scan_barcodes(image_data)
            
            return VisionResult(
                text=text.strip(),
                description=description.strip(),
                confidence=float(confidence),
                qr_codes=qr_codes,
            )
            
        except httpx.TimeoutException:
            logger.error("Vision LLM request timed out")
            raise RuntimeError("Vision LLM timeout")
        except Exception as e:
            logger.error(f"Vision LLM error: {e}")
            raise
    
    async def extract_from_file(self, file_path: str) -> VisionResult:
        """
        Extract text from an image file.
        
        Args:
            file_path: Path to image file (PNG, JPEG, etc.)
            
        Returns:
            VisionResult with extracted text and description
        """
        with open(file_path, "rb") as f:
            image_data = f.read()
        
        filename = os.path.basename(file_path)
        return await self.extract_from_image(image_data, filename)
    
    async def extract_from_pdf_pages(
        self,
        file_path: str,
        dpi: int = 200,
        min_text_per_page: int = 50,
    ) -> Dict[str, Any]:
        """
        Smart PDF text extraction: extract text directly from text pages,
        only send image/scanned pages to Vision LLM.
        
        Args:
            file_path: Path to PDF file
            dpi: Resolution for image pages
            min_text_per_page: Minimum chars to consider page as "text page"
            
        Returns:
            Dict with text, sections, page_count, metadata
        """
        import pypdfium2 as pdfium
        import io
        
        pdf = None
        try:
            logger.info(f"Processing PDF with smart text/image detection: {file_path}")
            pdf = pdfium.PdfDocument(file_path)
            page_count = len(pdf)
            
            # First pass: extract text from all pages and identify image pages
            text_pages = []  # (page_num, text) for pages with sufficient text
            image_pages = []  # (page_num, image_bytes) for pages needing Vision LLM
            
            scale = dpi / 72  # PDFium uses 72 DPI as base
            
            for page_num in range(page_count):
                page = pdf[page_num]
                
                # Try to extract text directly from the page
                page_text = ""
                try:
                    textpage = page.get_textpage()
                    # Try get_text_bounded first (full page text)
                    page_text = textpage.get_text_bounded().strip()
                    
                    # If that's empty, try get_text_range as fallback
                    if not page_text:
                        char_count = textpage.count_chars()
                        if char_count > 0:
                            page_text = textpage.get_text_range(0, char_count).strip()
                            logger.debug(f"Page {page_num + 1}: used get_text_range ({char_count} chars)")
                    
                except Exception as text_err:
                    logger.warning(f"Text extraction failed for page {page_num + 1}: {text_err}")
                    page_text = ""
                
                logger.info(f"Page {page_num + 1}: extracted {len(page_text)} chars via text extraction")
                
                if len(page_text) >= min_text_per_page:
                    # Page has enough text, use it directly
                    text_pages.append((page_num + 1, page_text))
                    logger.debug(f"Page {page_num + 1}: text page ({len(page_text)} chars)")
                else:
                    # Page is likely an image/scan, render it for Vision LLM
                    bitmap = page.render(scale=scale)
                    pil_image = bitmap.to_pil()
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    img_data = img_buffer.getvalue()
                    image_pages.append((page_num + 1, img_data))
                    logger.debug(f"Page {page_num + 1}: image page ({len(page_text)} chars, sending to Vision LLM)")
            
            # Close PDF after extraction
            pdf.close()
            pdf = None
            
            logger.info(
                f"PDF analysis: {len(text_pages)} text pages, "
                f"{len(image_pages)} image pages to process with Vision LLM"
            )
            
            # Build results dict for all pages
            page_results = {}  # page_num -> (text, confidence, source, qr_codes)
            
            # Add text pages directly
            for page_num, text in text_pages:
                page_results[page_num] = (text, 1.0, "text_extraction", [])
            
            # Process image pages with Vision LLM
            for page_num, img_data in image_pages:
                try:
                    result = await self.extract_from_image(
                        img_data,
                        f"page_{page_num}.png"
                    )
                    
                    page_text = result.text
                    if result.description:
                        page_text = f"{result.text}\n\n[Image Description: {result.description}]"
                    
                    page_results[page_num] = (page_text, result.confidence, "vision_llm", result.qr_codes)
                    
                    logger.debug(
                        f"Vision LLM page {page_num}: {len(result.text)} chars, "
                        f"confidence: {result.confidence:.2f}, qr_codes: {len(result.qr_codes)}"
                    )
                    
                except Exception as e:
                    logger.warning(f"Vision LLM failed for page {page_num}: {e}")
                    page_results[page_num] = ("", 0.0, "vision_llm_error", [])
            
            # Assemble results in page order
            page_texts = []
            sections = []
            total_confidence = 0.0
            vision_pages_count = 0
            
            for page_num in range(1, page_count + 1):
                text, confidence, source, qr_codes = page_results.get(page_num, ("", 0.0, "missing", []))
                page_texts.append(text)
                total_confidence += confidence
                
                if source == "vision_llm":
                    vision_pages_count += 1
                
                if text.strip():
                    section_data = {
                        "text": text,
                        "title": f"Page {page_num}",
                        "page_number": page_num,
                        "section_path": f"Page {page_num}",
                        "confidence": confidence,
                        "source": source,
                    }
                    if qr_codes:
                        section_data["qr_codes"] = qr_codes
                    sections.append(section_data)
            
            full_text = "\n\n".join(page_texts)
            avg_confidence = total_confidence / page_count if page_count > 0 else 0.0
            
            logger.info(
                f"PDF processing complete: {len(full_text)} chars from {page_count} pages "
                f"({len(text_pages)} text, {vision_pages_count} vision), "
                f"avg confidence: {avg_confidence:.2f}"
            )
            
            return {
                "text": full_text,
                "sections": sections,
                "page_count": page_count,
                "metadata": {
                    "parser": "hybrid_vision_llm",
                    "dpi": dpi,
                    "confidence": avg_confidence,
                    "text_pages": len(text_pages),
                    "image_pages": len(image_pages),
                },
            }
            
        except Exception as e:
            logger.error(f"Vision LLM PDF processing failed: {e}")
            raise
        finally:
            if pdf is not None:
                try:
                    pdf.close()
                except:
                    pass


# Singleton accessor
_vision_client: Optional[VisionLLMClient] = None


def get_vision_llm_client() -> VisionLLMClient:
    """Get the Vision LLM client singleton."""
    global _vision_client
    if _vision_client is None:
        _vision_client = VisionLLMClient()
    return _vision_client
