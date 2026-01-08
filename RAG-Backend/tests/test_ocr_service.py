"""
Tests for OCR Service using EasyOCR.
"""
import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from app.services.ocr_service import (
    OCRService,
    OCRResult,
    OCRDetectionResult,
    OCRAwareDocumentParser,
    get_ocr_service,
    get_ocr_detection_service,
    get_ocr_aware_parser,
)


class TestOCRServiceBasic:
    """Basic OCR service tests without GPU/model loading."""
    
    def test_ocr_service_init(self):
        """Test OCR service initialization."""
        service = OCRService()
        assert service.languages == ['en']
        assert service._reader is None  # Lazy loaded
        
    def test_ocr_service_custom_languages(self):
        """Test OCR service with custom languages."""
        service = OCRService(languages=['en', 'fr'])
        assert service.languages == ['en', 'fr']
    
    def test_ocr_result_dataclass(self):
        """Test OCRResult dataclass."""
        result = OCRResult(
            text="Hello World",
            confidence=0.95,
            bounding_boxes=[{"x": 0, "y": 0}],
            language="en",
        )
        assert result.text == "Hello World"
        assert result.confidence == 0.95
        assert len(result.bounding_boxes) == 1
        
        # Test to_dict
        d = result.to_dict()
        assert d["text"] == "Hello World"
        assert d["confidence"] == 0.95
    
    def test_ocr_detection_result_dataclass(self):
        """Test OCRDetectionResult dataclass."""
        result = OCRDetectionResult(
            needs_ocr=True,
            reason="Low text density",
            text_density=0.005,
            text_per_page=20.0,
        )
        assert result.needs_ocr is True
        assert "Low text density" in result.reason


class TestOCRDetection:
    """Test OCR need detection logic."""
    
    def test_detect_ocr_need_non_pdf(self, tmp_path):
        """Non-PDF files should not need OCR."""
        service = OCRService()
        service.enabled = True
        
        # Create a dummy text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")
        
        result = service.detect_ocr_need(
            str(txt_file),
            "Some extracted text",
            1,
        )
        assert result.needs_ocr is False
        assert "Not a PDF" in result.reason
    
    def test_detect_ocr_need_disabled(self, tmp_path):
        """OCR detection disabled should return False."""
        service = OCRService()
        service.enabled = False
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        
        result = service.detect_ocr_need(str(pdf_file), "", 1)
        assert result.needs_ocr is False
        assert "disabled" in result.reason
    
    def test_detect_ocr_need_low_text_density(self, tmp_path):
        """Low text density PDF should need OCR."""
        service = OCRService()
        service.enabled = True
        service.min_text_threshold = 50
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 " + b"x" * 10000)  # Large file
        
        result = service.detect_ocr_need(
            str(pdf_file),
            "Short text",  # Only ~10 chars
            1,  # 1 page = 10 chars/page < 50 threshold
        )
        assert result.needs_ocr is True
        assert "Low text density" in result.reason
    
    def test_detect_ocr_need_sufficient_text(self, tmp_path):
        """PDF with sufficient text should not need OCR."""
        service = OCRService()
        service.enabled = True
        service.min_text_threshold = 50
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        
        result = service.detect_ocr_need(
            str(pdf_file),
            "A" * 200,  # 200 chars / 1 page = 200 chars/page
            1,
        )
        assert result.needs_ocr is False
        assert "Sufficient" in result.reason
    
    def test_needs_ocr_backward_compat(self, tmp_path):
        """Test backward compatible needs_ocr method."""
        service = OCRService()
        service.enabled = True
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        
        # Should return bool
        needs = service.needs_ocr(str(pdf_file), "Short", 1)
        assert isinstance(needs, bool)


class TestOCRServiceAvailability:
    """Test OCR service availability checks."""
    
    def test_is_available_disabled(self):
        """Disabled service should not be available."""
        service = OCRService()
        service.enabled = False
        assert service.is_available() is False
    
    def test_check_available_easyocr_installed(self):
        """EasyOCR should be detected as installed."""
        service = OCRService()
        # This actually checks if easyocr can be imported
        available = service._check_available()
        assert available is True  # We just installed it


@pytest.mark.slow
class TestOCRProcessing:
    """Tests that actually run OCR (slower, load models)."""
    
    @pytest.fixture
    def ocr_service(self):
        """Get enabled OCR service."""
        service = OCRService(languages=['en'])
        service.enabled = True
        return service
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample image with text."""
        # Create image with text
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw text - use default font
        draw.text((20, 30), "Hello EasyOCR Test", fill='black')
        
        # Save
        img_path = tmp_path / "test_image.png"
        img.save(str(img_path))
        return str(img_path)
    
    @pytest.fixture
    def sample_image_array(self):
        """Create a sample numpy array image with text."""
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "Array Test 123", fill='black')
        return np.array(img)
    
    @pytest.mark.asyncio
    async def test_process_image_file(self, ocr_service, sample_image):
        """Test OCR on image file."""
        result = await ocr_service.process_image(sample_image)
        
        assert isinstance(result, OCRResult)
        assert len(result.text) > 0
        assert result.confidence > 0
        assert result.language == 'en'
        # Should detect some of the text
        text_lower = result.text.lower()
        assert 'hello' in text_lower or 'easyocr' in text_lower or 'test' in text_lower
    
    @pytest.mark.asyncio
    async def test_process_image_array(self, ocr_service, sample_image_array):
        """Test OCR on numpy array."""
        result = await ocr_service.process_image(sample_image_array)
        
        assert isinstance(result, OCRResult)
        assert len(result.text) > 0
    
    @pytest.mark.asyncio
    async def test_process_image_detail_levels(self, ocr_service, sample_image):
        """Test different detail levels."""
        # Paragraph mode
        result0 = await ocr_service.process_image(sample_image, detail_level=0)
        # Line mode
        result1 = await ocr_service.process_image(sample_image, detail_level=1)
        
        # Both should return text
        assert len(result0.text) > 0
        assert len(result1.text) > 0
    
    @pytest.mark.asyncio
    async def test_ocr_image_file_backward_compat(self, ocr_service, sample_image):
        """Test backward compatible ocr_image_file method."""
        text = await ocr_service.ocr_image_file(sample_image)
        
        assert isinstance(text, str)
        assert len(text) > 0


class TestOCRAwareParser:
    """Test OCR-aware document parser."""
    
    def test_parser_init(self):
        """Test parser initialization."""
        parser = OCRAwareDocumentParser()
        assert parser.ocr_service is not None
        assert parser._basic_parser is None  # Lazy loaded
    
    def test_parser_custom_languages(self):
        """Test parser with custom languages."""
        parser = OCRAwareDocumentParser(languages=['en', 'de'])
        assert parser.ocr_service.languages == ['en', 'de']


class TestSingletonGetters:
    """Test singleton getter functions."""
    
    def test_get_ocr_service(self):
        """Test get_ocr_service returns consistent instance."""
        # Reset singleton for test
        import app.services.ocr_service as ocr_module
        ocr_module._ocr_service = None
        
        service1 = get_ocr_service()
        service2 = get_ocr_service()
        
        # Same instance
        assert service1 is service2
    
    def test_get_ocr_detection_service_alias(self):
        """Test backward compatible alias."""
        service = get_ocr_detection_service()
        assert isinstance(service, OCRService)
    
    def test_get_ocr_aware_parser(self):
        """Test get_ocr_aware_parser returns consistent instance."""
        import app.services.ocr_service as ocr_module
        ocr_module._ocr_parser = None
        
        parser1 = get_ocr_aware_parser()
        parser2 = get_ocr_aware_parser()
        
        assert parser1 is parser2


class TestOCRServiceNotAvailable:
    """Test behavior when OCR is not available."""
    
    @pytest.mark.asyncio
    async def test_process_image_not_available(self, tmp_path):
        """Should raise error when not available."""
        service = OCRService()
        service.enabled = False  # Disabled
        
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(str(img_path))
        
        with pytest.raises(RuntimeError, match="not available"):
            await service.process_image(str(img_path))


# Mark slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (loading OCR models)"
    )
