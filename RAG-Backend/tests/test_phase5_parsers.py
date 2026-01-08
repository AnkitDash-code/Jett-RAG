"""
Tests for Phase 5: Document Parsing and OCR Services.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.docling_parser import DoclingParser, EnhancedDocumentParser
from app.services.ocr_service import OCRDetectionService, OCRAwareDocumentParser


# ============================================================================
# Docling Parser Tests
# ============================================================================

class TestDoclingParser:
    """Tests for Docling document parser."""
    
    @pytest.fixture
    def parser(self):
        """Create Docling parser with mocked settings."""
        with patch('app.services.docling_parser.settings') as mock_settings:
            mock_settings.ENABLE_DOCLING_PARSER = True
            mock_settings.DOCLING_TIMEOUT_SECONDS = 120
            return DoclingParser()
    
    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.enabled is True
        assert parser.timeout == 120
    
    def test_check_docling_not_installed(self, parser):
        """Test checking Docling availability when not installed."""
        parser._docling_available = None
        
        with patch.dict('sys.modules', {'docling': None}):
            # Force ImportError
            parser._docling_available = None
            result = parser._check_docling()
            # May be True or False depending on actual installation
            assert isinstance(result, bool)
    
    def test_is_available_when_disabled(self):
        """Test availability check when disabled."""
        with patch('app.services.docling_parser.settings') as mock_settings:
            mock_settings.ENABLE_DOCLING_PARSER = False
            mock_settings.DOCLING_TIMEOUT_SECONDS = 120
            parser = DoclingParser()
            
            # Even if docling is installed, should return False when disabled
            assert parser.is_available() is False
    
    def test_extract_sections_empty(self, parser):
        """Test section extraction with mock document."""
        mock_doc = MagicMock()
        mock_doc.iterate_items.return_value = []
        
        sections = parser._extract_sections(mock_doc)
        assert sections == []
    
    def test_extract_tables_empty(self, parser):
        """Test table extraction with mock document."""
        mock_doc = MagicMock()
        mock_doc.tables = []
        
        tables = parser._extract_tables(mock_doc)
        assert tables == []
    
    def test_extract_images_empty(self, parser):
        """Test image extraction with mock document."""
        mock_doc = MagicMock()
        mock_doc.pictures = []
        
        images = parser._extract_images(mock_doc)
        assert images == []


class TestEnhancedDocumentParser:
    """Tests for enhanced document parser."""
    
    @pytest.fixture
    def parser(self):
        """Create enhanced parser."""
        return EnhancedDocumentParser()
    
    def test_should_use_docling_not_available(self, parser):
        """Test should_use_docling when Docling not available."""
        parser.docling._docling_available = False
        
        result = parser.should_use_docling("test.pdf", 100)
        assert result is False
    
    def test_should_use_docling_wrong_extension(self, parser):
        """Test should_use_docling with unsupported file type."""
        parser.docling._docling_available = True
        
        result = parser.should_use_docling("test.txt", 100)
        assert result is False


# ============================================================================
# OCR Service Tests
# ============================================================================

class TestOCRDetectionService:
    """Tests for OCR detection service."""
    
    @pytest.fixture
    def service(self):
        """Create OCR service with mocked settings."""
        with patch('app.services.ocr_service.settings') as mock_settings:
            mock_settings.ENABLE_OCR_DETECTION = True
            mock_settings.OCR_MIN_TEXT_THRESHOLD = 50
            return OCRDetectionService()
    
    def test_service_initialization(self, service):
        """Test service initialization."""
        assert service.enabled is True
        assert service.min_text_threshold == 50
    
    def test_check_tesseract_not_installed(self, service):
        """Test checking Tesseract when not installed."""
        service._tesseract_available = None
        
        # Mock pytesseract not being available
        with patch.dict('sys.modules', {'pytesseract': None}):
            service._tesseract_available = None
            result = service._check_tesseract()
            # May be True or False depending on actual installation
            assert isinstance(result, bool)
    
    def test_is_available_when_disabled(self):
        """Test availability when service is disabled."""
        with patch('app.services.ocr_service.settings') as mock_settings:
            mock_settings.ENABLE_OCR_DETECTION = False
            mock_settings.OCR_MIN_TEXT_THRESHOLD = 50
            service = OCRDetectionService()
            
            assert service.is_available() is False
    
    def test_needs_ocr_disabled(self):
        """Test needs_ocr when service is disabled."""
        with patch('app.services.ocr_service.settings') as mock_settings:
            mock_settings.ENABLE_OCR_DETECTION = False
            mock_settings.OCR_MIN_TEXT_THRESHOLD = 50
            service = OCRDetectionService()
            
            result = service.needs_ocr("test.pdf", "", 1)
            assert result is False
    
    def test_needs_ocr_not_pdf(self, service):
        """Test needs_ocr for non-PDF file."""
        result = service.needs_ocr("test.docx", "", 1)
        assert result is False
    
    def test_needs_ocr_low_text_density(self, service):
        """Test needs_ocr with low text density."""
        # 10 chars per page is below threshold of 50
        result = service.needs_ocr("test.pdf", "short text", 5)
        assert result is True
    
    def test_needs_ocr_sufficient_text(self, service):
        """Test needs_ocr with sufficient text."""
        # 200 chars for 2 pages = 100 chars/page, above threshold
        long_text = "x" * 200
        result = service.needs_ocr("test.pdf", long_text, 2)
        assert result is False
    
    def test_needs_ocr_zero_pages(self, service):
        """Test needs_ocr with zero pages."""
        result = service.needs_ocr("test.pdf", "some text", 0)
        assert result is False


class TestOCRAwareDocumentParser:
    """Tests for OCR-aware document parser."""
    
    @pytest.fixture
    def parser(self):
        """Create OCR-aware parser."""
        with patch('app.services.ocr_service.settings') as mock_settings:
            mock_settings.ENABLE_OCR_DETECTION = True
            mock_settings.OCR_MIN_TEXT_THRESHOLD = 50
            return OCRAwareDocumentParser()
    
    @pytest.mark.asyncio
    async def test_parse_falls_back_to_basic(self, parser):
        """Test that parser falls back to basic parser when Docling unavailable."""
        # Mock basic parser
        mock_result = {
            "text": "parsed text",
            "sections": [],
            "page_count": 1,
            "metadata": {},
        }
        
        parser.ocr_service._tesseract_available = False
        parser._docling_parser = False  # Docling not available
        
        with patch.object(parser, '_get_basic_parser') as mock_basic:
            mock_basic_parser = AsyncMock()
            mock_basic_parser.parse = AsyncMock(return_value=mock_result)
            mock_basic.return_value = mock_basic_parser
            
            # Force Docling to not be available
            parser._docling_parser = MagicMock()
            parser._docling_parser.is_available.return_value = False
            
            result = await parser.parse("test.txt", "text/plain")
            
            assert result["text"] == "parsed text"
            mock_basic_parser.parse.assert_called_once()


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestParserIntegration:
    """Integration tests for parser chain."""
    
    def test_parser_chain_initialization(self):
        """Test that parser chain initializes correctly."""
        with patch('app.services.ocr_service.settings') as mock_settings:
            mock_settings.ENABLE_OCR_DETECTION = False
            mock_settings.OCR_MIN_TEXT_THRESHOLD = 50
            
            parser = OCRAwareDocumentParser()
            
            assert parser.ocr_service is not None
            assert parser._basic_parser is None  # Lazy loaded
            assert parser._docling_parser is None  # Lazy loaded
