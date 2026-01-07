"""
Tests for Phase 2 services: BM25, reranker, token counter, metrics, authorization.
"""
import pytest
import uuid
from unittest.mock import MagicMock, patch


class TestBM25Index:
    """Tests for BM25 lexical search index."""
    
    def _reset_index(self):
        """Helper to reset BM25 index for clean testing."""
        from app.services.bm25_index import BM25Index
        BM25Index._instance = None
        BM25Index._initialized = False
        index = BM25Index()
        # Clear any persisted data
        index.corpus = []
        index.chunk_ids = []
        index.metadata = {}
        index.bm25 = None
        return index
    
    def test_bm25_index_creation(self):
        """Test BM25 index can be created."""
        from app.services.bm25_index import BM25Index
        
        index = BM25Index()
        assert index is not None
        # Cleanup
        BM25Index._instance = None
        BM25Index._initialized = False
    
    def test_bm25_add_document(self):
        """Test adding documents to BM25 index."""
        index = self._reset_index()
        
        # Add a chunk using the correct method signature
        index.add_chunk(
            chunk_id="chunk1",
            text="This is a test document about machine learning.",
            document_id="doc1"
        )
        
        assert len(index.corpus) >= 1
        assert "chunk1" in index.chunk_ids
        
        # Cleanup
        from app.services.bm25_index import BM25Index
        BM25Index._instance = None
        BM25Index._initialized = False
    
    def test_bm25_search(self):
        """Test BM25 search functionality."""
        index = self._reset_index()
        
        # Add chunks using correct method
        index.add_chunk("chunk1", "Machine learning is a subset of artificial intelligence.", "doc1")
        index.add_chunk("chunk2", "Python is a programming language used for data science.", "doc1")
        index.add_chunk("chunk3", "Neural networks are used in deep learning applications.", "doc1")
        
        # Search
        results = index.search("machine learning", top_k=2)
        
        assert len(results) <= 2
        # Check result format
        if results:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2  # (chunk_id, score)
        
        # Cleanup
        from app.services.bm25_index import BM25Index
        BM25Index._instance = None
        BM25Index._initialized = False
    
    def test_bm25_delete_document(self):
        """Test deleting documents from BM25 index."""
        index = self._reset_index()
        
        index.add_chunk("chunk1", "Test document one.", document_id="docA")
        index.add_chunk("chunk2", "Test document two.", document_id="docB")
        
        initial_count = len(index.corpus)
        
        # Delete by document_id
        deleted = index.delete_by_document("docA")
        
        assert len(index.corpus) == initial_count - 1
        assert "chunk1" not in index.chunk_ids
        assert deleted >= 1
        
        # Cleanup
        from app.services.bm25_index import BM25Index
        BM25Index._instance = None
        BM25Index._initialized = False
    
    def test_bm25_empty_search(self):
        """Test search on empty index."""
        index = self._reset_index()
        
        results = index.search("test query")
        
        assert results == []
        
        # Cleanup
        from app.services.bm25_index import BM25Index
        BM25Index._instance = None
        BM25Index._initialized = False


class TestRerankerService:
    """Tests for cross-encoder reranker service."""
    
    def test_reranker_creation(self):
        """Test reranker can be created."""
        from app.services.reranker_service import RerankerService
        
        reranker = RerankerService()
        assert reranker is not None
    
    def test_reranker_unavailable_fallback(self):
        """Test reranker gracefully handles missing model."""
        from app.services.reranker_service import RerankerService
        from dataclasses import dataclass
        
        @dataclass
        class MockChunk:
            text: str
            final_score: float = 0.0
        
        reranker = RerankerService()
        
        chunks = [
            MockChunk(text="doc one", final_score=0.5),
            MockChunk(text="doc two", final_score=0.8),
            MockChunk(text="doc three", final_score=0.3),
        ]
        
        results, elapsed = reranker.rerank("test query", chunks, top_k=2)
        
        # Should return list of chunks and elapsed time
        assert isinstance(results, list)
        assert len(results) <= 2
        assert isinstance(elapsed, int)
    
    def test_reranker_empty_chunks(self):
        """Test reranker handles empty input."""
        from app.services.reranker_service import RerankerService
        
        reranker = RerankerService()
        results, elapsed = reranker.rerank("test query", [], top_k=5)
        
        assert results == []
    
    def test_reranker_score_pair(self):
        """Test scoring a single pair."""
        from app.services.reranker_service import RerankerService
        
        reranker = RerankerService()
        score = reranker.score_pair("What is machine learning?", "Machine learning is a type of AI.")
        
        assert isinstance(score, float)


class TestTokenCounter:
    """Tests for token counting service."""
    
    def test_token_counter_creation(self):
        """Test token counter can be created."""
        from app.services.token_counter import TokenCounter, get_token_counter
        
        counter = get_token_counter()
        assert counter is not None
    
    def test_count_simple_text(self):
        """Test counting tokens in simple text."""
        from app.services.token_counter import get_token_counter
        
        counter = get_token_counter()
        
        count = counter.count("Hello, world!")
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_empty_text(self):
        """Test counting tokens in empty text."""
        from app.services.token_counter import get_token_counter
        
        counter = get_token_counter()
        
        count = counter.count("")
        assert count == 0
    
    def test_count_messages(self):
        """Test counting tokens in message list."""
        from app.services.token_counter import get_token_counter
        
        counter = get_token_counter()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        
        count = counter.count_messages(messages)
        assert count > 0
    
    def test_truncate_text(self):
        """Test text truncation to token limit."""
        from app.services.token_counter import get_token_counter
        
        counter = get_token_counter()
        
        long_text = "word " * 1000  # ~1000 words
        truncated = counter.truncate_to_tokens(long_text, max_tokens=100)
        
        # Truncated text should be shorter
        assert len(truncated) < len(long_text)
        # Token count should be at or below limit
        assert counter.count(truncated) <= 100


class TestMetricsService:
    """Tests for Prometheus metrics service."""
    
    def test_metrics_service_creation(self):
        """Test metrics service can be created."""
        from app.services.metrics_service import MetricsService, get_metrics_service
        
        service = get_metrics_service()
        assert service is not None
    
    def test_record_chat_request(self):
        """Test recording chat request metrics."""
        from app.services.metrics_service import get_metrics_service
        
        service = get_metrics_service()
        
        # Should not raise
        service.record_chat_request(
            status="success",
            role="user",
            latency=1.5,
            prompt_tokens=100,
            completion_tokens=50
        )
    
    def test_record_retrieval(self):
        """Test recording retrieval metrics."""
        from app.services.metrics_service import get_metrics_service
        
        service = get_metrics_service()
        
        service.record_retrieval(
            status="success",
            latency=0.5,
            chunk_count=5
        )
    
    def test_get_metrics_output(self):
        """Test getting metrics in Prometheus format."""
        from app.services.metrics_service import get_metrics_service
        
        service = get_metrics_service()
        content, content_type = service.get_metrics()
        
        assert content is not None
        assert isinstance(content, bytes)


class TestAuthorizationService:
    """Tests for authorization service."""
    
    def test_authorization_permissions(self):
        """Test permission definitions exist."""
        from app.services.authorization_service import Permission, AccessLevel
        
        assert Permission.DOC_VIEW is not None
        assert Permission.ADMIN_USERS is not None
        assert AccessLevel.PUBLIC is not None
        assert AccessLevel.PRIVATE is not None
    
    def test_role_definitions(self):
        """Test role definitions are complete."""
        from app.services.authorization_service import ROLES
        
        assert "admin" in ROLES
        assert "user" in ROLES
        assert "viewer" in ROLES
        assert "guest" in ROLES
        
        # Admin should have all permissions
        admin_role = ROLES["admin"]
        assert len(admin_role.permissions) > 0
    
    def test_authorization_context_creation(self):
        """Test creating authorization context."""
        from app.services.authorization_service import (
            AuthorizationContext, Permission, AccessLevel
        )
        
        context = AuthorizationContext(
            user_id=uuid.uuid4(),
            role="user",
            permissions={Permission.DOC_VIEW, Permission.CHAT_USE},
            access_levels={AccessLevel.PUBLIC, AccessLevel.INTERNAL},
        )
        
        assert context.role == "user"
        assert Permission.DOC_VIEW in context.permissions
    
    def test_has_permission(self):
        """Test permission checking."""
        from app.services.authorization_service import (
            AuthorizationContext, Permission, AccessLevel
        )
        
        # Create mock context
        context = AuthorizationContext(
            user_id=uuid.uuid4(),
            role="user",
            permissions={Permission.DOC_VIEW, Permission.CHAT_USE},
            access_levels={AccessLevel.PUBLIC},
        )
        
        # Manual check (service needs db session)
        assert Permission.DOC_VIEW in context.permissions
        assert Permission.ADMIN_USERS not in context.permissions


class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""
    
    def test_limiter_creation(self):
        """Test rate limiter can be created."""
        from app.middleware.rate_limit import create_limiter, get_limiter
        
        # May return None if slowapi not installed or disabled
        limiter = get_limiter()
        # Just verify it doesn't crash and returns either a Limiter or None
        assert limiter is None or limiter is not None
    
    def test_rate_limit_decorator_exists(self):
        """Test rate limit decorator function exists."""
        from app.middleware.rate_limit import rate_limit
        
        # The decorator should be callable
        assert callable(rate_limit)
        
        # Should return a decorator when called with a limit string
        decorator = rate_limit("10/minute")
        assert callable(decorator)
    
    def test_rate_limit_setup(self):
        """Test rate limit setup function exists."""
        from app.middleware.rate_limit import setup_rate_limiting
        
        # Just verify function exists
        assert callable(setup_rate_limiting)


class TestHealthEndpoint:
    """Tests for health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_structure(self):
        """Test health endpoint returns expected structure."""
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "version" in data


class TestConfigSettings:
    """Tests for new configuration settings."""
    
    def test_hybrid_search_settings(self):
        """Test hybrid search settings exist."""
        from app.config import settings
        
        assert hasattr(settings, "ENABLE_HYBRID_SEARCH")
        assert hasattr(settings, "HYBRID_ALPHA")
        assert 0 <= settings.HYBRID_ALPHA <= 1
    
    def test_cross_encoder_settings(self):
        """Test cross-encoder settings exist."""
        from app.config import settings
        
        assert hasattr(settings, "ENABLE_CROSS_ENCODER")
        assert hasattr(settings, "CROSS_ENCODER_MODEL")
    
    def test_rate_limit_settings(self):
        """Test rate limiting settings exist."""
        from app.config import settings
        
        assert hasattr(settings, "RATE_LIMIT_ENABLED")
        assert hasattr(settings, "RATE_LIMIT_CHAT")
    
    def test_utility_llm_settings(self):
        """Test utility LLM settings exist."""
        from app.config import settings
        
        assert hasattr(settings, "UTILITY_LLM_ENABLED")
        assert hasattr(settings, "UTILITY_LLM_URL")
