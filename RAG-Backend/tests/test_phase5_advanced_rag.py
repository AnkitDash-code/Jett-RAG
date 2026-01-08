"""
Tests for Phase 5: Advanced RAG Services.
"""
import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.hierarchical_retrieval_service import HierarchicalRetrievalService
from app.services.auto_retry_service import AutoRetryService, AutoRetryResult
from app.services.community_detection_service import CommunityDetectionService
from app.services.stream_cancellation_service import StreamCancellationService, StreamContext


# ============================================================================
# Hierarchical Retrieval Tests
# ============================================================================

class TestHierarchicalRetrievalService:
    """Tests for hierarchical chunk retrieval."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        return db
    
    @pytest.fixture
    def service(self, mock_db):
        """Create service with hierarchical chunking enabled."""
        with patch('app.services.hierarchical_retrieval_service.settings') as mock_settings:
            mock_settings.ENABLE_HIERARCHICAL_CHUNKING = True
            return HierarchicalRetrievalService(mock_db)
    
    @pytest.mark.asyncio
    async def test_expand_to_parents_empty_input(self, service):
        """Test expanding with empty chunk list."""
        result = await service.expand_to_parents([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_expand_to_children_empty_input(self, service):
        """Test expanding with empty chunk list."""
        result = await service.expand_to_children([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_expand_siblings_empty_input(self, service):
        """Test expanding with empty chunk list."""
        result = await service.expand_siblings([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_disabled_service(self, mock_db):
        """Test that disabled service returns empty results."""
        with patch('app.services.hierarchical_retrieval_service.settings') as mock_settings:
            mock_settings.ENABLE_HIERARCHICAL_CHUNKING = False
            service = HierarchicalRetrievalService(mock_db)
            
            result = await service.expand_to_parents([uuid.uuid4()])
            assert result == []


# ============================================================================
# Auto-Retry Service Tests
# ============================================================================

class TestAutoRetryService:
    """Tests for auto-retry with self-reflection."""
    
    @pytest.fixture
    def service(self):
        """Create auto-retry service."""
        with patch('app.services.auto_retry_service.settings') as mock_settings:
            mock_settings.ENABLE_AUTO_RETRY = True
            mock_settings.MAX_RETRY_ATTEMPTS = 2
            mock_settings.RETRY_RELEVANCE_THRESHOLD = 0.4
            
            with patch('app.services.auto_retry_service.get_relevance_grader'):
                with patch('app.services.auto_retry_service.get_query_expansion_service'):
                    with patch('app.services.auto_retry_service.get_utility_llm'):
                        return AutoRetryService()
    
    @pytest.mark.asyncio
    async def test_should_retry_no_chunks(self, service):
        """Test that empty chunks triggers retry."""
        result = await service.should_retry("test query", [])
        assert result is True
    
    @pytest.mark.asyncio
    async def test_should_retry_disabled(self):
        """Test that disabled service doesn't retry."""
        with patch('app.services.auto_retry_service.settings') as mock_settings:
            mock_settings.ENABLE_AUTO_RETRY = False
            mock_settings.MAX_RETRY_ATTEMPTS = 2
            mock_settings.RETRY_RELEVANCE_THRESHOLD = 0.4
            
            with patch('app.services.auto_retry_service.get_relevance_grader'):
                with patch('app.services.auto_retry_service.get_query_expansion_service'):
                    with patch('app.services.auto_retry_service.get_utility_llm'):
                        service = AutoRetryService()
            
            result = await service.should_retry("test query", ["chunk1"])
            assert result is False
    
    def test_auto_retry_result_structure(self):
        """Test AutoRetryResult dataclass."""
        result = AutoRetryResult(
            final_query="reformulated query",
            original_query="original query",
        )
        
        assert result.final_query == "reformulated query"
        assert result.original_query == "original query"
        assert result.used_retry is False
        assert result.total_attempts == 1
        assert result.attempts == []


# ============================================================================
# Community Detection Tests
# ============================================================================

class TestCommunityDetectionService:
    """Tests for community detection algorithms."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        return db
    
    @pytest.fixture
    def service(self, mock_db):
        """Create community detection service."""
        with patch('app.services.community_detection_service.settings') as mock_settings:
            mock_settings.COMMUNITY_DETECTION_ALGORITHM = 'louvain'
            mock_settings.LOUVAIN_RESOLUTION = 1.0
            mock_settings.MIN_COMMUNITY_SIZE = 2
            mock_settings.ENABLE_COMMUNITY_SUMMARIES = False
            return CommunityDetectionService(mock_db)
    
    def test_connected_components_simple(self, service):
        """Test connected components on simple graph."""
        # Simple graph: 0-1-2, 3-4
        graph = {
            0: {1},
            1: {0, 2},
            2: {1},
            3: {4},
            4: {3},
        }
        
        communities = service._connected_components_detect(graph)
        
        assert len(communities) == 2
        # Each component should be complete
        for community in communities:
            if 0 in community:
                assert community == {0, 1, 2}
            else:
                assert community == {3, 4}
    
    def test_label_propagation_simple(self, service):
        """Test label propagation on simple graph."""
        # Fully connected triangle
        graph = {
            0: {1, 2},
            1: {0, 2},
            2: {0, 1},
        }
        
        communities = service._label_propagation_detect(graph)
        
        # Should find one community with all nodes
        assert len(communities) >= 1
        total_nodes = sum(len(c) for c in communities)
        assert total_nodes == 3
    
    def test_custom_louvain_simple(self, service):
        """Test custom Louvain implementation."""
        # Two cliques connected by one edge
        graph = {
            0: {1, 2},
            1: {0, 2},
            2: {0, 1, 3},
            3: {2, 4, 5},
            4: {3, 5},
            5: {3, 4},
        }
        
        communities, modularity = service._custom_louvain(graph)
        
        assert len(communities) >= 1
        assert modularity is not None
        # All nodes should be assigned
        all_nodes = set()
        for c in communities:
            all_nodes.update(c)
        assert all_nodes == {0, 1, 2, 3, 4, 5}
    
    def test_modularity_calculation(self, service):
        """Test modularity calculation."""
        graph = {
            0: {1},
            1: {0},
        }
        communities = {0: {0, 1}}
        
        modularity = service._calculate_modularity(graph, communities, m=1.0)
        assert isinstance(modularity, float)
    
    def test_empty_graph(self, service):
        """Test handling empty graph."""
        graph = {}
        
        communities = service._connected_components_detect(graph)
        assert communities == []


# ============================================================================
# Stream Cancellation Tests
# ============================================================================

class TestStreamCancellationService:
    """Tests for stream cancellation."""
    
    @pytest.fixture
    def service(self):
        """Create fresh stream cancellation service."""
        # Reset singleton for testing
        StreamCancellationService._instance = None
        with patch('app.services.stream_cancellation_service.settings') as mock_settings:
            mock_settings.STREAM_TIMEOUT_SECONDS = 120
            return StreamCancellationService()
    
    @pytest.mark.asyncio
    async def test_register_stream(self, service):
        """Test registering a new stream."""
        user_id = uuid.uuid4()
        
        stream_id = await service.register(user_id)
        
        assert stream_id is not None
        assert stream_id in service._active_streams
    
    @pytest.mark.asyncio
    async def test_unregister_stream(self, service):
        """Test unregistering a stream."""
        user_id = uuid.uuid4()
        
        stream_id = await service.register(user_id)
        await service.unregister(stream_id)
        
        assert stream_id not in service._active_streams
    
    @pytest.mark.asyncio
    async def test_cancel_stream(self, service):
        """Test cancelling a stream."""
        user_id = uuid.uuid4()
        
        stream_id = await service.register(user_id)
        cancelled = await service.cancel(stream_id, "Test cancellation")
        
        assert cancelled is True
        assert service.is_cancelled(stream_id) is True
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_stream(self, service):
        """Test cancelling non-existent stream."""
        cancelled = await service.cancel("nonexistent-id")
        assert cancelled is False
    
    @pytest.mark.asyncio
    async def test_cancel_user_streams(self, service):
        """Test cancelling all user streams."""
        user_id = uuid.uuid4()
        
        # Register multiple streams
        stream_ids = []
        for _ in range(3):
            stream_id = await service.register(user_id)
            stream_ids.append(stream_id)
        
        cancelled = await service.cancel_user_streams(user_id)
        
        assert cancelled == 3
        for stream_id in stream_ids:
            assert service.is_cancelled(stream_id) is True
    
    def test_is_cancelled_nonexistent(self, service):
        """Test checking cancellation of non-existent stream."""
        result = service.is_cancelled("nonexistent")
        assert result is True  # Non-existent streams are considered cancelled
    
    def test_update_progress(self, service):
        """Test updating stream progress."""
        # Register synchronously for this test
        stream = MagicMock()
        stream.tokens_generated = 0
        service._active_streams["test-id"] = stream
        
        service.update_progress("test-id", tokens=5)
        
        assert stream.tokens_generated == 5
    
    @pytest.mark.asyncio
    async def test_get_stream_info(self, service):
        """Test getting stream info."""
        user_id = uuid.uuid4()
        
        stream_id = await service.register(user_id, conversation_id="conv-123")
        info = service.get_stream_info(stream_id)
        
        assert info is not None
        assert info["stream_id"] == stream_id
        assert info["user_id"] == str(user_id)
        assert info["conversation_id"] == "conv-123"
        assert info["cancelled"] is False
    
    @pytest.mark.asyncio
    async def test_get_active_streams(self, service):
        """Test getting all active streams."""
        user_id = uuid.uuid4()
        
        await service.register(user_id)
        await service.register(user_id)
        
        streams = service.get_active_streams(user_id=user_id)
        
        assert len(streams) == 2
    
    @pytest.mark.asyncio
    async def test_cleanup_stale_streams(self, service):
        """Test cleaning up stale streams."""
        # This would require mocking time, simplified test
        cleaned = await service.cleanup_stale_streams()
        assert cleaned >= 0


class TestStreamContext:
    """Tests for StreamContext context manager."""
    
    @pytest.mark.asyncio
    async def test_stream_context_lifecycle(self):
        """Test stream context manager lifecycle."""
        # Reset singleton
        StreamCancellationService._instance = None
        
        with patch('app.services.stream_cancellation_service.settings') as mock_settings:
            mock_settings.STREAM_TIMEOUT_SECONDS = 120
            
            user_id = uuid.uuid4()
            
            async with StreamContext(user_id, "conv-123") as stream_id:
                assert stream_id is not None
                
                # Get service and check stream exists
                from app.services.stream_cancellation_service import get_stream_cancellation_service
                service = get_stream_cancellation_service()
                
                assert stream_id in service._active_streams
            
            # After context, stream should be unregistered
            assert stream_id not in service._active_streams
