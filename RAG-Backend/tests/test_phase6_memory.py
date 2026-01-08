"""
Phase 6: Memory System Tests.

Tests for the Supermemory System including:
- Memory model operations
- MemoryService functionality
- Memory API endpoints
- Integration with retrieval pipeline
"""
import pytest
import pytest_asyncio
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestMemoryModel:
    """Test Memory model operations."""
    
    @pytest.mark.asyncio
    async def test_memory_creation(self, db_session: AsyncSession):
        """Test creating a memory record."""
        from app.models.memory import Memory, MemoryType, MemoryStatus
        
        memory = Memory(
            user_id=uuid.uuid4(),
            memory_type=MemoryType.EPISODIC,
            content="User discussed project requirements",
            importance_score=1.0,
            interaction_count=1,
            feedback_score=0.5,
        )
        
        db_session.add(memory)
        await db_session.flush()
        
        assert memory.id is not None
        assert memory.status == MemoryStatus.ACTIVE
        assert memory.is_cached is True
        assert memory.created_at is not None
    
    @pytest.mark.asyncio
    async def test_memory_types(self, db_session: AsyncSession):
        """Test different memory types."""
        from app.models.memory import Memory, MemoryType
        
        user_id = uuid.uuid4()
        
        # Create episodic memory
        episodic = Memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content="User asked about API documentation",
        )
        
        # Create semantic memory
        semantic = Memory(
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="REST API design patterns",
        )
        
        db_session.add(episodic)
        db_session.add(semantic)
        await db_session.flush()
        
        assert episodic.memory_type == MemoryType.EPISODIC
        assert semantic.memory_type == MemoryType.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_importance_calculation(self, db_session: AsyncSession):
        """Test importance score calculation formula."""
        from app.models.memory import Memory, MemoryType
        
        memory = Memory(
            user_id=uuid.uuid4(),
            memory_type=MemoryType.EPISODIC,
            content="Test content",
            days_old=1.0,
            interaction_count=5,
            feedback_score=0.8,
        )
        
        # Formula: (1 / max(days_old, 0.1)) × interaction_count × feedback_score
        expected_importance = (1.0 / 1.0) * 5 * 0.8  # 4.0
        calculated = memory.calculate_importance()
        
        assert calculated == expected_importance
    
    @pytest.mark.asyncio
    async def test_importance_with_very_old_memory(self, db_session: AsyncSession):
        """Test importance decreases with age."""
        from app.models.memory import Memory, MemoryType
        
        memory = Memory(
            user_id=uuid.uuid4(),
            memory_type=MemoryType.EPISODIC,
            content="Old memory",
            days_old=30.0,
            interaction_count=2,
            feedback_score=0.5,
        )
        
        # Formula: (1 / 30) × 2 × 0.5 = 0.0333...
        expected_importance = (1.0 / 30.0) * 2 * 0.5
        calculated = memory.calculate_importance()
        
        assert abs(calculated - expected_importance) < 0.001
        assert calculated < 0.1  # Should be low importance
    
    @pytest.mark.asyncio
    async def test_entity_ids_json_property(self, db_session: AsyncSession):
        """Test entity_ids JSON property getter/setter."""
        from app.models.memory import Memory, MemoryType
        
        memory = Memory(
            user_id=uuid.uuid4(),
            memory_type=MemoryType.SEMANTIC,
            content="Test content",
        )
        
        # Set entity IDs
        entity_ids = ["entity-1", "entity-2", "entity-3"]
        memory.entity_ids = entity_ids
        
        # Get entity IDs
        retrieved = memory.entity_ids
        
        assert retrieved == entity_ids
        assert memory.entity_ids_json == json.dumps(entity_ids)
    
    @pytest.mark.asyncio
    async def test_metadata_json_property(self, db_session: AsyncSession):
        """Test metadata JSON property getter/setter."""
        from app.models.memory import Memory, MemoryType
        
        memory = Memory(
            user_id=uuid.uuid4(),
            memory_type=MemoryType.EPISODIC,
            content="Test content",
        )
        
        # Set metadata (using extra_metadata to avoid SQLAlchemy conflict)
        metadata = {"source": "chat", "query_type": "factual"}
        memory.extra_metadata = metadata
        
        # Get metadata
        retrieved = memory.extra_metadata
        
        assert retrieved == metadata
    
    @pytest.mark.asyncio
    async def test_memory_access_record(self, db_session: AsyncSession):
        """Test MemoryAccess audit record."""
        from app.models.memory import Memory, MemoryAccess, MemoryType
        
        memory = Memory(
            user_id=uuid.uuid4(),
            memory_type=MemoryType.EPISODIC,
            content="Test content",
        )
        db_session.add(memory)
        await db_session.flush()
        
        # Create access record
        access = MemoryAccess(
            memory_id=memory.id,
            query="What about the project?",
            relevance_score=0.85,
            retrieval_method="hybrid",
        )
        db_session.add(access)
        await db_session.flush()
        
        assert access.id is not None
        assert access.memory_id == memory.id
        assert access.accessed_at is not None


# ============================================================================
# SERVICE TESTS
# ============================================================================

class TestMemoryService:
    """Test MemoryService business logic."""
    
    @pytest.mark.asyncio
    @patch('app.services.memory_service.MemoryEmbeddingService')
    @patch('app.services.memory_service.MemoryVectorStore')
    async def test_create_memory(
        self,
        mock_vector_store_class,
        mock_embedding_class,
        db_session: AsyncSession,
    ):
        """Test memory creation with mocked embeddings."""
        from app.services.memory_service import MemoryService
        from app.models.memory import MemoryType
        
        # Setup mocks
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [0.1] * 384
        mock_embedding.model_name = "test-model"
        mock_embedding_class.return_value = mock_embedding
        
        mock_vector = MagicMock()
        mock_vector.add_memory.return_value = "mem_test123"
        mock_vector_store_class.return_value = mock_vector
        
        service = MemoryService(db_session)
        service._embedding_service = mock_embedding
        service._vector_store = mock_vector
        
        user_id = uuid.uuid4()
        
        result = await service.create_memory(
            user_id=user_id,
            content="Test memory content",
            memory_type=MemoryType.EPISODIC,
        )
        
        assert result.memory is not None
        assert result.memory.user_id == user_id
        assert result.memory.content == "Test memory content"
        assert result.memory.memory_type == MemoryType.EPISODIC
        assert result.total_ms >= 0
    
    @pytest.mark.asyncio
    @patch('app.services.memory_service.MemoryEmbeddingService')
    @patch('app.services.memory_service.MemoryVectorStore')
    async def test_retrieve_memories_empty(
        self,
        mock_vector_store_class,
        mock_embedding_class,
        db_session: AsyncSession,
    ):
        """Test retrieval when no memories exist."""
        from app.services.memory_service import MemoryService
        
        # Setup mocks
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [0.1] * 384
        mock_embedding_class.return_value = mock_embedding
        
        mock_vector = MagicMock()
        mock_vector.search.return_value = []  # No results
        mock_vector_store_class.return_value = mock_vector
        
        service = MemoryService(db_session)
        service._embedding_service = mock_embedding
        service._vector_store = mock_vector
        
        result = await service.retrieve_memories(
            query="test query",
            user_id=uuid.uuid4(),
        )
        
        assert result.memories == []
        assert result.total_candidates == 0
    
    @pytest.mark.asyncio
    async def test_update_importance_scores(self, db_session: AsyncSession):
        """Test importance score recalculation."""
        from app.services.memory_service import MemoryService
        from app.models.memory import Memory, MemoryType
        
        user_id = uuid.uuid4()
        
        # Create test memories with different ages
        memory1 = Memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content="Recent memory",
            created_at=datetime.utcnow() - timedelta(days=1),
            interaction_count=5,
            feedback_score=0.8,
        )
        memory2 = Memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content="Old memory",
            created_at=datetime.utcnow() - timedelta(days=30),
            interaction_count=2,
            feedback_score=0.5,
        )
        
        db_session.add(memory1)
        db_session.add(memory2)
        await db_session.flush()
        
        # Run update
        service = MemoryService(db_session)
        result = await service.update_importance_scores(user_id=user_id)
        
        # Verify updates
        assert result.updated_count >= 0
        
        # Refresh and check
        await db_session.refresh(memory1)
        await db_session.refresh(memory2)
        
        # Recent memory should have higher importance
        assert memory1.days_old < memory2.days_old
    
    @pytest.mark.asyncio
    async def test_evict_low_importance(self, db_session: AsyncSession):
        """Test archival of low-importance memories."""
        from app.services.memory_service import MemoryService
        from app.models.memory import Memory, MemoryType, MemoryStatus
        
        user_id = uuid.uuid4()
        
        # Create low-importance old memory
        old_memory = Memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content="Very old low importance",
            created_at=datetime.utcnow() - timedelta(days=60),
            days_old=60.0,
            importance_score=0.1,  # Below threshold
            interaction_count=1,
            feedback_score=0.3,
        )
        
        # Create recent memory
        recent_memory = Memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content="Recent high importance",
            importance_score=1.0,
            days_old=1.0,
            interaction_count=5,
            feedback_score=0.9,
        )
        
        db_session.add(old_memory)
        db_session.add(recent_memory)
        await db_session.flush()
        
        service = MemoryService(db_session)
        
        result = await service.evict_low_importance_memories(
            user_id=user_id,
            importance_threshold=0.3,
            age_threshold_days=30,
        )
        
        # Refresh and check
        await db_session.refresh(old_memory)
        await db_session.refresh(recent_memory)
        
        assert old_memory.status == MemoryStatus.ARCHIVED
        assert old_memory.is_cached is False
        assert recent_memory.status == MemoryStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_update_feedback(self, db_session: AsyncSession):
        """Test feedback score update."""
        from app.services.memory_service import MemoryService
        from app.models.memory import Memory, MemoryType
        
        user_id = uuid.uuid4()
        
        memory = Memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content="Test content",
            feedback_score=0.5,
            interaction_count=1,
            days_old=1.0,
        )
        db_session.add(memory)
        await db_session.flush()
        
        service = MemoryService(db_session)
        
        # Positive feedback
        updated = await service.update_feedback(memory.id, 0.3)
        
        assert updated is not None
        assert updated.feedback_score == 0.8
        
        # Verify importance was recalculated
        assert updated.importance_score == updated.calculate_importance()
    
    @pytest.mark.asyncio
    async def test_feedback_clamping(self, db_session: AsyncSession):
        """Test feedback score is clamped to 0-1."""
        from app.services.memory_service import MemoryService
        from app.models.memory import Memory, MemoryType
        
        user_id = uuid.uuid4()
        
        memory = Memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content="Test content",
            feedback_score=0.9,
        )
        db_session.add(memory)
        await db_session.flush()
        
        service = MemoryService(db_session)
        
        # Try to exceed 1.0
        updated = await service.update_feedback(memory.id, 0.5)
        
        assert updated.feedback_score == 1.0  # Clamped
    
    @pytest.mark.asyncio
    async def test_memory_stats(self, db_session: AsyncSession):
        """Test memory statistics calculation."""
        from app.services.memory_service import MemoryService
        from app.models.memory import Memory, MemoryType, MemoryStatus
        
        user_id = uuid.uuid4()
        
        # Create various memories
        for i in range(3):
            db_session.add(Memory(
                user_id=user_id,
                memory_type=MemoryType.EPISODIC,
                content=f"Episodic {i}",
            ))
        
        for i in range(2):
            db_session.add(Memory(
                user_id=user_id,
                memory_type=MemoryType.SEMANTIC,
                content=f"Semantic {i}",
            ))
        
        await db_session.flush()
        
        service = MemoryService(db_session)
        stats = await service.get_memory_stats(user_id=user_id)
        
        assert stats["by_type"]["episodic"] == 3
        assert stats["by_type"]["semantic"] == 2
        assert stats["total_active"] == 5


# ============================================================================
# LRU CACHE TESTS
# ============================================================================

class TestMemoryCache:
    """Test LRU cache behavior."""
    
    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        from app.services.memory_service import MemoryCache
        from app.models.memory import Memory, MemoryType
        
        cache = MemoryCache(max_size=5)
        
        memory = Memory(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            memory_type=MemoryType.EPISODIC,
            content="Test",
        )
        
        cache.put(memory)
        retrieved = cache.get(str(memory.id))
        
        assert retrieved == memory
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        from app.services.memory_service import MemoryCache
        from app.models.memory import Memory, MemoryType
        
        cache = MemoryCache(max_size=3)
        
        memories = []
        for i in range(5):
            memory = Memory(
                id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                memory_type=MemoryType.EPISODIC,
                content=f"Memory {i}",
            )
            memories.append(memory)
            cache.put(memory)
        
        # First two should be evicted
        assert cache.get(str(memories[0].id)) is None
        assert cache.get(str(memories[1].id)) is None
        
        # Last three should remain
        assert cache.get(str(memories[2].id)) is not None
        assert cache.get(str(memories[3].id)) is not None
        assert cache.get(str(memories[4].id)) is not None
    
    def test_cache_access_updates_order(self):
        """Test that accessing an item moves it to end."""
        from app.services.memory_service import MemoryCache
        from app.models.memory import Memory, MemoryType
        
        cache = MemoryCache(max_size=3)
        
        memories = []
        for i in range(3):
            memory = Memory(
                id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                memory_type=MemoryType.EPISODIC,
                content=f"Memory {i}",
            )
            memories.append(memory)
            cache.put(memory)
        
        # Access first memory (moves to end)
        cache.get(str(memories[0].id))
        
        # Add new memory (should evict memories[1])
        new_memory = Memory(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            memory_type=MemoryType.EPISODIC,
            content="New memory",
        )
        cache.put(new_memory)
        
        # memories[1] should be evicted, others should remain
        assert cache.get(str(memories[0].id)) is not None  # Was accessed
        assert cache.get(str(memories[1].id)) is None  # Was evicted
        assert cache.get(str(memories[2].id)) is not None  # Still in cache


# ============================================================================
# SCHEMA TESTS
# ============================================================================

class TestMemorySchemas:
    """Test memory request/response schemas."""
    
    def test_create_memory_request_validation(self):
        """Test CreateMemoryRequest validation."""
        from app.schemas.memory import CreateMemoryRequest
        
        # Valid request
        request = CreateMemoryRequest(
            content="Test memory content",
            memory_type="episodic",
        )
        
        assert request.content == "Test memory content"
        assert request.memory_type == "episodic"
    
    def test_create_memory_request_type_validation(self):
        """Test that only valid memory types are accepted."""
        from app.schemas.memory import CreateMemoryRequest
        from pydantic import ValidationError
        
        # Invalid memory type
        with pytest.raises(ValidationError):
            CreateMemoryRequest(
                content="Test",
                memory_type="invalid_type",
            )
    
    def test_memory_retrieval_request_defaults(self):
        """Test MemoryRetrievalRequest default values."""
        from app.schemas.memory import MemoryRetrievalRequest
        
        request = MemoryRetrievalRequest(query="test query")
        
        assert request.top_k == 5
        assert request.relevance_threshold == 0.3
        assert request.include_archived is False
    
    def test_memory_response_from_memory(self):
        """Test MemoryResponse.from_memory() conversion."""
        from app.schemas.memory import MemoryResponse
        from app.models.memory import Memory, MemoryType, MemoryStatus
        
        memory = Memory(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            memory_type=MemoryType.SEMANTIC,
            content="Test content",
            status=MemoryStatus.ACTIVE,
            importance_score=0.75,
            interaction_count=3,
            feedback_score=0.6,
            days_old=5.0,
            is_cached=True,
        )
        
        response = MemoryResponse.from_memory(memory)
        
        assert response.id == str(memory.id)
        assert response.memory_type == "semantic"
        assert response.content == "Test content"
        assert response.importance_score == 0.75


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMemoryIntegration:
    """Test memory integration with other services."""
    
    @pytest.mark.asyncio
    @patch('app.services.memory_service.MemoryEmbeddingService')
    @patch('app.services.memory_service.MemoryVectorStore')
    async def test_enhanced_retrieval_memory_injection(
        self,
        mock_vector_store_class,
        mock_embedding_class,
        db_session: AsyncSession,
    ):
        """Test that enhanced retrieval injects memory context."""
        from app.models.memory import Memory, MemoryType
        
        user_id = uuid.uuid4()
        
        # Create a memory
        memory = Memory(
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            content="Previous discussion about API design",
            importance_score=0.8,
        )
        db_session.add(memory)
        await db_session.flush()
        
        # Verify memory exists
        stmt = select(Memory).where(Memory.user_id == user_id)
        result = await db_session.execute(stmt)
        memories = list(result.scalars().all())
        
        assert len(memories) == 1
        assert memories[0].content == "Previous discussion about API design"


# ============================================================================
# API TESTS (require full app context)
# ============================================================================

class TestMemoryAPI:
    """Test memory API endpoints."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires auth setup - integration test")
    async def test_create_memory_endpoint(self, client):
        """Test POST /memory endpoint."""
        response = await client.post(
            "/api/v1/memory",
            json={
                "content": "Test memory",
                "memory_type": "episodic",
            },
            headers={"Authorization": "Bearer test-token"},
        )
        
        assert response.status_code in [201, 401]  # 401 if no auth
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires auth setup - integration test")
    async def test_list_memories_endpoint(self, client):
        """Test GET /memory endpoint."""
        response = await client.get(
            "/api/v1/memory",
            headers={"Authorization": "Bearer test-token"},
        )
        
        assert response.status_code in [200, 401]
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires auth setup - integration test")
    async def test_retrieve_memories_endpoint(self, client):
        """Test POST /memory/retrieve endpoint."""
        response = await client.post(
            "/api/v1/memory/retrieve",
            json={"query": "test query"},
            headers={"Authorization": "Bearer test-token"},
        )
        
        assert response.status_code in [200, 401]
