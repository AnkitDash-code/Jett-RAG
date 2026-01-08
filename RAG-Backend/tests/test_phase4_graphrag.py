"""
Phase 4: GraphRAG Tests

Comprehensive test suite for the GraphRAG implementation including:
- Entity extraction
- Graph store operations
- Permission inheritance
- Graph indexing
- Graph traversal
- Admin endpoints
- Hybrid scoring integration
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_user_id():
    return uuid4()


@pytest.fixture
def sample_tenant_id():
    return str(uuid4())


@pytest.fixture
def sample_document_id():
    return uuid4()


@pytest.fixture
def sample_chunk_id():
    return uuid4()


@pytest.fixture
def sample_folder_id():
    return uuid4()


@pytest.fixture
def sample_text():
    return """
    Dr. Sarah Chen leads the research team at MIT's AI Laboratory. 
    She collaborated with Professor James Wilson from Stanford University 
    on a groundbreaking paper about neural networks. The research was 
    funded by DARPA and published in Nature journal.
    """


@pytest.fixture
def sample_entities():
    return [
        {"name": "Dr. Sarah Chen", "type": "PERSON", "confidence": 0.95},
        {"name": "MIT", "type": "ORGANIZATION", "confidence": 0.92},
        {"name": "AI Laboratory", "type": "ORGANIZATION", "confidence": 0.88},
        {"name": "Professor James Wilson", "type": "PERSON", "confidence": 0.94},
        {"name": "Stanford University", "type": "ORGANIZATION", "confidence": 0.96},
        {"name": "DARPA", "type": "ORGANIZATION", "confidence": 0.97},
        {"name": "Nature", "type": "PUBLICATION", "confidence": 0.90},
    ]


@pytest.fixture
def sample_relationships():
    return [
        {"source": "Dr. Sarah Chen", "target": "MIT", "type": "WORKS_AT", "confidence": 0.9},
        {"source": "Dr. Sarah Chen", "target": "Professor James Wilson", "type": "COLLABORATES_WITH", "confidence": 0.85},
        {"source": "Professor James Wilson", "target": "Stanford University", "type": "WORKS_AT", "confidence": 0.92},
    ]


@pytest.fixture
def mock_db():
    """Create a mock AsyncSession for testing."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    return db


# ============================================================================
# Module-Level Function Tests
# ============================================================================

class TestLevenshteinDistance:
    """Tests for the Levenshtein distance function."""
    
    def test_same_string_returns_zero(self):
        """Same strings should have distance 0."""
        from app.services.entity_extraction_service import _levenshtein_distance
        
        assert _levenshtein_distance("test", "test") == 0
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("hello world", "hello world") == 0
    
    def test_single_character_difference(self):
        """Single character differences should return 1."""
        from app.services.entity_extraction_service import _levenshtein_distance
        
        assert _levenshtein_distance("test", "tests") == 1  # insertion
        assert _levenshtein_distance("tests", "test") == 1  # deletion
        assert _levenshtein_distance("test", "tast") == 1   # substitution
    
    def test_multiple_differences(self):
        """Multiple differences should accumulate correctly."""
        from app.services.entity_extraction_service import _levenshtein_distance
        
        assert _levenshtein_distance("cat", "dog") == 3
        assert _levenshtein_distance("kitten", "sitting") == 3
    
    def test_empty_strings(self):
        """Empty string comparisons."""
        from app.services.entity_extraction_service import _levenshtein_distance
        
        assert _levenshtein_distance("", "test") == 4
        assert _levenshtein_distance("test", "") == 4
    
    def test_similar_entity_names(self):
        """Test with entity-like names that might need fuzzy matching."""
        from app.services.entity_extraction_service import _levenshtein_distance
        
        # These should be close enough to match with threshold 2
        assert _levenshtein_distance("Dr. Sarah Chen", "Dr Sarah Chen") <= 2
        assert _levenshtein_distance("MIT", "M.I.T.") <= 3


# ============================================================================
# Graph Models Tests
# ============================================================================

class TestGraphModels:
    """Tests for graph model definitions."""
    
    def test_entity_type_enum(self):
        """Test EntityType enum values exist."""
        from app.models.graph import EntityType
        
        assert hasattr(EntityType, "PERSON")
        assert hasattr(EntityType, "ORGANIZATION")
        assert hasattr(EntityType, "LOCATION")
        assert hasattr(EntityType, "CONCEPT")
        assert hasattr(EntityType, "EVENT")
    
    def test_relationship_type_enum(self):
        """Test RelationshipType enum values exist."""
        from app.models.graph import RelationshipType
        
        assert hasattr(RelationshipType, "WORKS_FOR")
        assert hasattr(RelationshipType, "LOCATED_IN")
        assert hasattr(RelationshipType, "RELATED_TO")
        assert hasattr(RelationshipType, "PART_OF")
    
    def test_permission_type_enum(self):
        """Test PermissionType enum values exist."""
        from app.models.graph import PermissionType
        
        assert hasattr(PermissionType, "VIEW")
        assert hasattr(PermissionType, "EDIT")
        assert hasattr(PermissionType, "ADMIN")
    
    def test_entity_model_fields(self):
        """Test Entity model has required fields."""
        from app.models.graph import Entity
        
        # Check field names exist in the model
        field_names = [c.key for c in Entity.__table__.columns]
        assert "id" in field_names
        assert "name" in field_names
        assert "normalized_name" in field_names
        assert "entity_type" in field_names
        assert "tenant_id" in field_names
    
    def test_entity_relationship_model_fields(self):
        """Test EntityRelationship model has required fields."""
        from app.models.graph import EntityRelationship
        
        field_names = [c.key for c in EntityRelationship.__table__.columns]
        assert "source_entity_id" in field_names
        assert "target_entity_id" in field_names
        assert "relationship_type" in field_names
    
    def test_community_model_fields(self):
        """Test Community model has required fields."""
        from app.models.graph import Community
        
        field_names = [c.key for c in Community.__table__.columns]
        assert "id" in field_names
        assert "name" in field_names
        assert "tenant_id" in field_names


# ============================================================================
# GraphStoreService Tests (with mock DB)
# ============================================================================

class TestGraphStoreService:
    """Tests for the GraphStoreService."""
    
    def test_service_initialization(self, mock_db):
        """Test that GraphStoreService initializes correctly."""
        from app.services.graph_store import GraphStoreService
        
        service = GraphStoreService(mock_db)
        
        assert service is not None
        assert service.db == mock_db
    
    def test_is_available_property(self, mock_db):
        """Test the is_available property."""
        from app.services.graph_store import GraphStoreService, RUSTWORKX_AVAILABLE
        
        service = GraphStoreService(mock_db)
        
        # Should depend on rustworkx availability and settings
        if RUSTWORKX_AVAILABLE:
            # Result depends on ENABLE_GRAPH_RAG setting
            assert isinstance(service.is_available, bool)
        else:
            assert service.is_available == False
    
    @pytest.mark.asyncio
    async def test_load_graph_empty_database(self, mock_db):
        """Test loading graph from an empty database."""
        from app.services.graph_store import GraphStoreService, RUSTWORKX_AVAILABLE
        
        if not RUSTWORKX_AVAILABLE:
            pytest.skip("rustworkx not installed")
        
        service = GraphStoreService(mock_db)
        
        await service.load_graph()
        
        # Graph should be created even if empty
        assert service._graph is not None
    
    def test_graph_node_dataclass(self):
        """Test GraphNode dataclass creation."""
        from app.services.graph_store import GraphNode
        from app.models.graph import EntityType
        
        node = GraphNode(
            entity_id=uuid4(),
            name="Test Entity",
            normalized_name="test entity",
            entity_type=EntityType.PERSON,
            tenant_id=None,
            access_level="public",
        )
        
        assert node.name == "Test Entity"
        assert node.centrality == 0.0  # default
        
        # Test to_dict
        d = node.to_dict()
        assert "entity_id" in d
        assert d["name"] == "Test Entity"
    
    def test_rbac_params_dataclass(self):
        """Test RBACParams dataclass creation."""
        from app.services.graph_store import RBACParams
        
        user_id = uuid4()
        tenant_id = str(uuid4())
        
        params = RBACParams(
            user_id=user_id,
            tenant_ids={tenant_id},
            access_levels={"public", "internal"},
        )
        
        assert params.user_id == user_id
        assert tenant_id in params.tenant_ids
        assert "public" in params.access_levels


# ============================================================================
# EntityExtractionService Tests
# ============================================================================

class TestEntityExtractionService:
    """Tests for the EntityExtractionService."""
    
    def test_service_initialization(self, mock_db):
        """Test that EntityExtractionService initializes correctly."""
        from app.services.entity_extraction_service import EntityExtractionService
        
        service = EntityExtractionService(mock_db)
        
        assert service is not None
        assert service.db == mock_db
    
    def test_extracted_entity_dataclass(self):
        """Test ExtractedEntity dataclass."""
        from app.services.entity_extraction_service import ExtractedEntity
        from app.models.graph import EntityType
        
        entity = ExtractedEntity(
            name="Test Entity",
            normalized_name="test entity",
            entity_type=EntityType.PERSON,
            confidence=0.95,
            mention_text="Test Entity mentioned here",
        )
        
        assert entity.name == "Test Entity"
        assert entity.confidence == 0.95
        assert entity.start_pos is None  # optional
    
    def test_extraction_result_dataclass(self):
        """Test ExtractionResult dataclass."""
        from app.services.entity_extraction_service import ExtractionResult
        
        result = ExtractionResult(
            chunk_id=uuid4(),
            entities=[],
            relationships=[],
            processing_ms=100,
            extraction_method="llm",
        )
        
        assert result.processing_ms == 100
        assert result.extraction_method == "llm"


# ============================================================================
# PermissionInheritanceService Tests
# ============================================================================

class TestPermissionInheritanceService:
    """Tests for the PermissionInheritanceService."""
    
    def test_service_initialization(self, mock_db):
        """Test that PermissionInheritanceService initializes correctly."""
        from app.services.permission_inheritance_service import PermissionInheritanceService
        
        service = PermissionInheritanceService(mock_db)
        
        assert service is not None
    
    def test_resource_type_enum(self):
        """Test ResourceType enum values exist."""
        from app.services.permission_inheritance_service import ResourceType
        
        assert hasattr(ResourceType, "TENANT")
        assert hasattr(ResourceType, "FOLDER")
        assert hasattr(ResourceType, "DOCUMENT")
        assert hasattr(ResourceType, "CHUNK")
        assert hasattr(ResourceType, "ENTITY")
    
    def test_effective_permissions_dataclass(self):
        """Test EffectivePermissions dataclass."""
        from app.services.permission_inheritance_service import EffectivePermissions, ResourceType
        from app.models.graph import PermissionType
        
        perms = EffectivePermissions(
            user_id=uuid4(),
            resource_type=ResourceType.DOCUMENT,
            resource_id=str(uuid4()),
            permissions={PermissionType.VIEW},
            is_owner=False,
        )
        
        assert PermissionType.VIEW in perms.permissions
        assert perms.is_owner == False


# ============================================================================
# GraphIndexService Tests
# ============================================================================

class TestGraphIndexService:
    """Tests for the GraphIndexService."""
    
    def test_service_initialization(self, mock_db):
        """Test that GraphIndexService initializes correctly."""
        from app.services.graph_index_service import GraphIndexService
        
        service = GraphIndexService(mock_db)
        
        assert service is not None
    
    def test_task_status_enum(self):
        """Test TaskStatus enum values exist."""
        from app.services.graph_index_service import TaskStatus
        
        assert hasattr(TaskStatus, "PENDING")
        assert hasattr(TaskStatus, "PROCESSING")
        assert hasattr(TaskStatus, "COMPLETED")
        assert hasattr(TaskStatus, "FAILED")
    
    def test_indexing_stats_dataclass(self):
        """Test IndexingStats dataclass."""
        from app.services.graph_index_service import IndexingStats
        
        stats = IndexingStats(
            chunks_processed=10,
            entities_created=20,
            entities_linked=5,
            relationships_created=15,
            errors=2,
            processing_ms=100,
        )
        
        assert stats.chunks_processed == 10
        assert stats.entities_created == 20
        assert stats.processing_ms == 100


# ============================================================================
# GraphTraversalService Tests
# ============================================================================

class TestGraphTraversalService:
    """Tests for the GraphTraversalService."""
    
    def test_service_initialization(self, mock_db):
        """Test that GraphTraversalService initializes correctly."""
        from app.services.graph_traversal_service import GraphTraversalService
        
        service = GraphTraversalService(mock_db)
        
        assert service is not None
    
    def test_graph_chunk_dataclass(self):
        """Test GraphChunk dataclass."""
        from app.services.graph_traversal_service import GraphChunk
        
        chunk = GraphChunk(
            chunk_id=uuid4(),
            text="Sample text",
            document_id=uuid4(),
            graph_score=0.85,
            hop_distance=1,
            matched_entities=["Entity1", "Entity2"],
        )
        
        assert chunk.graph_score == 0.85
        assert chunk.hop_distance == 1
        assert len(chunk.matched_entities) == 2
    
    def test_graph_retrieval_result_dataclass(self):
        """Test GraphRetrievalResult dataclass."""
        from app.services.graph_traversal_service import GraphRetrievalResult
        
        result = GraphRetrievalResult(
            chunks=[],
            communities=[],
            query_entities=["Entity1"],
            entities_found=1,
            traversal_ms=50,
            method="k-hop",
        )
        
        assert result.traversal_ms == 50
        assert "Entity1" in result.query_entities


# ============================================================================
# Configuration Tests
# ============================================================================

class TestGraphRAGConfig:
    """Tests for GraphRAG configuration settings."""
    
    def test_config_settings_exist(self):
        """Test that GraphRAG settings exist in config."""
        from app.config import settings
        
        assert hasattr(settings, "ENABLE_GRAPH_RAG")
        assert hasattr(settings, "GRAPH_CACHE_TTL_SECONDS")
        assert hasattr(settings, "GRAPH_MAX_HOPS")
        assert hasattr(settings, "GRAPH_SCORE_WEIGHT")
        assert hasattr(settings, "GRAPH_ENTITY_FUZZY_THRESHOLD")
    
    def test_default_values_reasonable(self):
        """Test that default config values are reasonable."""
        from app.config import settings
        
        assert settings.GRAPH_CACHE_TTL_SECONDS >= 60  # At least 1 minute
        assert settings.GRAPH_MAX_HOPS >= 1
        assert 0 <= settings.GRAPH_SCORE_WEIGHT <= 1
        assert settings.GRAPH_ENTITY_FUZZY_THRESHOLD >= 0


# ============================================================================
# Hybrid Scoring Integration Tests
# ============================================================================

class TestHybridScoring:
    """Tests for hybrid scoring (vector + rerank + graph)."""
    
    def test_score_calculation_formula(self):
        """Test hybrid score calculation formula."""
        # Formula: 0.4*vector + 0.3*rerank + 0.3*graph
        vector_score = 0.8
        rerank_score = 0.7
        graph_score = 0.6
        
        expected = 0.4 * vector_score + 0.3 * rerank_score + 0.3 * graph_score
        
        assert expected == pytest.approx(0.71, rel=0.01)
    
    def test_score_without_graph(self):
        """Test scoring when graph is disabled."""
        vector_score = 0.8
        rerank_score = 0.7
        
        # When graph disabled: 0.6*vector + 0.4*rerank
        expected = 0.6 * vector_score + 0.4 * rerank_score
        
        assert expected == pytest.approx(0.76, rel=0.01)
    
    def test_enhanced_retrieval_result_has_graph_fields(self):
        """Test EnhancedRetrievalResult includes graph fields."""
        from app.services.enhanced_retrieval_service import EnhancedRetrievalResult
        
        result = EnhancedRetrievalResult(
            base_result=MagicMock(),
            graph_entities=["Test Entity"],
            graph_traversal_ms=50,
        )
        
        assert result.graph_entities == ["Test Entity"]
        assert result.graph_traversal_ms == 50
        assert hasattr(result, "graph_context")
        assert hasattr(result, "graph_communities")


# ============================================================================
# Import and Structure Tests
# ============================================================================

class TestModuleImports:
    """Tests that all modules can be imported correctly."""
    
    def test_import_graph_models(self):
        """Test importing graph models."""
        from app.models.graph import (
            Entity, EntityMention, EntityRelationship,
            Community, CommunityMembership, CommunityContext,
            Group, GroupMembership, FolderPermission,
            GraphProcessingTask, EntityType, RelationshipType,
            PermissionType, GroupRole
        )
        
        assert Entity is not None
        assert Community is not None
    
    def test_import_graph_store(self):
        """Test importing graph store service."""
        from app.services.graph_store import (
            GraphStoreService, GraphNode, GraphEdge,
            TraversalResult, RBACParams
        )
        
        assert GraphStoreService is not None
    
    def test_import_entity_extraction(self):
        """Test importing entity extraction service."""
        from app.services.entity_extraction_service import (
            EntityExtractionService, ExtractedEntity,
            ExtractedRelationship, ExtractionResult
        )
        
        assert EntityExtractionService is not None
    
    def test_import_permission_inheritance(self):
        """Test importing permission inheritance service."""
        from app.services.permission_inheritance_service import (
            PermissionInheritanceService, EffectivePermissions,
            ResourceType
        )
        
        assert PermissionInheritanceService is not None
    
    def test_import_graph_index(self):
        """Test importing graph index service."""
        from app.services.graph_index_service import (
            GraphIndexService, TaskStatus, IndexingStats
        )
        
        assert GraphIndexService is not None
    
    def test_import_graph_traversal(self):
        """Test importing graph traversal service."""
        from app.services.graph_traversal_service import (
            GraphTraversalService, GraphChunk, GraphRetrievalResult
        )
        
        assert GraphTraversalService is not None


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_entity_name_levenshtein(self):
        """Test Levenshtein with empty names."""
        from app.services.entity_extraction_service import _levenshtein_distance
        
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("a", "") == 1
    
    def test_unicode_entity_names(self):
        """Test with unicode characters in entity names."""
        from app.services.entity_extraction_service import _levenshtein_distance
        
        # Should handle unicode correctly
        assert _levenshtein_distance("北京", "北京") == 0
        assert _levenshtein_distance("北京", "上海") == 2
    
    def test_case_sensitivity(self):
        """Test case sensitivity in string matching."""
        from app.services.entity_extraction_service import _levenshtein_distance
        
        # These should be different (case matters for Levenshtein)
        assert _levenshtein_distance("MIT", "mit") > 0


# ============================================================================
# Integration Smoke Tests
# ============================================================================

class TestIntegrationSmoke:
    """Smoke tests to verify components work together."""
    
    @pytest.mark.asyncio
    async def test_graph_store_with_mock_db(self, mock_db):
        """Test GraphStoreService with mocked database."""
        from app.services.graph_store import GraphStoreService, RUSTWORKX_AVAILABLE
        
        if not RUSTWORKX_AVAILABLE:
            pytest.skip("rustworkx not installed")
        
        service = GraphStoreService(mock_db)
        loaded = await service.ensure_loaded()
        
        # Should succeed with empty mock data
        assert isinstance(loaded, bool)
    
    def test_all_services_instantiable(self, mock_db):
        """Test that all services can be instantiated with mock db."""
        from app.services.graph_store import GraphStoreService
        from app.services.entity_extraction_service import EntityExtractionService
        from app.services.permission_inheritance_service import PermissionInheritanceService
        from app.services.graph_index_service import GraphIndexService
        from app.services.graph_traversal_service import GraphTraversalService
        
        services = [
            GraphStoreService(mock_db),
            EntityExtractionService(mock_db),
            PermissionInheritanceService(mock_db),
            GraphIndexService(mock_db),
            GraphTraversalService(mock_db),
        ]
        
        for service in services:
            assert service is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
