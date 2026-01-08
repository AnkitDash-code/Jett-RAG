"""
Tests for Phase 3: Query Intelligence & Dual LLM services.

Tests:
- UtilityLLMClient
- QueryExpansionService
- QueryClassificationService
- RelevanceGrader
- ConversationContextService
- EnhancedRetrievalService
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid


# ===========================================================================
# Utility LLM Client Tests
# ===========================================================================

class TestUtilityLLMClient:
    """Tests for the UtilityLLMClient."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        from app.services import utility_llm_client
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
        yield
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
    
    def test_singleton_pattern(self):
        """Test that UtilityLLMClient is a singleton."""
        from app.services.utility_llm_client import UtilityLLMClient
        
        client1 = UtilityLLMClient()
        client2 = UtilityLLMClient()
        
        assert client1 is client2
    
    def test_is_available(self):
        """Test is_available reflects enabled state."""
        from app.services.utility_llm_client import get_utility_llm
        
        client = get_utility_llm()
        # Should be enabled by default in updated config
        assert client.is_available() == client.enabled
    
    def test_heuristic_classification(self):
        """Test fallback heuristic classification."""
        from app.services.utility_llm_client import get_utility_llm
        
        client = get_utility_llm()
        
        # Comparison query
        result = client._heuristic_classification("Compare Python vs Java")
        assert result["type"] == "COMPARISON"
        
        # Summary query
        result = client._heuristic_classification("Give me a summary of the document")
        assert result["type"] == "SUMMARY"
        
        # Troubleshooting query
        result = client._heuristic_classification("How to fix this error?")
        assert result["type"] == "TROUBLESHOOTING"
        
        # Entity query
        result = client._heuristic_classification("Who is the CEO?")
        assert result["type"] == "ENTITY"
        
        # Simple query defaults to FACT
        result = client._heuristic_classification("Hello")
        assert result["type"] == "FACT"
    
    def test_heuristic_complexity(self):
        """Test complexity detection."""
        from app.services.utility_llm_client import get_utility_llm
        
        client = get_utility_llm()
        
        # Simple query
        result = client._heuristic_classification("What is Python?")
        assert result["complexity"] == "simple"
        
        # Complex query (many words)
        long_query = "Can you explain in detail how machine learning algorithms work and compare different approaches"
        result = client._heuristic_classification(long_query)
        assert result["complexity"] in ["moderate", "complex"]


# ===========================================================================
# Query Expansion Service Tests
# ===========================================================================

class TestQueryExpansionService:
    """Tests for QueryExpansionService."""
    
    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singletons before each test."""
        from app.services import query_expansion_service, utility_llm_client
        query_expansion_service._expansion_service = None
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
        yield
        query_expansion_service._expansion_service = None
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
    
    @pytest.mark.asyncio
    async def test_heuristic_expansion(self):
        """Test heuristic query expansion."""
        from app.services.query_expansion_service import QueryExpansionService
        
        # Disable LLM to force heuristic
        with patch('app.services.utility_llm_client.settings') as mock_settings:
            mock_settings.UTILITY_LLM_ENABLED = False
            mock_settings.UTILITY_LLM_URL = "http://localhost:8080"
            mock_settings.UTILITY_LLM_MODEL = "test"
            mock_settings.UTILITY_LLM_MAX_TOKENS = 256
            mock_settings.UTILITY_LLM_TEMPERATURE = 0.2
            
            service = QueryExpansionService()
            result = await service.expand("What is machine learning?", num_variants=2)
        
        assert result.original == "What is machine learning?"
        assert len(result.variants) >= 1  # At least original
        assert result.method in ["llm", "heuristic"]
    
    def test_heuristic_keywords(self):
        """Test keyword extraction."""
        from app.services.query_expansion_service import QueryExpansionService
        
        service = QueryExpansionService()
        keywords = service._heuristic_keywords("What is machine learning in Python?")
        
        assert "machine" in keywords
        assert "learning" in keywords
        assert "python" in keywords
        # Stopwords should be excluded
        assert "what" not in keywords
        assert "is" not in keywords
    
    @pytest.mark.asyncio
    async def test_expand_for_hybrid_search(self):
        """Test expansion for hybrid search format."""
        from app.services.query_expansion_service import QueryExpansionService
        
        service = QueryExpansionService()
        result = await service.expand_for_hybrid_search("How does neural network work?")
        
        assert "semantic_queries" in result
        assert "keyword_queries" in result
        assert "original" in result
        assert result["original"] == "How does neural network work?"


# ===========================================================================
# Query Classification Service Tests
# ===========================================================================

class TestQueryClassificationService:
    """Tests for QueryClassificationService."""
    
    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singletons."""
        from app.services import query_classification_service, utility_llm_client
        query_classification_service._classification_service = None
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
        yield
        query_classification_service._classification_service = None
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
    
    @pytest.mark.asyncio
    async def test_classification_routing(self):
        """Test that classification generates proper routing."""
        from app.services.query_classification_service import QueryClassificationService
        
        service = QueryClassificationService()
        
        # Test comparison query routing
        result = await service.classify("Compare Python and Java")
        
        assert result.query_type.value == "COMPARISON"
        assert "num_chunks" in result.routing
        assert "use_hybrid" in result.routing
        assert result.routing["use_graph"] == True  # Comparisons benefit from graph
    
    @pytest.mark.asyncio
    async def test_default_classification(self):
        """Test default classification for empty query."""
        from app.services.query_classification_service import QueryClassificationService
        
        service = QueryClassificationService()
        result = await service.classify("")
        
        assert result.confidence == 0.0
        assert result.query_type.value == "FACT"
    
    @pytest.mark.asyncio
    async def test_get_routing_for_query(self):
        """Test convenience method for getting routing."""
        from app.services.query_classification_service import QueryClassificationService
        
        service = QueryClassificationService()
        routing = await service.get_routing_for_query("Summarize the document")
        
        assert "query_type" in routing
        assert "complexity" in routing
        assert routing["query_type"] == "SUMMARY"


# ===========================================================================
# Relevance Grader Tests
# ===========================================================================

class TestRelevanceGrader:
    """Tests for RelevanceGrader."""
    
    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singletons."""
        from app.services import relevance_grader, utility_llm_client
        relevance_grader._relevance_grader = None
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
        yield
        relevance_grader._relevance_grader = None
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
    
    @pytest.mark.asyncio
    async def test_grade_empty_chunks(self):
        """Test grading with no chunks."""
        from app.services.relevance_grader import RelevanceGrader
        
        grader = RelevanceGrader()
        result = await grader.grade("test query", [])
        
        assert result.is_relevant == False
        assert result.should_retry == True
        assert result.suggestion is not None
    
    @pytest.mark.asyncio
    async def test_heuristic_grade_relevant(self):
        """Test heuristic grading with relevant chunks."""
        from app.services.relevance_grader import RelevanceGrader
        
        grader = RelevanceGrader()
        
        chunks = [
            "Python is a programming language used for machine learning.",
            "Machine learning algorithms use Python extensively.",
        ]
        
        result = await grader.grade("Python machine learning", chunks, threshold=0.3)
        
        assert result.is_relevant == True
        assert len(result.relevant_indices) > 0
    
    @pytest.mark.asyncio
    async def test_heuristic_grade_irrelevant(self):
        """Test heuristic grading with irrelevant chunks."""
        from app.services.relevance_grader import RelevanceGrader
        
        grader = RelevanceGrader()
        
        chunks = [
            "The weather today is sunny and warm.",
            "Cooking recipes for Italian pasta dishes.",
        ]
        
        result = await grader.grade("quantum computing algorithms", chunks, threshold=0.5)
        
        # Low overlap should trigger retry suggestion
        assert result.confidence < 1.0
    
    def test_filter_chunks(self):
        """Test chunk filtering based on grading."""
        from app.services.relevance_grader import RelevanceGrader, GradingResult
        
        grader = RelevanceGrader()
        
        chunks = ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]
        grading = GradingResult(
            is_relevant=True,
            confidence=0.8,
            relevant_indices=[0, 2],  # Only 0 and 2 are relevant
            filtered_count=2,
            suggestion=None,
            should_retry=False,
        )
        
        filtered = grader.filter_chunks(chunks, grading)
        
        assert len(filtered) == 2
        assert filtered[0] == "chunk 0"
        assert filtered[1] == "chunk 2"


# ===========================================================================
# Conversation Context Service Tests
# ===========================================================================

class TestConversationContextService:
    """Tests for ConversationContextService."""
    
    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singletons."""
        from app.services import conversation_context_service, utility_llm_client
        conversation_context_service._context_service = None
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
        yield
        conversation_context_service._context_service = None
        utility_llm_client._utility_llm = None
        utility_llm_client.UtilityLLMClient._instance = None
    
    def test_add_and_get_message(self):
        """Test adding and retrieving messages."""
        from app.services.conversation_context_service import ConversationContextService
        
        service = ConversationContextService()
        
        service.add_message("user1", "user", "Hello!")
        service.add_message("user1", "assistant", "Hi there!")
        
        history = service.get_history("user1")
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"
        assert history[1]["role"] == "assistant"
    
    def test_session_isolation(self):
        """Test that sessions are isolated."""
        from app.services.conversation_context_service import ConversationContextService
        
        service = ConversationContextService()
        
        service.add_message("user1", "user", "Message for user1")
        service.add_message("user2", "user", "Message for user2")
        
        history1 = service.get_history("user1")
        history2 = service.get_history("user2")
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0]["content"] != history2[0]["content"]
    
    def test_clear_session(self):
        """Test clearing a session."""
        from app.services.conversation_context_service import ConversationContextService
        
        service = ConversationContextService()
        
        service.add_message("user1", "user", "Hello!")
        service.add_message("user1", "assistant", "Hi!")
        
        assert service.get_session_message_count("user1") == 2
        
        service.clear_session("user1")
        
        assert service.get_session_message_count("user1") == 0
    
    def test_max_history_limit(self):
        """Test that history is limited to max size."""
        from app.services.conversation_context_service import ConversationContextService
        
        service = ConversationContextService(max_history_per_session=5)
        
        for i in range(10):
            service.add_message("user1", "user", f"Message {i}")
        
        history = service.get_history("user1")
        
        assert len(history) == 5
        # Should have latest messages
        assert history[-1]["content"] == "Message 9"
    
    @pytest.mark.asyncio
    async def test_get_context_for_query_no_history(self):
        """Test context retrieval with no history."""
        from app.services.conversation_context_service import ConversationContextService
        
        service = ConversationContextService()
        
        context = await service.get_context_for_query(
            query="What is Python?",
            user_id="user1",
        )
        
        assert context.resolved_query == "What is Python?"
        assert context.is_followup == False
        assert len(context.relevant_history) == 0
    
    @pytest.mark.asyncio
    async def test_heuristic_followup_detection(self):
        """Test heuristic follow-up detection."""
        from app.services.conversation_context_service import ConversationContextService
        
        service = ConversationContextService()
        
        # Disable LLM so heuristic is used
        service._utility_llm._initialized = True
        service._utility_llm.enabled = False
        
        # Add previous context
        service.add_message("user1", "user", "Tell me about Python")
        service.add_message("user1", "assistant", "Python is a programming language.")
        
        context = await service.get_context_for_query(
            query="Tell me more about it",
            user_id="user1",
        )
        
        # "it" is a follow-up indicator
        assert context.is_followup == True
    
    def test_get_formatted_context(self):
        """Test formatting context for LLM prompt."""
        from app.services.conversation_context_service import (
            ConversationContextService,
            ConversationContext,
        )
        
        service = ConversationContextService()
        
        context = ConversationContext(
            resolved_query="What is Python?",
            is_followup=False,
            referenced_entities=["Python", "Programming"],
            summary="User asked about Python.",
            relevant_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )
        
        formatted = service.get_formatted_context(context)
        
        assert "Python" in formatted
        assert "summary" in formatted.lower() or "Python" in formatted


# ===========================================================================
# Enhanced Retrieval Service Tests
# ===========================================================================

class TestEnhancedRetrievalService:
    """Tests for EnhancedRetrievalService integration."""
    
    @pytest.mark.asyncio
    @patch('app.services.enhanced_retrieval_service.get_query_classification_service')
    @patch('app.services.enhanced_retrieval_service.get_query_expansion_service')
    @patch('app.services.enhanced_retrieval_service.get_relevance_grader')
    @patch('app.services.enhanced_retrieval_service.get_conversation_context_service')
    async def test_enhanced_retrieval_structure(
        self,
        mock_context_svc,
        mock_grader,
        mock_expansion_svc,
        mock_classification_svc,
    ):
        """Test that enhanced retrieval returns proper structure."""
        from app.services.enhanced_retrieval_service import EnhancedRetrievalService
        from app.services.retrieval_service import RetrievalResult, QueryContext
        from app.services.query_classification_service import (
            QueryClassification,
            QueryType,
            QueryComplexity,
        )
        
        # Setup mock classification result
        mock_classification = QueryClassification(
            query_type=QueryType.FACT,
            complexity=QueryComplexity.SIMPLE,
            needs_graph=False,
            is_followup=False,
            confidence=0.9,
            routing={
                "num_chunks": 5,
                "use_hybrid": False,
                "use_graph": False,
                "use_expansion": False,
            },
        )
        
        # Configure mock classification service
        mock_class_instance = MagicMock()
        mock_class_instance.classify = AsyncMock(return_value=mock_classification)
        mock_classification_svc.return_value = mock_class_instance
        
        # Configure mock grader
        mock_grader_instance = MagicMock()
        mock_grader_instance.grade = AsyncMock(return_value={})
        mock_grader.return_value = mock_grader_instance
        
        # Configure mock context service
        mock_ctx_instance = MagicMock()
        mock_ctx_instance.resolve_query = MagicMock(return_value=("What is Python?", False, ""))
        mock_ctx_instance.add_turn = MagicMock()
        mock_context_svc.return_value = mock_ctx_instance
        
        # Configure mock expansion service
        mock_expansion_svc.return_value = MagicMock()
        
        # Mock the database session
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        
        user_id = uuid.uuid4()
        
        service = EnhancedRetrievalService(mock_db)
        
        # Mock base service with proper QueryContext
        mock_query_context = QueryContext(
            raw_query="What is Python?",
            user_id=user_id,
        )
        mock_base_result = RetrievalResult(
            chunks=[],
            query_context=mock_query_context,
            total_ms=100,
        )
        service._base_service.retrieve = AsyncMock(return_value=mock_base_result)
        
        result = await service.retrieve(
            query="What is Python?",
            user_id=user_id,
            enable_expansion=False,
            enable_grading=False,
            enable_context=False,
        )
        
        # Check structure
        assert hasattr(result, "base_result")
        assert hasattr(result, "query_classification")
        assert hasattr(result, "expanded_queries")
        assert hasattr(result, "relevance_grading")
        assert result.total_ms >= 0
    
    def test_add_assistant_response(self):
        """Test adding assistant response to conversation via context service."""
        from app.services.conversation_context_service import (
            ConversationContextService,
        )
        
        # Create a fresh context service for testing
        context_service = ConversationContextService()
        
        user_id = uuid.uuid4()
        session_hint = "test_session"
        
        # Add an assistant turn
        context_service.add_message(
            user_id=str(user_id),
            session_hint=session_hint,
            role="assistant",
            content="This is the assistant response.",
        )
        
        # Check it was added to context service
        history = context_service.get_history(str(user_id), session_hint)
        
        assert len(history) == 1
        assert history[0]["role"] == "assistant"
        assert history[0]["content"] == "This is the assistant response."


# ===========================================================================
# Integration Tests
# ===========================================================================

@pytest.mark.skip(reason="Integration tests require LLM connection")
class TestPhase3Integration:
    """Integration tests for Phase 3 services working together."""
    
    def test_services_importable(self):
        """Test that all Phase 3 services can be imported and initialized."""
        from app.services.utility_llm_client import get_utility_llm
        from app.services.query_expansion_service import get_query_expansion_service
        from app.services.query_classification_service import get_query_classification_service
        from app.services.relevance_grader import get_relevance_grader
        from app.services.conversation_context_service import get_conversation_context_service
        from app.services.enhanced_retrieval_service import EnhancedRetrievalService
        
        utility_llm = get_utility_llm()
        assert utility_llm is not None
        
        expander = get_query_expansion_service()
        assert expander is not None
        
        classifier = get_query_classification_service()
        assert classifier is not None
        
        grader = get_relevance_grader()
        assert grader is not None
        
        context_service = get_conversation_context_service()
        assert context_service is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
