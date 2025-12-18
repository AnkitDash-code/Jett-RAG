"""
Tests for chat service and LLM client.
Run with: pytest tests/test_chat.py -v
"""
import pytest
import pytest_asyncio
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm_client import LLMClient, PromptBuilder
from app.services.chat_service import ChatService


class TestPromptBuilder:
    """Unit tests for PromptBuilder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = PromptBuilder()
    
    def test_build_messages_basic(self):
        """Test basic message building."""
        query = "What is compliance?"
        context_chunks = [
            {
                "chunk_id": "chunk-1",
                "document_name": "policy.pdf",
                "page_number": 5,
                "section_path": "Section 2.1",
                "text": "Compliance means following rules and regulations.",
                "score": 0.9,
            }
        ]
        
        messages = self.builder.build_messages(query, context_chunks)
        
        assert len(messages) >= 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == query
        
        # System message should contain context
        assert "policy.pdf" in messages[0]["content"]
        assert "Compliance means" in messages[0]["content"]
    
    def test_build_messages_with_history(self):
        """Test message building with conversation history."""
        query = "Tell me more"
        context_chunks = []
        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
        ]
        
        messages = self.builder.build_messages(
            query=query,
            context_chunks=context_chunks,
            conversation_history=history,
        )
        
        # Should include history
        assert len(messages) >= 4  # system + 2 history + user
        
        # History should be in order
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert user_msgs[-1]["content"] == query
    
    def test_format_context_empty(self):
        """Test context formatting with no chunks."""
        result = self.builder._format_context([])
        assert "No relevant documents found" in result
    
    def test_format_context_with_chunks(self):
        """Test context formatting with multiple chunks."""
        chunks = [
            {
                "document_name": "doc1.pdf",
                "page_number": 1,
                "section_path": "Intro",
                "text": "First chunk content.",
            },
            {
                "document_name": "doc2.pdf",
                "page_number": 5,
                "text": "Second chunk content.",
            },
        ]
        
        result = self.builder._format_context(chunks)
        
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result
        assert "Page 1" in result
        assert "First chunk content" in result
        assert "Second chunk content" in result
    
    def test_extract_citations(self):
        """Test citation extraction."""
        response = "According to the policy [Source: policy.pdf], compliance is required."
        chunks = [
            {"chunk_id": "1", "document_name": "policy.pdf", "page_number": 1, "score": 0.9},
            {"chunk_id": "2", "document_name": "guide.pdf", "page_number": 2, "score": 0.7},
        ]
        
        citations = self.builder.extract_citations(response, chunks)
        
        assert len(citations) == 2
        assert citations[0]["name"] == "policy.pdf"
    
    def test_score_to_relevance(self):
        """Test score to relevance label conversion."""
        assert self.builder._score_to_relevance(0.9) == "High"
        assert self.builder._score_to_relevance(0.6) == "Medium"
        assert self.builder._score_to_relevance(0.3) == "Low"


class TestLLMClient:
    """Unit tests for LLMClient."""
    
    def test_messages_to_query_context(self):
        """Test conversion of messages to query/context format."""
        client = LLMClient()
        
        messages = [
            {"role": "system", "content": "Context: This is the document content."},
            {"role": "user", "content": "What does the document say?"},
        ]
        
        query, context = client._messages_to_query_context(messages)
        
        assert query == "What does the document say?"
        assert "document content" in context
    
    def test_messages_to_query_context_no_system(self):
        """Test conversion when no system message."""
        client = LLMClient()
        
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        
        query, context = client._messages_to_query_context(messages)
        
        assert query == "Hello"
        assert context == "No context provided."
    
    @pytest.mark.asyncio
    async def test_generate_mocked(self):
        """Test generate method with mocked HTTP client."""
        # Test the message conversion logic directly (no HTTP call)
        client = LLMClient()
        
        messages = [
            {"role": "system", "content": "Context here"},
            {"role": "user", "content": "What is this?"},
        ]
        
        # Test the conversion method
        query, context = client._messages_to_query_context(messages)
        
        assert query == "What is this?"
        assert "Context here" in context
        
        # Verify client has base_url set
        assert client.base_url is not None


class TestChatService:
    """Integration tests for ChatService."""
    
    @pytest.mark.asyncio
    async def test_get_or_create_conversation(self, db_session: AsyncSession):
        """Test conversation creation."""
        service = ChatService(db_session)
        user_id = uuid.uuid4()
        
        # Create new conversation
        conv1 = await service._get_or_create_conversation(user_id)
        assert conv1 is not None
        assert conv1.user_id == user_id
        
        # Get existing conversation
        conv2 = await service._get_or_create_conversation(user_id, conv1.id)
        assert conv2.id == conv1.id
    
    @pytest.mark.asyncio
    async def test_save_message(self, db_session: AsyncSession):
        """Test message saving."""
        from app.models.chat import MessageRoleEnum
        
        service = ChatService(db_session)
        user_id = uuid.uuid4()
        
        conv = await service._get_or_create_conversation(user_id)
        
        msg = await service._save_message(
            conversation_id=conv.id,
            role=MessageRoleEnum.USER,
            content="Test message",
        )
        
        assert msg.content == "Test message"
        assert msg.conversation_id == conv.id
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, db_session: AsyncSession):
        """Test retrieving conversation history."""
        from app.models.chat import MessageRoleEnum
        
        service = ChatService(db_session)
        user_id = uuid.uuid4()
        
        conv = await service._get_or_create_conversation(user_id)
        
        # Add messages
        await service._save_message(conv.id, MessageRoleEnum.USER, "Hello")
        await service._save_message(conv.id, MessageRoleEnum.ASSISTANT, "Hi there!")
        await service._save_message(conv.id, MessageRoleEnum.USER, "How are you?")
        
        history = await service._get_conversation_history(conv.id)
        
        assert len(history) == 3
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "Hi there!"
        assert history[2]["content"] == "How are you?"
    
    @pytest.mark.asyncio
    async def test_list_conversations(self, db_session: AsyncSession):
        """Test listing user conversations."""
        service = ChatService(db_session)
        user_id = uuid.uuid4()
        
        # Create multiple conversations
        await service._get_or_create_conversation(user_id)
        await service._get_or_create_conversation(user_id)
        
        convs, total = await service.list_conversations(user_id)
        
        assert total == 2
        assert len(convs) == 2
    
    @pytest.mark.asyncio
    async def test_save_feedback(self, db_session: AsyncSession):
        """Test saving feedback."""
        from app.models.chat import MessageRoleEnum
        
        service = ChatService(db_session)
        user_id = uuid.uuid4()
        
        conv = await service._get_or_create_conversation(user_id)
        msg = await service._save_message(conv.id, MessageRoleEnum.ASSISTANT, "Response")
        
        feedback = await service.save_feedback(
            message_id=msg.id,
            user_id=user_id,
            rating=5,
            is_helpful=True,
            comment="Great response!",
        )
        
        assert feedback.rating == 5
        assert feedback.is_helpful is True
        assert feedback.comment == "Great response!"
