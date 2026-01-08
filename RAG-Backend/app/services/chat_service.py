"""
Chat service orchestrating retrieval and LLM generation.
"""
import uuid
import json
import logging
import time
from datetime import datetime
from typing import Optional, List, AsyncIterator, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.models.chat import (
    Conversation,
    Message,
    QueryLog,
    Feedback,
    MessageRoleEnum,
    QueryTypeEnum,
)
from app.services.retrieval_service import RetrievalService, RetrievedChunk
from app.services.llm_client import LLMClient, PromptBuilder
from app.middleware.error_handler import NotFoundError

logger = logging.getLogger(__name__)


class ChatService:
    """
    Main chat orchestration service.
    Coordinates retrieval, prompt building, and LLM generation.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.retrieval = RetrievalService(db)
        self.llm = LLMClient()
        self.prompt_builder = PromptBuilder()
    
    async def chat(
        self,
        query: str,
        user_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response with sources.
        Non-streaming version.
        """
        start_time = time.perf_counter()
        
        # Get or create conversation
        conv_uuid = uuid.UUID(conversation_id) if conversation_id else None
        conversation = await self._get_or_create_conversation(user_id, conv_uuid)
        
        # Get conversation history
        history = await self._get_conversation_history(conversation.id)
        
        # Step 1: Retrieval
        retrieval_result = await self.retrieval.retrieve(
            query=query,
            user_id=user_id,
            tenant_id=tenant_id,
            document_ids=document_ids,
            conversation_id=conversation.id,
        )
        
        # Step 2: Build prompt
        context_chunks = [
            {
                "chunk_id": str(rc.chunk.id),
                "document_name": rc.document_name,
                "page_number": rc.page_number,
                "section_path": rc.section_path,
                "text": rc.chunk.text,
                "score": rc.final_score,
            }
            for rc in retrieval_result.chunks
        ]
        
        messages = self.prompt_builder.build_messages(
            query=query,
            context_chunks=context_chunks,
            conversation_history=history,
        )
        
        # Step 3: Generate response
        llm_result = await self.llm.generate(
            messages=messages,
            temperature=temperature or settings.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
        )
        
        # Step 4: Extract citations
        citations = self.prompt_builder.extract_citations(
            llm_result["content"],
            context_chunks,
        )
        
        # Step 5: Save messages
        total_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Save user message
        user_message = await self._save_message(
            conversation_id=conversation.id,
            role=MessageRoleEnum.USER,
            content=query,
            query_type=retrieval_result.query_context.query_type.value,
        )
        
        # Save assistant message
        assistant_message = await self._save_message(
            conversation_id=conversation.id,
            role=MessageRoleEnum.ASSISTANT,
            content=llm_result["content"],
            sources=citations,
            retrieval_latency_ms=retrieval_result.total_ms,
            llm_latency_ms=llm_result["latency_ms"],
            total_latency_ms=total_ms,
            input_tokens=0,  # TODO: Calculate from prompt
            output_tokens=llm_result.get("tokens_used", 0),
        )
        
        # Save query log
        await self._save_query_log(
            message=user_message,
            user_id=user_id,
            retrieval_result=retrieval_result,
            context_used=json.dumps(context_chunks),
        )
        
        # Update conversation
        conversation.message_count += 2
        conversation.last_message_at = datetime.utcnow()
        conversation.updated_at = datetime.utcnow()
        
        # Phase 6: Create episodic memory of this interaction
        await self._create_episodic_memory(
            user_id=user_id,
            query=query,
            response=llm_result["content"],
            conversation_id=conversation.id,
            citations=citations,
        )
        
        return {
            "conversation_id": str(conversation.id),
            "message_id": str(assistant_message.id),
            "content": llm_result["content"],
            "sources": citations,
            "model": settings.LLM_MODEL_NAME,
            "tokens_used": llm_result.get("tokens_used", 0),
            "latency_ms": total_ms,
        }
    
    async def chat_stream(
        self,
        query: str,
        user_id: uuid.UUID,
        tenant_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a chat message with streaming response.
        Yields events: {type: "token"|"sources"|"done"|"error", data: ...}
        """
        start_time = time.perf_counter()
        
        try:
            # Get or create conversation
            conv_uuid = uuid.UUID(conversation_id) if conversation_id else None
            conversation = await self._get_or_create_conversation(user_id, conv_uuid)
            
            # Get conversation history
            history = await self._get_conversation_history(conversation.id)
            
            # Step 1: Retrieval
            retrieval_result = await self.retrieval.retrieve(
                query=query,
                user_id=user_id,
                tenant_id=tenant_id,
                document_ids=document_ids,
                conversation_id=conversation.id,
            )
            
            # Build context
            context_chunks = [
                {
                    "chunk_id": str(rc.chunk.id),
                    "document_name": rc.document_name,
                    "page_number": rc.page_number,
                    "section_path": rc.section_path,
                    "text": rc.chunk.text,
                    "score": rc.final_score,
                }
                for rc in retrieval_result.chunks
            ]
            
            # Build prompt
            messages = self.prompt_builder.build_messages(
                query=query,
                context_chunks=context_chunks,
                conversation_history=history,
            )
            
            # Step 2: Stream response
            full_response = ""
            token_count = 0
            
            logger.info(f"[ChatStream] Starting LLM streaming for query: {query[:50]}...")
            
            async for token in self.llm.generate_stream(
                messages=messages,
                temperature=temperature or settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            ):
                token_count += 1
                full_response += token
                if token_count <= 5:
                    logger.debug(f"[ChatStream] Token {token_count}: {repr(token)}")
                yield {"event": "token", "data": token}
            
            logger.info(f"[ChatStream] Streaming complete. Total tokens: {token_count}")
            
            # Step 3: Send sources
            citations = self.prompt_builder.extract_citations(
                full_response,
                context_chunks,
            )
            yield {"event": "sources", "data": json.dumps(citations)}
            
            # Step 4: Save messages
            total_ms = int((time.perf_counter() - start_time) * 1000)
            
            user_message = await self._save_message(
                conversation_id=conversation.id,
                role=MessageRoleEnum.USER,
                content=query,
            )
            
            assistant_message = await self._save_message(
                conversation_id=conversation.id,
                role=MessageRoleEnum.ASSISTANT,
                content=full_response,
                sources=citations,
                total_latency_ms=total_ms,
            )
            
            # Update conversation
            conversation.message_count += 2
            conversation.last_message_at = datetime.utcnow()
            await self.db.commit()
            
            # Send done event
            yield {
                "event": "done",
                "data": json.dumps({
                    "conversation_id": str(conversation.id),
                    "message_id": str(assistant_message.id),
                    "latency_ms": total_ms,
                })
            }
            
        except Exception as e:
            logger.exception(f"Chat stream error: {e}")
            yield {"event": "error", "data": str(e)}
    
    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[Conversation]:
        """Get a conversation by ID for a user."""
        result = await self.db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()
    
    async def list_conversations(
        self,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Conversation], int]:
        """List conversations for a user."""
        from sqlmodel import func
        
        # Count
        count_result = await self.db.execute(
            select(func.count(Conversation.id)).where(
                Conversation.user_id == user_id,
                Conversation.is_archived == False,
            )
        )
        total = count_result.scalar_one()
        
        # Fetch
        offset = (page - 1) * page_size
        result = await self.db.execute(
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.is_archived == False,
            )
            .order_by(Conversation.last_message_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        
        return list(result.scalars().all()), total
    
    async def get_messages(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 50,
    ) -> List[Message]:
        """Get messages for a conversation."""
        # Verify ownership
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            raise NotFoundError("Conversation not found")
        
        result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        
        return list(result.scalars().all())
    
    async def save_feedback(
        self,
        message_id: uuid.UUID,
        user_id: uuid.UUID,
        rating: int,
        is_helpful: Optional[bool] = None,
        is_accurate: Optional[bool] = None,
        comment: Optional[str] = None,
        issues: Optional[List[str]] = None,
    ) -> Feedback:
        """Save user feedback on a message."""
        feedback = Feedback(
            message_id=message_id,
            user_id=user_id,
            rating=rating,
            is_helpful=is_helpful,
            is_accurate=is_accurate,
            comment=comment,
            issues_json=json.dumps(issues or []),
        )
        self.db.add(feedback)
        await self.db.flush()
        return feedback
    
    # ==================== Private Methods ====================
    
    async def _get_or_create_conversation(
        self,
        user_id: uuid.UUID,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> Conversation:
        """Get existing conversation or create new one."""
        if conversation_id:
            result = await self.db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                return conversation
        
        # Create new conversation
        conversation = Conversation(
            user_id=user_id,
            title="New Conversation",
        )
        self.db.add(conversation)
        await self.db.flush()
        
        return conversation
    
    async def _get_conversation_history(
        self,
        conversation_id: uuid.UUID,
        limit: int = 10,
    ) -> List[Dict[str, str]]:
        """Get recent messages for context."""
        result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        
        # Convert to API format and reverse to chronological order
        history = []
        for msg in reversed(messages):
            history.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        
        return history
    
    async def _save_message(
        self,
        conversation_id: uuid.UUID,
        role: MessageRoleEnum,
        content: str,
        sources: Optional[List[Dict]] = None,
        query_type: Optional[str] = None,
        retrieval_latency_ms: Optional[int] = None,
        llm_latency_ms: Optional[int] = None,
        total_latency_ms: Optional[int] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Message:
        """Save a message to the database."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources_json=json.dumps(sources or []),
            query_type=QueryTypeEnum(query_type) if query_type else None,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.db.add(message)
        await self.db.flush()
        return message
    
    async def _save_query_log(
        self,
        message: Message,
        user_id: uuid.UUID,
        retrieval_result,
        context_used: str,
    ) -> QueryLog:
        """Save detailed query log for analysis."""
        log = QueryLog(
            message_id=message.id,
            user_id=user_id,
            raw_query=retrieval_result.query_context.raw_query,
            normalized_query=retrieval_result.query_context.normalized_query,
            query_type=QueryTypeEnum(retrieval_result.query_context.query_type.value),
            expanded_queries_json=json.dumps(
                retrieval_result.query_context.expanded_queries
            ),
            retrieval_mode="hybrid",
            chunks_retrieved=len(retrieval_result.chunks),
            chunks_after_rerank=len(retrieval_result.chunks),
            normalization_ms=retrieval_result.normalization_ms,
            embedding_ms=retrieval_result.embedding_ms,
            vector_search_ms=retrieval_result.vector_search_ms,
            reranking_ms=retrieval_result.reranking_ms,
            top_chunk_score=retrieval_result.top_score,
            mean_chunk_score=retrieval_result.mean_score,
            relevance_grade=retrieval_result.relevance_grade,
            context_used=context_used,
        )
        self.db.add(log)
        await self.db.flush()
        return log
    
    async def _create_episodic_memory(
        self,
        user_id: uuid.UUID,
        query: str,
        response: str,
        conversation_id: uuid.UUID,
        citations: Optional[List[Dict]] = None,
    ) -> None:
        """
        Create an episodic memory of the chat interaction (Phase 6).
        
        Episodic memories capture specific events/interactions and enable
        cross-session recall of what was discussed.
        """
        try:
            from app.services.memory_service import get_memory_service
            from app.models.memory import MemoryType
            
            memory_service = get_memory_service(self.db)
            
            # Build memory content (truncate long responses)
            response_summary = response[:300] + "..." if len(response) > 300 else response
            memory_content = f"User asked: {query}\nAssistant responded: {response_summary}"
            
            # Extract related chunk IDs from citations
            related_chunk_ids = []
            if citations:
                related_chunk_ids = [
                    c.get("id", "") for c in citations if c.get("id")
                ][:5]  # Limit to 5
            
            # Extract source names for metadata
            source_names = []
            if citations:
                source_names = [
                    c.get("name", "") for c in citations if c.get("name")
                ][:3]
            
            await memory_service.create_memory(
                user_id=user_id,
                content=memory_content,
                memory_type=MemoryType.EPISODIC,
                conversation_id=conversation_id,
                related_chunk_ids=related_chunk_ids,
                metadata={
                    "source": "chat_interaction",
                    "auto_generated": True,
                    "query_length": len(query),
                    "response_length": len(response),
                    "citation_count": len(citations) if citations else 0,
                    "source_documents": source_names,
                },
            )
            
            logger.debug(f"Created episodic memory for user {user_id} (conversation: {conversation_id})")
            
        except Exception as e:
            # Don't fail chat if memory creation fails
            logger.warning(f"Failed to create episodic memory: {e}")
