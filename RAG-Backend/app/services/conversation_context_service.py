"""
Conversation Context Service.

Manages conversation history, follow-up detection, and context summarization.
Enables multi-turn conversation support in RAG.
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from app.services.utility_llm_client import get_utility_llm
from app.services.token_counter import get_token_counter

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context information for the current query."""
    resolved_query: str        # Query with references resolved
    is_followup: bool          # Whether this is a follow-up question
    referenced_entities: List[str]  # Entities from previous context
    summary: Optional[str]     # Conversation summary if available
    relevant_history: List[Dict[str, str]]  # Recent relevant messages


class ConversationContextService:
    """
    Service for managing conversation context.
    
    Features:
    - Message history management per session
    - Follow-up detection and reference resolution
    - Conversation summarization for token efficiency
    - Context retrieval for queries
    """
    
    def __init__(self, max_history_per_session: int = 20):
        self._utility_llm = get_utility_llm()
        self._token_counter = get_token_counter()
        self._max_history = max_history_per_session
        
        # Session storage: session_id -> list of messages
        self._sessions: Dict[str, List[ConversationMessage]] = {}
        self._session_summaries: Dict[str, str] = {}
        
        logger.info(f"ConversationContextService initialized (LLM: {self._utility_llm.is_available()})")
    
    def _get_session_id(self, user_id: str, session_hint: Optional[str] = None) -> str:
        """Generate or use session ID."""
        if session_hint:
            return f"{user_id}:{session_hint}"
        return user_id
    
    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_hint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to conversation history.
        
        Args:
            user_id: User identifier
            role: "user" or "assistant"
            content: Message content
            session_hint: Optional session identifier
            metadata: Additional message metadata
        """
        session_id = self._get_session_id(user_id, session_hint)
        
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        
        self._sessions[session_id].append(message)
        
        # Trim if exceeding max
        if len(self._sessions[session_id]) > self._max_history:
            # Remove oldest messages
            self._sessions[session_id] = self._sessions[session_id][-self._max_history:]
            # Invalidate summary
            self._session_summaries.pop(session_id, None)
    
    def get_history(
        self,
        user_id: str,
        session_hint: Optional[str] = None,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Returns:
            List of {"role": "...", "content": "..."} dicts
        """
        session_id = self._get_session_id(user_id, session_hint)
        messages = self._sessions.get(session_id, [])
        
        if max_messages:
            messages = messages[-max_messages:]
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    def clear_session(
        self,
        user_id: str,
        session_hint: Optional[str] = None,
    ) -> None:
        """Clear conversation history for a session."""
        session_id = self._get_session_id(user_id, session_hint)
        self._sessions.pop(session_id, None)
        self._session_summaries.pop(session_id, None)
    
    async def get_context_for_query(
        self,
        query: str,
        user_id: str,
        session_hint: Optional[str] = None,
        max_context_tokens: int = 500,
    ) -> ConversationContext:
        """
        Get conversation context for a query.
        
        Handles:
        - Follow-up detection
        - Reference resolution
        - History summarization
        
        Args:
            query: Current user query
            user_id: User identifier
            session_hint: Optional session identifier
            max_context_tokens: Token budget for context
            
        Returns:
            ConversationContext with resolved query and relevant history
        """
        session_id = self._get_session_id(user_id, session_hint)
        history = self._sessions.get(session_id, [])
        
        if not history:
            return ConversationContext(
                resolved_query=query,
                is_followup=False,
                referenced_entities=[],
                summary=None,
                relevant_history=[],
            )
        
        # Convert to dict format for LLM
        history_dicts = [
            {"role": msg.role, "content": msg.content}
            for msg in history[-6:]  # Last 3 Q&A pairs
        ]
        
        # Detect follow-up and resolve references
        if self._utility_llm.is_available():
            try:
                followup_result = await self._utility_llm.detect_followup(
                    query, history_dicts
                )
                
                is_followup = followup_result.get("is_followup", False)
                resolved_query = followup_result.get("resolved_query", query)
                referenced_entities = followup_result.get("referenced_entities", [])
                
            except Exception as e:
                logger.warning(f"Follow-up detection failed: {e}")
                is_followup, resolved_query, referenced_entities = self._heuristic_followup(query, history_dicts)
        else:
            is_followup, resolved_query, referenced_entities = self._heuristic_followup(query, history_dicts)
        
        # Get or generate summary if history is long
        summary = None
        if len(history) > 4:
            summary = await self._get_or_generate_summary(session_id, history_dicts)
        
        # Select relevant history within token budget
        relevant_history = self._select_relevant_history(
            history_dicts, 
            max_context_tokens,
            query
        )
        
        return ConversationContext(
            resolved_query=resolved_query,
            is_followup=is_followup,
            referenced_entities=referenced_entities,
            summary=summary,
            relevant_history=relevant_history,
        )
    
    def _heuristic_followup(
        self,
        query: str,
        history: List[Dict[str, str]],
    ) -> tuple:
        """Detect follow-up using simple heuristics."""
        query_lower = query.lower()
        
        # Check for pronouns/references
        followup_indicators = [
            "it", "this", "that", "they", "them", "those", "these",
            "the same", "what about", "and also", "more about",
            "tell me more", "explain further", "why is that",
        ]
        
        is_followup = any(indicator in query_lower for indicator in followup_indicators)
        
        # Try to resolve "it" / "this" from last assistant response
        resolved_query = query
        referenced_entities = []
        
        if is_followup and history:
            # Find last assistant message
            last_response = None
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    last_response = msg.get("content", "")
                    break
            
            if last_response:
                # Extract potential entities (simple: capitalize words)
                import re
                potential_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', last_response)
                referenced_entities = list(set(potential_entities))[:3]
        
        return is_followup, resolved_query, referenced_entities
    
    async def _get_or_generate_summary(
        self,
        session_id: str,
        history: List[Dict[str, str]],
    ) -> Optional[str]:
        """Get cached summary or generate new one."""
        # Check cache
        if session_id in self._session_summaries:
            return self._session_summaries[session_id]
        
        # Generate summary
        if self._utility_llm.is_available():
            try:
                summary = await self._utility_llm.summarize_conversation(history)
                if summary:
                    self._session_summaries[session_id] = summary
                    return summary
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
        
        return None
    
    def _select_relevant_history(
        self,
        history: List[Dict[str, str]],
        max_tokens: int,
        query: str,
    ) -> List[Dict[str, str]]:
        """Select history messages within token budget."""
        if not history:
            return []
        
        selected = []
        current_tokens = 0
        
        # Start from most recent
        for msg in reversed(history):
            msg_tokens = self._token_counter.count(msg.get("content", ""))
            
            if current_tokens + msg_tokens <= max_tokens:
                selected.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return selected
    
    def get_formatted_context(
        self,
        context: ConversationContext,
        include_summary: bool = True,
    ) -> str:
        """
        Format conversation context as string for LLM prompt.
        
        Returns:
            Formatted context string
        """
        parts = []
        
        if include_summary and context.summary:
            parts.append(f"Previous conversation summary: {context.summary}")
        
        if context.relevant_history:
            history_lines = []
            for msg in context.relevant_history:
                role = msg.get("role", "user").title()
                content = msg.get("content", "")[:200]  # Truncate
                history_lines.append(f"{role}: {content}")
            
            if history_lines:
                parts.append("Recent conversation:\n" + "\n".join(history_lines))
        
        if context.referenced_entities:
            parts.append(f"Referenced entities: {', '.join(context.referenced_entities)}")
        
        return "\n\n".join(parts) if parts else ""
    
    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)
    
    def get_session_message_count(
        self,
        user_id: str,
        session_hint: Optional[str] = None,
    ) -> int:
        """Get number of messages in a session."""
        session_id = self._get_session_id(user_id, session_hint)
        return len(self._sessions.get(session_id, []))


# Module-level singleton
_context_service: Optional[ConversationContextService] = None


def get_conversation_context_service() -> ConversationContextService:
    """Get the conversation context service singleton."""
    global _context_service
    if _context_service is None:
        _context_service = ConversationContextService()
    return _context_service
