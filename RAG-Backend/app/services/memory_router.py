"""
Memory Router for automatic classification of memories.
Routes inputs to episodic (events/actions) or semantic (facts/concepts) stores.
"""
import logging
import re
from typing import Optional, Tuple

from app.models.memory import MemoryType

logger = logging.getLogger(__name__)


class MemoryRouter:
    """
    Routes memory content to appropriate memory type.
    
    Episodic: Specific events, actions, time-bound experiences
    Semantic: Facts, concepts, general knowledge
    """
    
    # Keywords indicating episodic memory
    EPISODIC_PATTERNS = [
        r'\b(yesterday|today|last week|ago|just now|recently)\b',
        r'\b(I|we|user) (did|asked|clicked|uploaded|searched|requested)\b',
        r'\b(happened|occurred|took place|was done)\b',
        r'\b(conversation|session|query|request|interaction)\b',
        r'\b(at \d+:\d+|on \d+/\d+|\d+ (minutes?|hours?|days?) ago)\b',
    ]
    
    # Keywords indicating semantic memory
    SEMANTIC_PATTERNS = [
        r'\b(is defined as|means|refers to|is a type of)\b',
        r'\b(concept|definition|principle|theory|fact)\b',
        r'\b(always|never|generally|typically|usually)\b',
        r'\b(according to|states that|explains|describes)\b',
        r'\b(knowledge|information|data about)\b',
    ]
    
    def __init__(self, utility_llm: Optional[object] = None):
        """
        Initialize memory router.
        
        Args:
            utility_llm: Optional UtilityLLM client for advanced classification
        """
        self.utility_llm = utility_llm
        self._compiled_episodic = [re.compile(p, re.IGNORECASE) for p in self.EPISODIC_PATTERNS]
        self._compiled_semantic = [re.compile(p, re.IGNORECASE) for p in self.SEMANTIC_PATTERNS]
    
    def route_to_memory_type(self, content: str) -> MemoryType:
        """
        Classify content as episodic or semantic memory.
        
        Uses heuristic pattern matching first, then optional LLM classification
        for ambiguous cases.
        
        Args:
            content: The memory content to classify
            
        Returns:
            MemoryType.EPISODIC or MemoryType.SEMANTIC
        """
        episodic_score, semantic_score = self._calculate_scores(content)
        
        # Clear winner
        if episodic_score > semantic_score + 2:
            logger.debug(f"Memory classified as EPISODIC (score: {episodic_score} vs {semantic_score})")
            return MemoryType.EPISODIC
        elif semantic_score > episodic_score + 2:
            logger.debug(f"Memory classified as SEMANTIC (score: {semantic_score} vs {episodic_score})")
            return MemoryType.SEMANTIC
        
        # Ambiguous case - use additional heuristics
        return self._classify_ambiguous(content, episodic_score, semantic_score)
    
    def _calculate_scores(self, content: str) -> Tuple[int, int]:
        """Calculate episodic and semantic scores based on pattern matching."""
        episodic_score = 0
        semantic_score = 0
        
        for pattern in self._compiled_episodic:
            if pattern.search(content):
                episodic_score += 1
        
        for pattern in self._compiled_semantic:
            if pattern.search(content):
                semantic_score += 1
        
        return episodic_score, semantic_score
    
    def _classify_ambiguous(
        self, 
        content: str, 
        episodic_score: int, 
        semantic_score: int
    ) -> MemoryType:
        """
        Classify ambiguous content using additional heuristics.
        
        Additional rules:
        - Short, action-oriented → EPISODIC
        - Contains "what is", "how does" → SEMANTIC
        - Contains personal pronouns → EPISODIC
        - Contains technical terms → SEMANTIC
        """
        content_lower = content.lower()
        
        # Personal pronouns suggest episodic
        personal_pronouns = ['i ', 'me ', 'my ', 'we ', 'our ', 'user ']
        if any(p in content_lower for p in personal_pronouns):
            episodic_score += 1
        
        # Question patterns suggest semantic (factual queries)
        question_patterns = ['what is', 'how does', 'why does', 'explain', 'define']
        if any(p in content_lower for p in question_patterns):
            semantic_score += 1
        
        # Short content with verbs → action → episodic
        if len(content.split()) < 10:
            action_verbs = ['clicked', 'uploaded', 'searched', 'asked', 'viewed', 'downloaded']
            if any(v in content_lower for v in action_verbs):
                episodic_score += 1
        
        # Summary or excerpt → semantic
        if 'summary' in content_lower or 'excerpt' in content_lower:
            semantic_score += 1
        
        # Final decision
        if episodic_score >= semantic_score:
            logger.debug(f"Ambiguous memory classified as EPISODIC (score: {episodic_score} vs {semantic_score})")
            return MemoryType.EPISODIC
        else:
            logger.debug(f"Ambiguous memory classified as SEMANTIC (score: {semantic_score} vs {episodic_score})")
            return MemoryType.SEMANTIC
    
    async def route_with_llm(self, content: str) -> MemoryType:
        """
        Use UtilityLLM for classification (for complex cases).
        
        Falls back to heuristic if LLM unavailable.
        """
        if not self.utility_llm:
            return self.route_to_memory_type(content)
        
        try:
            prompt = f"""Classify this memory content as either 'episodic' or 'semantic'.

Episodic memories are specific events, actions, or time-bound experiences.
Examples: "User uploaded document X yesterday", "Searched for 'machine learning'"

Semantic memories are facts, concepts, or general knowledge.
Examples: "Machine learning is a subset of AI", "The company policy states..."

Content: {content[:500]}

Classification (reply with only 'episodic' or 'semantic'):"""
            
            result = await self.utility_llm.generate(prompt, max_tokens=10)
            result = result.strip().lower()
            
            if 'episodic' in result:
                return MemoryType.EPISODIC
            elif 'semantic' in result:
                return MemoryType.SEMANTIC
            else:
                # LLM didn't give clear answer, fall back to heuristic
                return self.route_to_memory_type(content)
                
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to heuristic")
            return self.route_to_memory_type(content)


# Singleton instance
_memory_router: Optional[MemoryRouter] = None


def get_memory_router() -> MemoryRouter:
    """Get or create memory router instance."""
    global _memory_router
    if _memory_router is None:
        _memory_router = MemoryRouter()
    return _memory_router
