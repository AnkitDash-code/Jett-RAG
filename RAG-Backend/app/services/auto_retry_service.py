"""
Auto-Retry Service with Self-Reflection for Phase 5.

Implements the self-reflection pattern where:
1. Initial retrieval is graded for relevance
2. If relevance is low, LLM reformulates the query
3. Retry with reformulated query
4. Optionally expand to different retrieval strategies

This significantly improves answer quality by catching poor retrievals
before they reach the answer generation stage.
"""
import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import uuid

from app.config import settings
from app.services.relevance_grader import get_relevance_grader, GradingResult
from app.services.query_expansion_service import get_query_expansion_service
from app.services.utility_llm_client import get_utility_llm

logger = logging.getLogger(__name__)


@dataclass
class RetryAttempt:
    """Record of a retrieval attempt."""
    attempt_number: int
    query: str
    strategy: str  # "original", "expanded", "reformulated", "fallback"
    grading: Optional[GradingResult] = None
    chunk_count: int = 0
    elapsed_ms: int = 0


@dataclass
class AutoRetryResult:
    """Result of auto-retry retrieval."""
    final_query: str
    original_query: str
    attempts: List[RetryAttempt] = field(default_factory=list)
    used_retry: bool = False
    total_attempts: int = 1
    final_confidence: float = 0.0
    strategy_used: str = "original"
    improvement_note: Optional[str] = None


class AutoRetryService:
    """
    Service for automatic query retry with self-reflection.
    
    When initial retrieval has low relevance, this service:
    1. Uses LLM to analyze why retrieval failed
    2. Reformulates the query based on feedback
    3. Retries with new query or different strategy
    4. Limits retries to prevent infinite loops
    """
    
    def __init__(self):
        self.enabled = getattr(settings, 'ENABLE_AUTO_RETRY', True)
        self.max_retries = getattr(settings, 'MAX_RETRY_ATTEMPTS', 2)
        self.relevance_threshold = getattr(settings, 'RETRY_RELEVANCE_THRESHOLD', 0.4)
        
        self._grader = get_relevance_grader()
        self._expander = get_query_expansion_service()
        self._utility_llm = get_utility_llm()
        
        logger.info(
            f"AutoRetryService initialized: enabled={self.enabled}, "
            f"max_retries={self.max_retries}, threshold={self.relevance_threshold}"
        )
    
    async def should_retry(
        self,
        query: str,
        chunks: List[str],
        grading: Optional[GradingResult] = None,
    ) -> bool:
        """
        Determine if retrieval should be retried.
        
        Args:
            query: Original query
            chunks: Retrieved chunk texts
            grading: Optional pre-computed grading result
            
        Returns:
            True if retry is recommended
        """
        if not self.enabled:
            return False
        
        if not chunks:
            return True  # No chunks = definitely retry
        
        # Get or compute grading
        if grading is None:
            grading = await self._grader.grade(
                query, chunks, threshold=self.relevance_threshold
            )
        
        # Check retry conditions
        if grading.should_retry:
            logger.info(
                f"Retry recommended: confidence={grading.confidence:.2f}, "
                f"relevant={len(grading.relevant_indices)}/{len(chunks)}"
            )
            return True
        
        if grading.confidence < self.relevance_threshold:
            logger.info(f"Low confidence ({grading.confidence:.2f}), recommending retry")
            return True
        
        return False
    
    async def reformulate_query(
        self,
        original_query: str,
        chunks: List[str],
        grading: GradingResult,
    ) -> str:
        """
        Use LLM to reformulate query based on failed retrieval.
        
        Args:
            original_query: Original user query
            chunks: Retrieved (but low-relevance) chunks
            grading: Grading result with feedback
            
        Returns:
            Reformulated query string
        """
        if not self._utility_llm.is_available():
            # Fallback: use query expansion
            expanded = await self._expander.expand(original_query, max_expansions=1)
            if expanded and expanded != [original_query]:
                return expanded[0]
            return original_query
        
        # Build context about what went wrong
        context_snippets = [c[:200] for c in chunks[:3]]  # First 3 chunks, truncated
        
        system_prompt = (
            "You are a query reformulation expert. When a search query doesn't "
            "find relevant results, you rewrite it to be more effective."
        )
        
        prompt = f"""The following query did not retrieve relevant results:
Query: {original_query}

Relevance assessment:
- Confidence: {grading.confidence:.1%}
- Suggestion: {grading.suggestion or 'None provided'}

Retrieved content snippets (low relevance):
{chr(10).join(f'- {s}...' for s in context_snippets)}

Please reformulate the query to find more relevant information.
Focus on:
1. Using different keywords/synonyms
2. Being more specific or more general as needed
3. Clarifying ambiguous terms

Reformulated query:"""
        
        try:
            response = await self._utility_llm._call_llm(
                prompt,
                system_prompt,
                temperature=0.5,
                max_tokens=100,
            )
            
            reformulated = response.strip().strip('"').strip("'")
            
            if reformulated and reformulated != original_query:
                logger.info(f"Query reformulated: '{original_query}' -> '{reformulated}'")
                return reformulated
                
        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")
        
        return original_query
    
    async def get_retry_strategy(
        self,
        attempt_number: int,
        original_query: str,
        last_grading: GradingResult,
    ) -> Dict[str, Any]:
        """
        Determine strategy for retry attempt.
        
        Returns:
            Dict with 'strategy', 'query', 'params'
        """
        if attempt_number == 1:
            # First retry: try query expansion
            expanded = await self._expander.expand(original_query, max_expansions=3)
            if expanded and len(expanded) > 1:
                return {
                    "strategy": "expanded",
                    "query": original_query,
                    "queries": expanded,
                    "use_multi_query": True,
                }
        
        elif attempt_number == 2:
            # Second retry: reformulate with LLM
            reformulated = await self.reformulate_query(
                original_query, [], last_grading
            )
            return {
                "strategy": "reformulated",
                "query": reformulated,
                "use_multi_query": False,
            }
        
        # Fallback: semantic similarity search
        return {
            "strategy": "fallback",
            "query": original_query,
            "increase_top_k": True,
            "use_multi_query": False,
        }
    
    async def execute_with_retry(
        self,
        query: str,
        retrieve_func,
        grade_func=None,
    ) -> AutoRetryResult:
        """
        Execute retrieval with automatic retry on low relevance.
        
        Args:
            query: User query
            retrieve_func: Async function(query) -> List[chunks]
            grade_func: Optional async function(query, chunks) -> GradingResult
            
        Returns:
            AutoRetryResult with final retrieval info
        """
        result = AutoRetryResult(
            final_query=query,
            original_query=query,
        )
        
        current_query = query
        last_grading = None
        
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            
            # Retrieve
            try:
                chunks = await retrieve_func(current_query)
            except Exception as e:
                logger.error(f"Retrieval failed on attempt {attempt+1}: {e}")
                chunks = []
            
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            
            # Grade
            if grade_func:
                grading = await grade_func(current_query, chunks)
            else:
                chunk_texts = [c.text if hasattr(c, 'text') else str(c) for c in chunks]
                grading = await self._grader.grade(
                    current_query, chunk_texts, threshold=self.relevance_threshold
                )
            
            # Record attempt
            strategy = "original" if attempt == 0 else result.strategy_used
            attempt_record = RetryAttempt(
                attempt_number=attempt + 1,
                query=current_query,
                strategy=strategy,
                grading=grading,
                chunk_count=len(chunks),
                elapsed_ms=elapsed_ms,
            )
            result.attempts.append(attempt_record)
            
            # Check if we're done
            if not self.enabled or attempt >= self.max_retries:
                break
            
            if not await self.should_retry(current_query, chunks, grading):
                break
            
            # Get retry strategy
            retry_info = await self.get_retry_strategy(
                attempt + 1, query, grading
            )
            
            result.used_retry = True
            result.strategy_used = retry_info["strategy"]
            
            if retry_info.get("query"):
                current_query = retry_info["query"]
            
            last_grading = grading
            logger.info(
                f"Retry {attempt + 1}: strategy={retry_info['strategy']}, "
                f"query='{current_query[:50]}...'"
            )
        
        # Finalize result
        result.final_query = current_query
        result.total_attempts = len(result.attempts)
        
        if result.attempts:
            final_attempt = result.attempts[-1]
            if final_attempt.grading:
                result.final_confidence = final_attempt.grading.confidence
        
        if result.used_retry:
            improvement = (
                result.final_confidence - 
                (result.attempts[0].grading.confidence if result.attempts[0].grading else 0)
            )
            if improvement > 0:
                result.improvement_note = f"Improved confidence by {improvement:.1%}"
        
        return result


# Singleton getter
_auto_retry_service: Optional[AutoRetryService] = None


def get_auto_retry_service() -> AutoRetryService:
    """Get auto-retry service instance."""
    global _auto_retry_service
    if _auto_retry_service is None:
        _auto_retry_service = AutoRetryService()
    return _auto_retry_service
