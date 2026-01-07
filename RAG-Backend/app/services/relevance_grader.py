"""
Relevance Grader Service for Self-Reflection.

Grades retrieved chunks for relevance and determines if retrieval should be retried.
Implements the self-reflection pattern from advanced RAG.
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from app.services.utility_llm_client import get_utility_llm

logger = logging.getLogger(__name__)


@dataclass
class GradingResult:
    """Result of relevance grading."""
    is_relevant: bool
    confidence: float
    relevant_indices: List[int]  # Indices of relevant chunks
    filtered_count: int          # Number of chunks filtered out
    suggestion: Optional[str]    # Query refinement suggestion
    should_retry: bool           # Whether to retry with different strategy


@dataclass
class ChunkGrade:
    """Grade for a single chunk."""
    index: int
    is_relevant: bool
    score: float
    reason: Optional[str] = None


class RelevanceGrader:
    """
    Grades relevance of retrieved chunks to the query.
    
    Features:
    - LLM-based grading (when utility LLM available)
    - Heuristic fallback grading
    - Retry suggestions for low relevance
    - Chunk filtering
    """
    
    def __init__(self):
        self._utility_llm = get_utility_llm()
        self._min_confidence_threshold = 0.3
        self._retry_threshold = 0.4
        logger.info(f"RelevanceGrader initialized (LLM: {self._utility_llm.is_available()})")
    
    async def grade(
        self,
        query: str,
        chunks: List[str],
        threshold: float = 0.5,
    ) -> GradingResult:
        """
        Grade relevance of chunks to the query.
        
        Args:
            query: User query
            chunks: Retrieved chunk texts
            threshold: Minimum relevance threshold
            
        Returns:
            GradingResult with relevance assessment
        """
        if not chunks:
            return GradingResult(
                is_relevant=False,
                confidence=0.0,
                relevant_indices=[],
                filtered_count=0,
                suggestion="No chunks retrieved. Try a different query.",
                should_retry=True,
            )
        
        if not query.strip():
            return GradingResult(
                is_relevant=True,
                confidence=1.0,
                relevant_indices=list(range(len(chunks))),
                filtered_count=0,
                suggestion=None,
                should_retry=False,
            )
        
        # Try LLM grading
        if self._utility_llm.is_available():
            try:
                llm_result = await self._utility_llm.grade_relevance(
                    query, chunks, threshold
                )
                return self._process_llm_result(llm_result, chunks)
            except Exception as e:
                logger.warning(f"LLM grading failed: {e}")
        
        # Fallback to heuristic grading
        return self._heuristic_grade(query, chunks, threshold)
    
    def _process_llm_result(
        self,
        llm_result: Dict[str, Any],
        chunks: List[str],
    ) -> GradingResult:
        """Process LLM grading result into GradingResult."""
        relevant_indices = llm_result.get("relevant_chunks", [])
        confidence = llm_result.get("confidence", 0.5)
        is_relevant = llm_result.get("is_relevant", True)
        
        # Determine if retry is needed
        should_retry = (
            not is_relevant or 
            confidence < self._retry_threshold or
            len(relevant_indices) == 0
        )
        
        filtered_count = len(chunks) - len(relevant_indices)
        
        return GradingResult(
            is_relevant=is_relevant,
            confidence=confidence,
            relevant_indices=relevant_indices,
            filtered_count=filtered_count,
            suggestion=llm_result.get("suggestion"),
            should_retry=should_retry,
        )
    
    def _heuristic_grade(
        self,
        query: str,
        chunks: List[str],
        threshold: float,
    ) -> GradingResult:
        """
        Grade chunks using keyword overlap heuristics.
        """
        query_words = set(query.lower().split())
        
        # Remove stopwords
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'what', 'how', 'why',
            'when', 'where', 'who', 'which', 'that', 'this', 'it', 'to', 'for',
            'of', 'in', 'on', 'at', 'by', 'with', 'and', 'or', 'but', 'if',
        }
        query_keywords = query_words - stopwords
        
        if not query_keywords:
            # No keywords to match, assume all relevant
            return GradingResult(
                is_relevant=True,
                confidence=0.5,
                relevant_indices=list(range(len(chunks))),
                filtered_count=0,
                suggestion=None,
                should_retry=False,
            )
        
        relevant_indices = []
        scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            
            # Calculate overlap score
            overlap = len(query_keywords & chunk_words)
            score = overlap / len(query_keywords) if query_keywords else 0
            scores.append(score)
            
            if score >= threshold:
                relevant_indices.append(i)
        
        # Calculate overall confidence
        avg_score = sum(scores) / len(scores) if scores else 0
        confidence = min(avg_score + 0.3, 1.0)  # Boost confidence a bit
        
        is_relevant = len(relevant_indices) > 0
        filtered_count = len(chunks) - len(relevant_indices)
        
        # Generate suggestion if low relevance
        suggestion = None
        if not is_relevant:
            suggestion = "Retrieved chunks have low keyword overlap. Consider using more specific terms."
        
        should_retry = not is_relevant or confidence < self._retry_threshold
        
        return GradingResult(
            is_relevant=is_relevant,
            confidence=confidence,
            relevant_indices=relevant_indices if relevant_indices else list(range(len(chunks))),
            filtered_count=filtered_count,
            suggestion=suggestion,
            should_retry=should_retry,
        )
    
    async def grade_with_retry(
        self,
        query: str,
        chunks: List[str],
        retrieval_func,
        max_retries: int = 2,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Grade chunks and retry retrieval if needed.
        
        Args:
            query: User query
            chunks: Initially retrieved chunks
            retrieval_func: Async function to call for retry (takes query, returns chunks)
            max_retries: Maximum retry attempts
            threshold: Relevance threshold
            
        Returns:
            Dict with final chunks and grading metadata
        """
        current_chunks = chunks
        current_query = query
        attempts = 0
        grading_history = []
        
        while attempts <= max_retries:
            result = await self.grade(current_query, current_chunks, threshold)
            grading_history.append({
                "attempt": attempts,
                "query": current_query,
                "chunk_count": len(current_chunks),
                "is_relevant": result.is_relevant,
                "confidence": result.confidence,
            })
            
            # If relevant enough or no suggestion, stop
            if not result.should_retry or not result.suggestion:
                break
            
            # Try query expansion for retry
            if attempts < max_retries:
                # Modify query based on suggestion or just try different formulation
                if result.suggestion and "specific" in result.suggestion.lower():
                    # Keep original query, expand instead
                    current_query = query
                
                # Retry retrieval with expanded parameters
                try:
                    current_chunks = await retrieval_func(current_query)
                except Exception as e:
                    logger.warning(f"Retry retrieval failed: {e}")
                    break
            
            attempts += 1
        
        # Filter chunks based on final grading
        final_result = await self.grade(current_query, current_chunks, threshold)
        
        if final_result.relevant_indices:
            filtered_chunks = [current_chunks[i] for i in final_result.relevant_indices]
        else:
            filtered_chunks = current_chunks
        
        return {
            "chunks": filtered_chunks,
            "grading": final_result,
            "attempts": attempts,
            "history": grading_history,
        }
    
    def filter_chunks(
        self,
        chunks: List[str],
        grading_result: GradingResult,
    ) -> List[str]:
        """
        Filter chunks based on grading result.
        
        Returns only the relevant chunks.
        """
        if not grading_result.relevant_indices:
            return chunks  # Return all if no filtering info
        
        return [
            chunks[i] 
            for i in grading_result.relevant_indices 
            if i < len(chunks)
        ]


# Module-level singleton
_relevance_grader: Optional[RelevanceGrader] = None


def get_relevance_grader() -> RelevanceGrader:
    """Get the relevance grader singleton."""
    global _relevance_grader
    if _relevance_grader is None:
        _relevance_grader = RelevanceGrader()
    return _relevance_grader
