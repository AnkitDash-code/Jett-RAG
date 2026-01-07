"""
Query Expansion Service for Multi-Query RAG.

Generates multiple query variants to improve retrieval recall.
Uses the utility LLM when available, with fallback to heuristics.
"""
import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from app.services.utility_llm_client import get_utility_llm

logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original: str
    variants: List[str]
    keywords: List[str]
    method: str  # "llm" or "heuristic"


class QueryExpansionService:
    """
    Service for generating query variants.
    
    Supports:
    - LLM-based paraphrasing (when utility LLM is available)
    - Keyword extraction for BM25 boost
    - Heuristic fallback expansion
    """
    
    def __init__(self):
        self._utility_llm = get_utility_llm()
        logger.info(f"QueryExpansionService initialized (LLM enabled: {self._utility_llm.is_available()})")
    
    async def expand(
        self,
        query: str,
        num_variants: int = 3,
        extract_keywords: bool = True,
    ) -> ExpandedQuery:
        """
        Expand a query into multiple variants.
        
        Args:
            query: Original user query
            num_variants: Number of variants to generate
            extract_keywords: Whether to extract keywords
            
        Returns:
            ExpandedQuery with variants and keywords
        """
        query = query.strip()
        
        if not query:
            return ExpandedQuery(
                original=query,
                variants=[query] if query else [],
                keywords=[],
                method="none",
            )
        
        # Try LLM expansion first
        if self._utility_llm.is_available():
            try:
                variants = await self._utility_llm.expand_query(query, num_variants)
                keywords = []
                
                if extract_keywords:
                    keywords = await self._utility_llm.extract_keywords(query)
                
                return ExpandedQuery(
                    original=query,
                    variants=variants,
                    keywords=keywords,
                    method="llm",
                )
            except Exception as e:
                logger.warning(f"LLM expansion failed, using heuristics: {e}")
        
        # Fallback to heuristic expansion
        variants = self._heuristic_expand(query, num_variants)
        keywords = self._heuristic_keywords(query) if extract_keywords else []
        
        return ExpandedQuery(
            original=query,
            variants=variants,
            keywords=keywords,
            method="heuristic",
        )
    
    def _heuristic_expand(self, query: str, num_variants: int) -> List[str]:
        """
        Generate query variants using heuristic rules.
        
        Strategies:
        - Remove question words
        - Rephrase as statement
        - Extract key phrases
        """
        variants = [query]  # Always include original
        query_lower = query.lower()
        
        # Strategy 1: Remove question words and punctuation
        question_patterns = [
            (r'^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does)\s+', ''),
            (r'\?$', ''),
        ]
        variant = query
        for pattern, replacement in question_patterns:
            variant = re.sub(pattern, replacement, variant, flags=re.IGNORECASE)
        variant = variant.strip()
        if variant and variant != query and variant not in variants:
            variants.append(variant)
        
        # Strategy 2: "Explain X" / "Tell me about X" versions
        # Extract the core topic
        core_topic = query
        for prefix in ["what is", "what are", "how to", "how do i", "explain", "tell me about"]:
            if query_lower.startswith(prefix):
                core_topic = query[len(prefix):].strip()
                break
        
        if core_topic != query:
            # Create "explain" version
            explain_variant = f"explain {core_topic}"
            if explain_variant not in variants:
                variants.append(explain_variant)
        
        # Strategy 3: Add "information about" prefix
        info_variant = f"information about {core_topic if core_topic != query else query}"
        info_variant = re.sub(r'\?$', '', info_variant).strip()
        if info_variant not in variants and len(variants) < num_variants + 1:
            variants.append(info_variant)
        
        # Strategy 4: Keyword-focused query
        keywords = self._heuristic_keywords(query)
        if keywords and len(variants) < num_variants + 1:
            keyword_query = " ".join(keywords[:4])
            if keyword_query not in variants:
                variants.append(keyword_query)
        
        return variants[:num_variants + 1]
    
    def _heuristic_keywords(self, query: str) -> List[str]:
        """Extract keywords using simple heuristics."""
        # Stopwords to filter out
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'what', 'how', 'why', 'when',
            'where', 'who', 'which', 'that', 'this', 'these', 'those', 'it', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'they', 'their', 'them',
            'and', 'or', 'but', 'if', 'then', 'so', 'as', 'of', 'at', 'by', 'for',
            'with', 'about', 'into', 'to', 'from', 'in', 'on', 'up', 'out', 'all',
            'some', 'any', 'no', 'not', 'only', 'just', 'also', 'very', 'too',
            'tell', 'explain', 'describe', 'show', 'give', 'please', 'help',
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:5]
    
    async def expand_for_hybrid_search(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Expand query specifically for hybrid search.
        
        Returns dict optimized for hybrid retriever:
        - semantic_queries: Queries for vector search
        - keyword_queries: Terms for BM25 search
        """
        expanded = await self.expand(query, num_variants=2, extract_keywords=True)
        
        return {
            "semantic_queries": expanded.variants,
            "keyword_queries": expanded.keywords,
            "original": expanded.original,
            "method": expanded.method,
        }


# Module-level singleton
_expansion_service: Optional[QueryExpansionService] = None


def get_query_expansion_service() -> QueryExpansionService:
    """Get the query expansion service singleton."""
    global _expansion_service
    if _expansion_service is None:
        _expansion_service = QueryExpansionService()
    return _expansion_service
