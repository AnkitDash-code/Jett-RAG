"""
Query Classification & Routing Service.

Classifies queries by type and complexity to determine optimal retrieval strategy.
"""
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from app.services.utility_llm_client import get_utility_llm

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries with different retrieval needs."""
    FACT = "FACT"                    # Simple factual lookup
    ENTITY = "ENTITY"                # Entity/definition lookup
    COMPARISON = "COMPARISON"        # Compare multiple items
    SUMMARY = "SUMMARY"              # Summarization request
    TROUBLESHOOTING = "TROUBLESHOOTING"  # Problem-solving
    CONVERSATIONAL = "CONVERSATIONAL"    # Follow-up/conversational


class QueryComplexity(str, Enum):
    """Complexity levels affecting retrieval depth."""
    SIMPLE = "simple"       # Single retrieval pass
    MODERATE = "moderate"   # May need reranking
    COMPLEX = "complex"     # Multi-pass, expansion needed


@dataclass
class QueryClassification:
    """Classification result for a query."""
    query_type: QueryType
    complexity: QueryComplexity
    needs_graph: bool
    is_followup: bool
    confidence: float
    routing: Dict[str, Any]  # Retrieval strategy parameters


class QueryClassificationService:
    """
    Service for classifying queries and determining retrieval strategy.
    
    Classification determines:
    - Query type (FACT, ENTITY, COMPARISON, etc.)
    - Complexity (affects num_chunks, reranking)
    - Graph needs (for entity relationship queries)
    - Follow-up detection (for conversation context)
    """
    
    def __init__(self):
        self._utility_llm = get_utility_llm()
        logger.info(f"QueryClassificationService initialized (LLM: {self._utility_llm.is_available()})")
    
    async def classify(self, query: str) -> QueryClassification:
        """
        Classify a query and determine retrieval routing.
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification with type, complexity, and routing
        """
        query = query.strip()
        
        if not query:
            return self._default_classification()
        
        # Try LLM classification
        if self._utility_llm.is_available():
            try:
                result = await self._utility_llm.classify_query(query)
                classification = self._build_classification(result, query)
                return classification
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
        
        # Fallback to heuristic classification
        result = self._utility_llm._heuristic_classification(query)
        return self._build_classification(result, query)
    
    def _build_classification(
        self,
        llm_result: Dict[str, Any],
        query: str,
    ) -> QueryClassification:
        """Build classification object from LLM/heuristic result."""
        # Map type string to enum
        type_str = llm_result.get("type", "FACT")
        try:
            query_type = QueryType(type_str)
        except ValueError:
            query_type = QueryType.FACT
        
        # Map complexity
        complexity_str = llm_result.get("complexity", "simple")
        try:
            complexity = QueryComplexity(complexity_str)
        except ValueError:
            complexity = QueryComplexity.SIMPLE
        
        # Build routing based on classification
        routing = self._determine_routing(query_type, complexity, query)
        
        return QueryClassification(
            query_type=query_type,
            complexity=complexity,
            needs_graph=llm_result.get("needs_graph", False),
            is_followup=llm_result.get("is_followup", False),
            confidence=llm_result.get("confidence", 0.7),
            routing=routing,
        )
    
    def _determine_routing(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        query: str,
    ) -> Dict[str, Any]:
        """
        Determine retrieval routing parameters based on classification.
        
        Returns:
            Dict with retrieval parameters
        """
        # Base configuration
        routing = {
            "num_chunks": 5,
            "use_hybrid": True,
            "use_reranking": True,
            "use_expansion": False,
            "use_graph": False,
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
            "rerank_top_n": 10,
        }
        
        # Adjust based on query type
        if query_type == QueryType.FACT:
            routing["num_chunks"] = 3
            routing["use_expansion"] = False
            routing["semantic_weight"] = 0.6
            routing["keyword_weight"] = 0.4
        
        elif query_type == QueryType.ENTITY:
            routing["num_chunks"] = 5
            routing["use_graph"] = True  # Entity queries benefit from graph
            routing["use_expansion"] = True
        
        elif query_type == QueryType.COMPARISON:
            routing["num_chunks"] = 8
            routing["use_expansion"] = True
            routing["use_graph"] = True
            routing["rerank_top_n"] = 15
        
        elif query_type == QueryType.SUMMARY:
            routing["num_chunks"] = 10
            routing["use_expansion"] = False
            routing["semantic_weight"] = 0.8
            routing["keyword_weight"] = 0.2
        
        elif query_type == QueryType.TROUBLESHOOTING:
            routing["num_chunks"] = 7
            routing["use_expansion"] = True
            routing["keyword_weight"] = 0.5  # Keywords important for error messages
            routing["semantic_weight"] = 0.5
        
        elif query_type == QueryType.CONVERSATIONAL:
            routing["num_chunks"] = 5
            routing["use_expansion"] = True  # Expand to handle context
        
        # Adjust based on complexity
        if complexity == QueryComplexity.COMPLEX:
            routing["num_chunks"] = min(routing["num_chunks"] + 3, 15)
            routing["use_expansion"] = True
            routing["use_reranking"] = True
            routing["rerank_top_n"] = min(routing["rerank_top_n"] + 5, 20)
        
        elif complexity == QueryComplexity.SIMPLE:
            routing["use_expansion"] = False
            routing["use_reranking"] = False  # Skip reranking for simple queries
        
        return routing
    
    def _default_classification(self) -> QueryClassification:
        """Return default classification for empty/invalid queries."""
        return QueryClassification(
            query_type=QueryType.FACT,
            complexity=QueryComplexity.SIMPLE,
            needs_graph=False,
            is_followup=False,
            confidence=0.0,
            routing={
                "num_chunks": 5,
                "use_hybrid": True,
                "use_reranking": True,
                "use_expansion": False,
                "use_graph": False,
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "rerank_top_n": 10,
            },
        )
    
    async def get_routing_for_query(
        self,
        query: str,
        user_role: str = "user",
    ) -> Dict[str, Any]:
        """
        Get retrieval routing configuration for a query.
        
        Convenience method that returns only routing config.
        """
        classification = await self.classify(query)
        
        # Add classification metadata to routing
        routing = classification.routing.copy()
        routing["query_type"] = classification.query_type.value
        routing["complexity"] = classification.complexity.value
        routing["is_followup"] = classification.is_followup
        
        return routing


# Module-level singleton
_classification_service: Optional[QueryClassificationService] = None


def get_query_classification_service() -> QueryClassificationService:
    """Get the query classification service singleton."""
    global _classification_service
    if _classification_service is None:
        _classification_service = QueryClassificationService()
    return _classification_service
