"""
Cross-encoder reranking service.
Provides high-precision reranking of retrieval candidates.
"""
import logging
import time
from typing import List, Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Cross-encoder based reranking service.
    Uses sentence-transformers CrossEncoder for precise query-document scoring.
    """
    
    _instance = None
    _model = None
    _model_name = None
    
    def __new__(cls):
        """Singleton pattern for model caching."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if RerankerService._model is not None and RerankerService._model_name == self.model_name:
            return RerankerService._model
        
        try:
            import torch
            from sentence_transformers import CrossEncoder
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading cross-encoder model on {device}: {self.model_name}")
            
            RerankerService._model = CrossEncoder(
                self.model_name,
                device=device,
            )
            RerankerService._model_name = self.model_name
            
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
            return RerankerService._model
            
        except ImportError as e:
            logger.warning(f"sentence-transformers not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self._load_model() is not None
    
    def rerank(
        self,
        query: str,
        chunks: List[Any],
        top_k: int = None,
        score_field: str = "rerank_score",
    ) -> tuple[List[Any], int]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: The search query
            chunks: List of chunk objects (must have 'text' or 'chunk.text' attribute)
            top_k: Number of top results to return (default: settings.RERANK_TOP_K)
            score_field: Name of field to store rerank score
        
        Returns:
            Tuple of (reranked chunks, elapsed_ms)
        """
        if not chunks:
            return chunks, 0
        
        top_k = top_k or settings.RERANK_TOP_K
        start_time = time.perf_counter()
        
        model = self._load_model()
        if model is None:
            # Fallback: sort by existing score
            logger.warning("Cross-encoder not available, using fallback sorting")
            sorted_chunks = sorted(
                chunks,
                key=lambda c: getattr(c, 'final_score', 0) or getattr(c, 'vector_score', 0),
                reverse=True
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return sorted_chunks[:top_k], elapsed_ms
        
        try:
            # Extract text from chunks
            pairs = []
            for chunk in chunks:
                # Handle different chunk formats
                if hasattr(chunk, 'chunk') and hasattr(chunk.chunk, 'text'):
                    text = chunk.chunk.text
                elif hasattr(chunk, 'text'):
                    text = chunk.text
                elif isinstance(chunk, dict):
                    text = chunk.get('text', '')
                else:
                    text = str(chunk)
                
                pairs.append((query, text))
            
            # Score all pairs
            scores = model.predict(pairs, show_progress_bar=False)
            
            # Attach scores to chunks
            for chunk, score in zip(chunks, scores):
                if hasattr(chunk, score_field):
                    setattr(chunk, score_field, float(score))
                elif hasattr(chunk, '__setattr__'):
                    setattr(chunk, score_field, float(score))
                
                # Also update final_score for downstream use
                if hasattr(chunk, 'final_score'):
                    chunk.final_score = float(score)
            
            # Sort by rerank score
            reranked = sorted(
                chunks,
                key=lambda c: getattr(c, score_field, 0),
                reverse=True
            )
            
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug(f"Reranked {len(chunks)} chunks in {elapsed_ms}ms")
            
            return reranked[:top_k], elapsed_ms
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return chunks[:top_k], elapsed_ms
    
    def score_pair(self, query: str, text: str) -> float:
        """Score a single query-document pair."""
        model = self._load_model()
        if model is None:
            return 0.0
        
        try:
            score = model.predict([(query, text)])[0]
            return float(score)
        except Exception as e:
            logger.error(f"Error scoring pair: {e}")
            return 0.0


# Singleton instance
reranker_service = RerankerService()
