"""
KoboldCpp Embedding Client for GGUF models.

Uses local KoboldCpp instance running Qwen3-Embedding model.
Provides embeddings via koboldcpp's /api/extra/generate/embeddings endpoint.
"""
import logging
import asyncio
from typing import List, Optional, Union
import numpy as np

import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class KoboldEmbeddingService:
    """
    Embedding service using local KoboldCpp with GGUF model.
    
    Replaces sentence-transformers with koboldcpp for offline GGUF embeddings.
    """
    
    _instance = None
    _model_loaded = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.base_url = getattr(settings, 'EMBEDDING_KOBOLD_URL', 'http://localhost:8001')
        self.timeout = 60.0
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
        self._dimension: Optional[int] = None
        self._initialized = True
        
        logger.info(f"KoboldEmbeddingService initialized: {self.base_url}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._sync_client
    
    async def close(self):
        """Close HTTP clients."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
    
    def is_available(self) -> bool:
        """Check if koboldcpp embedding server is available."""
        try:
            client = self._get_sync_client()
            response = client.get("/api/v1/model")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Kobold embedding server not available: {e}")
            return False
    
    async def embed_async(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings asynchronously.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            numpy array of embeddings (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        client = await self._get_client()
        embeddings = []
        
        for text in texts:
            try:
                # KoboldCpp embedding endpoint
                response = await client.post(
                    "/api/extra/generate/embeddings",
                    json={"content": text}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Handle different response formats
                    if "embedding" in result:
                        embedding = result["embedding"]
                    elif "embeddings" in result:
                        embedding = result["embeddings"][0] if result["embeddings"] else []
                    elif "data" in result:
                        embedding = result["data"][0]["embedding"]
                    else:
                        embedding = result
                    
                    embeddings.append(embedding)
                    
                    # Cache dimension
                    if self._dimension is None:
                        self._dimension = len(embedding)
                else:
                    logger.error(f"Embedding request failed: {response.status_code}")
                    # Return zero vector on error
                    embeddings.append([0.0] * (self._dimension or 384))
                    
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                embeddings.append([0.0] * (self._dimension or 384))
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings synchronously.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            numpy array of embeddings (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        client = self._get_sync_client()
        embeddings = []
        
        for text in texts:
            try:
                response = client.post(
                    "/api/extra/generate/embeddings",
                    json={"content": text}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "embedding" in result:
                        embedding = result["embedding"]
                    elif "embeddings" in result:
                        embedding = result["embeddings"][0] if result["embeddings"] else []
                    elif "data" in result:
                        embedding = result["data"][0]["embedding"]
                    else:
                        embedding = result
                    
                    embeddings.append(embedding)
                    
                    if self._dimension is None:
                        self._dimension = len(embedding)
                else:
                    embeddings.append([0.0] * (self._dimension or 384))
                    
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                embeddings.append([0.0] * (self._dimension or 384))
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Try to get dimension by embedding a test string
            try:
                test_embedding = self.embed("test")
                self._dimension = test_embedding.shape[1]
            except:
                self._dimension = 384  # Default fallback
        return self._dimension
    
    @classmethod
    def preload_model(cls):
        """Check if koboldcpp is available (model is loaded externally)."""
        instance = cls()
        if instance.is_available():
            logger.info("Kobold embedding server is available")
            KoboldEmbeddingService._model_loaded = True
        else:
            logger.warning(
                "Kobold embedding server not available. "
                "Start it with: start_embedding.bat"
            )


# Singleton accessor
_kobold_embedding_service: Optional[KoboldEmbeddingService] = None


def get_kobold_embedding_service() -> KoboldEmbeddingService:
    """Get the Kobold embedding service singleton."""
    global _kobold_embedding_service
    if _kobold_embedding_service is None:
        _kobold_embedding_service = KoboldEmbeddingService()
    return _kobold_embedding_service
