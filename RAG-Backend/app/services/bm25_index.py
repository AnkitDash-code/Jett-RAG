"""
BM25 index for lexical search.
Provides keyword-based retrieval to complement vector similarity search.
"""
import logging
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 lexical index for hybrid search.
    Uses rank_bm25 library for efficient keyword matching.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.corpus: List[List[str]] = []  # Tokenized documents
            self.chunk_ids: List[str] = []  # Parallel list of chunk IDs
            self.metadata: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata
            self.bm25 = None
            self._load()
            BM25Index._initialized = True
    
    @property
    def _persist_dir(self) -> Path:
        """Get persistence directory."""
        base_dir = Path(settings.UPLOAD_DIR).parent
        persist_dir = base_dir / "vector_store"
        persist_dir.mkdir(parents=True, exist_ok=True)
        return persist_dir
    
    @property
    def _index_path(self) -> Path:
        return self._persist_dir / "bm25_index.pkl"
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        Simple whitespace + lowercase tokenization with basic cleaning.
        """
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split on whitespace and filter empty
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens
    
    def _rebuild_bm25(self):
        """Rebuild BM25 index from corpus."""
        if not self.corpus:
            self.bm25 = None
            return
        
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus)
            logger.info(f"Rebuilt BM25 index with {len(self.corpus)} documents")
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 search disabled.")
            self.bm25 = None
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {e}")
            self.bm25 = None
    
    def _load(self):
        """Load index from disk."""
        try:
            if self._index_path.exists():
                with open(self._index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.corpus = data.get("corpus", [])
                    self.chunk_ids = data.get("chunk_ids", [])
                    self.metadata = data.get("metadata", {})
                
                self._rebuild_bm25()
                logger.info(f"Loaded BM25 index with {len(self.corpus)} documents")
            else:
                logger.info("No existing BM25 index found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            self.corpus = []
            self.chunk_ids = []
            self.metadata = {}
    
    def _save(self):
        """Persist index to disk."""
        try:
            with open(self._index_path, 'wb') as f:
                pickle.dump({
                    "corpus": self.corpus,
                    "chunk_ids": self.chunk_ids,
                    "metadata": self.metadata,
                }, f)
        except Exception as e:
            logger.error(f"Error saving BM25 index: {e}")
    
    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        document_id: str,
        tenant_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        access_level: str = "private",
    ):
        """Add a single chunk to the index."""
        if chunk_id in self.metadata:
            logger.debug(f"Chunk {chunk_id} already in BM25 index, skipping")
            return
        
        tokens = self._tokenize(text)
        if not tokens:
            logger.debug(f"Chunk {chunk_id} has no tokens, skipping")
            return
        
        self.corpus.append(tokens)
        self.chunk_ids.append(chunk_id)
        self.metadata[chunk_id] = {
            "document_id": document_id,
            "tenant_id": tenant_id,
            "owner_id": str(owner_id) if owner_id else None,
            "access_level": access_level,
        }
        
        # Rebuild BM25 (relatively fast for BM25Okapi)
        self._rebuild_bm25()
        self._save()
    
    def add_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """Add multiple chunks to the index."""
        added = 0
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in self.metadata:
                continue
            
            text = chunk.get("text", "")
            tokens = self._tokenize(text)
            if not tokens:
                continue
            
            self.corpus.append(tokens)
            self.chunk_ids.append(chunk_id)
            self.metadata[chunk_id] = {
                "document_id": str(chunk.get("document_id", "")),
                "tenant_id": chunk.get("tenant_id"),
                "owner_id": str(chunk["owner_id"]) if chunk.get("owner_id") else None,
                "access_level": chunk.get("access_level", "private"),
            }
            added += 1
        
        if added > 0:
            self._rebuild_bm25()
            self._save()
            logger.info(f"Added {added} chunks to BM25 index")
        
        return added
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        access_levels: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for chunks matching query.
        Returns list of (chunk_id, bm25_score).
        """
        if self.bm25 is None or not self.corpus:
            return []
        
        tokens = self._tokenize(query)
        if not tokens:
            return []
        
        try:
            scores = self.bm25.get_scores(tokens)
            
            # Create (idx, score) pairs and filter
            results = []
            for idx, score in enumerate(scores):
                if score <= 0:
                    continue
                
                chunk_id = self.chunk_ids[idx]
                meta = self.metadata.get(chunk_id, {})
                
                # Apply RBAC filters
                if tenant_id and meta.get("tenant_id") != tenant_id:
                    continue
                
                chunk_access = meta.get("access_level", "private")
                chunk_owner = meta.get("owner_id")
                
                if chunk_access == "private":
                    if owner_id and chunk_owner and chunk_owner != owner_id:
                        continue
                elif access_levels and chunk_access not in access_levels:
                    continue
                
                if document_ids and meta.get("document_id") not in document_ids:
                    continue
                
                results.append((chunk_id, float(score)))
            
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    
    def delete_by_document(self, document_id: str) -> int:
        """Remove all chunks for a document."""
        to_delete = [
            cid for cid, meta in self.metadata.items()
            if meta.get("document_id") == document_id
        ]
        
        if not to_delete:
            return 0
        
        # Rebuild corpus without deleted chunks
        new_corpus = []
        new_chunk_ids = []
        
        for i, chunk_id in enumerate(self.chunk_ids):
            if chunk_id not in to_delete:
                new_corpus.append(self.corpus[i])
                new_chunk_ids.append(chunk_id)
        
        # Update state
        self.corpus = new_corpus
        self.chunk_ids = new_chunk_ids
        for cid in to_delete:
            self.metadata.pop(cid, None)
        
        self._rebuild_bm25()
        self._save()
        
        logger.info(f"Removed {len(to_delete)} chunks from BM25 index for document {document_id}")
        return len(to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_documents": len(self.corpus),
            "total_chunks": len(self.chunk_ids),
            "has_bm25": self.bm25 is not None,
        }


# Singleton instance
bm25_index = BM25Index()
