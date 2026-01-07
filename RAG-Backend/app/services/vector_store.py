"""
FAISS vector store service for document chunk embeddings.
Handles indexing and similarity search with disk persistence.

Current: FAISS (no external dependencies, works on Windows)
Phase 2: Weaviate with Docker for production hybrid search
"""
import logging
import os
import json
import pickle
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store with disk persistence.
    Uses cosine similarity (via normalized vectors + inner product).
    """
    
    _instance = None
    _index = None
    _metadata: Dict[str, Dict[str, Any]] = {}
    _embeddings: Dict[str, List[float]] = {}  # chunk_id -> embedding (for rebuild)
    _id_to_idx: Dict[str, int] = {}  # chunk_id -> FAISS index position
    _idx_to_id: Dict[int, str] = {}  # FAISS index position -> chunk_id
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._load()
            FAISSVectorStore._initialized = True
    
    @property
    def _persist_dir(self) -> Path:
        """Get persistence directory."""
        base_dir = Path(settings.UPLOAD_DIR).parent
        persist_dir = base_dir / "vector_store"
        persist_dir.mkdir(parents=True, exist_ok=True)
        return persist_dir
    
    @property
    def _index_path(self) -> Path:
        return self._persist_dir / "faiss.index"
    
    @property
    def _metadata_path(self) -> Path:
        return self._persist_dir / "metadata.json"
    
    @property
    def _mappings_path(self) -> Path:
        return self._persist_dir / "mappings.pkl"
    
    @property
    def _embeddings_path(self) -> Path:
        return self._persist_dir / "embeddings.pkl"
    
    def _load(self):
        """Load index and metadata from disk."""
        try:
            import faiss
            
            if self._index_path.exists():
                self._index = faiss.read_index(str(self._index_path))
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
            else:
                # Create empty index (Inner Product for cosine similarity on normalized vectors)
                self._index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
                logger.info("Created new FAISS index")
            
            if self._metadata_path.exists():
                with open(self._metadata_path, 'r') as f:
                    self._metadata = json.load(f)
            
            if self._mappings_path.exists():
                with open(self._mappings_path, 'rb') as f:
                    mappings = pickle.load(f)
                    self._id_to_idx = mappings.get("id_to_idx", {})
                    self._idx_to_id = mappings.get("idx_to_id", {})
            
            # Load embeddings for rebuild capability
            if self._embeddings_path.exists():
                with open(self._embeddings_path, 'rb') as f:
                    self._embeddings = pickle.load(f)
                    
        except ImportError:
            logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            # Initialize empty
            import faiss
            self._index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
            self._metadata = {}
            self._embeddings = {}
            self._id_to_idx = {}
            self._idx_to_id = {}
    
    def _save(self):
        """Persist index and metadata to disk."""
        try:
            import faiss
            
            faiss.write_index(self._index, str(self._index_path))
            
            with open(self._metadata_path, 'w') as f:
                json.dump(self._metadata, f)
            
            with open(self._mappings_path, 'wb') as f:
                pickle.dump({
                    "id_to_idx": self._id_to_idx,
                    "idx_to_id": self._idx_to_id,
                }, f)
            
            # Save embeddings for rebuild capability
            with open(self._embeddings_path, 'wb') as f:
                pickle.dump(self._embeddings, f)
                
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def is_available(self) -> bool:
        """Check if FAISS is available."""
        return self._index is not None
    
    async def index_chunk(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        embedding: List[float],
        document_name: str = "",
        page_number: Optional[int] = None,
        section_path: Optional[str] = None,
        chunk_index: int = 0,
        tenant_id: Optional[str] = None,
        access_level: str = "private",
        department: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> bool:
        """Index a single chunk with its embedding."""
        if self._index is None:
            return False
        
        try:
            # Check if already exists
            if chunk_id in self._id_to_idx:
                logger.debug(f"Chunk {chunk_id} already indexed, skipping")
                return True
            
            # Store embedding for rebuild capability
            self._embeddings[chunk_id] = embedding
            
            # Normalize and add to index
            vector = np.array([embedding], dtype=np.float32)
            vector = self._normalize(vector)
            
            idx = self._index.ntotal
            self._index.add(vector)
            
            # Store mappings
            self._id_to_idx[chunk_id] = idx
            self._idx_to_id[idx] = chunk_id
            
            # Store metadata
            self._metadata[chunk_id] = {
                "document_id": document_id,
                "text": text,
                "document_name": document_name,
                "page_number": page_number,
                "section_path": section_path,
                "chunk_index": chunk_index,
                "tenant_id": tenant_id,
                "access_level": access_level,
                "department": department,
                "owner_id": str(owner_id) if owner_id else None,
            }
            
            self._save()
            return True
            
        except Exception as e:
            logger.error(f"Error indexing chunk {chunk_id}: {e}")
            return False
    
    async def index_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """Batch index multiple chunks."""
        if self._index is None or not chunks:
            return 0
        
        try:
            vectors = []
            new_chunks = []
            
            for chunk in chunks:
                chunk_id = str(chunk["chunk_id"])
                if chunk_id in self._id_to_idx:
                    continue  # Skip already indexed
                new_chunks.append(chunk)
                vectors.append(chunk["embedding"])
                # Store embedding for rebuild
                self._embeddings[chunk_id] = chunk["embedding"]
            
            if not vectors:
                return 0
            
            # Normalize and add to index
            vectors = np.array(vectors, dtype=np.float32)
            vectors = self._normalize(vectors)
            
            start_idx = self._index.ntotal
            self._index.add(vectors)
            
            # Store mappings and metadata
            for i, chunk in enumerate(new_chunks):
                chunk_id = str(chunk["chunk_id"])
                idx = start_idx + i
                
                self._id_to_idx[chunk_id] = idx
                self._idx_to_id[idx] = chunk_id
                
                self._metadata[chunk_id] = {
                    "document_id": str(chunk["document_id"]),
                    "text": chunk.get("text", ""),
                    "document_name": chunk.get("document_name", ""),
                    "page_number": chunk.get("page_number"),
                    "section_path": chunk.get("section_path"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "tenant_id": chunk.get("tenant_id"),
                    "access_level": chunk.get("access_level", "private"),
                    "department": chunk.get("department"),
                    "owner_id": str(chunk["owner_id"]) if chunk.get("owner_id") else None,
                }
            
            self._save()
            logger.info(f"Indexed {len(new_chunks)} chunks in FAISS")
            return len(new_chunks)
            
        except Exception as e:
            logger.error(f"Error batch indexing: {e}")
            return 0
    
    async def search(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,  # Kept for API compatibility (not used in pure vector search)
        tenant_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        access_levels: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        alpha: float = 0.5,  # Kept for API compatibility
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search with RBAC filtering.
        Returns list of {chunk_id, score, text, metadata}.
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        
        try:
            # Normalize query vector
            query = np.array([query_embedding], dtype=np.float32)
            query = self._normalize(query)
            
            # Search more than top_k to allow for filtering
            search_k = min(top_k * 3, self._index.ntotal)
            scores, indices = self._index.search(query, search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                chunk_id = self._idx_to_id.get(idx)
                if not chunk_id:
                    continue
                
                meta = self._metadata.get(chunk_id, {})
                
                # Apply RBAC filters
                if tenant_id and meta.get("tenant_id") != tenant_id:
                    continue
                
                # For private documents, owner can always access
                # For non-private, check access_levels
                chunk_access = meta.get("access_level", "private")
                chunk_owner = meta.get("owner_id")
                
                # Skip if:
                # - It's private AND owner doesn't match (if owner_id specified)
                # - It's not private AND access_level not in allowed list
                if chunk_access == "private":
                    # Private chunks: owner can access, or if no owner_id filter specified, allow all
                    if owner_id and chunk_owner and chunk_owner != owner_id:
                        continue
                elif access_levels and chunk_access not in access_levels:
                    # Non-private chunks: check access level
                    continue
                
                if document_ids and meta.get("document_id") not in document_ids:
                    continue
                
                results.append({
                    "chunk_id": chunk_id,
                    "document_id": meta.get("document_id"),
                    "text": meta.get("text", ""),
                    "score": float(score),  # Cosine similarity (0-1)
                    "document_name": meta.get("document_name", ""),
                    "page_number": meta.get("page_number"),
                    "section_path": meta.get("section_path"),
                    "access_level": meta.get("access_level"),
                })
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []
    
    async def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        Rebuilds the FAISS index from persisted embeddings.
        """
        if self._index is None:
            return 0
        
        try:
            # Find chunks to delete
            to_delete = [
                cid for cid, meta in self._metadata.items()
                if meta.get("document_id") == document_id
            ]
            
            if not to_delete:
                return 0
            
            # Remove from metadata and embeddings
            for cid in to_delete:
                self._metadata.pop(cid, None)
                self._embeddings.pop(cid, None)
            
            # Rebuild FAISS index from remaining embeddings
            await self._rebuild_index()
            
            self._save()
            logger.info(f"Deleted {len(to_delete)} chunks for document {document_id}")
            return len(to_delete)
            
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {e}")
            return 0
    
    async def _rebuild_index(self):
        """Rebuild FAISS index from stored embeddings."""
        import faiss
        
        # Clear current index
        self._index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
        self._id_to_idx = {}
        self._idx_to_id = {}
        
        if not self._embeddings:
            logger.info("No embeddings to rebuild, index is now empty")
            return
        
        # Get remaining chunk IDs (those still in metadata)
        remaining_ids = [cid for cid in self._embeddings.keys() if cid in self._metadata]
        
        if not remaining_ids:
            # Clean up orphaned embeddings
            self._embeddings = {}
            return
        
        # Build vectors array
        vectors = []
        for cid in remaining_ids:
            vectors.append(self._embeddings[cid])
        
        vectors = np.array(vectors, dtype=np.float32)
        vectors = self._normalize(vectors)
        
        # Add to index
        self._index.add(vectors)
        
        # Rebuild mappings
        for i, cid in enumerate(remaining_ids):
            self._id_to_idx[cid] = i
            self._idx_to_id[i] = cid
        
        # Remove orphaned embeddings
        self._embeddings = {cid: self._embeddings[cid] for cid in remaining_ids}
        
        logger.info(f"Rebuilt FAISS index with {len(remaining_ids)} vectors")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "backend": "faiss",
            "total_vectors": self._index.ntotal if self._index else 0,
            "total_chunks": len(self._metadata),
            "dimension": settings.EMBEDDING_DIMENSION,
        }
    
    def close(self):
        """Save and close."""
        self._save()


# Singleton instance
faiss_service = FAISSVectorStore()

# Aliases for backward compatibility
weaviate_service = faiss_service
milvus_service = faiss_service
WeaviateService = FAISSVectorStore
MilvusService = FAISSVectorStore
