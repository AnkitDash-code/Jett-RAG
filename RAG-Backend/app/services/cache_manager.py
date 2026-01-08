"""
Cache Manager for Phase 7.
Unified in-memory caching with TTL, LRU eviction, and invalidation hooks.
"""
import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, List, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    value: T
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """Update access stats."""
        self.access_count += 1
        self.last_accessed_at = time.time()


class Cache(Generic[T]):
    """
    Thread-safe in-memory cache with TTL and LRU eviction.
    
    Features:
    - TTL (Time To Live) expiration
    - LRU (Least Recently Used) eviction
    - Tag-based invalidation
    - Statistics tracking
    """
    
    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        default_ttl_seconds: Optional[float] = 300,  # 5 minutes
    ):
        self.name = name
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Returns None if key doesn't exist or is expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired:
                self._remove(key)
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._hits += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: T,
        ttl_seconds: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live (None uses default, 0 means no expiration)
            tags: Tags for grouped invalidation
        """
        with self._lock:
            # Calculate expiration
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds
            
            expires_at = None
            if ttl_seconds is not None and ttl_seconds > 0:
                expires_at = time.time() + ttl_seconds
            
            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                tags=tags or [],
            )
            
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            # Store
            self._cache[key] = entry
            self._cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        with self._lock:
            return self._remove(key)
    
    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all entries with a specific tag.
        
        Returns number of entries invalidated.
        """
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._cache.items()
                if tag in entry.tags
            ]
            
            for key in keys_to_delete:
                self._remove(key)
            
            logger.debug(f"Cache '{self.name}': Invalidated {len(keys_to_delete)} entries with tag '{tag}'")
            return len(keys_to_delete)
    
    def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Invalidate all entries with keys starting with prefix.
        
        Returns number of entries invalidated.
        """
        with self._lock:
            keys_to_delete = [
                key for key in self._cache.keys()
                if key.startswith(prefix)
            ]
            
            for key in keys_to_delete:
                self._remove(key)
            
            return len(keys_to_delete)
    
    def clear(self) -> int:
        """Clear all entries from the cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache '{self.name}': Cleared {count} entries")
            return count
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.expires_at and entry.expires_at < now
            ]
            
            for key in expired_keys:
                self._remove(key)
            
            return len(expired_keys)
    
    def _remove(self, key: str) -> bool:
        """Remove a key from the cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove(key)
            self._evictions += 1
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "name": self.name,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self._evictions,
        }


class CacheManager:
    """
    Central manager for multiple caches.
    
    Provides a unified interface for creating and managing caches
    with support for invalidation hooks and monitoring.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._caches: Dict[str, Cache] = {}
            cls._instance._invalidation_hooks: Dict[str, List[Callable]] = {}
            cls._instance._cleanup_task: Optional[asyncio.Task] = None
        return cls._instance
    
    def get_cache(
        self,
        name: str,
        max_size: int = 1000,
        default_ttl_seconds: Optional[float] = 300,
    ) -> Cache:
        """Get or create a cache by name."""
        if name not in self._caches:
            self._caches[name] = Cache(
                name=name,
                max_size=max_size,
                default_ttl_seconds=default_ttl_seconds,
            )
        return self._caches[name]
    
    def register_invalidation_hook(
        self,
        cache_name: str,
        hook: Callable[[str], None],
    ) -> None:
        """
        Register a hook to be called when a cache is invalidated.
        
        Args:
            cache_name: Name of the cache
            hook: Function(tag) to call on invalidation
        """
        if cache_name not in self._invalidation_hooks:
            self._invalidation_hooks[cache_name] = []
        self._invalidation_hooks[cache_name].append(hook)
    
    def invalidate(
        self,
        cache_name: str,
        tag: Optional[str] = None,
        key: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            cache_name: Name of the cache
            tag: Invalidate by tag
            key: Invalidate specific key
            prefix: Invalidate by key prefix
            
        Returns:
            Number of entries invalidated
        """
        cache = self._caches.get(cache_name)
        if not cache:
            return 0
        
        count = 0
        
        if key:
            count += 1 if cache.delete(key) else 0
        
        if tag:
            count += cache.invalidate_by_tag(tag)
            # Call hooks
            for hook in self._invalidation_hooks.get(cache_name, []):
                try:
                    hook(tag)
                except Exception as e:
                    logger.error(f"Invalidation hook error: {e}")
        
        if prefix:
            count += cache.invalidate_by_prefix(prefix)
        
        return count
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            name: cache.stats
            for name, cache in self._caches.items()
        }
    
    def clear_all(self) -> Dict[str, int]:
        """Clear all caches."""
        return {
            name: cache.clear()
            for name, cache in self._caches.items()
        }
    
    async def start_cleanup_task(self, interval_seconds: float = 60) -> None:
        """Start background task to cleanup expired entries."""
        if self._cleanup_task is not None:
            return
        
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                for cache in self._caches.values():
                    cache.cleanup_expired()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Cache cleanup task started")
    
    def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return CacheManager()


def get_cache(
    name: str,
    max_size: int = 1000,
    default_ttl_seconds: Optional[float] = 300,
) -> Cache:
    """Get or create a cache by name."""
    return get_cache_manager().get_cache(name, max_size, default_ttl_seconds)


# ============================================================================
# PRE-CONFIGURED CACHES
# ============================================================================

# User/Auth cache
USER_CACHE = get_cache("users", max_size=500, default_ttl_seconds=300)

# Document metadata cache
DOCUMENT_CACHE = get_cache("documents", max_size=1000, default_ttl_seconds=600)

# Query result cache
QUERY_CACHE = get_cache("queries", max_size=200, default_ttl_seconds=120)

# Entity graph cache
ENTITY_CACHE = get_cache("entities", max_size=500, default_ttl_seconds=900)

# LLM response cache
LLM_CACHE = get_cache("llm", max_size=100, default_ttl_seconds=1800)
