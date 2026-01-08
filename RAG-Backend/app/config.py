"""
Application configuration using Pydantic Settings.
Loads from environment variables or .env file.
"""
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    APP_NAME: str = "RAG Backend"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    
    # API
    API_V1_PREFIX: str = "/v1"
    # Allow all origins in dev mode for port-forwarding/tunneling
    CORS_ORIGINS: list[str] = ["*"]
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./rag_backend.db"
    DATABASE_ECHO: bool = False
    
    # Redis (for caching and task queue)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # LLM Backend (your FastAPI wrapper on port 8080)
    LLM_API_BASE_URL: str = "http://localhost:8080"
    LLM_MODEL_NAME: str = "mistral-7b-instruct"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1
    
    # Utility LLM (uses same API as main LLM but with different parameters)
    # Same endpoint, but can be configured for lighter model if available
    UTILITY_LLM_URL: str = "http://localhost:8080"  # Same as main LLM API
    UTILITY_LLM_MODEL: str = "mistral-7b-instruct"  # Same model, different params
    UTILITY_LLM_MAX_TOKENS: int = 256  # Shorter responses for utility tasks
    UTILITY_LLM_TEMPERATURE: float = 0.2  # Lower temp for deterministic outputs
    UTILITY_LLM_ENABLED: bool = True  # Enable for Phase 3 query intelligence
    
    # Vector Database (FAISS - current implementation)
    # Phase 2: Weaviate with Docker for production hybrid search
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # GraphRAG Settings (rustworkx + SQLite)
    ENABLE_GRAPH_RAG: bool = False
    GRAPH_CACHE_TTL_SECONDS: int = 300  # 5 minutes
    GRAPH_MAX_HOPS: int = 1  # Max traversal depth
    GRAPH_SCORE_WEIGHT: float = 0.3  # Weight in final score (vector=0.4, rerank=0.3, graph=0.3)
    GRAPH_PROCESSING_WORKERS: int = 2  # ThreadPoolExecutor workers
    GRAPH_BATCH_SIZE: int = 10  # Chunks per batch for graph indexing
    ENABLE_COMMUNITY_SUMMARIES: bool = True  # Generate community summaries
    ENABLE_GROUP_PERMISSIONS: bool = True  # Enable group-based RBAC
    COMMUNITY_REBUILD_THRESHOLD: int = 50  # Rebuild communities after N new entities
    GRAPH_ENTITY_FUZZY_THRESHOLD: int = 2  # Levenshtein distance for entity matching
    COMMUNITY_DETECTION_ALGORITHM: str = "louvain"  # "connected_components" | "louvain"
    LOUVAIN_RESOLUTION: float = 1.0  # Resolution parameter for Louvain
    MIN_COMMUNITY_SIZE: int = 3  # Minimum entities for a community
    
    # Phase 5: Task Queue Settings
    TASK_QUEUE_WORKERS: int = 4  # Number of async worker tasks
    TASK_QUEUE_MAX_SIZE: int = 1000  # Max pending tasks in queue
    JOB_CLEANUP_DAYS: int = 7  # Delete old jobs after N days
    
    # Phase 5: Hierarchical Chunking
    ENABLE_HIERARCHICAL_CHUNKING: bool = True
    PARENT_CHUNK_SIZE: int = 1024  # Parent chunk token size
    HIERARCHY_LEVELS: int = 2  # 0=leaf, 1=section, 2=document
    
    # Phase 5: Auto-Retry Self-Reflection
    ENABLE_AUTO_RETRY: bool = True
    MAX_RETRY_ATTEMPTS: int = 2
    RETRY_RELEVANCE_THRESHOLD: float = 0.6
    
    # Phase 5: Parser Settings
    ENABLE_DOCLING_PARSER: bool = True
    DOCLING_TIMEOUT_SECONDS: int = 60
    ENABLE_OCR_DETECTION: bool = True
    OCR_MIN_TEXT_THRESHOLD: int = 100  # Chars below this triggers OCR
    
    # Phase 5: Stream Cancellation
    STREAM_TIMEOUT_SECONDS: int = 300  # Auto-cancel streams after this
    
    # Phase 5: Maintenance Jobs
    ENABLE_MAINTENANCE_JOBS: bool = True
    MAINTENANCE_CLEANUP_ENABLED: bool = True
    MAINTENANCE_COMMUNITY_REBUILD_ENABLED: bool = True
    MAINTENANCE_EMBEDDING_REFRESH_ENABLED: bool = True
    
    # Document Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx", ".doc", ".txt", ".md"]
    
    # Retrieval Settings
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Hybrid Search Settings
    HYBRID_ALPHA: float = 0.5  # Weight for vector vs BM25 (1.0 = pure vector, 0.0 = pure BM25)
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_CROSS_ENCODER: bool = True
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATES: int = 30  # Number of candidates to rerank
    
    # Query Expansion
    ENABLE_QUERY_EXPANSION: bool = True
    QUERY_EXPANSION_COUNT: int = 3
    
    # Relevance Threshold
    RELEVANCE_THRESHOLD: float = 0.3
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_CHAT: str = "30/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_DEFAULT: str = "100/minute"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    
    # ========================================================================
    # PHASE 7: RESILIENCE & PRODUCTION SETTINGS
    # ========================================================================
    
    # Circuit Breaker Settings
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5  # Failures before opening
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = 2  # Successes to close
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: float = 30.0  # Time before half-open
    
    # LLM Circuit Breaker (more lenient)
    LLM_CIRCUIT_FAILURE_THRESHOLD: int = 3
    LLM_CIRCUIT_TIMEOUT_SECONDS: float = 60.0
    
    # Tracing Settings
    ENABLE_TRACING: bool = True
    TRACING_LOG_REQUESTS: bool = True
    TRACING_LOG_RESPONSES: bool = True
    TRACING_SLOW_REQUEST_MS: float = 1000.0  # Log warning for slow requests
    
    # Session Settings
    SESSION_DURATION_DAYS: int = 30
    MAX_SESSIONS_PER_USER: int = 10
    
    # Audit Settings
    AUDIT_ENABLED: bool = True
    AUDIT_RETENTION_DAYS: int = 365
    
    # Cache Settings
    CACHE_ENABLED: bool = True
    CACHE_DEFAULT_TTL_SECONDS: int = 300  # 5 minutes
    CACHE_MAX_SIZE: int = 1000
    
    # User/Auth Cache
    USER_CACHE_TTL_SECONDS: int = 300
    USER_CACHE_MAX_SIZE: int = 500
    
    # Document Cache
    DOCUMENT_CACHE_TTL_SECONDS: int = 600
    DOCUMENT_CACHE_MAX_SIZE: int = 1000
    
    # Query Cache
    QUERY_CACHE_TTL_SECONDS: int = 120
    QUERY_CACHE_MAX_SIZE: int = 200
    
    # LLM Response Cache
    LLM_CACHE_TTL_SECONDS: int = 1800  # 30 minutes
    LLM_CACHE_MAX_SIZE: int = 100
    
    # Graceful Degradation Settings
    GRACEFUL_DEGRADATION_ENABLED: bool = True
    FALLBACK_TO_SIMPLE_SEARCH: bool = True  # Use simple vector search if hybrid fails
    FALLBACK_TO_CACHED_RESPONSE: bool = True  # Use cached response if LLM fails
    DEGRADED_MODE_LOG_LEVEL: str = "WARNING"
    
    # Health Check Settings
    HEALTH_CHECK_TIMEOUT_SECONDS: float = 5.0
    HEALTH_CHECK_INTERVAL_SECONDS: float = 30.0
    
    # Background Task Settings
    CLEANUP_INTERVAL_SECONDS: float = 60.0  # Cache cleanup interval
    METRICS_FLUSH_INTERVAL_SECONDS: float = 15.0


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
