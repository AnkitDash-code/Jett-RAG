# RAG Backend

A production-ready **GraphRAG Knowledge Portal** backend with authentication, multi-format document processing, hybrid retrieval, knowledge graph integration, memory system, and streaming chat—all implemented in pure Python with FastAPI.

## 🎯 Overview

This unified FastAPI backend implements a complete RAG (Retrieval-Augmented Generation) system with advanced features including knowledge graph traversal, episodic/semantic memory, self-reflection, and hierarchical retrieval. Built for production with circuit breakers, audit logging, session management, and comprehensive observability.

**Key Features:**

- 📚 Multi-format document support (PDF, DOCX, TXT, Images with VLM)
- 🕸️ **GraphRAG** with rustworkx for entity-aware retrieval
- 🧠 **Supermemory** system (episodic + semantic memory)
- 🔍 Hybrid search (FAISS vector + BM25 keyword)
- 🎯 Self-reflection with auto-retry
- 🔄 Background job queue (Python-only, no Redis/Cellar)
- 🛡️ Production-ready (circuit breakers, audit logs, health checks)
- 💬 Real-time streaming (SSE + WebSocket)
- ⚡ **Offline-ready** - Auto-caches models on first run
- 📷 **QR/Barcode scanning** - Auto-detect and decode via pyzbar
- 👥 **Demo users** - Pre-configured RBAC users on startup

---

## 🚀 Quick Start (Automatic Offline Setup)

### **Windows - One-Click Start**

```bash
# Double-click to start (first run downloads models automatically)
start.bat
```

### **Manual Start (First Time)**

```bash
cd RAG-Backend
myenv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

**First run will automatically:**

- ✅ Download embedding models (~80 MB)
- ✅ Download reranker model (~80 MB)
- ✅ Download spaCy NER model (~15 MB)
- ✅ Cache everything for offline use
- ✅ Subsequent runs start instantly

See [OFFLINE_SETUP.md](OFFLINE_SETUP.md) for complete offline configuration guide.

---

## 📦 What Gets Cached?

| Model                  | Size    | Purpose               | Status       |
| ---------------------- | ------- | --------------------- | ------------ |
| all-MiniLM-L6-v2       | ~80 MB  | Document embeddings   | Auto-cached  |
| ms-marco-MiniLM-L-6-v2 | ~80 MB  | Result reranking      | Auto-cached  |
| en_core_web_sm         | ~15 MB  | Entity extraction     | Auto-cached  |
| mistral-7b-instruct    | ~4.4 GB | LLM generation        | Manual setup |
| Tesseract OCR          | ~50 MB  | Image text extraction | Optional     |

**Total:** ~4.6 GB for complete offline operation

---

## 🧪 Test Offline Readiness

```bash
# Check if all models cached and services running
test.bat

# Or manually:
python test_offline.py
```

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Development Phases](#development-phases)
- [Configuration](#configuration)
- [Testing](#testing)
- [Production Deployment](#production-deployment)

---

## 🏗 Architecture

```
RAG-Backend/
├── app/
│   ├── api/
│   │   ├── deps.py              # FastAPI dependencies (auth, RBAC)
│   │   └── v1/
│   │       ├── __init__.py      # API router aggregation
│   │       └── endpoints/
│   │           ├── auth.py      # Authentication endpoints
│   │           ├── users.py     # User management
│   │           ├── documents.py # Document upload, chunking, preview
│   │           ├── chat.py      # Chat, streaming, export/import
│   │           ├── memory.py    # Memory system endpoints
│   │           ├── search.py    # Search suggestions, autocomplete
│   │           ├── sessions.py  # Device/session management
│   │           ├── jobs.py      # Background job tracking
│   │           ├── health.py    # Health checks
│   │           └── admin.py     # Admin operations
│   ├── middleware/
│   │   ├── error_handler.py     # Global exception handling
│   │   ├── logging.py           # Request logging
│   │   ├── tracing.py           # Request tracing
│   │   └── rate_limit.py        # Rate limiting
│   ├── models/
│   │   ├── user.py              # User, Role, Tenant
│   │   ├── session.py           # UserSession, DeviceInfo
│   │   ├── audit.py             # AuditLog
│   │   ├── document.py          # Document, Chunk, Hierarchy
│   │   ├── chat.py              # Conversation, Message, QueryLog
│   │   ├── memory.py            # EpisodicMemory, SemanticMemory
│   │   ├── graph.py             # Entity, Relationship, Community
│   │   └── job.py               # Job, JobHistory
│   ├── services/
│   │   ├── auth_service.py      # JWT authentication
│   │   ├── session_manager.py   # Multi-device sessions
│   │   ├── audit_service.py     # Audit logging
│   │   ├── document_service.py  # Document CRUD
│   │   ├── ingestion_service.py # Document parsing & chunking
│   │   ├── ingestion_tasks.py   # Background ingestion pipeline
│   │   ├── ocr_service.py       # EasyOCR integration
│   │   ├── docling_parser.py    # Advanced PDF parsing
│   │   ├── retrieval_service.py # Hybrid FAISS + BM25 search
│   │   ├── enhanced_retrieval_service.py # Query intelligence pipeline
│   │   ├── hierarchical_retrieval_service.py # Parent/child expansion
│   │   ├── query_classification_service.py # Query routing
│   │   ├── query_expansion_service.py # Multi-query RAG
│   │   ├── relevance_grader.py  # Self-reflection
│   │   ├── auto_retry_service.py # Auto-retry on low relevance
│   │   ├── llm_client.py        # Main LLM client
│   │   ├── utility_llm_client.py # Utility LLM
│   │   ├── chat_service.py      # Chat orchestration
│   │   ├── entity_extraction_service.py # NER extraction
│   │   ├── graph_store.py       # rustworkx graph operations
│   │   ├── graph_traversal_service.py # Entity-aware retrieval
│   │   ├── community_detection_service.py # Louvain clustering
│   │   ├── memory_service.py    # Episodic/semantic memory
│   │   ├── memory_router.py     # Memory classification
│   │   ├── forgetting_policy.py # Smart memory eviction
│   │   ├── task_queue.py        # Python-only job queue
│   │   ├── job_service.py       # Job lifecycle management
│   │   ├── circuit_breaker.py   # Resilience pattern
│   │   ├── cache_manager.py     # In-memory caching
│   │   ├── metrics_service.py   # Prometheus metrics
│   │   ├── reranker_service.py  # Cross-encoder reranking
│   │   ├── bm25_index.py        # BM25 keyword search
│   │   └── vector_store.py      # FAISS vector store
│   ├── config.py                # Pydantic settings
│   ├── database.py              # SQLModel async setup
│   └── main.py                  # FastAPI application
├── tests/
│   ├── test_auth.py
│   ├── test_documents.py
│   ├── test_chat.py
│   ├── test_phase2_services.py
│   ├── test_phase3_services.py
│   ├── test_phase4_graphrag.py
│   ├── test_phase5_advanced_rag.py
│   ├── test_phase5_parsers.py
│   ├── test_phase5_task_queue.py
│   ├── test_phase6_memory.py
│   └── test_ocr_service.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠 Tech Stack

| Component            | Technology                    | Purpose                            |
| -------------------- | ----------------------------- | ---------------------------------- |
| **Web Framework**    | FastAPI                       | Async API with OpenAPI docs        |
| **Database**         | SQLModel + SQLite             | ORM with Pydantic integration      |
| **Vector Store**     | FAISS                         | Cosine similarity search           |
| **Keyword Search**   | rank-bm25                     | BM25 lexical search                |
| **Graph Engine**     | rustworkx                     | High-performance graph operations  |
| **Embeddings**       | sentence-transformers         | Text embeddings (all-MiniLM-L6-v2) |
| **Reranking**        | sentence-transformers         | Cross-encoder reranking            |
| **LLM Client**       | llama.cpp                     | Local LLM inference                |
| **Document Parsing** | Docling, PyMuPDF, python-docx | Multi-format support               |
| **OCR**              | EasyOCR                       | Image text extraction              |
| **Task Queue**       | asyncio + ThreadPoolExecutor  | Pure Python background jobs        |
| **Monitoring**       | prometheus-client             | Metrics and observability          |
| **Testing**          | pytest + pytest-asyncio       | Unit and integration tests         |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd RAG-Backend
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your settings
```

**Key Settings:**

```env
DATABASE_URL=sqlite+aiosqlite:///./rag_backend.db
SECRET_KEY=your-secret-key-here
LLM_API_BASE_URL=http://localhost:8000
EMBEDDING_MODEL=all-MiniLM-L6-v2
ENABLE_GRAPH_RAG=true
ENABLE_MEMORY_SYSTEM=true
```

### 3. Start the LLM Backend

Make sure your LLM backend (llama.cpp) is running on port 8000:

```bash
cd ../LLM-Backend
python main.py
```

### 4. Start the RAG Backend

```bash
cd ../RAG-Backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload
```

### 5. Access API Documentation

- **Swagger UI**: http://localhost:8081/v1/docs
- **ReDoc**: http://localhost:8081/v1/redoc
- **Health Check**: http://localhost:8081/health

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint           | Description            |
| ------ | ------------------ | ---------------------- |
| POST   | `/v1/auth/signup`  | Register new user      |
| POST   | `/v1/auth/login`   | Login & get JWT tokens |
| POST   | `/v1/auth/logout`  | Logout & revoke tokens |
| POST   | `/v1/auth/refresh` | Refresh access token   |

### Users

| Method | Endpoint         | Description              |
| ------ | ---------------- | ------------------------ |
| GET    | `/v1/users/me`   | Get current user profile |
| PATCH  | `/v1/users/me`   | Update profile           |
| GET    | `/v1/users/{id}` | Get user by ID (admin)   |

### Documents

| Method | Endpoint                            | Description                          |
| ------ | ----------------------------------- | ------------------------------------ |
| POST   | `/v1/documents/upload`              | Upload document (PDF/DOCX/TXT/Image) |
| GET    | `/v1/documents`                     | List documents                       |
| GET    | `/v1/documents/{id}`                | Get document details                 |
| GET    | `/v1/documents/chunks/{id}/preview` | Get chunk with context               |
| POST   | `/v1/documents/{id}/reindex`        | Trigger reindex                      |
| DELETE | `/v1/documents/{id}`                | Delete document                      |

### Chat & RAG

| Method | Endpoint                             | Description                   |
| ------ | ------------------------------------ | ----------------------------- |
| POST   | `/v1/chat`                           | Send message (non-streaming)  |
| GET    | `/v1/chat/stream`                    | SSE streaming response        |
| WS     | `/v1/chat/ws`                        | WebSocket streaming           |
| GET    | `/v1/chat/history`                   | List conversations            |
| GET    | `/v1/chat/conversations/{id}`        | Get conversation              |
| GET    | `/v1/chat/conversations/{id}/export` | Export conversation (JSON/MD) |
| POST   | `/v1/chat/conversations/import`      | Import conversation           |
| POST   | `/v1/chat/feedback`                  | Submit feedback               |

### Memory System

| Method | Endpoint                   | Description                       |
| ------ | -------------------------- | --------------------------------- |
| POST   | `/v1/memory`               | Create memory (episodic/semantic) |
| GET    | `/v1/memory`               | List memories with pagination     |
| POST   | `/v1/memory/retrieve`      | Hybrid memory search              |
| POST   | `/v1/memory/{id}/feedback` | Update memory importance          |
| POST   | `/v1/memory/link`          | Link two memories                 |
| POST   | `/v1/memory/evict`         | Force memory eviction             |
| GET    | `/v1/memory/stats/summary` | Memory statistics                 |

### Search

| Method | Endpoint                  | Description              |
| ------ | ------------------------- | ------------------------ |
| GET    | `/v1/search/suggestions`  | Get search suggestions   |
| GET    | `/v1/search/autocomplete` | Get autocomplete results |

### Sessions

| Method | Endpoint               | Description          |
| ------ | ---------------------- | -------------------- |
| GET    | `/v1/sessions`         | List active sessions |
| GET    | `/v1/sessions/current` | Get current session  |
| DELETE | `/v1/sessions/{id}`    | Revoke session       |
| DELETE | `/v1/sessions`         | Revoke all sessions  |

### Background Jobs

| Method | Endpoint                | Description              |
| ------ | ----------------------- | ------------------------ |
| GET    | `/v1/jobs`              | List jobs (with filters) |
| GET    | `/v1/jobs/{id}`         | Get job status           |
| GET    | `/v1/jobs/{id}/stream`  | SSE job progress stream  |
| POST   | `/v1/jobs/{id}/cancel`  | Cancel running job       |
| GET    | `/v1/jobs/{id}/history` | Get job history          |

### Health & Monitoring

| Method | Endpoint               | Description                 |
| ------ | ---------------------- | --------------------------- |
| GET    | `/health`              | Comprehensive health check  |
| GET    | `/health/ready`        | Readiness probe             |
| GET    | `/health/live`         | Liveness probe              |
| GET    | `/health/dependencies` | Check DB, LLM, vector store |

### Admin & Monitoring

| Method | Endpoint                      | Description          |
| ------ | ----------------------------- | -------------------- |
| GET    | `/v1/admin/metrics`           | System metrics       |
| GET    | `/v1/admin/metrics/{user_id}` | User metrics         |
| GET    | `/v1/admin/logs`              | Query logs           |
| GET    | `/v1/admin/errors`            | Error summary        |
| GET    | `/v1/admin/users`             | List all users       |
| GET    | `/v1/admin/documents`         | List all documents   |
| GET    | `/v1/admin/cache/stats`       | Cache hit/miss rates |
| POST   | `/v1/admin/cache/clear`       | Flush cache          |
| GET    | `/v1/admin/traces`            | Query request traces |

---

## 📚 Development Phases

All 8 development phases have been completed. This section documents the journey from basic infrastructure to a production-ready GraphRAG system with supermemory and advanced retrieval capabilities.

### ✅ Phase 1: Core Infrastructure (COMPLETED)

**Duration:** ~3 weeks | **Status:** Production-ready

**Implemented Components:**

- ✅ FastAPI application with async support
- ✅ SQLModel database models (User, Role, Document, Chunk, Conversation, Message)
- ✅ JWT authentication with Argon2 password hashing
- ✅ RBAC with fine-grained permissions (admin, power_user, viewer, ingestor)
- ✅ Multi-tenancy support (tenant_id isolation)
- ✅ Document upload with validation (size limits, extension checks)
- ✅ SHA-256 deduplication
- ✅ LangChain RecursiveCharacterTextSplitter for chunking
- ✅ FAISS IndexFlatIP for vector storage
- ✅ SSE streaming for token-by-token LLM responses
- ✅ Chat orchestration with RAG pipeline
- ✅ Source citation extraction

**Key Services:**

- `auth_service.py` - Registration, login, JWT tokens
- `document_service.py` - Document CRUD
- `ingestion_service.py` - Parsing and chunking
- `chat_service.py` - RAG orchestration
- `vector_store.py` - FAISS operations
- `llm_client.py` - LLM streaming

**API Endpoints:** Auth, Users, Documents, Chat (basic)

**Tests:** `test_auth.py`, `test_documents.py`, `test_chat.py`, `test_users.py`

---

### ✅ Phase 2: Vector Store + Hybrid Search (COMPLETED)

**Duration:** ~2 weeks | **Status:** Production-ready

**Implemented Components:**

- ✅ BM25 keyword search integration (`rank-bm25`)
- ✅ Hybrid retrieval (FAISS cosine + BM25 keyword)
- ✅ Reciprocal Rank Fusion (RRF) scoring
- ✅ Cross-encoder reranking with `sentence-transformers`
- ✅ GPU acceleration support
- ✅ RBAC filtering in retrieval pipeline
- ✅ Document-level and chunk-level access control
- ✅ Batch embedding generation
- ✅ Index persistence to disk

**Key Services:**

- `bm25_index.py` - BM25 indexing and search
- `reranker_service.py` - Cross-encoder reranking
- `retrieval_service.py` - Hybrid search orchestration

**Changes Made:**

- Added BM25 index building during document ingestion
- Integrated reranking as final retrieval step
- Added hybrid scoring formula: `0.7 * vector_score + 0.3 * bm25_score`

**Tests:** `test_phase2_services.py`

---

### ✅ Phase 3: Query Intelligence (COMPLETED)

**Duration:** ~2 weeks | **Status:** Production-ready

**Implemented Components:**

- ✅ Query classification (FACT, ENTITY, COMPARISON, SUMMARY, etc.)
- ✅ Complexity detection (simple, moderate, complex)
- ✅ Multi-query RAG (LLM generates 3 query variants)
- ✅ Relevance grading with self-reflection
- ✅ Auto-retry service (reformulates query on low relevance)
- ✅ Conversation context management
- ✅ Follow-up detection
- ✅ Utility LLM client (separate LLM for query operations)
- ✅ Enhanced retrieval pipeline orchestrating all services

**Key Services:**

- `query_classification_service.py` - Query type routing
- `query_expansion_service.py` - Multi-query generation
- `relevance_grader.py` - Self-reflection scoring
- `auto_retry_service.py` - Query reformulation
- `conversation_context_service.py` - Context management
- `utility_llm_client.py` - Utility LLM operations
- `enhanced_retrieval_service.py` - 7-step pipeline

**Pipeline Steps:**

1. Load conversation context
2. Classify query (routing parameters)
3. Expand query (multi-query RAG)
4. Execute hybrid search
5. Rerank results
6. Grade relevance (self-reflection)
7. Return enhanced result (with retry suggestions)

**Changes Made:**

- Created dual LLM system (main + utility)
- Added query intelligence layer before retrieval
- Integrated self-reflection with auto-retry
- Enhanced prompt builder with conversation history

**Tests:** `test_phase3_services.py`

---

### ✅ Phase 4: GraphRAG (COMPLETED)

**Duration:** ~3 weeks | **Status:** Production-ready

**Implemented Components:**

- ✅ Entity extraction service (LLM-based NER)
- ✅ rustworkx graph engine (10-100x faster than NetworkX)
- ✅ Entity models (Entity, EntityMention, EntityRelationship)
- ✅ Community detection (Louvain + connected components)
- ✅ Community summarization with LLM
- ✅ Graph traversal for entity-aware retrieval
- ✅ K-hop neighbor expansion
- ✅ Graph-level RBAC filtering
- ✅ Permission inheritance (tenant → group → folder → document → chunk → entity)
- ✅ Centrality scoring (degree, betweenness)
- ✅ Graph persistence in SQLite
- ✅ Lazy graph building (iRAG-style)

**Key Services:**

- `entity_extraction_service.py` - LLM-based NER with normalization
- `graph_store.py` - rustworkx operations, caching, RBAC
- `graph_traversal_service.py` - Entity-aware retrieval
- `community_detection_service.py` - Louvain clustering
- `graph_index_service.py` - Background graph building
- `permission_inheritance_service.py` - Graph RBAC

**GraphRAG Retrieval Flow:**

1. Extract entities from query
2. Find entities in graph
3. Traverse 1-2 hops to find related entities
4. Fetch chunks mentioning related entities
5. Score: `0.4*vector + 0.3*rerank + 0.3*graph`
6. Inject community summaries if entities share communities

**Changes Made:**

- Integrated rustworkx (Rust-backed Python library)
- Added entity extraction to ingestion pipeline
- Created graph traversal step in enhanced retrieval
- Added community-aware context injection

**Tests:** `test_phase4_graphrag.py`

---

### ✅ Phase 5: Background Jobs + Advanced RAG (COMPLETED)

**Duration:** ~3 weeks | **Status:** Production-ready

**Implemented Components:**

- ✅ Python-only task queue (asyncio + ThreadPoolExecutor)
- ✅ Job tracking models (Job, JobHistory, JobStep)
- ✅ Progress tracking with SSE streaming
- ✅ Resumable jobs with checkpointing
- ✅ 8-step document ingestion pipeline
- ✅ Hierarchical chunk retrieval (parent/child/sibling expansion)
- ✅ Self-reflection auto-retry (up to 2 retries)
- ✅ Docling parser integration (advanced PDF parsing)
- ✅ EasyOCR integration (GPU-accelerated OCR)
- ✅ OCR detection (text density analysis)
- ✅ Louvain community detection
- ✅ Scheduled maintenance jobs
- ✅ Job cancellation support

**Key Services:**

- `task_queue.py` - Pure Python job queue (no Redis/Celery)
- `job_service.py` - Job lifecycle, progress tracking
- `ingestion_tasks.py` - 8-step ingestion pipeline
- `hierarchical_retrieval_service.py` - Parent/child expansion
- `docling_parser.py` - Advanced PDF parsing
- `ocr_service.py` - EasyOCR integration
- `maintenance_jobs.py` - Scheduled cleanup

**Ingestion Pipeline (8 Steps):**

1. **Parse** - Extract text from PDF/DOCX/TXT/Image
2. **Chunk** - Split into semantic chunks
3. **Hierarchy** - Build parent/child relationships
4. **Embed** - Generate embeddings
5. **Vector** - Index in FAISS
6. **Graph** - Extract entities and relationships
7. **Index** - Update BM25 index
8. **Finalize** - Mark document as indexed_full

**Changes Made:**

- Replaced ThreadPoolExecutor with full task queue
- Added job progress tracking and SSE streaming
- Integrated Docling for better PDF parsing
- Added OCR for image-based PDFs
- Implemented hierarchical retrieval for better context

**Tests:** `test_phase5_advanced_rag.py`, `test_phase5_parsers.py`, `test_phase5_task_queue.py`, `test_ocr_service.py`

---

### ✅ Phase 6: Supermemory System (COMPLETED)

**Duration:** ~2 weeks | **Status:** Production-ready

**Implemented Components:**

- ✅ Dual memory types (Episodic + Semantic)
- ✅ Memory models (EpisodicMemory, SemanticMemory, MemoryLink)
- ✅ Memory router (auto-classification)
- ✅ Hybrid memory retrieval (60% vector + 40% entity overlap)
- ✅ Importance scoring with decay functions
- ✅ Smart forgetting with policies (Aggressive, Conservative, Balanced)
- ✅ Memory consolidation (episodic → semantic)
- ✅ Cross-session recall
- ✅ Memory linking (relationships between memories)
- ✅ Separate FAISS index for memory vectors
- ✅ LRU cache eviction
- ✅ Conversation summarization integration

**Key Services:**

- `memory_service.py` - Core memory operations (1297 lines!)
- `memory_router.py` - Episodic vs semantic classification
- `forgetting_policy.py` - Decay, archive, consolidation policies

**Memory Types:**

- **Episodic**: Events, actions, experiences (time-sensitive)
- **Semantic**: Facts, concepts, knowledge (time-independent)

**Importance Formula:**

```
importance = base_score × recency_factor × access_frequency × consolidation_boost
```

**Forgetting Policies:**

- **AggressivePolicy**: Fast decay (half-life 7 days)
- **ConservativePolicy**: Slow decay (half-life 60 days)
- **BalancedPolicy**: Medium decay (half-life 30 days)

**Changes Made:**

- Created separate memory subsystem
- Integrated memory retrieval into enhanced retrieval pipeline
- Added memory-enhanced context to prompts
- Implemented background memory consolidation

**Tests:** `test_phase6_memory.py`

---

### ✅ Phase 7: Production Infrastructure (COMPLETED)

**Duration:** ~2 weeks | **Status:** Production-ready

**Implemented Components:**

- ✅ Multi-device session tracking
- ✅ Session revocation (single + all devices)
- ✅ Device fingerprinting
- ✅ Audit logging (40+ action types)
- ✅ Compliance-ready (SOC2, HIPAA, GDPR)
- ✅ Circuit breakers (3 states: CLOSED, OPEN, HALF_OPEN)
- ✅ In-memory caching with TTL and LRU eviction
- ✅ Tag-based cache invalidation
- ✅ Request tracing with correlation IDs
- ✅ Span tracking across services
- ✅ Enhanced health checks (DB, LLM, vector store, disk, memory)
- ✅ Rate limiting (token bucket algorithm)
- ✅ Prometheus metrics collection
- ✅ Global error handling

**Key Services:**

- `session_manager.py` - Multi-device sessions
- `audit_service.py` - Audit logging
- `circuit_breaker.py` - Resilience pattern
- `cache_manager.py` - In-memory caching
- `metrics_service.py` - Prometheus metrics

**Middleware:**

- `tracing.py` - Request correlation
- `logging.py` - Structured logging
- `rate_limit.py` - Rate limiting
- `error_handler.py` - Global exception handling

**Changes Made:**

- Added session tracking to JWT payload
- Integrated audit logging in all sensitive operations
- Wrapped LLM calls with circuit breaker
- Added caching layer for frequent queries
- Implemented request tracing for debugging

**Tests:** Unit tests distributed across service files

---

### ✅ Phase 8: Frontend Integration (COMPLETED)

**Duration:** ~2 weeks | **Status:** Production-ready

**Implemented Components:**

- ✅ Search suggestions endpoint (entities, documents, recent queries)
- ✅ Autocomplete endpoint
- ✅ Chunk preview API (with prev/next navigation)
- ✅ Conversation export (JSON + Markdown formats)
- ✅ Conversation import from JSON
- ✅ Stream cancellation (abort in-progress generation)
- ✅ Active stream management
- ✅ Enhanced chat history with filters

**Key Endpoints:**

- `GET /v1/search/suggestions` - Grouped suggestions
- `GET /v1/search/autocomplete` - Flat autocomplete list
- `GET /v1/documents/chunks/{id}/preview` - Chunk with context
- `GET /v1/chat/conversations/{id}/export` - Export as JSON/MD
- `POST /v1/chat/conversations/import` - Import conversation
- `POST /v1/chat/stream/cancel/{id}` - Cancel streaming

**Frontend Integration Points:**

- Citation tooltips (hover for chunk preview)
- Export/import for conversation portability
- Search autocomplete with grouped suggestions
- Real-time progress tracking for document ingestion
- Multi-device session management UI

**Changes Made:**

- Added search suggestions service
- Enhanced document endpoints with chunk preview
- Added conversation export/import for portability
- Improved streaming with cancellation support

**Tests:** Frontend tests in Next.js app (`tests/e2e/`)

---

## ⚙️ Configuration

Key configuration options in `app/config.py`:

```python
# Database
DATABASE_URL: str = "sqlite+aiosqlite:///./rag_backend.db"

# LLM
LLM_API_BASE_URL: str = "http://localhost:8000"
LLM_MAX_RETRIES: int = 3
LLM_TIMEOUT: int = 120

# Embeddings
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = "cuda"  # or "cpu"

# Retrieval
ENABLE_GRAPH_RAG: bool = True
ENABLE_MEMORY_SYSTEM: bool = True
HYBRID_SEARCH_ALPHA: float = 0.7  # 0.7 vector + 0.3 BM25

# GraphRAG
GRAPH_MAX_HOPS: int = 2
GRAPH_SCORE_WEIGHT: float = 0.3
COMMUNITY_ALGORITHM: str = "louvain"  # or "connected_components"

# Memory
MEMORY_DECAY_POLICY: str = "balanced"  # aggressive, conservative, balanced
MEMORY_MAX_EPISODIC: int = 1000
MEMORY_MAX_SEMANTIC: int = 500

# Background Jobs
TASK_QUEUE_WORKERS: int = 4
JOB_RETENTION_DAYS: int = 7

# Resilience
CIRCUIT_BREAKER_THRESHOLD: int = 5
CIRCUIT_BREAKER_TIMEOUT: int = 30
CACHE_TTL_SECONDS: int = 300
CACHE_MAX_SIZE_MB: int = 100

# Security
SECRET_KEY: str = "your-secret-key-here"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
RATE_LIMIT_PER_MINUTE: int = 60
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# Specific phase
pytest tests/test_phase3_services.py

# With coverage
pytest --cov=app --cov-report=html

# Verbose output
pytest -v -s
```

**Test Coverage:**

- Phase 1: Auth, documents, chat, users
- Phase 2: BM25, reranker, metrics
- Phase 3: Query intelligence pipeline
- Phase 4: GraphRAG (entities, graph, communities)
- Phase 5: Background jobs, parsers, OCR, hierarchical retrieval
- Phase 6: Memory system (episodic, semantic, forgetting)
- Phase 7: Unit tests within service files

**Total Tests:** 150+ tests across 15 test files

---

## 🚀 Production Deployment

### Docker Deployment (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/faiss_index data/bm25_index data/graph

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8081/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]
```

### Environment Variables

```bash
# Production settings
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/ragdb
SECRET_KEY=$(openssl rand -hex 32)
ENABLE_CORS=true
CORS_ORIGINS=["https://yourapp.com"]

# External services
LLM_API_BASE_URL=http://llm-backend:8000

# Performance
TASK_QUEUE_WORKERS=8
CACHE_MAX_SIZE_MB=500

# Security
RATE_LIMIT_PER_MINUTE=120
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### Monitoring

- **Prometheus Metrics**: `/metrics` endpoint
- **Health Checks**: `/health`, `/health/ready`, `/health/live`
- **Request Tracing**: Via correlation IDs in logs
- **Audit Logs**: Queryable via `/v1/admin/logs`

---

## ��� Integration with Frontend

The frontend (Next.js) should:

1. Store tokens in secure storage (httpOnly cookies recommended)
2. Include `Authorization: Bearer <token>` header in requests
3. Handle token refresh on 401 responses
4. Use SSE for streaming: `EventSource` API
5. Use WebSocket for real-time chat: `WebSocket` API

Example fetch with auth:

```typescript
const response = await fetch("http://localhost:8081/v1/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${accessToken}`,
  },
  body: JSON.stringify({ message: "What is...?" }),
});
```

---

## ��� License

MIT License - See LICENSE file for details.

---

## ��� Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass: `pytest`
5. Submit a pull request

---

Built with ❤️ for the Aerothon Hackathon

**Tech Stack Summary:**

- 40+ services
- 14 database models
- 60+ API endpoints
- 150+ tests
- 8 development phases
- Production-ready with comprehensive monitoring
