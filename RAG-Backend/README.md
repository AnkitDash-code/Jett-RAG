# RAG Backend

A unified FastAPI backend for RAG (Retrieval-Augmented Generation) with authentication, document processing, hybrid retrieval, and streaming chat.

## Architecture

```
RAG-Backend/
├── app/
│   ├── api/
│   │   ├── deps.py              # FastAPI dependencies (auth, RBAC)
│   │   └── v1/
│   │       ├── __init__.py      # API router aggregation
│   │       └── endpoints/
│   │           ├── auth.py      # /auth/* endpoints
│   │           ├── users.py     # /users/* endpoints
│   │           ├── documents.py # /documents/* endpoints
│   │           ├── chat.py      # /chat/* endpoints
│   │           └── admin.py     # /admin/* endpoints
│   ├── middleware/
│   │   ├── error_handler.py     # Global exception handling
│   │   └── logging.py           # Request logging
│   ├── models/
│   │   ├── user.py              # User, Role, RefreshToken, APIKey
│   │   ├── document.py          # Document, DocumentVersion, Chunk
│   │   └── chat.py              # Conversation, Message, QueryLog, Feedback
│   ├── schemas/
│   │   ├── auth.py              # Auth request/response schemas
│   │   ├── user.py              # User schemas
│   │   ├── document.py          # Document schemas
│   │   └── chat.py              # Chat schemas
│   ├── services/
│   │   ├── auth_service.py      # Authentication & JWT
│   │   ├── user_service.py      # User management
│   │   ├── document_service.py  # Document CRUD
│   │   ├── ingestion_service.py # Document parsing & chunking
│   │   ├── retrieval_service.py # Hybrid search & reranking
│   │   ├── llm_client.py        # LLM API client
│   │   ├── chat_service.py      # Chat orchestration
│   │   └── monitoring_service.py# Metrics & logging
│   ├── config.py                # Pydantic settings
│   ├── database.py              # SQLAlchemy async setup
│   └── main.py                  # FastAPI application
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd RAG-Backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your settings
```

### 3. Start the LLM Backend

Make sure your existing LLM-Backend (KoboldCpp) is running on port 8000:

```bash
cd ../LLM-Backend
python init.py
```

### 4. Start the RAG Backend

```bash
cd ../RAG-Backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 5. Access API Documentation

- Swagger UI: http://localhost:8080/v1/docs
- ReDoc: http://localhost:8080/v1/redoc

## API Endpoints

### Authentication

| Method | Endpoint           | Description            |
| ------ | ------------------ | ---------------------- |
| POST   | `/v1/auth/signup`  | Register new user      |
| POST   | `/v1/auth/login`   | Login & get tokens     |
| POST   | `/v1/auth/logout`  | Logout & revoke tokens |
| POST   | `/v1/auth/refresh` | Refresh access token   |

### Users

| Method | Endpoint         | Description              |
| ------ | ---------------- | ------------------------ |
| GET    | `/v1/users/me`   | Get current user profile |
| PATCH  | `/v1/users/me`   | Update profile           |
| GET    | `/v1/users/{id}` | Get user by ID (admin)   |

### Documents

| Method | Endpoint                     | Description          |
| ------ | ---------------------------- | -------------------- |
| POST   | `/v1/documents/upload`       | Upload document      |
| GET    | `/v1/documents`              | List documents       |
| GET    | `/v1/documents/{id}`         | Get document details |
| POST   | `/v1/documents/{id}/reindex` | Trigger reindex      |
| DELETE | `/v1/documents/{id}`         | Delete document      |

### Chat & RAG

| Method | Endpoint                      | Description                  |
| ------ | ----------------------------- | ---------------------------- |
| POST   | `/v1/chat`                    | Send message (non-streaming) |
| GET    | `/v1/chat/stream`             | SSE streaming response       |
| WS     | `/v1/chat/ws`                 | WebSocket streaming          |
| GET    | `/v1/chat/history`            | List conversations           |
| GET    | `/v1/chat/conversations/{id}` | Get conversation             |
| POST   | `/v1/chat/feedback`           | Submit feedback              |

### Admin & Monitoring

| Method | Endpoint                      | Description        |
| ------ | ----------------------------- | ------------------ |
| GET    | `/v1/admin/metrics`           | System metrics     |
| GET    | `/v1/admin/metrics/{user_id}` | User metrics       |
| GET    | `/v1/admin/logs`              | Query logs         |
| GET    | `/v1/admin/errors`            | Error summary      |
| GET    | `/v1/admin/users`             | List all users     |
| GET    | `/v1/admin/documents`         | List all documents |

## Vector Database

### Current: FAISS (Phase 1)

The current implementation uses **FAISS** (Facebook AI Similarity Search) for vector storage and retrieval:

- ✅ No external dependencies or Docker required
- ✅ Works natively on Windows, Linux, and macOS
- ✅ Persists to disk (`vector_store/` directory)
- ✅ Fast cosine similarity search
- ⚠️ Pure vector search only (no hybrid BM25+vector)

### Planned: Weaviate (Phase 2)

In Phase 2, we will migrate to **Weaviate** for production use with Docker:

```bash
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.28.2
```

Weaviate benefits:

- Hybrid search (BM25 keyword + vector similarity)
- Multi-tenancy support
- GraphQL API
- Better scalability for large document collections

## Development Roadmap

### Phase 1 (Current) ✅

- [x] Project structure & configuration
- [x] Database models (User, Document, Chunk, Conversation, Message)
- [x] Authentication with JWT tokens
- [x] RBAC with role-based permissions
- [x] Document upload & basic parsing
- [x] Chunking service
- [x] Basic retrieval (DB fallback)
- [x] Chat orchestration
- [x] Streaming responses (SSE + WebSocket)
- [x] Monitoring & metrics

### Phase 2 (TODO)

- [ ] Weaviate integration with Docker for hybrid search
- [ ] Cross-encoder reranking
- [ ] Query expansion (multi-query RAG)
- [ ] Redis caching layer
- [ ] Background job queue (Celery)

### Phase 3 (TODO)

- [ ] Neo4j integration for GraphRAG
- [ ] Entity extraction & knowledge graph
- [ ] Community detection & summaries
- [ ] Advanced document parsing (Docling)
- [ ] OpenTelemetry tracing

## Integration with Frontend

The frontend (Next.js) should:

1. Store tokens in secure storage (httpOnly cookies recommended)
2. Include `Authorization: Bearer <token>` header in requests
3. Handle token refresh on 401 responses
4. Use SSE for streaming: `EventSource` API
5. Use WebSocket for real-time chat: `WebSocket` API

Example fetch with auth:

```typescript
const response = await fetch("http://localhost:8080/v1/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${accessToken}`,
  },
  body: JSON.stringify({ message: "What is...?" }),
});
```

## License

MIT
