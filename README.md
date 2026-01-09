# 🚀 GraphRAG Knowledge Portal

A full-stack **Retrieval-Augmented Generation (RAG)** system with real-time streaming LLM responses, role-based access control, and a modern Next.js frontend.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Frontend](https://img.shields.io/badge/Frontend-Next.js%2015-black)
![Backend](https://img.shields.io/badge/Backend-FastAPI-green)
![LLM](https://img.shields.io/badge/LLM-Llama.cpp-orange)
![Vector DB](https://img.shields.io/badge/Vector%20DB-FAISS-red)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## ✨ Features

### Core RAG Capabilities

- 📄 **Multi-Format Document Upload** - PDF, DOCX, TXT, images with OCR (EasyOCR)
- 🔍 **Semantic Search** - FAISS-powered vector similarity search with autocomplete
- 🧠 **Smart Chunking** - LangChain RecursiveCharacterTextSplitter for optimal context
- 💬 **Real-time Streaming** - Token-by-token LLM response streaming via SSE
- 📚 **Source Citations** - View exact sources with chunk preview and context
- 🕸️ **Knowledge Graph** - Entity extraction and relationship mapping with visualization

### User Experience

- 🔐 **Authentication** - JWT-based secure login/registration with session management
- 👥 **Role-Based Access** - Admin and User roles with different permissions
- 💾 **Chat History** - Persistent conversation storage, export/import, and retrieval
- 🎨 **Modern UI** - Clean, responsive design with dark theme
- 🧠 **Memory System** - Episodic and semantic memory with importance ranking

### Admin Features

- 📊 **Analytics Dashboard** - Usage statistics and charts with Recharts
- 👤 **User Management** - View and manage users
- 📁 **Document Management** - Upload, permissions, and chunk inspection
- 🔗 **Knowledge Graph Viewer** - Interactive force-directed graph visualization
- ⚙️ **Settings** - Configure system preferences and session management

### Advanced Capabilities

- 🔎 **Search Suggestions** - Autocomplete with entities, documents, and recent queries
- 📤 **Export/Import** - Conversation export (JSON/Markdown/TXT) with full import support
- 🖼️ **OCR Processing** - Extract text from images using EasyOCR
- 🏷️ **Entity Extraction** - Automatic extraction and linking of entities
- 📊 **Chunk Preview** - View document chunks with surrounding context and navigation
- 🌐 **Distributed LLM** - Load balancing across multiple LLM instances via ngrok
- 🔄 **Auto Failover** - Automatic retry with different LLM on failure
- 💊 **Health Monitoring** - Continuous health checks of all LLM workers
- 🎯 **Smart Routing** - Task-based routing (utility vs chat operations)

---

## 🏗 Architecture

### Distributed Architecture with Ngrok Tunnels

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INTERNET (Ngrok)                             │
│   https://orchestrator-12345.ngrok-free.app (Orchestrator)          │
│   https://worker-llm-12345.ngrok-free.app (Worker LLM)              │
└────────┬───────────────────────────────────────┬───────────────────┘
         │                                       │
         │ HTTPS (from anywhere)                 │
         │                                       │
┌────────▼─────────────────────────┐  ┌─────────▼──────────────────────┐
│      DEVICE 1 (Orchestrator)     │  │      DEVICE 2 (Worker)         │
│                                  │  │                                │
│  ┌────────────────────────────┐  │  │  ┌──────────────────────────┐ │
│  │   Frontend (Port 3000)     │  │  │  │  LLM Backend (8000)      │ │
│  │   Next.js 15 Dashboard     │  │  │  │  + pyngrok (auto-tunnel) │ │
│  └────────────────────────────┘  │  │  └──────────────────────────┘ │
│                                  │  │                                │
│  ┌────────────────────────────┐  │  │  ┌──────────────────────────┐ │
│  │ RAG Backend (Port 8081)    │◄─┼──┤  │   Mistral-7B Model       │ │
│  │ + pyngrok (auto-tunnel)    │  │  │  │   Q4_K_M Quantized       │ │
│  │                            │  │  │  └──────────────────────────┘ │
│  │ Components:                │  │  │                                │
│  │ • LLM Router               │  │  └────────────────────────────────┘
│  │ • Load Balancer            │  │
│  │ • Health Monitor           │  │
│  │ • GraphRAG Engine          │  │
│  │ • Memory System            │  │
│  │ • Vector Store (FAISS)     │  │
│  │ • Knowledge Graph          │  │
│  └────────┬───────────────────┘  │
│           │                      │
│  ┌────────▼───────────────────┐  │
│  │ Local LLM Backend (8000)   │  │
│  │ Mistral-7B (Primary)       │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘

Key Features:
• Round-robin load balancing between Device 1 & 2 LLMs
• Automatic failover if one LLM becomes unavailable
• Health monitoring every 60 seconds
• Utility operations prefer local LLM for low latency
• Chat generation distributes across all healthy LLMs
• Public access via free ngrok static domains
```

---

## 🛠 Tech Stack

### Frontend

| Technology           | Purpose                         |
| -------------------- | ------------------------------- |
| Next.js 15           | React framework with App Router |
| TypeScript           | Type-safe development           |
| Recharts             | Analytics visualizations        |
| react-force-graph-2d | Knowledge graph visualization   |
| react-dropzone       | File upload interface           |
| Sonner               | Toast notifications             |
| fetch-event-source   | SSE streaming client            |

### RAG-Backend

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| FastAPI               | High-performance async API          |
| SQLModel              | Async ORM with Pydantic             |
| FAISS                 | Vector similarity search            |
| Sentence Transformers | Text embeddings                     |
| LangChain             | Document processing & chunking      |
| PyMuPDF               | PDF parsing                         |
| EasyOCR               | Image text extraction               |
| python-docx           | Word document parsing               |
| JWT                   | Authentication tokens               |
| pyngrok               | Public tunnels for distributed LLMs |
| rustworkx             | Knowledge graph operations          |

### LLM-Backend

| Technology        | Purpose                     |
| ----------------- | --------------------------- |
| FastAPI           | API wrapper for llama.cpp   |
| OpenAI Python SDK | Compatible client interface |
| llama.cpp         | Local LLM inference         |
| Mistral-7B        | Language model              |

---

## 📦 Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **CUDA Toolkit** (for GPU acceleration)
- **llama.cpp** server compiled with CUDA support
- **~8GB VRAM** (for Mistral-7B Q4)

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Code
```

### 2. Setup RAG-Backend

```bash
cd RAG-Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your settings
```

### 3. Setup LLM-Backend

```bash
cd ../LLM-Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the model (or place your own GGUF file)
# Model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 4. Setup Frontend

```bash
cd ../Frontend/next-app

# Install dependencies
npm install
```

---

## ⚙️ Configuration

### RAG-Backend (.env)

```env
# Database
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# JWT
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Ngrok Configuration (for distributed setup)
NGROK_AUTH_TOKEN=your_ngrok_authtoken_here
NGROK_STATIC_DOMAIN=orchestrator-12345.ngrok-free.app
PORT=8081

# Primary LLM (Local)
LLM_API_BASE_URL=http://localhost:8000

# Worker LLMs (Remote via ngrok)
WORKER_LLM_URLS=["https://worker-llm-12345.ngrok-free.app"]

# LLM Routing
LLM_ROUTING_STRATEGY=round_robin  # round_robin, random, or failover
LLM_HEALTH_CHECK_INTERVAL=60

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda

# RAG Features
ENABLE_GRAPH_RAG=true
ENABLE_MEMORY_SYSTEM=true

# FAISS
FAISS_INDEX_PATH=./data/faiss_index
```

### LLM-Backend (.env)

```env
# Ngrok Configuration
NGROK_AUTH_TOKEN=your_ngrok_authtoken_here
NGROK_STATIC_DOMAIN=worker-llm-12345.ngrok-free.app  # Optional

# Server
PORT=8000
```

### LLM-Backend

Configure `Settings.kcpps` for llama.cpp server settings.

---

## 🚀 Running the Application

### Option A: Single Device (Local Development)

#### 1. Start Local LLM

```bash
cd LLM-Backend
myenv\Scripts\activate  # On Windows
python main.py
# Runs on http://localhost:8000
```

#### 2. Start RAG-Backend

```bash
cd RAG-Backend
myenv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload
# Runs on http://localhost:8081
```

#### 3. Start Frontend

```bash
cd Frontend/next-app
npm run dev
# Runs on http://localhost:3000
```

#### 4. Access Locally

Open browser: **http://localhost:3000**

---

### Option B: Distributed Setup (Multi-Device with Ngrok)

#### Device 2 (Worker LLM):

```bash
cd LLM-Backend

# 1. Configure ngrok in .env
echo "NGROK_AUTH_TOKEN=your_token" > .env
echo "PORT=8000" >> .env

# 2. Start worker (automatically creates ngrok tunnel)
myenv\Scripts\activate
python main.py

# 3. Copy the ngrok URL from output:
# 🌐 LLM Worker is now publicly accessible at:
#    https://worker-llm-12345.ngrok-free.app
```

#### Device 1 (Orchestrator):

```bash
cd RAG-Backend

# 1. Configure .env with worker URL
echo "NGROK_AUTH_TOKEN=your_token" > .env.orchestrator
echo "WORKER_LLM_URLS=[\"https://worker-llm-12345.ngrok-free.app\"]" >> .env.orchestrator
echo "LLM_API_BASE_URL=http://localhost:8000" >> .env.orchestrator
echo "PORT=8081" >> .env.orchestrator

# 2. Start local LLM
cd ../LLM-Backend
myenv\Scripts\activate
python main.py  # Runs locally without ngrok

# 3. Start RAG Backend (new terminal)
cd ../RAG-Backend
myenv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8081

# 4. Copy orchestrator URL from output:
# 🌐 RAG Backend (Orchestrator) is publicly accessible at:
#    https://orchestrator-12345.ngrok-free.app

# 5. Update frontend .env.local
cd ../Frontend/next-app
echo "NEXT_PUBLIC_API_URL=https://orchestrator-12345.ngrok-free.app" > .env.local

# 6. Start frontend
npm run dev
```

#### Access from Anywhere:

Open browser: **http://localhost:3000**  
API accessible at: **https://orchestrator-12345.ngrok-free.app**

**Default Admin Credentials:**

- Email: `admin@graphrag.com`
- Password: `admin123`

---

### Architecture Benefits

| Feature              | Single Device | Distributed        |
| -------------------- | ------------- | ------------------ |
| **Setup Complexity** | Simple        | Moderate           |
| **LLM Performance**  | 1x            | 2x (load balanced) |
| **Remote Access**    | No            | Yes (via ngrok)    |
| **Failover**         | No            | Yes (automatic)    |
| **GPU Sharing**      | N/A           | Use multiple GPUs  |
| **Best For**         | Development   | Production/Demo    |

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint            | Description            |
| ------ | ------------------- | ---------------------- |
| POST   | `/v1/auth/register` | Register new user      |
| POST   | `/v1/auth/login`    | Login and get JWT      |
| GET    | `/v1/auth/me`       | Get current user info  |
| POST   | `/v1/auth/logout`   | Logout current session |

### Documents

| Method | Endpoint                            | Description                          |
| ------ | ----------------------------------- | ------------------------------------ |
| POST   | `/v1/documents/upload`              | Upload document (PDF/DOCX/TXT/Image) |
| GET    | `/v1/documents`                     | List all documents                   |
| DELETE | `/v1/documents/{id}`                | Delete a document                    |
| GET    | `/v1/documents/chunks/{id}/preview` | Get chunk with context               |

### Chat

| Method | Endpoint                             | Description                   |
| ------ | ------------------------------------ | ----------------------------- |
| POST   | `/v1/chat`                           | Send message (non-streaming)  |
| GET    | `/v1/chat/stream`                    | Stream chat response (SSE)    |
| GET    | `/v1/chat/history`                   | Get conversation history      |
| GET    | `/v1/chat/conversations/{id}`        | Get specific conversation     |
| GET    | `/v1/chat/conversations/{id}/export` | Export conversation (JSON/MD) |
| POST   | `/v1/chat/conversations/import`      | Import conversation           |

### Search

| Method | Endpoint                  | Description              |
| ------ | ------------------------- | ------------------------ |
| GET    | `/v1/search/suggestions`  | Get search suggestions   |
| GET    | `/v1/search/autocomplete` | Get autocomplete results |

### Memory

| Method | Endpoint              | Description            |
| ------ | --------------------- | ---------------------- |
| GET    | `/v1/memory/episodic` | Get episodic memories  |
| GET    | `/v1/memory/semantic` | Get semantic concepts  |
| POST   | `/v1/memory/episodic` | Create episodic memory |

### Knowledge Graph

| Method | Endpoint                        | Description                      |
| ------ | ------------------------------- | -------------------------------- |
| GET    | `/v1/graph/entities`            | List entities                    |
| GET    | `/v1/graph/relationships`       | Get entity relationships         |
| POST   | `/v1/graph/extract`             | Extract entities from text       |
| GET    | `/v1/admin/graph/visualization` | Get graph data for visualization |
| GET    | `/v1/admin/graph/stats`         | Get graph statistics             |
| POST   | `/v1/admin/graph/reindex`       | Trigger graph reindexing         |

### Admin

| Method | Endpoint             | Description           |
| ------ | -------------------- | --------------------- |
| GET    | `/v1/admin/users`    | List all users        |
| GET    | `/v1/admin/stats`    | Get system statistics |
| GET    | `/v1/admin/sessions` | Get active sessions   |

---

## 📁 Project Structure

```
Code/
├── Frontend/
│   └── next-app/
│       ├── src/
│       │   ├── app/              # Next.js App Router pages
│       │   │   ├── (dashboard)/  # Protected dashboard routes
│       │   │   │   ├── admin/    # Admin panel & graph viewer
│       │   │   │   ├── analytics/# Analytics dashboard
│       │   │   │   ├── chat/     # Main chat interface
│       │   │   │   ├── dashboard/# User dashboard
│       │   │   │   ├── documents/# Document management
│       │   │   │   ├── memory/   # Memory explorer
│       │   │   │   └── settings/ # Settings & sessions
│       │   │   ├── sign-in/      # Login page
│       │   │   └── create-account/# Registration
│       │   ├── components/       # React components
│       │   │   ├── Sidebar.tsx   # Navigation sidebar
│       │   │   ├── CitationCard.tsx # Citation display
│       │   │   ├── SearchBar.tsx # Autocomplete search
│       │   │   ├── ChunkPreviewModal.tsx # Chunk context viewer
│       │   │   └── ConversationExporter.tsx # Export/import chats
│       │   ├── contexts/         # React contexts (Auth)
│       │   ├── hooks/            # Custom hooks (useChatStream)
│       │   ├── lib/              # Utilities (API client)
│       │   └── types/            # TypeScript types
│       └── package.json
│
├── RAG-Backend/
│   ├── app/
│   │   ├── api/v1/endpoints/     # API route handlers
│   │   │   ├── auth.py           # Authentication endpoints
│   │   │   ├── chat.py           # Chat & conversation endpoints
│   │   │   ├── documents.py      # Document upload & management
│   │   │   ├── memory.py         # Memory system endpoints
│   │   │   ├── graph.py          # Knowledge graph endpoints
│   │   │   └── search.py         # Search & autocomplete
│   │   ├── core/                 # Config, security, auth
│   │   ├── models/               # SQLModel models
│   │   │   ├── user.py           # User & session models
│   │   │   ├── document.py       # Document & chunk models
│   │   │   ├── conversation.py   # Conversation & message models
│   │   │   ├── memory.py         # Memory models
│   │   │   └── entity.py         # Entity & relationship models
│   │   ├── schemas/              # Pydantic schemas
│   │   └── services/             # Business logic
│   │       ├── auth_service.py   # Authentication
│   │       ├── chat_service.py   # Chat & RAG orchestration
│   │       ├── document_service.py# Document processing
│   │       ├── ocr_service.py    # OCR with EasyOCR
│   │       ├── entity_service.py # Entity extraction
│   │       ├── memory_service.py # Memory management
│   │       ├── llm_client.py     # LLM API client
│   │       ├── llm_router.py     # Multi-LLM load balancer
│   │       └── retrieval_service.py# Vector search
│   ├── tests/                    # Pytest test suite
│   │   ├── test_ocr_service.py   # OCR tests
│   │   ├── test_document_service.py
│   │   └── ...
│   ├── data/                     # SQLite DB & FAISS index
│   └── requirements.txt
│
├── LLM-Backend/
│   ├── main.py                   # FastAPI wrapper for llama.cpp
│   ├── mistral-7b-instruct-v0.2.Q4_K_M.gguf  # Model file
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## 🗺️ Development Roadmap

### Completed Phases

| Phase       | Description                      | Status      |
| ----------- | -------------------------------- | ----------- |
| **Phase 1** | Core Infrastructure Setup        | ✅ Complete |
| **Phase 2** | Authentication & User Management | ✅ Complete |
| **Phase 3** | Document Processing Pipeline     | ✅ Complete |
| **Phase 4** | RAG Implementation               | ✅ Complete |
| **Phase 5** | Knowledge Graph Integration      | ✅ Complete |
| **Phase 6** | Memory System                    | ✅ Complete |
| **Phase 7** | Testing & Optimization           | ✅ Complete |
| **Phase 8** | Frontend Integration             | ✅ Complete |

### Phase Details

#### Phase 1: Core Infrastructure ✅

- FastAPI backend with async support
- SQLModel database models
- FAISS vector store integration
- LLM backend wrapper for llama.cpp

#### Phase 2: Authentication ✅

- JWT-based authentication
- User registration and login
- Role-based access control (Admin/User)
- Session management with device tracking

#### Phase 3: Document Processing ✅

- Multi-format support (PDF, DOCX, TXT, Images)
- OCR processing with EasyOCR
- Smart chunking with LangChain
- Sentence Transformers embeddings

#### Phase 4: RAG Implementation ✅

- Vector similarity search with FAISS
- Context retrieval and ranking
- Real-time streaming responses (SSE)
- Source citation with chunk context

#### Phase 5: Knowledge Graph ✅

- Entity extraction from documents
- Relationship mapping
- Graph storage and querying
- Interactive graph visualization

#### Phase 6: Memory System ✅

- Episodic memory (conversation history)
- Semantic memory (concept extraction)
- Memory importance ranking
- Memory search and retrieval

#### Phase 7: Testing & Optimization ✅

- Pytest test suite (21+ tests)
- OCR service tests
- Document processing tests
- Performance optimization

#### Phase 8: Frontend Integration ✅

- **Chat Interface** - Real-time streaming with SSE, citation cards, chunk preview
- **Document Management** - Upload UI with drag-drop, OCR processing, status tracking
- **Admin Dashboard** - User management, system statistics, analytics charts
- **Knowledge Graph Viewer** - Force-directed interactive graph with D3.js
- **Memory Explorer** - Episodic and semantic memory timeline
- **Global Search** - Autocomplete search bar in dashboard header
- **Session Management** - Device tracking, active sessions, logout all
- **Export/Import** - Conversation export (JSON/Markdown/TXT) with import
- **Chunk Preview Modal** - View chunks with surrounding context and entities
- **Citation Cards** - Source display with relevance badges and actions
- **LLM Router** - Load balancing and failover for distributed LLMs
- **Ngrok Integration** - Automatic public tunnels with pyngrok
- **Health Monitoring** - Real-time status of all LLM workers

---

## 🔒 Security Notes

- Change the `SECRET_KEY` in production
- Use HTTPS in production deployments
- Configure proper CORS origins for production
- Store sensitive credentials in environment variables
- The default admin account should be changed after first login

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🌐 Distributed Setup Guide

### Getting Ngrok Static Domains (Free)

1. Sign up at https://ngrok.com
2. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Go to https://dashboard.ngrok.com/domains
4. Click "Create Domain" (1 free static domain per account)
5. You'll get a domain like: `your-name-12345.ngrok-free.app`

### Load Balancing Strategies

**Round Robin** (Default)

- Distributes requests evenly across all LLMs
- Best for: Equal hardware on all devices

**Random**

- Randomly selects an LLM for each request
- Best for: Simple load distribution

**Failover**

- Uses primary LLM, workers only on failure
- Best for: Primary device has better hardware

### Adding More Workers

```env
# In orchestrator .env
WORKER_LLM_URLS=[
  "https://worker1.ngrok-free.app",
  "https://worker2.ngrok-free.app",
  "https://worker3.ngrok-free.app"
]
```

### Health Monitoring

Check LLM health status:

```bash
curl https://orchestrator-12345.ngrok-free.app/health
```

Response includes:

- `primary`: Local LLM status
- `workers`: Each worker's status
- `healthy_count`: Number of working LLMs
- `routing_strategy`: Current load balancing method

---

## 📞 Support

For issues and questions, please open a GitHub issue.

---

Built with ❤️ for the Aerothon Hackathon
