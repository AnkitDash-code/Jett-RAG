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

- 📄 **PDF Document Upload** - Upload and process PDF documents
- 🔍 **Semantic Search** - FAISS-powered vector similarity search
- 🧠 **Smart Chunking** - LangChain RecursiveCharacterTextSplitter for optimal context
- 💬 **Real-time Streaming** - Token-by-token LLM response streaming via SSE
- 📚 **Source Citations** - View exact sources used for each answer

### User Experience

- 🔐 **Authentication** - JWT-based secure login/registration
- 👥 **Role-Based Access** - Admin and User roles with different permissions
- 💾 **Chat History** - Persistent conversation storage and retrieval
- 🎨 **Modern UI** - Clean, responsive design with dark theme

### Admin Features

- 📊 **Analytics Dashboard** - Usage statistics and charts
- 👤 **User Management** - View and manage users
- 📁 **Document Permissions** - Control document access per user
- ⚙️ **Settings** - Configure system preferences

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND                                   │
│                    Next.js 15 (Port 3000)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │    Chat     │  │  Dashboard  │  │   Admin     │  │  Analytics │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
└─────────┼────────────────┼────────────────┼───────────────┼─────────┘
          │                │                │               │
          └────────────────┴────────────────┴───────────────┘
                                    │
                                    ▼ HTTP/SSE
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG-BACKEND                                  │
│                    FastAPI (Port 8081)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │    Auth     │  │    Chat     │  │  Documents  │  │  Retrieval │ │
│  │   Service   │  │   Service   │  │   Service   │  │   Service  │ │
│  └─────────────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│                          │                │               │         │
│  ┌───────────────────────┴────────────────┴───────────────┘         │
│  │                                                                   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  │   SQLite    │  │    FAISS    │  │  Sentence Transformers  │  │
│  │  │  (Users,    │  │   Vector    │  │      (Embeddings)       │  │
│  │  │   Chats)    │  │    Store    │  │                         │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└──┼──────────────────────────────────────────────────────────────────┘
   │
   ▼ HTTP Streaming
┌─────────────────────────────────────────────────────────────────────┐
│                         LLM-BACKEND                                  │
│                    FastAPI (Port 8080)                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              OpenAI-compatible API Wrapper                   │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼ HTTP
┌─────────────────────────────────────────────────────────────────────┐
│                         LLAMA.CPP                                    │
│                    Server (Port 8000)                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           Mistral-7B-Instruct (Q4_K_M Quantized)            │   │
│  │                    GPU Accelerated (CUDA)                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

### Frontend

| Technology         | Purpose                         |
| ------------------ | ------------------------------- |
| Next.js 15         | React framework with App Router |
| TypeScript         | Type-safe development           |
| Recharts           | Analytics visualizations        |
| Sonner             | Toast notifications             |
| fetch-event-source | SSE streaming client            |

### RAG-Backend

| Technology            | Purpose                        |
| --------------------- | ------------------------------ |
| FastAPI               | High-performance async API     |
| SQLAlchemy            | Async ORM for database         |
| FAISS                 | Vector similarity search       |
| Sentence Transformers | Text embeddings                |
| LangChain             | Document processing & chunking |
| PyMuPDF               | PDF parsing                    |
| JWT                   | Authentication tokens          |

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

# LLM Backend
LLM_API_BASE_URL=http://localhost:8080

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda

# FAISS
FAISS_INDEX_PATH=./data/faiss_index
```

### LLM-Backend

Configure `Settings.kcpps` for llama.cpp server settings.

---

## 🚀 Running the Application

### 1. Start llama.cpp Server

```bash
# In LLM-Backend folder
./llama-server -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -ngl 99 -c 4096 --port 8000
```

### 2. Start LLM-Backend

```bash
cd LLM-Backend
python main.py
# Runs on http://localhost:8080
```

### 3. Start RAG-Backend

```bash
cd RAG-Backend
python -m app.main
# Runs on http://localhost:8081
```

### 4. Start Frontend

```bash
cd Frontend/next-app
npm run dev
# Runs on http://localhost:3000
```

### 5. Access the Application

Open your browser and navigate to: **http://localhost:3000**

**Default Admin Credentials:**

- Email: `admin@graphrag.com`
- Password: `admin123`

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint            | Description           |
| ------ | ------------------- | --------------------- |
| POST   | `/v1/auth/register` | Register new user     |
| POST   | `/v1/auth/login`    | Login and get JWT     |
| GET    | `/v1/auth/me`       | Get current user info |

### Documents

| Method | Endpoint               | Description         |
| ------ | ---------------------- | ------------------- |
| POST   | `/v1/documents/upload` | Upload PDF document |
| GET    | `/v1/documents`        | List all documents  |
| DELETE | `/v1/documents/{id}`   | Delete a document   |

### Chat

| Method | Endpoint                     | Description                  |
| ------ | ---------------------------- | ---------------------------- |
| POST   | `/v1/chat`                   | Send message (non-streaming) |
| GET    | `/v1/chat/stream`            | Stream chat response (SSE)   |
| GET    | `/v1/chat/history`           | Get conversation history     |
| GET    | `/v1/chat/conversation/{id}` | Get specific conversation    |

### Admin

| Method | Endpoint          | Description           |
| ------ | ----------------- | --------------------- |
| GET    | `/v1/admin/users` | List all users        |
| GET    | `/v1/admin/stats` | Get system statistics |

---

## 📁 Project Structure

```
Code/
├── Frontend/
│   └── next-app/
│       ├── src/
│       │   ├── app/              # Next.js App Router pages
│       │   │   ├── (dashboard)/  # Protected dashboard routes
│       │   │   │   ├── admin/    # Admin panel
│       │   │   │   ├── analytics/# Analytics page
│       │   │   │   ├── chat/     # Main chat interface
│       │   │   │   └── dashboard/# User dashboard
│       │   │   ├── sign-in/      # Login page
│       │   │   └── create-account/# Registration
│       │   ├── components/       # React components
│       │   ├── contexts/         # React contexts (Auth)
│       │   ├── hooks/            # Custom hooks (useChatStream)
│       │   ├── lib/              # Utilities (API client)
│       │   └── types/            # TypeScript types
│       └── package.json
│
├── RAG-Backend/
│   ├── app/
│   │   ├── api/v1/endpoints/     # API route handlers
│   │   ├── core/                 # Config, security, auth
│   │   ├── models/               # SQLAlchemy models
│   │   ├── schemas/              # Pydantic schemas
│   │   └── services/             # Business logic
│   │       ├── auth_service.py   # Authentication
│   │       ├── chat_service.py   # Chat & RAG orchestration
│   │       ├── document_service.py# Document processing
│   │       ├── llm_client.py     # LLM API client
│   │       └── retrieval_service.py# Vector search
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

## 📞 Support

For issues and questions, please open a GitHub issue.

---

Built with ❤️ for the Aerothon Hackathon
