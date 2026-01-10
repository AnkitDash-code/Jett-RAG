# LLM Backend (KoboldCpp + GGUF)

Local LLM inference server using **KoboldCpp** with **GGUF** models for the GraphRAG Knowledge Portal.

## 📦 Contents

| File | Purpose |
|------|---------|
| `koboldcpp.exe` | KoboldCpp inference engine |
| `mistral-7b-instruct-v0.2.Q4_K_M.gguf` | Main chat model (~4.4 GB) |
| `qwen.gguf` | Utility model for query expansion (~2 GB) |
| `main.py` | FastAPI wrapper with auto-start |
| `Settings.kcpps` | KoboldCpp configuration |
| `start.bat` / `start.sh` | Startup scripts |

## 🚀 Quick Start

### Windows
```bash
# One-click start (auto-configures and launches)
start.bat
```

### Manual
```bash
cd LLM-Backend
myenv\Scripts\activate
python main.py
```

Server runs on: **http://localhost:8000**

## 📋 Features

- ✅ **Auto-start** - Automatically launches KoboldCpp on startup
- ✅ **Dual models** - Chat model + utility model support
- ✅ **Health checks** - `/health` endpoint for monitoring
- ✅ **OpenAI-compatible API** - Works with standard LLM clients
- ✅ **GPU acceleration** - CUDA support for NVIDIA GPUs

## ⬇️ Model Downloads

### 1. KoboldCpp
- Download: https://github.com/LostRuins/koboldcpp/releases
- Place `koboldcpp.exe` in this folder

### 2. Chat Model (Mistral 7B)
- Download: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
- File: `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (~4.4 GB)

### 3. Utility Model (Optional)
- For query expansion and utility tasks
- Smaller model like Qwen 2.5B recommended

## 🔧 Configuration

Edit `Settings.kcpps` for:
- GPU layers (`--gpulayers`)
- Context size (`--contextsize`)
- Port (`--port`)

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `POST /v1/completions` | Text completion |
| `GET /health` | Health check |

## 🔗 Links

- KoboldCpp: https://github.com/LostRuins/koboldcpp
- Mistral 7B GGUF: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
