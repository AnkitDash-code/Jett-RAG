# 🚀 Complete RAG System - Startup Guide

## Quick Start (3 Commands)

### **First Time Setup (While Online)**

Open 3 terminals and run:

```bash
# Terminal 1 - LLM Backend
cd LLM-Backend
myenv\Scripts\activate
python main.py

# Terminal 2 - RAG Backend (auto-downloads models ~175MB on first run)
cd RAG-Backend
start.bat

# Terminal 3 - Frontend
cd Frontend\next-app
npm run dev
```

**Wait for model caching to complete on first run (~2-3 minutes)**

Access: http://localhost:3000

---

## Subsequent Runs (Instant Startup)

Same commands, but no downloads - starts in seconds!

---

## ✅ Pre-Flight Check

Before starting, verify everything is ready:

```bash
cd RAG-Backend
test.bat
```

**Should show:**

```
🎉 All systems operational! Ready for offline demo.
```

---

## 🎯 What Happens on First Run?

### **Terminal 2 (RAG Backend) Output:**

```
🚀 RAG Backend - Model Caching (First Run Setup)
═══════════════════════════════════════════════════════════════════

This will download ~175 MB of models for offline operation.
This only happens once. Please wait...

📥 Caching embedding model (all-MiniLM-L6-v2)...
   ✅ Embedding model cached (dim: 384)

📥 Caching reranker model (ms-marco-MiniLM-L-6-v2)...
   ✅ Reranker model cached

📥 Caching spaCy model (en_core_web_sm)...
   ✅ spaCy model downloaded and cached

🔍 Checking LLM model...
   ✅ LLM model found: 4.37 GB

🔍 Checking Tesseract OCR...
   ⚠️  Tesseract OCR not found (optional for OCR features)

═══════════════════════════════════════════════════════════════════
📊 Model Cache Summary
═══════════════════════════════════════════════════════════════════
✅ LLM Model: LLM model ready (4.37 GB)
✅ Embedding Model: Embedding model ready
✅ Reranker Model: Reranker model ready
✅ spaCy Model: spaCy model ready
⚠️  Tesseract OCR: (optional)

🎉 All critical models cached! Ready for offline operation.

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Computer (Offline)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Frontend (Port 3000)                                       │
│    ↓                                                        │
│  RAG Backend (Port 8001)                                    │
│    ├── Embeddings (all-MiniLM-L6-v2) ✅ Cached            │
│    ├── Reranker (ms-marco) ✅ Cached                       │
│    ├── spaCy NER ✅ Cached                                 │
│    └── Vector Store (FAISS) + Graph (rustworkx)            │
│    ↓                                                        │
│  LLM Backend (Port 8080)                                    │
│    └── FastAPI Wrapper                                      │
│    ↓                                                        │
│  llama.cpp Server (Port 8000)                               │
│    └── Mistral-7B (4.4 GB GGUF) ✅ Local                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

No Internet Required After First Setup! 🎉
```

---

## 🔍 Troubleshooting

### **Problem: Models won't download**

**Solution:**

```bash
# Check internet connection
ping google.com

# Manually cache
cd RAG-Backend
python cache_models.py
```

### **Problem: "Port already in use"**

**Solution:**

```bash
# Find process using port
netstat -ano | findstr :8001

# Kill process
taskkill /PID <PID> /F
```

### **Problem: Frontend won't start**

**Solution:**

```bash
cd Frontend\next-app
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### **Problem: Want to verify everything**

**Solution:**

```bash
cd RAG-Backend
test.bat
```

### **Problem: LLM Backend not responding**

**Check:**

1. Is `main.py` running in LLM-Backend?
2. Is port 8080 free?
3. Is mistral model file present?

```bash
cd LLM-Backend
python main.py
```

---

## 📦 Disk Space Required

| Component             | Size        | Location             |
| --------------------- | ----------- | -------------------- |
| Mistral-7B GGUF       | 4.4 GB      | LLM-Backend/         |
| Embedding Model       | 80 MB       | HF Cache             |
| Reranker Model        | 80 MB       | HF Cache             |
| spaCy Model           | 15 MB       | Python site-packages |
| Frontend node_modules | 500 MB      | Frontend/next-app/   |
| **Total**             | **~5.1 GB** |                      |

---

## ✅ Success Indicators

### **LLM Backend (Terminal 1):**

```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### **RAG Backend (Terminal 2):**

```
🎉 All critical models cached! Ready for offline operation.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### **Frontend (Terminal 3):**

```
✓ Ready in 2.3s
○ Local:   http://localhost:3000
```

---

## 🎬 Demo Workflow

1. **Start all services** (3 terminals)
2. **Open browser** → http://localhost:3000
3. **Sign in:** admin@graphrag.com / admin123
4. **Upload documents** (PDF, DOCX, TXT)
5. **Chat with documents** (streaming responses)
6. **View knowledge graph** (Admin → Graph Visualization)
7. **Export conversations** (JSON/Markdown)

---

## 🌐 Offline Mode

After first run, you can:

- ✅ Disconnect from internet
- ✅ Start all services
- ✅ Full RAG functionality works
- ✅ Upload, embed, retrieve, chat - all local!

**Perfect for:**

- Conference demos
- Client presentations
- Classroom teaching
- Air-gapped environments
- Travel/remote locations

---

## 🎯 Port Summary

| Service                 | Port | URL                           |
| ----------------------- | ---- | ----------------------------- |
| Frontend                | 3000 | http://localhost:3000         |
| RAG Backend API         | 8001 | http://localhost:8001/v1/docs |
| LLM Backend             | 8080 | http://localhost:8080         |
| llama.cpp (if separate) | 8000 | http://localhost:8000         |

---

## 📚 Documentation

- [AUTO_CACHE_SETUP.md](AUTO_CACHE_SETUP.md) - Auto-caching explanation
- [OFFLINE_SETUP.md](OFFLINE_SETUP.md) - Complete offline guide
- [TESSERACT_INSTALL.md](TESSERACT_INSTALL.md) - OCR setup (optional)
- [README.md](README.md) - Full project docs

---

## 🆘 Need Help?

1. **Check test results:** `cd RAG-Backend && test.bat`
2. **View logs:** Each terminal shows error messages
3. **Verify ports:** `netstat -ano | findstr "3000 8001 8080"`
4. **Re-cache models:** `cd RAG-Backend && cache.bat`

---

## 🎉 You're Ready!

**System Status:**

- ✅ Auto-caching configured
- ✅ Offline mode enabled
- ✅ All models integrated
- ✅ Production-ready
- ✅ Demo-ready

**Just run the 3 terminals and you're live!** 🚀
