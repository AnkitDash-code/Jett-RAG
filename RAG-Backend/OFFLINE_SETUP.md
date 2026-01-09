# RAG Backend - Offline Setup Guide

## 🚀 Quick Start (Automatic Model Caching)

### **First Time Setup (While Online)**

The RAG Backend will **automatically download and cache all models** when you run it for the first time:

```bash
cd RAG-Backend
myenv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

On first startup, you'll see:

```
🚀 RAG Backend - Model Caching (First Run Setup)
This will download ~175 MB of models for offline operation.
This only happens once. Please wait...

📥 Caching embedding model (all-MiniLM-L6-v2)...
✅ Embedding model cached (dim: 384)
📥 Caching reranker model (ms-marco-MiniLM-L-6-v2)...
✅ Reranker model cached
📥 Caching spaCy model (en_core_web_sm)...
✅ spaCy model downloaded and cached
✅ LLM model found: 4.37 GB
⚠️  Tesseract OCR not found (optional for OCR features)

🎉 All critical models cached! Ready for offline operation.
```

**That's it!** The models are now cached and you can run offline.

---

## 🔧 Manual Model Caching (Optional)

If you want to pre-cache models before starting the server:

```bash
cd RAG-Backend
myenv\Scripts\activate
python cache_models.py
```

This explicitly downloads all models without starting the server.

---

## ✅ Verify Offline Readiness

Test that all models and services work:

```bash
cd RAG-Backend
myenv\Scripts\activate
python test_offline.py
```

**Expected Output:**

```
🧪 RAG Backend - Offline Readiness Test

📦 Testing Cached Models:
🧪 Testing Embedding Model...
   ✅ Embedding model working (dimension: 384)
🧪 Testing Reranker Model...
   ✅ Reranker model working (score: 0.1234)
🧪 Testing spaCy NER Model...
   ✅ spaCy model working (found 3 entities)
🧪 Testing Tesseract OCR...
   ⚠️  Tesseract OCR not found (optional)

🌐 Testing Services:
🧪 Testing LLM Backend connection...
   ✅ LLM Backend responding on port 8080
🧪 Testing RAG Backend connection...
   ✅ RAG Backend responding on port 8001
🧪 Testing Frontend connection...
   ✅ Frontend responding on port 3000

📊 Test Summary
Models:
  ✅ Embedding Model
  ✅ Reranker Model
  ✅ spaCy Model
  ⚠️  Tesseract OCR

Services:
  ✅ LLM Backend (8080)
  ✅ RAG Backend (8001)
  ✅ Frontend (3000)

🎉 All systems operational! Ready for offline demo.
```

---

## 🏃 Complete Offline Startup

### **1. Start LLM Backend** (Terminal 1)

```bash
cd LLM-Backend
myenv\Scripts\activate
python main.py
```

✅ Should show: `Uvicorn running on http://0.0.0.0:8080`

### **2. Start RAG Backend** (Terminal 2)

```bash
cd RAG-Backend
myenv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

✅ First run will cache models, subsequent runs will skip (instant start)

### **3. Start Frontend** (Terminal 3)

```bash
cd Frontend/next-app
npm run dev
```

✅ Should show: `Ready on http://localhost:3000`

---

## 📦 What Gets Cached?

| Component                  | Size    | Purpose               | Required    |
| -------------------------- | ------- | --------------------- | ----------- |
| **all-MiniLM-L6-v2**       | ~80 MB  | Document embeddings   | ✅ Yes      |
| **ms-marco-MiniLM-L-6-v2** | ~80 MB  | Reranking results     | ✅ Yes      |
| **en_core_web_sm**         | ~15 MB  | Entity extraction     | ✅ Yes      |
| **mistral-7b-instruct**    | ~4.4 GB | LLM generation        | ✅ Yes      |
| **Tesseract OCR**          | ~50 MB  | Image text extraction | ⚠️ Optional |

**Total Required:** ~4.6 GB  
**Cache Location:** `C:\Users\ASUS\.cache\huggingface\hub\`

---

## 🔍 Troubleshooting

### **Models Don't Download**

```bash
# Manually cache all models
cd RAG-Backend
python cache_models.py
```

### **Check What's Missing**

```bash
python test_offline.py
```

### **Re-cache Models**

```bash
# Delete cache status file
del .model_cache_status

# Restart server (will re-download)
python -m uvicorn app.main:app --port 8001
```

### **Offline Mode Test**

1. Cache all models (while online)
2. Disconnect from internet
3. Run `test_offline.py`
4. Start all services
5. Should work completely offline ✅

---

## 🎯 Model Cache Status

The system tracks whether models are cached in:

```
RAG-Backend/.model_cache_status
```

- **Exists:** Models cached, skip download
- **Missing:** First run, download models

To force re-download, delete this file.

---

## 📍 Architecture

```
Internet (First Run Only)
    ↓ (Downloads models)
┌─────────────────────┐
│  Model Cache Dir    │
│  ~/.cache/...       │
└─────────────────────┘
    ↓ (Loads from cache)
┌─────────────────────┐     ┌─────────────────┐     ┌──────────────┐
│   RAG Backend       │────→│   LLM Backend   │────→│  llama.cpp   │
│   (Port 8001)       │     │   (Port 8080)   │     │  (Port 8000) │
│                     │     │                 │     │              │
│ • Embeddings        │     │ • Mistral-7B    │     │ • GGUF Model │
│ • Reranker          │     │ • OpenAI API    │     │ • Local Gen  │
│ • spaCy NER         │     │                 │     │              │
└─────────────────────┘     └─────────────────┘     └──────────────┘
    ↓
┌─────────────────────┐
│   Frontend          │
│   (Port 3000)       │
└─────────────────────┘
```

**Offline Operation:**

- All models loaded from local cache
- No internet required after first setup
- Complete RAG pipeline runs locally

---

## ✨ Features

- ✅ **Automatic caching** on first run
- ✅ **Zero configuration** - just start the server
- ✅ **Offline-ready** after initial setup
- ✅ **Smart detection** - skips re-download
- ✅ **Comprehensive testing** - verify before demo
- ✅ **Clear logging** - see exactly what's happening

---

## 🎉 Ready to Demo!

After first run:

1. ✅ All models cached (~4.6 GB)
2. ✅ No internet needed
3. ✅ Start services in any order
4. ✅ Full RAG pipeline works offline
5. ✅ Upload documents, chat, retrieve - all local!

**Demo anywhere - no WiFi needed!** 🚀
