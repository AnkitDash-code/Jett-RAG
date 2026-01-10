# Distributed LLM Backend (Vision LLM Server)

Vision Language Model (VLM) server for extracting text and descriptions from images using **Granite VLM**.

## 📦 Contents

| File | Purpose |
|------|---------|
| `api.py` | FastAPI server for vision extraction |
| `vision_extractor.py` | VLM processing logic |
| `API_USAGE.md` | API documentation |
| `Start_Server.bat` | Windows startup script |

## 🚀 Quick Start

### Windows
```bash
Start_Server.bat
```

### Manual
```bash
cd Distributed-LLM-Backend
python api.py
```

Server runs on: **http://localhost:8088**

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vision/extract` | POST | Extract text from image |
| `/health` | GET | Health check |

### Extract Text from Image

```bash
curl -X POST "http://localhost:8088/vision/extract" \
  -F "file=@document.png"
```

**Response:**
```json
{
  "full_text": "Extracted text from the image...",
  "description": "This image contains a document with...",
  "confidence": 0.95
}
```

## 🔧 Configuration

The server connects to a Granite VLM model. Configure the model endpoint in the environment or code.

## 🔗 Integration

This server is used by the RAG-Backend's `vision_llm_client.py` to:
1. Extract text from scanned PDFs and images
2. Generate image descriptions
3. Detect QR codes and barcodes (via pyzbar)

## 📋 Requirements

- Python 3.10+
- Granite VLM model or compatible vision model
- CUDA GPU (recommended)
