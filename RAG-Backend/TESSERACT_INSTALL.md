# Installing Tesseract OCR (Optional)

Tesseract is required for extracting text from images in documents (PDFs with scanned pages, screenshots, etc.).

## Windows Installation

### **Option 1: Official Installer (Recommended)**

1. **Download:** https://github.com/UB-Mannheim/tesseract/wiki

   - Get `tesseract-ocr-w64-setup-5.3.3.20231005.exe` (or latest)

2. **Install:**

   - Run installer
   - Default path: `C:\Program Files\Tesseract-OCR`
   - ✅ Check "Add to PATH" during installation

3. **Verify:**
   ```bash
   tesseract --version
   # Should show: tesseract v5.3.3
   ```

### **Option 2: Manual Setup (if PATH not added)**

If you didn't add to PATH during installation, configure manually:

**In RAG-Backend, create `.env.local`:**

```bash
# Tesseract executable path
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Or set in Python code:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## Test Installation

```bash
cd RAG-Backend
myenv\Scripts\activate
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

**Expected:** `5.3.3` (or your installed version)

---

## What if I skip Tesseract?

✅ **RAG Backend will still work!**

Without Tesseract:

- ❌ Cannot extract text from images in PDFs
- ❌ Cannot process screenshot uploads
- ✅ All other features work normally
- ✅ Text-based PDFs, DOCX, PPTX work fine

Tesseract is marked as **optional** in the offline setup.

---

## Download Size

- Installer: ~50 MB
- Language data (English): ~5 MB (included)

---

## Troubleshooting

### **"tesseract not found"**

```bash
# Add to PATH manually
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
# Restart terminal
```

### **"Failed loading language 'eng'"**

During installation, make sure to install **English language data**.

### **Still not working?**

```bash
# Check where it's installed
where tesseract
# Should show: C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## Alternative: Skip OCR Features

If you don't need OCR, you can skip Tesseract entirely. The RAG Backend will log a warning but continue working:

```
⚠️  Tesseract OCR not found. OCR features will be disabled.
```

All other document processing (text extraction from DOCX, PPTX, text-based PDFs) will work normally.
