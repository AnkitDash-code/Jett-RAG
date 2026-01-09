"""
Automatic model caching for offline operation.
Downloads and caches all required models on first run.
"""
import os
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Model cache status file
CACHE_STATUS_FILE = Path(__file__).parent.parent.parent / ".model_cache_status"


class ModelCacheManager:
    """Manages automatic caching of all required models."""
    
    def __init__(self):
        self.cache_complete = self._check_cache_status()
        
    def _check_cache_status(self) -> bool:
        """Check if models have been cached previously."""
        return CACHE_STATUS_FILE.exists()
    
    def _mark_cache_complete(self):
        """Mark models as cached."""
        CACHE_STATUS_FILE.touch()
        logger.info("✅ All models cached successfully")
    
    def cache_embedding_model(self) -> Tuple[bool, str]:
        """Cache sentence transformer embedding model."""
        try:
            logger.info("📥 Caching embedding model (all-MiniLM-L6-v2)...")
            from sentence_transformers import SentenceTransformer
            
            # Force download if not cached
            model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device='cpu'  # Ensure CPU compatibility
            )
            
            # Test it works
            test_embedding = model.encode("test")
            
            logger.info(f"✅ Embedding model cached (dim: {len(test_embedding)})")
            return True, "Embedding model ready"
            
        except Exception as e:
            error_msg = f"Failed to cache embedding model: {e}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def cache_reranker_model(self) -> Tuple[bool, str]:
        """Cache cross-encoder reranker model."""
        try:
            logger.info("📥 Caching reranker model (ms-marco-MiniLM-L-6-v2)...")
            from sentence_transformers import CrossEncoder
            
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Test it works
            test_score = model.predict([("test query", "test document")])
            
            logger.info(f"✅ Reranker model cached")
            return True, "Reranker model ready"
            
        except Exception as e:
            error_msg = f"Failed to cache reranker model: {e}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def cache_spacy_model(self) -> Tuple[bool, str]:
        """Cache spaCy NER model (OPTIONAL - not used in RAG pipeline)."""
        try:
            logger.info("📥 Checking spaCy model (optional, not required)...")
            import spacy
            
            # Try to load
            try:
                nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model available (optional)")
                return True, "spaCy model ready (optional)"
            except OSError:
                # Download if not present
                logger.info("⚠️  spaCy model not found (optional, skipping)")
                return True, "spaCy not installed (optional - entity extraction uses LLM)"
                
        except Exception as e:
            # spaCy is optional - Python 3.14+ has compatibility issues
            warning = f"spaCy not available: {str(e)[:100]}... (optional - entity extraction uses LLM)"
            logger.warning(f"⚠️  {warning}")
            return True, warning  # Return True since it's optional
    
    def cache_easyocr_model(self) -> Tuple[bool, str]:
        """Cache EasyOCR model for image-based PDFs and scanned documents."""
        try:
            logger.info("📥 Caching EasyOCR model (for scanned documents)...")
            import easyocr
            
            # Initialize reader - downloads models on first use (~100MB)
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            
            logger.info("✅ EasyOCR model cached")
            return True, "EasyOCR model ready"
            
        except ImportError:
            warning = "EasyOCR not installed. Scanned document support disabled. Install with: pip install easyocr"
            logger.warning(f"⚠️  {warning}")
            return False, warning
        except Exception as e:
            error_msg = f"Failed to cache EasyOCR model: {e}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def verify_tesseract(self) -> Tuple[bool, str]:
        """Verify Tesseract OCR installation."""
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract OCR found: {version}")
            return True, f"Tesseract {version}"
        except Exception as e:
            warning = (
                "Tesseract OCR not found. OCR features will be disabled.\n"
                "Install from: https://github.com/UB-Mannheim/tesseract/wiki"
            )
            logger.warning(f"⚠️  {warning}")
            return False, warning
    
    def verify_llm_backend(self) -> Tuple[bool, str]:
        """Check if LLM model file exists."""
        llm_model_paths = [
            Path(__file__).parent.parent.parent.parent / "LLM-Backend" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            Path("../LLM-Backend/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        ]
        
        for llm_path in llm_model_paths:
            if llm_path.exists():
                size_gb = llm_path.stat().st_size / (1024**3)
                logger.info(f"✅ LLM model found: {size_gb:.2f} GB")
                return True, f"LLM model ready ({size_gb:.2f} GB)"
        
        warning = "LLM model not found. Make sure LLM-Backend is set up."
        logger.warning(f"⚠️  {warning}")
        return False, warning
    
    def cache_all_models(self, force: bool = False) -> Dict[str, Tuple[bool, str]]:
        """
        Cache all required models for offline operation.
        
        Args:
            force: Force re-caching even if already cached
            
        Returns:
            Dict of {model_name: (success, message)}
        """
        if self.cache_complete and not force:
            logger.info("✅ Models already cached (use force=True to re-cache)")
            return {"status": (True, "All models already cached")}
        
        logger.info("=" * 70)
        logger.info("🚀 RAG Backend - Model Caching (First Run Setup)")
        logger.info("=" * 70)
        logger.info("")
        logger.info("This will download ~175 MB of models for offline operation.")
        logger.info("This only happens once. Please wait...")
        logger.info("")
        
        results = {
            "LLM Model": self.verify_llm_backend(),
            "Embedding Model": self.cache_embedding_model(),
            "Reranker Model": self.cache_reranker_model(),
            "EasyOCR Model": self.cache_easyocr_model(),
            "spaCy Model": self.cache_spacy_model(),
            "Tesseract OCR": self.verify_tesseract(),
        }
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 Model Cache Summary")
        logger.info("=" * 70)
        
        all_critical_success = True
        for name, (success, message) in results.items():
            icon = "✅" if success else "❌"
            logger.info(f"{icon} {name}: {message}")
            
            # spaCy, Tesseract, and EasyOCR are optional
            if name not in ["Tesseract OCR", "spaCy Model", "EasyOCR Model"] and not success:
                all_critical_success = False
        
        logger.info("")
        
        if all_critical_success:
            logger.info("🎉 All critical models cached! Ready for offline operation.")
            self._mark_cache_complete()
        else:
            logger.error("⚠️  Some critical models failed to cache. Check errors above.")
            logger.error("The application may not work properly.")
        
        logger.info("=" * 70)
        logger.info("")
        
        return results
    
    def ensure_models_cached(self):
        """Ensure models are cached, downloading if necessary."""
        if not self.cache_complete:
            self.cache_all_models()


# Global instance
_cache_manager = None

def get_cache_manager() -> ModelCacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ModelCacheManager()
    return _cache_manager


def ensure_models_cached():
    """Convenience function to ensure all models are cached."""
    manager = get_cache_manager()
    manager.ensure_models_cached()
