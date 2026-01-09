"""
Test script to verify offline readiness.
Run after caching models to ensure everything works.
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def test_llm_backend():
    """Test if LLM Backend is running."""
    logger.info("🧪 Testing LLM Backend connection...")
    try:
        import httpx
        response = httpx.get("http://localhost:8080/health", timeout=5.0)
        if response.status_code == 200:
            logger.info("   ✅ LLM Backend responding on port 8080")
            return True
        else:
            logger.info(f"   ❌ LLM Backend returned status {response.status_code}")
            return False
    except Exception as e:
        logger.info(f"   ❌ LLM Backend not running: {e}")
        logger.info("      Start it with: cd LLM-Backend && python main.py")
        return False


def test_embedding_model():
    """Test if embedding model loads."""
    logger.info("🧪 Testing Embedding Model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedding = model.encode("test sentence")
        logger.info(f"   ✅ Embedding model working (dimension: {len(embedding)})")
        return True
    except Exception as e:
        logger.info(f"   ❌ Embedding model failed: {e}")
        return False


def test_reranker_model():
    """Test if reranker model loads."""
    logger.info("🧪 Testing Reranker Model...")
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = model.predict([("test query", "test document")])
        logger.info(f"   ✅ Reranker model working (score: {scores[0]:.4f})")
        return True
    except Exception as e:
        logger.info(f"   ❌ Reranker model failed: {e}")
        return False


def test_spacy_model():
    """Test if spaCy model loads (OPTIONAL)."""
    logger.info("🧪 Testing spaCy NER Model (optional)...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logger.info(f"   ✅ spaCy model working (found {len(entities)} entities)")
        return True
    except Exception as e:
        logger.info(f"   ⚠️  spaCy not available (optional - entity extraction uses LLM)")
        return True  # Return True since it's optional


def test_rag_backend():
    """Test if RAG Backend is running."""
    logger.info("🧪 Testing RAG Backend connection...")
    try:
        import httpx
        response = httpx.get("http://localhost:8001/health", timeout=5.0)
        if response.status_code == 200:
            logger.info("   ✅ RAG Backend responding on port 8001")
            return True
        else:
            logger.info(f"   ❌ RAG Backend returned status {response.status_code}")
            return False
    except Exception as e:
        logger.info(f"   ❌ RAG Backend not running: {e}")
        logger.info("      Start it with: python -m uvicorn app.main:app --port 8001")
        return False


def test_frontend():
    """Test if Frontend is running."""
    logger.info("🧪 Testing Frontend connection...")
    try:
        import httpx
        response = httpx.get("http://localhost:3000", timeout=5.0)
        if response.status_code == 200:
            logger.info("   ✅ Frontend responding on port 3000")
            return True
        else:
            logger.info(f"   ❌ Frontend returned status {response.status_code}")
            return False
    except Exception as e:
        logger.info(f"   ❌ Frontend not running: {e}")
        logger.info("      Start it with: cd Frontend/next-app && npm run dev")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("🧪 RAG Backend - Offline Readiness Test")
    logger.info("=" * 70)
    logger.info("")
    
    # Test models (should work without internet)
    logger.info("📦 Testing Cached Models:")
    logger.info("")
    
    model_results = {
        "Embedding Model": test_embedding_model(),
        "Reranker Model": test_reranker_model(),
        "spaCy Model": test_spacy_model(),
    }
    
    logger.info("")
    logger.info("🌐 Testing Services:")
    logger.info("")
    
    service_results = {
        "LLM Backend (8080)": test_llm_backend(),
        "RAG Backend (8001)": test_rag_backend(),
        "Frontend (3000)": test_frontend(),
    }
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 Test Summary")
    logger.info("=" * 70)
    logger.info("")
    
    logger.info("Models:")
    for name, status in model_results.items():
        # spaCy is optional
        is_optional = name in ["spaCy Model"]
        icon = "✅" if status else ("⚠️ " if is_optional else "❌")
        logger.info(f"  {icon} {name}")
    
    logger.info("")
    logger.info("Services:")
    for name, status in service_results.items():
        icon = "✅" if status else "❌"
        logger.info(f"  {icon} {name}")
    
    logger.info("")
    
    # Check critical components (spaCy is optional)
    critical_models = all(
        status for name, status in model_results.items()
        if name not in ["spaCy Model"]
    )
    all_services = all(service_results.values())
    
    if critical_models and all_services:
        logger.info("🎉 All systems operational! Ready for offline demo.")
        logger.info("")
        return 0
    elif critical_models:
        logger.info("⚠️  Models ready but some services not running.")
        logger.info("   Start the missing services to begin.")
        logger.info("")
        return 0
    else:
        logger.info("❌ Some models not cached. Run: python cache_models.py")
        logger.info("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
