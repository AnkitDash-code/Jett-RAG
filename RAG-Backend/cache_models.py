"""
Standalone script to cache all models for offline operation.
Can be run manually: python cache_models.py

Or automatically runs on first startup of main.py
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.model_cache import get_cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Cache all models."""
    manager = get_cache_manager()
    
    # Force re-cache if script is run directly
    results = manager.cache_all_models(force=True)
    
    # Check if all critical models succeeded
    all_critical_ok = True
    for name, (success, _) in results.items():
        if name != "Tesseract OCR" and name != "status" and not success:
            all_critical_ok = False
            break
    
    if all_critical_ok:
        print("\n✅ SUCCESS: All critical models cached!")
        print("You can now run the RAG Backend offline.")
        return 0
    else:
        print("\n❌ FAILED: Some critical models could not be cached.")
        print("Check the errors above and ensure you have internet connection.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
