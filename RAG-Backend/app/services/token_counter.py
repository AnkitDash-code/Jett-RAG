"""
Token counting service for LLM usage tracking.
Provides accurate or estimated token counts for prompts and responses.
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counter using tiktoken or fallback estimation.
    """
    
    _instance = None
    _encoder = None
    _encoding_name = "cl100k_base"  # Works for most modern models
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._load_encoder()
    
    def _load_encoder(self):
        """Load tiktoken encoder if available."""
        if TokenCounter._encoder is not None:
            return
        
        try:
            import tiktoken
            TokenCounter._encoder = tiktoken.get_encoding(self._encoding_name)
            logger.info(f"Loaded tiktoken encoder: {self._encoding_name}")
        except ImportError:
            logger.warning("tiktoken not installed. Using estimation fallback.")
            TokenCounter._encoder = None
        except Exception as e:
            logger.error(f"Failed to load tiktoken: {e}")
            TokenCounter._encoder = None
    
    def count(self, text: str) -> int:
        """
        Count tokens in text.
        Uses tiktoken if available, else estimates.
        """
        if not text:
            return 0
        
        if TokenCounter._encoder is not None:
            try:
                return len(TokenCounter._encoder.encode(text))
            except Exception as e:
                logger.debug(f"Tiktoken encoding error: {e}")
        
        # Fallback: estimate ~4 characters per token (English average)
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count without tokenizer.
        Uses heuristic: ~4 characters per token for English.
        """
        # Simple heuristic based on character count
        # Average English word is ~5 chars, average token is ~4 chars
        return max(1, len(text) // 4)
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of chat messages.
        Accounts for message overhead.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += self.count(content)
            # Add overhead per message (role token + formatting)
            total += 4
        
        # Add response priming
        total += 3
        
        return total
    
    def count_with_breakdown(
        self,
        messages: List[Dict[str, str]],
        context: str = "",
    ) -> Dict[str, int]:
        """
        Count tokens with detailed breakdown.
        
        Returns dict with:
        - messages: tokens in messages
        - context: tokens in RAG context
        - overhead: formatting overhead
        - total: total tokens
        """
        message_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            message_tokens += self.count(content)
        
        context_tokens = self.count(context) if context else 0
        
        # Overhead: per-message formatting + response priming
        overhead = len(messages) * 4 + 3
        
        return {
            "messages": message_tokens,
            "context": context_tokens,
            "overhead": overhead,
            "total": message_tokens + context_tokens + overhead,
        }
    
    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "...",
    ) -> str:
        """
        Truncate text to fit within token limit.
        """
        if self.count(text) <= max_tokens:
            return text
        
        # Binary search for the right length
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[:mid] + suffix) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        
        return text[:low] + suffix if low > 0 else suffix
    
    def is_tiktoken_available(self) -> bool:
        """Check if tiktoken is available."""
        return TokenCounter._encoder is not None


# Singleton instance
token_counter = TokenCounter()


def get_token_counter() -> TokenCounter:
    """Get the token counter singleton instance."""
    return token_counter
