"""
Granite Docling VLM Client for Phase 8.

Communicates with Granite Docling 258M model via existing LLM API's /v1/convert endpoint.
Your friend integrated Ollama (ibm/granite-docling) into the same API server.

Endpoint: POST {GRANITE_API_URL}/v1/convert
- Accepts: multipart/form-data with PDF file
- Returns: {doctags, markdown, tables, images, page_count}
"""
import os
import re
import time
import logging
import hashlib
import asyncio
from typing import Dict, Any, Optional, List

import httpx
from app.config import settings
from app.services.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError

logger = logging.getLogger(__name__)


class GraniteDoclingClient:
    """
    Client for Granite Docling VLM via existing LLM API.
    
    Features:
    - Async HTTP communication with retry logic
    - Circuit breaker for resilience
    - Image-aware document parsing with DocTags
    - Uses same API as utility LLM (http://localhost:8080)
    """
    
    def __init__(self):
        self.api_url = getattr(settings, 'GRANITE_API_URL', 'http://localhost:8080')
        self.endpoint = getattr(settings, 'GRANITE_CONVERT_ENDPOINT', '/v1/convert')
        self.full_url = f"{self.api_url.rstrip('/')}{self.endpoint}"
        self.timeout = getattr(settings, 'GRANITE_SERVER_TIMEOUT', 300)
        self.max_retries = getattr(settings, 'GRANITE_MAX_RETRIES', 3)
        self.max_file_size_mb = getattr(settings, 'GRANITE_MAX_FILE_SIZE_MB', 100)
        
        # Circuit breaker configuration
        cb_threshold = getattr(settings, 'GRANITE_CIRCUIT_BREAKER_THRESHOLD', 5)
        cb_timeout = getattr(settings, 'GRANITE_CIRCUIT_TIMEOUT', 60.0)
        
        cb_config = CircuitBreakerConfig(
            failure_threshold=cb_threshold,
            success_threshold=2,
            timeout_seconds=cb_timeout,
            half_open_max_calls=2,
        )
        self.circuit_breaker = CircuitBreaker("granite_vlm", cb_config)
        
        # HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None
        
        # Stats
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time_ms": 0,
            "last_success": None,
            "last_failure": None,
        }
        
        self._lock = asyncio.Lock()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _check_file_size(self, file_path: str) -> bool:
        """Check if file is within size limits."""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 ** 2)
            if file_size_mb > self.max_file_size_mb:
                logger.warning(
                    f"File {file_path} ({file_size_mb:.1f}MB) exceeds max size "
                    f"({self.max_file_size_mb}MB)"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking file size: {e}")
            return False
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for caching."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    async def convert_document(
        self,
        file_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert document using Granite VLM via /v1/convert endpoint.
        
        Args:
            file_path: Path to PDF file
            extract_images: Whether to extract image descriptions
            extract_tables: Whether to extract table structures
        
        Returns:
            Dict with:
            - doctags: DocTags formatted output
            - markdown: Markdown formatted output
            - text: Plain text
            - tables: List of extracted tables with metadata
            - images: List of images with VLM descriptions
            - page_count: Number of pages processed
            - metadata: Processing metadata
        
        Raises:
            CircuitOpenError: If circuit breaker is open
            httpx.HTTPError: If API call fails after retries
        """
        # Check file size
        if not self._check_file_size(file_path):
            raise ValueError(f"File exceeds max size of {self.max_file_size_mb}MB")
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are supported for Granite VLM conversion")
        
        # Wrap in circuit breaker
        return await self.circuit_breaker.call(
            self._convert_with_retry,
            file_path,
        )
    
    async def _convert_with_retry(self, file_path: str) -> Dict[str, Any]:
        """Convert document with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Call /v1/convert endpoint
                result = await self._call_convert_api(file_path)
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Update stats
                async with self._lock:
                    self._stats["total_requests"] += 1
                    self._stats["successful_requests"] += 1
                    self._stats["total_processing_time_ms"] += processing_time_ms
                    self._stats["last_success"] = time.time()
                
                logger.info(
                    f"Granite VLM conversion successful: {os.path.basename(file_path)} "
                    f"({processing_time_ms:.0f}ms, {result.get('page_count', 0)} pages)"
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Update stats
                async with self._lock:
                    self._stats["total_requests"] += 1
                    self._stats["failed_requests"] += 1
                    self._stats["last_failure"] = time.time()
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        f"Granite VLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Granite VLM call failed after {self.max_retries} attempts: {e}"
                    )
        
        # All retries failed
        raise last_exception
    
    async def _call_convert_api(self, file_path: str) -> Dict[str, Any]:
        """
        Call /v1/convert endpoint with PDF file.
        
        Request format (multipart/form-data):
        - file: PDF file upload
        
        Response format (JSON):
        {
            "doctags": str,
            "markdown": str,
            "tables": List[Dict],
            "images": List[Dict],
            "page_count": int
        }
        """
        client = await self._get_client()
        
        # Prepare multipart file upload
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "application/pdf")
            }
            
            # Send POST request
            response = await client.post(
                self.full_url,
                files=files,
            )
        
        # Check for errors
        response.raise_for_status()
        
        # Parse JSON response
        result_data = response.json()
        
        # Validate response structure
        if "doctags" not in result_data:
            logger.warning("Response missing 'doctags' field, using empty string")
            result_data["doctags"] = ""
        
        if "markdown" not in result_data:
            logger.warning("Response missing 'markdown' field, generating from doctags")
            result_data["markdown"] = self._doctags_to_markdown(result_data.get("doctags", ""))
        
        # Extract plain text from markdown or doctags
        text = self._extract_text(result_data.get("markdown", ""), result_data.get("doctags", ""))
        
        # Extract structured data from doctags
        doctags = result_data.get("doctags", "")
        tables = result_data.get("tables", []) or self._extract_tables_from_doctags(doctags)
        images = result_data.get("images", []) or self._extract_images_from_doctags(doctags)
        
        # Add computed fields
        return {
            "doctags": doctags,
            "markdown": result_data.get("markdown", ""),
            "text": text,
            "tables": tables,
            "images": images,
            "page_count": result_data.get("page_count", 1),
            "metadata": {
                "parser": "granite_vlm",
                "model": "ibm/granite-docling",
                "api_endpoint": self.full_url,
                "file_hash": self._compute_file_hash(file_path),
                "file_name": os.path.basename(file_path),
            }
        }
    
    def _extract_text(self, markdown: str, doctags: str) -> str:
        """Extract plain text from markdown or doctags."""
        if markdown:
            # Remove markdown formatting
            text = markdown
            text = re.sub(r'[#*`_~\[\]()]', '', text)  # Remove markdown symbols
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            return text.strip()
        elif doctags:
            # Strip outer <doctag> wrapper if present
            text = re.sub(r'^\\s*<doctag>\\s*', '', doctags, flags=re.IGNORECASE)
            text = re.sub(r'\\s*</doctag>\\s*$', '', text, flags=re.IGNORECASE)
            
            # Remove remaining XML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\\s+', ' ', text)
            return text.strip()
        return ""
    
    def _doctags_to_markdown(self, doctags: str) -> str:
        """Convert DocTags to markdown (fallback)."""
        if not doctags:
            return ""
        
        markdown = doctags
        
        # Convert headings
        def replace_heading(m):
            level = int(m.group(1))
            text = m.group(2)
            return f"\n{'#' * level} {text}\n"
        
        markdown = re.sub(
            r'<heading level="(\d+)">(.*?)</heading>',
            replace_heading,
            markdown,
            flags=re.DOTALL
        )
        
        # Convert paragraphs
        markdown = re.sub(r'<para>(.*?)</para>', r'\1\n\n', markdown, flags=re.DOTALL)
        
        # Convert lists
        markdown = re.sub(r'<item>(.*?)</item>', r'- \1\n', markdown)
        
        # Remove other tags (keep table, code, formula)
        markdown = re.sub(r'<(?!table|/table|code|/code|formula|/formula)([^>]+)>', '', markdown)
        
        return markdown.strip()
    
    def _extract_tables_from_doctags(self, doctags: str) -> List[Dict[str, Any]]:
        """Extract tables from DocTags format."""
        if not doctags:
            return []
        
        tables = []
        table_pattern = r'<table>(.*?)</table>'
        
        for idx, match in enumerate(re.finditer(table_pattern, doctags, re.DOTALL)):
            table_content = match.group(1).strip()
            tables.append({
                "index": idx,
                "content": table_content,
                "markdown": table_content,  # Often already in markdown format
                "start_pos": match.start(),
                "end_pos": match.end(),
            })
        
        return tables
    
    def _extract_images_from_doctags(self, doctags: str) -> List[Dict[str, Any]]:
        """Extract images with descriptions from DocTags (supports <image>, <picture>, <figure> tags)."""
        if not doctags:
            return []
        
        images = []
        
        # Granite VLM uses <picture> tag, also support standard <image> and <figure>
        image_patterns = [
            r'<image>(.*?)</image>',
            r'<picture>(.*?)</picture>',
            r'<figure>(.*?)</figure>',
        ]
        
        idx = 0
        for pattern in image_patterns:
            for match in re.finditer(pattern, doctags, re.DOTALL):
                image_content = match.group(1).strip()
                
                # Extract description if present
                desc_match = re.search(r'<desc>(.*?)</desc>', image_content, re.DOTALL)
                if not desc_match:
                    # Try to extract any text content
                    clean_content = re.sub(r'<[^>]+>', ' ', image_content)
                    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                    description = clean_content if clean_content else "Visual content"
                else:
                    description = desc_match.group(1).strip()
                
                images.append({
                    "index": idx,
                    "vlm_description": description,
                    "content": image_content,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                })
                idx += 1
        
        return images
    
    async def health_check(self) -> bool:
        """Check if Granite VLM endpoint is available."""
        try:
            client = await self._get_client()
            
            # Try to reach the base API health endpoint
            try:
                response = await client.get(f"{self.api_url}/health", timeout=10.0)
                response.raise_for_status()
                logger.info(f"Granite VLM API healthy: {self.api_url}")
                return True
            except Exception:
                # Try root endpoint as fallback
                response = await client.get(self.api_url, timeout=10.0)
                if response.status_code < 500:  # Any non-server-error is good
                    logger.info(f"Granite VLM API reachable: {self.api_url}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Granite VLM health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        stats = self._stats.copy()
        
        # Calculate averages
        if stats["successful_requests"] > 0:
            stats["avg_processing_time_ms"] = (
                stats["total_processing_time_ms"] / stats["successful_requests"]
            )
        else:
            stats["avg_processing_time_ms"] = 0
        
        if stats["total_requests"] > 0:
            stats["success_rate"] = (
                stats["successful_requests"] / stats["total_requests"]
            )
        else:
            stats["success_rate"] = 0
        
        # Circuit breaker state
        stats["circuit_breaker_state"] = self.circuit_breaker.state.value
        stats["circuit_breaker_stats"] = {
            "total_calls": self.circuit_breaker.stats.total_calls,
            "successful_calls": self.circuit_breaker.stats.successful_calls,
            "failed_calls": self.circuit_breaker.stats.failed_calls,
            "consecutive_failures": self.circuit_breaker.stats.consecutive_failures,
        }
        
        # API info
        stats["api_url"] = self.api_url
        stats["convert_endpoint"] = self.full_url
        
        return stats


# Singleton instance
_granite_client: Optional[GraniteDoclingClient] = None


def get_granite_client() -> GraniteDoclingClient:
    """Get singleton Granite client instance."""
    global _granite_client
    if _granite_client is None:
        _granite_client = GraniteDoclingClient()
    return _granite_client
