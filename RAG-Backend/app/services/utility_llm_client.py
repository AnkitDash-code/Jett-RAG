"""
Utility LLM client for lightweight query tasks.
Uses the same API as the main LLM but with different model/settings.

Tasks handled by utility LLM:
- Query expansion (paraphrasing)
- Query classification
- Relevance grading
- Conversation summarization
"""
import json
import logging
import re
from typing import Optional, Dict, Any, List

import httpx

from app.config import settings
from app.services.token_counter import get_token_counter
from app.services.metrics_service import get_metrics_service

logger = logging.getLogger(__name__)


class UtilityLLMClient:
    """
    Lightweight LLM client for utility tasks.
    Uses same API endpoint but with utility model settings.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.base_url = settings.UTILITY_LLM_URL
        self.model = settings.UTILITY_LLM_MODEL
        self.max_tokens = settings.UTILITY_LLM_MAX_TOKENS
        self.temperature = settings.UTILITY_LLM_TEMPERATURE
        self.enabled = settings.UTILITY_LLM_ENABLED
        self.timeout = 30.0  # Shorter timeout for utility tasks
        
        self._client: Optional[httpx.AsyncClient] = None
        self._token_counter = get_token_counter()
        self._metrics = get_metrics_service()
        self._initialized = True
        
        logger.info(f"UtilityLLMClient initialized: enabled={self.enabled}, model={self.model}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def is_available(self) -> bool:
        """Check if utility LLM is enabled."""
        return self.enabled
    
    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Make a call to the utility LLM.
        
        Returns:
            Response text from the LLM
        """
        if not self.enabled:
            logger.debug("Utility LLM disabled, returning empty")
            return ""
        
        client = await self._get_client()
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Build context from system prompt
            # Send system prompt as instruction, keep context empty to avoid formatting issues
            response = await client.post(
                "/generate",
                json={
                    "query": prompt,
                    "context": "",  # Empty context for utility tasks
                    "system_instruction": system_prompt or "You are a helpful assistant for query processing tasks.",
                }
            )
            response.raise_for_status()
            
            data = response.json()
            result = data.get("response", "")
            
            # Record metrics
            self._metrics.record_llm_request(
                llm_type="utility",
                status="success"
            )
            
            return result.strip()
            
        except httpx.HTTPError as e:
            logger.error(f"Utility LLM error: {e}")
            self._metrics.record_llm_request(
                llm_type="utility",
                status="error"
            )
            return ""
        except Exception as e:
            logger.error(f"Utility LLM unexpected error: {e}")
            return ""
    
    # ========================================================================
    # Query Expansion
    # ========================================================================
    
    async def expand_query(
        self,
        query: str,
        num_variants: int = 3,
    ) -> List[str]:
        """
        Generate query variants for multi-query RAG.
        
        Args:
            query: Original user query
            num_variants: Number of variants to generate
            
        Returns:
            List of query variants (including original)
        """
        if not self.enabled:
            return [query]
        
        system_prompt = "Generate alternative search queries. Return only the queries, one per line."
        
        prompt = f"""Rephrase this query in {num_variants} different ways:

{query}

Alternatives:"""
        
        try:
            response = await self._call_llm(prompt, system_prompt, temperature=0.7)
            
            if not response:
                return [query]
            
            # Parse response - split by newlines, clean up
            variants = [query]  # Always include original
            for line in response.strip().split("\n"):
                line = line.strip()
                # Remove numbering, bullets, etc.
                line = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)
                if line and line != query and len(line) > 5:
                    variants.append(line)
            
            # Limit to requested count + original
            return variants[:num_variants + 1]
            
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return [query]
    
    async def extract_keywords(self, query: str) -> List[str]:
        """
        Extract key terms from query for BM25 boost.
        
        Returns:
            List of important keywords
        """
        if not self.enabled:
            # Fallback: simple word extraction
            words = query.lower().split()
            stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who'}
            return [w for w in words if w not in stopwords and len(w) > 2]
        
        system_prompt = "You extract important keywords from queries. Return only the keywords separated by commas."
        
        prompt = f"Extract the 3-5 most important keywords from this query: {query}"
        
        try:
            response = await self._call_llm(prompt, system_prompt, temperature=0.1)
            
            if not response:
                return []
            
            # Parse comma-separated keywords
            keywords = [k.strip().lower() for k in response.split(",")]
            return [k for k in keywords if k and len(k) > 2]
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []
    
    # ========================================================================
    # Query Classification
    # ========================================================================
    
    async def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query type and complexity.
        
        Returns:
            Dict with 'type', 'complexity', 'needs_graph', 'is_followup'
        """
        if not self.enabled:
            return self._heuristic_classification(query)
        
        system_prompt = """You are a query classifier. Analyze queries and return a JSON object with:
- type: one of [FACT, ENTITY, COMPARISON, SUMMARY, TROUBLESHOOTING, CONVERSATIONAL]
- complexity: one of [simple, moderate, complex]
- needs_graph: boolean - true if query needs entity relationships
- is_followup: boolean - true if query references previous context

Return ONLY valid JSON, no other text."""
        
        prompt = f"Classify this query:\n\n{query}"
        
        try:
            response = await self._call_llm(prompt, system_prompt, temperature=0.1)
            
            if not response:
                return self._heuristic_classification(query)
            
            # Try to parse JSON
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            result = json.loads(response)
            
            # Validate and normalize
            valid_types = {"FACT", "ENTITY", "COMPARISON", "SUMMARY", "TROUBLESHOOTING", "CONVERSATIONAL"}
            if result.get("type") not in valid_types:
                result["type"] = "FACT"
            
            return {
                "type": result.get("type", "FACT"),
                "complexity": result.get("complexity", "simple"),
                "needs_graph": result.get("needs_graph", False),
                "is_followup": result.get("is_followup", False),
            }
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse classification response: {response}")
            return self._heuristic_classification(query)
        except Exception as e:
            logger.error(f"Query classification error: {e}")
            return self._heuristic_classification(query)
    
    def _heuristic_classification(self, query: str) -> Dict[str, Any]:
        """Fallback heuristic-based classification."""
        query_lower = query.lower()
        
        # Determine type
        query_type = "FACT"
        if any(w in query_lower for w in ["compare", "difference", "versus", "vs"]):
            query_type = "COMPARISON"
        elif any(w in query_lower for w in ["summarize", "summary", "overview", "brief"]):
            query_type = "SUMMARY"
        elif any(w in query_lower for w in ["error", "issue", "problem", "fix", "debug", "troubleshoot"]):
            query_type = "TROUBLESHOOTING"
        elif any(w in query_lower for w in ["who is", "what is", "define", "explain"]):
            query_type = "ENTITY"
        elif any(w in query_lower for w in ["what about", "and also", "tell me more", "how about"]):
            query_type = "CONVERSATIONAL"
        
        # Determine complexity
        word_count = len(query.split())
        complexity = "simple"
        if word_count > 20 or " and " in query_lower:
            complexity = "complex"
        elif word_count > 10:
            complexity = "moderate"
        
        # Check for graph needs
        needs_graph = query_type in {"ENTITY", "COMPARISON"}
        
        # Check for follow-up
        is_followup = any(w in query_lower for w in ["it", "this", "that", "they", "those", "the same"])
        
        return {
            "type": query_type,
            "complexity": complexity,
            "needs_graph": needs_graph,
            "is_followup": is_followup,
        }
    
    # ========================================================================
    # Relevance Grading
    # ========================================================================
    
    async def grade_relevance(
        self,
        query: str,
        chunks: List[str],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Grade relevance of retrieved chunks to the query.
        
        Args:
            query: User query
            chunks: List of retrieved chunk texts
            threshold: Minimum relevance threshold
            
        Returns:
            Dict with 'is_relevant', 'confidence', 'relevant_chunks', 'suggestion'
        """
        if not self.enabled or not chunks:
            return {
                "is_relevant": bool(chunks),
                "confidence": 0.5 if chunks else 0.0,
                "relevant_chunks": list(range(len(chunks))),
                "suggestion": None,
            }
        
        # Truncate chunks for prompt
        chunk_previews = []
        for i, chunk in enumerate(chunks[:5]):  # Max 5 chunks
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            chunk_previews.append(f"[{i+1}] {preview}")
        
        chunks_text = "\n\n".join(chunk_previews)
        
        system_prompt = """You are a LENIENT relevance grader. Given a query and retrieved chunks, 
evaluate if the chunks could be helpful for answering the query.

IMPORTANT: Be GENEROUS about relevance. If a chunk:
- Contains related topics, entities, or concepts
- Provides background or context that could be useful
- Has ANY connection to the query, even indirect
- Mentions terms that appear in or relate to the query

Then mark it as RELEVANT. Only mark as NOT relevant if the chunk is completely unrelated.

Return a JSON object with:
- is_relevant: boolean - true if ANY chunk could help answer the query (default to true)
- confidence: float 0-1 - how confident you are
- relevant_indices: list of 1-indexed chunk numbers that are relevant (include all unless clearly irrelevant)
- suggestion: string or null - only if NO chunks are relevant, suggest how to rephrase

Return ONLY valid JSON."""
        
        prompt = f"""Query: {query}

Retrieved chunks:
{chunks_text}

Evaluate relevance:"""
        
        try:
            response = await self._call_llm(prompt, system_prompt, temperature=0.1)
            
            if not response:
                return {
                    "is_relevant": True,
                    "confidence": 0.5,
                    "relevant_chunks": list(range(len(chunks))),
                    "suggestion": None,
                }
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            result = json.loads(response)
            
            # Convert 1-indexed to 0-indexed
            relevant_indices = result.get("relevant_indices", [])
            relevant_chunks = [i - 1 for i in relevant_indices if 0 < i <= len(chunks)]
            
            return {
                "is_relevant": result.get("is_relevant", True),
                "confidence": float(result.get("confidence", 0.5)),
                "relevant_chunks": relevant_chunks if relevant_chunks else list(range(len(chunks))),
                "suggestion": result.get("suggestion"),
            }
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse relevance response")
            return {
                "is_relevant": True,
                "confidence": 0.5,
                "relevant_chunks": list(range(len(chunks))),
                "suggestion": None,
            }
        except Exception as e:
            logger.error(f"Relevance grading error: {e}")
            return {
                "is_relevant": True,
                "confidence": 0.5,
                "relevant_chunks": list(range(len(chunks))),
                "suggestion": None,
            }
    
    # ========================================================================
    # Conversation Context
    # ========================================================================
    
    async def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 200,
    ) -> str:
        """
        Summarize conversation history to fit token budget.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            max_tokens: Target summary length
            
        Returns:
            Condensed summary of the conversation
        """
        if not self.enabled or not messages:
            return ""
        
        # Build conversation text
        conversation = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:500]  # Truncate long messages
            conversation.append(f"{role.title()}: {content}")
        
        conversation_text = "\n".join(conversation)
        
        system_prompt = "You create detailed conversation summaries for context preservation. Include all important facts, entities, and topics discussed."
        
        prompt = f"""Create a comprehensive summary of this conversation that preserves:

- All topics discussed
- Key facts, figures, and specific information mentioned
- Names, dates, and entities referenced
- Questions asked and answers provided
- Any decisions, conclusions, or action items

Be thorough - this summary maintains conversation context.

Conversation:
{conversation_text}

Detailed Summary:"""
        
        try:
            response = await self._call_llm(
                prompt, 
                system_prompt, 
                temperature=0.3,
                max_tokens=max_tokens
            )
            
            return response if response else ""
            
        except Exception as e:
            logger.error(f"Conversation summarization error: {e}")
            return ""
    
    async def summarize_text(
        self,
        text: str,
        max_length: int = 256,
    ) -> str:
        """
        Generate a summary of a text passage for hierarchical chunking.
        
        Args:
            text: The text to summarize
            max_length: Approximate max tokens for summary
            
        Returns:
            Condensed summary of the text
        """
        if not self.enabled or not text:
            # Fallback: extract first few sentences
            sentences = text.split(".")[:3]
            return ". ".join(s.strip() for s in sentences if s.strip()) + "."
        
        system_prompt = (
            "You are a concise summarization assistant. "
            "Create a brief, dense summary of the text. "
            "Do not repeat information. Do not hallucinate. "
            "Maximum length: 3 paragraphs."
        )
        
        prompt = f"""Summarize this text concisely:

Text:
{text[:4000]}

Summary:"""
        
        try:
            response = await self._call_llm(
                prompt,
                system_prompt,
                temperature=0.1,
                max_tokens=max_length
            )
            
            return response.strip() if response else text[:max_length]
            
        except Exception as e:
            logger.error(f"Text summarization error: {e}")
            # Fallback: first sentences
            sentences = text.split(".")[:3]
            return ". ".join(s.strip() for s in sentences if s.strip()) + "."
    
    async def detect_followup(
        self,
        current_query: str,
        previous_messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Detect if current query is a follow-up and resolve references.
        """
        if not self.enabled or not previous_messages:
            return {
                "is_followup": False,
                "resolved_query": current_query,
                "referenced_entities": [],
            }
        
        # Get last few messages for context
        recent = previous_messages[-4:]  # Last 2 Q&A pairs
        context_parts = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:300]
            context_parts.append(f"{role.title()}: {content}")
        
        context_text = "\n".join(context_parts)
        
        system_prompt = """You analyze if a query is a follow-up to previous conversation.
If it is, resolve any pronouns or references to create a standalone query.

Return a JSON object with:
- is_followup: boolean
- resolved_query: string - the query with references resolved
- referenced_entities: list of entities referenced from previous context

Return ONLY valid JSON."""
        
        prompt = f"""Previous conversation:
{context_text}

Current query: {current_query}

Analyze:"""
        
        try:
            response = await self._call_llm(prompt, system_prompt, temperature=0.1)
            
            if not response:
                return {
                    "is_followup": False,
                    "resolved_query": current_query,
                    "referenced_entities": [],
                }
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            result = json.loads(response)
            
            return {
                "is_followup": result.get("is_followup", False),
                "resolved_query": result.get("resolved_query", current_query),
                "referenced_entities": result.get("referenced_entities", []),
            }
            
        except json.JSONDecodeError:
            return {
                "is_followup": False,
                "resolved_query": current_query,
                "referenced_entities": [],
            }
        except Exception as e:
            logger.error(f"Follow-up detection error: {e}")
            return {
                "is_followup": False,
                "resolved_query": current_query,
                "referenced_entities": [],
            }
    
    # ========================================================================
    # Entity Extraction (Phase 4 - GraphRAG)
    # ========================================================================
    
    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract named entities from text for knowledge graph.
        """
        if not self.enabled:
            return {"entities": [], "extraction_method": "disabled"}
        
        # Truncate text if too long
        max_text_len = 2000
        truncated = text[:max_text_len] if len(text) > max_text_len else text
        
        types_hint = ""
        if entity_types:
            types_hint = f"\nFocus on these entity types: {', '.join(entity_types)}"
        
        system_prompt = f"""You are a comprehensive entity extraction system.
Extract ALL entities thoroughly.
{types_hint}

Entity types:
- PERSON
- ORGANIZATION
- LOCATION
- DATE
- EVENT
- PRODUCT
- TECHNOLOGY
- CONCEPT

Return valid JSON:
{{
  "entities": [
    {{"name": "entity name", "type": "ENTITY_TYPE", "confidence": 0.95}}
  ]
}}"""
        
        prompt = f"""Extract all named entities from this text:

{truncated}

Entities:"""
        
        try:
            response = await self._call_llm(prompt, system_prompt, temperature=0.1)
            
            if not response:
                return {"entities": [], "extraction_method": "llm_empty"}
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            result = json.loads(response)
            entities = result.get("entities", [])
            
            # Validate and normalize
            valid_types = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", 
                          "EVENT", "PRODUCT", "TECHNOLOGY", "CONCEPT", "OTHER"}
            validated = []
            for ent in entities:
                if isinstance(ent, dict) and "name" in ent:
                    ent_type = ent.get("type", "OTHER").upper()
                    if ent_type not in valid_types:
                        ent_type = "OTHER"
                    validated.append({
                        "name": str(ent.get("name", "")),
                        "type": ent_type,
                        "confidence": float(ent.get("confidence", 0.8)),
                    })
            
            return {"entities": validated, "extraction_method": "llm"}
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity extraction response: {e}")
            return {"entities": [], "extraction_method": "llm_parse_error"}
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {"entities": [], "extraction_method": "llm_error"}
    
    async def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extract relationships between entities in text.
        """
        if not self.enabled or not entities:
            return {"relationships": [], "extraction_method": "disabled"}
        
        # Build entity context
        entity_names = [e.get("name", "") for e in entities[:20]]  # Limit to 20
        entity_list = ", ".join(entity_names)
        
        # Truncate text
        max_text_len = 1500
        truncated = text[:max_text_len] if len(text) > max_text_len else text
        
        system_prompt = """You extract relationships between entities.

Relationship types:
- WORKS_FOR
- LOCATED_IN
- RELATED_TO
- PART_OF
- CREATED_BY
- OWNS
- MANAGES
- MENTIONED_WITH

Return ONLY valid JSON:
{
  "relationships": [
    {"source": "Entity A", "target": "Entity B", "type": "RELATIONSHIP_TYPE", "confidence": 0.9}
  ]
}"""
        
        prompt = f"""Given these entities: {entity_list}

Extract relationships from this text:

{truncated}

Relationships:"""
        
        try:
            response = await self._call_llm(prompt, system_prompt, temperature=0.1)
            
            if not response:
                return {"relationships": [], "extraction_method": "llm_empty"}
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            result = json.loads(response)
            relationships = result.get("relationships", [])
            
            # Validate
            valid_types = {"WORKS_FOR", "LOCATED_IN", "RELATED_TO", "PART_OF", 
                          "CREATED_BY", "OWNS", "MANAGES", "MENTIONED_WITH", "OTHER"}
            validated = []
            for rel in relationships:
                if isinstance(rel, dict) and "source" in rel and "target" in rel:
                    rel_type = rel.get("type", "RELATED_TO").upper()
                    if rel_type not in valid_types:
                        rel_type = "RELATED_TO"
                    validated.append({
                        "source": str(rel.get("source", "")),
                        "target": str(rel.get("target", "")),
                        "type": rel_type,
                        "confidence": float(rel.get("confidence", 0.7)),
                    })
            
            return {"relationships": validated, "extraction_method": "llm"}
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse relationship extraction response: {e}")
            return {"relationships": [], "extraction_method": "llm_parse_error"}
        except Exception as e:
            logger.error(f"Relationship extraction error: {e}")
            return {"relationships": [], "extraction_method": "llm_error"}
    
    async def extract_entities_and_relationships(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Combined extraction of entities and relationships in one call.
        More efficient than separate calls for large texts.
        
        Returns:
            Dict with 'entities' and 'relationships' lists
        """
        if not self.enabled:
            return {
                "entities": [],
                "relationships": [],
                "extraction_method": "disabled"
            }
        
        # Truncate text
        max_text_len = 2000
        truncated = text[:max_text_len] if len(text) > max_text_len else text

        system_prompt = """You are a knowledge graph extractor. Extract entities and relationships from the text.
Return ONLY valid JSON. No markdown formatting. No conversational text.

Refuse to output anything other than this JSON structure:
{
  "entities": [{"name": "Name", "type": "TYPE", "confidence": 0.9}],
  "relationships": [{"source": "Name A", "target": "Name B", "type": "TYPE", "confidence": 0.9}]
}

Entity types: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, PRODUCT, TECHNOLOGY, CONCEPT
Relationship types: WORKS_FOR, LOCATED_IN, RELATED_TO, PART_OF, CREATED_BY, OWNS, MANAGES, MENTIONED_WITH"""
        
        prompt = f"""Extract entities and relationships from:

{truncated}

JSON Output:"""
        
        try:
            response = await self._call_llm(prompt, system_prompt, temperature=0.1)
            
            if not response:
                return {
                    "entities": [],
                    "relationships": [],
                    "extraction_method": "llm_empty"
                }
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            result = json.loads(response)
            
            # Validate entities
            valid_entity_types = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", 
                                  "EVENT", "PRODUCT", "TECHNOLOGY", "CONCEPT", "OTHER"}
            validated_entities = []
            for ent in result.get("entities", []):
                if isinstance(ent, dict) and "name" in ent:
                    ent_type = ent.get("type", "OTHER").upper()
                    if ent_type not in valid_entity_types:
                        ent_type = "OTHER"
                    validated_entities.append({
                        "name": str(ent.get("name", "")),
                        "type": ent_type,
                        "confidence": float(ent.get("confidence", 0.8)),
                    })
            
            # Validate relationships
            valid_rel_types = {"WORKS_FOR", "LOCATED_IN", "RELATED_TO", "PART_OF", 
                              "CREATED_BY", "OWNS", "MANAGES", "MENTIONED_WITH", "OTHER"}
            validated_relationships = []
            for rel in result.get("relationships", []):
                if isinstance(rel, dict) and "source" in rel and "target" in rel:
                    rel_type = rel.get("type", "RELATED_TO").upper()
                    if rel_type not in valid_rel_types:
                        rel_type = "RELATED_TO"
                    validated_relationships.append({
                        "source": str(rel.get("source", "")),
                        "target": str(rel.get("target", "")),
                        "type": rel_type,
                        "confidence": float(rel.get("confidence", 0.7)),
                    })
            
            return {
                "entities": validated_entities,
                "relationships": validated_relationships,
                "extraction_method": "llm"
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse combined extraction response: {e}")
            return {
                "entities": [],
                "relationships": [],
                "extraction_method": "llm_parse_error"
            }
        except Exception as e:
            logger.error(f"Combined extraction error: {e}")
            return {
                "entities": [],
                "relationships": [],
                "extraction_method": "llm_error"
            }
    
    async def generate_community_summary(
        self,
        entity_names: List[str],
        entity_types: List[str],
        relationship_descriptions: List[str],
        max_tokens: int = 150,
    ) -> Dict[str, Any]:
        """
        Generate a summary for a community of entities.
        
        Args:
            entity_names: Names of entities in the community
            entity_types: Types of the entities
            relationship_descriptions: Descriptions of relationships
            max_tokens: Maximum tokens for summary
        
        Returns:
            Dict with 'summary', 'key_topics'
        """
        if not self.enabled:
            return {
                "summary": f"Community of {len(entity_names)} entities",
                "key_topics": entity_names[:3],
            }
        
        # Build context
        entities_desc = ", ".join(
            f"{name} ({etype})" 
            for name, etype in zip(entity_names[:10], entity_types[:10])
        )
        
        relationships_desc = "; ".join(relationship_descriptions[:10])
        
        system_prompt = "You generate concise summaries of entity clusters from a knowledge graph."
        
        prompt = f"""Generate a 2-3 sentence summary for this cluster of related entities:

Entities: {entities_desc}

Relationships: {relationships_desc}

Summary:"""
        
        try:
            response = await self._call_llm(
                prompt, 
                system_prompt, 
                temperature=0.3,
                max_tokens=max_tokens
            )
            
            if not response:
                return {
                    "summary": f"Community containing {', '.join(entity_names[:3])} and {len(entity_names)-3} more entities.",
                    "key_topics": entity_names[:5],
                }
            
            # Extract key topics (first few entity names)
            key_topics = entity_names[:5]
            
            return {
                "summary": response.strip(),
                "key_topics": key_topics,
            }
            
        except Exception as e:
            logger.error(f"Community summary generation error: {e}")
            return {
                "summary": f"Community of {len(entity_names)} related entities.",
                "key_topics": entity_names[:5],
            }


# Singleton accessor
_utility_llm: Optional[UtilityLLMClient] = None


def get_utility_llm() -> UtilityLLMClient:
    """Get the utility LLM client singleton."""
    global _utility_llm
    if _utility_llm is None:
        _utility_llm = UtilityLLMClient()
    return _utility_llm
