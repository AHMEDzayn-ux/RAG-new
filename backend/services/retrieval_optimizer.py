"""
Advanced Retrieval Optimization Service

Implements:
1. Hybrid Search (Vector + BM25 Keyword Search)
2. Re-Ranking with Cross-Encoder
3. Query Transformation (Rewriting + HyDE)

High ROI features for customer support RAG accuracy.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import re

from logger import get_logger

logger = get_logger(__name__)


class RetrievalOptimizer:
    """
    Advanced retrieval optimization for RAG systems.
    
    Features:
    - Hybrid Search: Combines vector similarity with keyword matching
    - Re-Ranking: Deep scoring of candidates with cross-encoder
    - Query Transformation: Rewrites queries for better retrieval
    """
    
    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        enable_reranking: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the retrieval optimizer.
        
        Args:
            vector_weight: Weight for vector search scores (0-1)
            keyword_weight: Weight for keyword search scores (0-1)
            enable_reranking: Whether to use cross-encoder re-ranking
            rerank_model: HuggingFace model for re-ranking
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.enable_reranking = enable_reranking
        
        # Validate weights
        if abs((vector_weight + keyword_weight) - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {vector_weight} + {keyword_weight}")
        
        # Initialize re-ranker if enabled
        self.reranker = None
        if enable_reranking:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(rerank_model)
                logger.info(f"Initialized re-ranker: {rerank_model}")
            except Exception as e:
                logger.error(f"Failed to load re-ranker: {e}")
                self.enable_reranking = False
        
        # BM25 index (will be built per collection)
        self.bm25_indexes: Dict[str, BM25Okapi] = {}
        self.bm25_documents: Dict[str, List[str]] = {}
        self.bm25_doc_ids: Dict[str, List[int]] = {}
        
        logger.info(f"RetrievalOptimizer initialized (vector={vector_weight}, keyword={keyword_weight})")
    
    def build_bm25_index(
        self,
        collection_name: str,
        documents: List[str],
        doc_ids: Optional[List[int]] = None
    ) -> None:
        """
        Build BM25 keyword search index for a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            doc_ids: Optional document IDs (for tracking)
        """
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        # Store
        self.bm25_indexes[collection_name] = bm25
        self.bm25_documents[collection_name] = documents
        self.bm25_doc_ids[collection_name] = doc_ids or list(range(len(documents)))
        
        logger.info(f"Built BM25 index for '{collection_name}': {len(documents)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        vector_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Search: Combine vector search with BM25 keyword search.
        
        Strategy:
        1. Get vector search results (already provided)
        2. Get BM25 keyword search results
        3. Fuse scores with weighted combination
        4. Return top-k by combined score
        
        Args:
            collection_name: Name of the collection
            query: User query
            vector_results: Results from vector search with 'distance' scores
            top_k: Number of results to return
            
        Returns:
            Hybrid search results sorted by combined score
        """
        if collection_name not in self.bm25_indexes:
            logger.warning(f"No BM25 index for '{collection_name}', returning vector results")
            return vector_results[:top_k]
        
        # Get BM25 results
        bm25_results = self._bm25_search(collection_name, query, top_k=50)
        
        # Normalize and combine scores
        combined = self._fuse_results(vector_results, bm25_results)
        
        # Sort by combined score (higher is better)
        combined.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return combined[:top_k]
    
    def _bm25_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            collection_name: Collection name
            query: Search query
            top_k: Number of results
            
        Returns:
            List of results with BM25 scores
        """
        bm25 = self.bm25_indexes[collection_name]
        documents = self.bm25_documents[collection_name]
        doc_ids = self.bm25_doc_ids[collection_name]
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)
        
        # Create results
        results = []
        for idx, (score, doc_id) in enumerate(zip(scores, doc_ids)):
            if score > 0:  # Only include matches
                results.append({
                    'doc_id': doc_id,
                    'text': documents[idx],
                    'bm25_score': float(score),
                    'index': idx
                })
        
        # Sort by BM25 score
        results.sort(key=lambda x: x['bm25_score'], reverse=True)
        
        return results[:top_k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Min-max normalize scores to 0-1 range.
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores or len(scores) == 1:
            return [1.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fuse vector and BM25 results with weighted combination.
        
        Uses Reciprocal Rank Fusion (RRF) + score normalization.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            
        Returns:
            Combined results with hybrid scores
        """
        # Build document map
        doc_map = {}
        
        # Add vector results (convert distance to similarity)
        vector_distances = [r.get('distance', 0) for r in vector_results]
        if vector_distances:
            # Convert L2 distance to similarity (lower distance = higher similarity)
            max_dist = max(vector_distances) if vector_distances else 1.0
            vector_scores = [1 - (d / (max_dist + 1e-6)) for d in vector_distances]
            norm_vector_scores = self._normalize_scores(vector_scores)
        else:
            norm_vector_scores = []
        
        for idx, (result, norm_score) in enumerate(zip(vector_results, norm_vector_scores)):
            doc_id = result.get('doc_id', idx)
            doc_map[doc_id] = {
                **result,
                'vector_score': norm_score,
                'bm25_score': 0.0,
                'vector_rank': idx + 1
            }
        
        # Add BM25 results
        bm25_scores = [r['bm25_score'] for r in bm25_results]
        norm_bm25_scores = self._normalize_scores(bm25_scores)
        
        for idx, (result, norm_score) in enumerate(zip(bm25_results, norm_bm25_scores)):
            doc_id = result.get('doc_id', result.get('index'))
            
            if doc_id in doc_map:
                # Document found in both - update BM25 score
                doc_map[doc_id]['bm25_score'] = norm_score
                doc_map[doc_id]['bm25_rank'] = idx + 1
            else:
                # Document only in BM25 results
                doc_map[doc_id] = {
                    **result,
                    'vector_score': 0.0,
                    'bm25_score': norm_score,
                    'bm25_rank': idx + 1
                }
        
        # Compute hybrid scores
        for doc_id, doc in doc_map.items():
            vector_score = doc.get('vector_score', 0)
            bm25_score = doc.get('bm25_score', 0)
            
            # Weighted combination
            hybrid_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * bm25_score
            )
            
            # Add RRF bonus (Reciprocal Rank Fusion)
            vector_rank = doc.get('vector_rank', 1000)
            bm25_rank = doc.get('bm25_rank', 1000)
            rrf_score = (1 / (60 + vector_rank)) + (1 / (60 + bm25_rank))
            
            # Final hybrid score
            doc['hybrid_score'] = hybrid_score + (0.1 * rrf_score)  # Small RRF boost
            doc['fusion_method'] = 'weighted_rrf'
        
        return list(doc_map.values())
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using Cross-Encoder for deep scoring.
        
        Strategy:
        1. Take top N candidates from initial retrieval (e.g., 50)
        2. Use Cross-Encoder to score each candidate deeply
        3. Return top-k by cross-encoder score
        
        This is expensive but highly accurate - use after fast retrieval.
        
        Args:
            query: User query
            documents: Candidate documents from initial retrieval
            top_k: Number of final results
            
        Returns:
            Re-ranked documents
        """
        if not self.enable_reranking or self.reranker is None:
            logger.warning("Re-ranking not available, returning original order")
            return documents[:top_k]
        
        if not documents:
            return []
        
        try:
            # Prepare query-document pairs
            pairs = [(query, doc.get('text', '')) for doc in documents]
            
            # Get cross-encoder scores
            scores = self.reranker.predict(pairs)
            
            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
                # Keep original scores for debugging
                doc['original_hybrid_score'] = doc.get('hybrid_score', doc.get('distance', 0))
            
            # Sort by rerank score
            documents.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Re-ranked {len(documents)} documents, returning top {top_k}")
            
            return documents[:top_k]
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return documents[:top_k]
    
    def rewrite_query(
        self,
        query: str,
        llm_service: Any,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Query Rewriting: Transform user's messy query into precise search query.
        
        Examples:
        - "help cant login" → "steps to troubleshoot login failure"
        - "Error 505" → "error code 505 meaning and troubleshooting"
        - "refund???" → "refund policy and process"
        
        Args:
            query: User's raw query
            llm_service: LLM service for rewriting
            conversation_history: Optional conversation context
            
        Returns:
            Rewritten search query
        """
        system_prompt = """You are a query optimization expert. Rewrite user queries to be clear, precise search queries.

Rules:
1. Fix typos and grammar
2. Add relevant search keywords
3. Make specific (e.g., "login error" → "steps to troubleshoot login authentication failure")
4. Keep it concise (under 20 words)
5. Preserve important specifics (error codes, product names, SKUs, etc.)
6. Remove filler words ("help", "please", "???")

Examples:
User: "help cant login"
Rewritten: "troubleshoot login authentication failure steps"

User: "Error 505 what mean"
Rewritten: "error code 505 meaning and resolution"

User: "refund???"
Rewritten: "refund policy process and requirements"

Now rewrite this query:"""
        
        try:
            # Use LLM to rewrite
            rewritten = llm_service.generate_response(
                query=f"Original query: {query}\n\nRewritten query:",
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            
            # Clean up response
            rewritten = rewritten.strip().strip('"').strip("'")
            
            # Safety check: if rewrite is too different or empty, use original
            if not rewritten or len(rewritten) > len(query) * 3:
                logger.warning(f"Query rewrite suspicious, using original. Original: '{query}', Rewritten: '{rewritten}'")
                return query
            
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
    
    def hyde_query(
        self,
        query: str,
        llm_service: Any,
        system_role: str = "helpful assistant"
    ) -> str:
        """
        HyDE (Hypothetical Document Embeddings): Generate ideal answer, search for similar docs.
        
        Strategy:
        1. Use LLM to generate a hypothetical "perfect answer" to the query
        2. Embed this hypothetical answer
        3. Search for real documents that look like this perfect answer
        4. Real docs that match the hypothetical answer are likely relevant
        
        This helps bridge the query-document gap (queries are questions, docs are answers).
        
        Args:
            query: User's question
            llm_service: LLM service for generation
            system_role: System role for LLM
            
        Returns:
            Hypothetical answer (to be embedded and searched)
        """
        system_prompt = f"""You are a {system_role}. Generate a detailed, comprehensive answer to the user's question as if you had perfect knowledge.

IMPORTANT:
- Write as if answering from documentation
- Include specific details, steps, explanations
- Use technical terms and keywords that would appear in real docs
- Keep it factual and structured (2-3 paragraphs, 100-200 words)
- This is NOT for the user - it's a search template

The goal is to create an "ideal document" that real documentation should resemble."""
        
        try:
            # Generate hypothetical answer
            hypothetical = llm_service.generate_response(
                query=query,
                system_prompt=system_prompt
            )
            
            logger.info(f"Generated HyDE answer ({len(hypothetical)} chars) for query: {query[:50]}...")
            logger.debug(f"HyDE result: {hypothetical[:200]}...")
            
            return hypothetical
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            # Fallback: use original query
            return query
    
    def optimize_retrieval(
        self,
        collection_name: str,
        query: str,
        vector_results: List[Dict[str, Any]],
        llm_service: Any,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        use_query_rewriting: bool = True,
        use_hyde: bool = False,
        top_k_initial: int = 50,
        top_k_final: int = 5,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Complete retrieval optimization pipeline.
        
        Full pipeline:
        1. Query Transformation (rewriting or HyDE)
        2. Hybrid Search (vector + BM25)
        3. Re-Ranking (cross-encoder)
        
        Args:
            collection_name: Collection name
            query: User query
            vector_results: Initial vector search results
            llm_service: LLM service
            use_hybrid: Enable hybrid search
            use_reranking: Enable re-ranking
            use_query_rewriting: Enable query rewriting
            use_hyde: Enable HyDE (mutually exclusive with rewriting)
            top_k_initial: Candidates for re-ranking
            top_k_final: Final number of results
            conversation_history: Optional conversation context
            
        Returns:
            Tuple of (optimized results, optimization metadata)
        """
        metadata = {
            'original_query': query,
            'transformations': [],
            'num_initial_results': len(vector_results)
        }
        
        # Step 1: Query Transformation
        search_query = query
        
        if use_hyde:
            # HyDE: Generate hypothetical answer
            search_query = self.hyde_query(query, llm_service)
            metadata['transformations'].append('hyde')
            metadata['hyde_query'] = search_query[:200] + '...'
            
        elif use_query_rewriting:
            # Query Rewriting: Clean up query
            search_query = self.rewrite_query(query, llm_service, conversation_history)
            metadata['transformations'].append('rewriting')
            metadata['rewritten_query'] = search_query
        
        # Step 2: Hybrid Search
        if use_hybrid:
            results = self.hybrid_search(
                collection_name,
                search_query,
                vector_results,
                top_k=top_k_initial
            )
            metadata['transformations'].append('hybrid_search')
            metadata['num_after_hybrid'] = len(results)
        else:
            results = vector_results[:top_k_initial]
        
        # Step 3: Re-Ranking
        if use_reranking and len(results) > top_k_final:
            # Use ORIGINAL query for re-ranking (not transformed one)
            results = self.rerank(query, results, top_k=top_k_final)
            metadata['transformations'].append('reranking')
            metadata['num_after_rerank'] = len(results)
        else:
            results = results[:top_k_final]
        
        metadata['final_count'] = len(results)
        
        logger.info(f"Retrieval optimization complete: {metadata['transformations']}")
        
        return results, metadata
