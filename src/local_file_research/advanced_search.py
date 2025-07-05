# filename: src/local_file_research/advanced_search.py
"""
Advanced search capabilities for Local File Deep Research.
"""

import os
import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import Counter
import string
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)

# --- Download NLTK resources ---
def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Download resources on module load
try:
    download_nltk_resources()
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")

# --- Query Expansion ---
class QueryExpander:
    """Expand search queries with synonyms and related terms."""

    def __init__(self, max_synonyms: int = 3, max_related: int = 2, include_hypernyms: bool = True, use_llm: bool = True):
        """
        Initialize the query expander.

        Args:
            max_synonyms: Maximum number of synonyms per term
            max_related: Maximum number of related terms per term
            include_hypernyms: Whether to include hypernyms (more general terms)
            use_llm: Whether to use LLM for query expansion
        """
        self.max_synonyms = max_synonyms
        self.max_related = max_related
        self.include_hypernyms = include_hypernyms
        self.use_llm = use_llm
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
             logger.warning("NLTK stopwords not found. Query expansion might be less effective.")
             self.stop_words = set()

    def expand_query(self, query: str) -> str:
        """
        Expand a query with synonyms and related terms.

        Args:
            query: Original query

        Returns:
            Expanded query
        """
        # Try LLM-based expansion first if enabled
        if self.use_llm:
            try:
                llm_expanded_query = self._expand_query_with_llm(query)
                if llm_expanded_query and llm_expanded_query != query:
                    logger.info(f"LLM expanded query: '{query}' -> '{llm_expanded_query}'")
                    return llm_expanded_query
            except Exception as e:
                logger.warning(f"LLM query expansion failed: {e}. Falling back to lexical expansion.")

        # Tokenize and filter query
        tokens = word_tokenize(query.lower())
        filtered_tokens = [token for token in tokens if token not in self.stop_words
                          and token not in string.punctuation
                          and len(token) > 2]

        # Lemmatize tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Get synonyms and related terms
        expanded_terms = set(lemmatized_tokens)

        for token in lemmatized_tokens:
            # Get WordNet synsets
            synsets = wordnet.synsets(token)

            # Add synonyms
            synonyms = set()
            for synset in synsets[:self.max_related]:
                for lemma in synset.lemmas()[:self.max_synonyms]:
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != token and synonym not in self.stop_words:
                        synonyms.add(synonym)

            # Add hypernyms if enabled
            if self.include_hypernyms:
                for synset in synsets[:self.max_related]:
                    hypernyms = synset.hypernyms()
                    for hypernym in hypernyms:
                        for lemma in hypernym.lemmas()[:self.max_synonyms]:
                            hypernym_name = lemma.name().replace('_', ' ')
                            if hypernym_name != token and hypernym_name not in self.stop_words:
                                synonyms.add(hypernym_name)

            # Add to expanded terms
            expanded_terms.update(synonyms)

        # Combine original query with expanded terms
        expanded_query = query + " " + " ".join(expanded_terms - set(lemmatized_tokens))

        logger.info(f"Lexical expanded query: '{query}' -> '{expanded_query}'")

        return expanded_query

    def _expand_query_with_llm(self, query: str) -> str:
        """
        Expand a query using an LLM.

        Args:
            query: Original query

        Returns:
            Expanded query or original query if expansion fails
        """
        # Try to use DSPy agent for query expansion
        from .dspy_config import DSPyAgentRegistry, DSPY_CONFIGURED
        if not DSPY_CONFIGURED:
            logger.warning("DSPy not configured. Cannot use LLM for query expansion.")
            return query # Return original query if DSPy isn't ready
        
        expanded_query_result = query # Default to original

        # Try QueryExpansionAgent first
        try:
            query_expansion_agent = DSPyAgentRegistry.get_agent("query_expansion")
            if query_expansion_agent:
                logger.info("Using DSPy QueryExpansionAgent")
                agent_inputs = {"query": query}
                # Use helper to ensure required fields are present
                prepared_data = DSPyAgentRegistry._ensure_required_fields("query_expansion", query_expansion_agent, agent_inputs)
                result = query_expansion_agent(**prepared_data)
                expanded_query_result = result.get("expanded_query", "").strip()
                if expanded_query_result and expanded_query_result != query:
                    logger.info("Expansion successful with QueryExpansionAgent.")
                    return expanded_query_result # Use the expanded query
                else:
                    logger.warning("QueryExpansionAgent did not return a different query.")
            # +++ ADD ELSE +++
            else:
                 logger.warning("QueryExpansionAgent not found in registry.")

        except Exception as e:
            logger.error(f"Error expanding query with QueryExpansionAgent: {e}", exc_info=True)

        # Fallback: Use TextGeneratorAgent if expansion agent failed or didn't work
        # +++ MODIFY: Check if expansion actually happened before fallback +++
        if expanded_query_result == query: # Only fallback if previous step failed
            try:
                text_generator = DSPyAgentRegistry.get_agent("text_generator")
                if text_generator:
                    logger.info("Using DSPy TextGeneratorAgent as fallback for query expansion")
                    prompt = f"""Expand the following search query to improve search results.
                    Include relevant synonyms, related concepts, and alternative phrasings.

                    Original query: {query}

                    Expanded query:"""

                    agent_inputs = {"prompt": prompt, "query": query}
                    # Use helper to ensure required fields are present
                    prepared_data = DSPyAgentRegistry._ensure_required_fields("text_generator", text_generator, agent_inputs)
                    result = text_generator(**prepared_data)
                    expanded_query_result = result.get("text", "").strip()
                    # Clean up potential LLM artifacts
                    if expanded_query_result.lower().startswith("expanded query:"):
                        expanded_query_result = expanded_query_result[len("expanded query:"):].strip()
                    if expanded_query_result and expanded_query_result != query:
                         logger.info("Expansion successful with TextGeneratorAgent fallback.")
                         return expanded_query_result # Use the expanded query
                    else:
                         logger.warning("TextGeneratorAgent did not return a different query.")
                # +++ ADD ELSE +++
                else:
                     logger.warning("TextGeneratorAgent not found for fallback.")

            except Exception as e:
                logger.error(f"Error expanding query with text generator: {e}", exc_info=True)


        # +++ MODIFY: Check final result before warning +++
        if expanded_query_result == query:
             logger.warning("All available LLM query expansion methods failed or yielded no change. Returning original query.")

        return expanded_query_result # Return original or whatever was last successful
    
    
    def get_expansion_details(self, query: str) -> Dict[str, List[str]]:
        """
        Get detailed expansion information for a query.

        Args:
            query: Original query

        Returns:
            Dictionary mapping original terms to expanded terms
        """
        # Tokenize and filter query
        tokens = word_tokenize(query.lower())
        filtered_tokens = [token for token in tokens if token not in self.stop_words
                          and token not in string.punctuation
                          and len(token) > 2]

        # Lemmatize tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Get synonyms and related terms
        expansion_details = {}

        for token in lemmatized_tokens:
            # Get WordNet synsets
            synsets = wordnet.synsets(token)

            # Get synonyms
            synonyms = set()
            for synset in synsets[:self.max_related]:
                for lemma in synset.lemmas()[:self.max_synonyms]:
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != token and synonym not in self.stop_words:
                        synonyms.add(synonym)

            # Get hypernyms if enabled
            hypernyms = set()
            if self.include_hypernyms:
                for synset in synsets[:self.max_related]:
                    for hypernym in synset.hypernyms():
                        for lemma in hypernym.lemmas()[:self.max_synonyms]:
                            hypernym_name = lemma.name().replace('_', ' ')
                            if hypernym_name != token and hypernym_name not in self.stop_words:
                                hypernyms.add(hypernym_name)

            # Add to expansion details
            expansion_details[token] = {
                "synonyms": list(synonyms),
                "hypernyms": list(hypernyms)
            }

        return expansion_details

# --- Semantic Search ---
class SemanticSearch:
    """Semantic search using TF-IDF and cosine similarity."""

    def __init__(self, documents: List[Dict[str, Any]] = None):
        """
        Initialize the semantic search.

        Args:
            documents: Optional list of documents to index
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        self.documents = documents or []
        self.document_vectors = None
        self.indexed = False

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the index.

        Args:
            documents: List of documents to add
        """
        self.documents.extend(documents)
        self.indexed = False

    def index_documents(self):
        """Index the documents for semantic search."""
        if not self.documents:
            logger.warning("No documents to index")
            return

        # Extract document content
        document_texts = [doc.get("content", "") for doc in self.documents]

        # Fit and transform the vectorizer
        self.document_vectors = self.vectorizer.fit_transform(document_texts)

        self.indexed = True
        logger.info(f"Indexed {len(self.documents)} documents for semantic search")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results
        """
        if not self.indexed:
            self.index_documents()

        if not self.indexed or not self.documents:
            return []

        # Transform query
        query_vector = self.vectorizer.transform([query])

        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Create results
        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:
                results.append({
                    "document": self.documents[idx],
                    "score": float(similarities[idx]),
                    "rank": i + 1
                })

        return results

# --- Hybrid Search ---
class HybridSearch:
    """Combine vector search with semantic search for better results."""

    def __init__(self, vector_store: Any, semantic_weight: float = 0.3):
        """
        Initialize the hybrid search.

        Args:
            vector_store: Vector store for embedding-based search
            semantic_weight: Weight for semantic search results (0-1)
        """
        self.vector_store = vector_store
        self.semantic_search = SemanticSearch()
        self.semantic_weight = max(0.0, min(1.0, semantic_weight))
        self.query_expander = QueryExpander()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to both search systems.

        Args:
            documents: List of documents to add
        """
        # Add to semantic search
        self.semantic_search.add_documents(documents)

        # Add to vector store
        for doc in documents:
            if "embedding" in doc:
                self.vector_store.add(doc["embedding"], doc)

    def search(self, query: str, top_k: int = 5, expand_query: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            expand_query: Whether to expand the query

        Returns:
            List of search results
        """
        # Expand query if enabled
        expanded_query = self.query_expander.expand_query(query) if expand_query else query

        # Get query embedding
        query_embedding = self.vector_store.get_embedding(expanded_query)

        # Perform vector search
        vector_results = self.vector_store.search(query_embedding, k=top_k * 2)

        # Perform semantic search
        semantic_results = self.semantic_search.search(expanded_query, top_k=top_k * 2)

        # Combine results
        combined_results = self._combine_results(vector_results, semantic_results, top_k)

        return combined_results

    def _combine_results(self, vector_results: List[Dict[str, Any]],
                        semantic_results: List[Dict[str, Any]],
                        top_k: int) -> List[Dict[str, Any]]:
        """
        Combine vector and semantic search results.

        Args:
            vector_results: Vector search results
            semantic_results: Semantic search results
            top_k: Number of results to return

        Returns:
            Combined search results
        """
        # Create document ID to score mapping
        document_scores = {}

        # Add vector search scores
        vector_weight = 1.0 - self.semantic_weight
        for i, result in enumerate(vector_results):
            doc_id = result.get("id", result.get("document", {}).get("id", f"v_{i}"))
            score = result.get("score", 0.0) * vector_weight
            document_scores[doc_id] = {"score": score, "document": result.get("document", result)}

        # Add semantic search scores
        for i, result in enumerate(semantic_results):
            doc_id = result.get("document", {}).get("id", f"s_{i}")
            semantic_score = result.get("score", 0.0) * self.semantic_weight

            if doc_id in document_scores:
                document_scores[doc_id]["score"] += semantic_score
            else:
                document_scores[doc_id] = {"score": semantic_score, "document": result.get("document", {})}

        # Sort by score and take top_k
        sorted_results = sorted(document_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        # Format results
        results = []
        for i, result in enumerate(sorted_results):
            results.append({
                "document": result["document"],
                "score": result["score"],
                "rank": i + 1
            })

        return results

# --- Context-Aware Search ---
class ContextAwareSearch:
    """Search that takes into account user context and previous searches."""

    def __init__(self, vector_store: Any, context_weight: float = 0.2, history_size: int = 5):
        """
        Initialize the context-aware search.

        Args:
            vector_store: Vector store for embedding-based search
            context_weight: Weight for contextual information (0-1)
            history_size: Number of previous queries to consider
        """
        self.vector_store = vector_store
        self.hybrid_search = HybridSearch(vector_store)
        self.context_weight = max(0.0, min(1.0, context_weight))
        self.history_size = history_size
        self.query_history = []
        self.result_history = []

    def search(self, query: str, user_context: Dict[str, Any] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform context-aware search.

        Args:
            query: Search query
            user_context: Optional user context information
            top_k: Number of results to return

        Returns:
            List of search results
        """
        # Perform hybrid search
        results = self.hybrid_search.search(query, top_k=top_k * 2)

        # Apply context-based reranking
        if user_context or self.query_history:
            results = self._rerank_with_context(results, query, user_context, top_k)

        # Update history
        self.query_history.append(query)
        self.result_history.append([r["document"].get("id") for r in results[:top_k]])

        # Trim history if needed
        if len(self.query_history) > self.history_size:
            self.query_history = self.query_history[-self.history_size:]
            self.result_history = self.result_history[-self.history_size:]

        return results[:top_k]

    def _rerank_with_context(self, results: List[Dict[str, Any]],
                            query: str,
                            user_context: Dict[str, Any],
                            top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank results based on context.

        Args:
            results: Search results to rerank
            query: Current query
            user_context: User context information
            top_k: Number of results to return

        Returns:
            Reranked search results
        """
        # Create document ID to result mapping
        doc_id_to_result = {r["document"].get("id"): r for r in results}

        # Calculate context scores
        context_scores = {}

        # Consider previous queries and results
        if self.query_history:
            # Calculate query similarity
            for prev_query in self.query_history:
                similarity = self._calculate_text_similarity(query, prev_query)

                # If similar enough, boost documents from previous results
                if similarity > 0.3:
                    for prev_results in self.result_history:
                        for doc_id in prev_results:
                            if doc_id in context_scores:
                                context_scores[doc_id] += similarity * 0.1
                            else:
                                context_scores[doc_id] = similarity * 0.1

        # Consider user context
        if user_context:
            # User interests
            if "interests" in user_context and isinstance(user_context["interests"], list):
                for interest in user_context["interests"]:
                    for result in results:
                        doc_id = result["document"].get("id")
                        doc_content = result["document"].get("content", "")

                        if interest.lower() in doc_content.lower():
                            if doc_id in context_scores:
                                context_scores[doc_id] += 0.2
                            else:
                                context_scores[doc_id] = 0.2

            # User role
            if "role" in user_context:
                role = user_context["role"]
                for result in results:
                    doc_id = result["document"].get("id")
                    doc_metadata = result["document"].get("metadata", {})

                    if "target_roles" in doc_metadata and role in doc_metadata["target_roles"]:
                        if doc_id in context_scores:
                            context_scores[doc_id] += 0.3
                        else:
                            context_scores[doc_id] = 0.3

        # Apply context scores to results
        for result in results:
            doc_id = result["document"].get("id")
            if doc_id in context_scores:
                result["score"] = result["score"] * (1 - self.context_weight) + context_scores[doc_id] * self.context_weight

        # Sort by updated scores
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity
        tokens1 = set(word_tokenize(text1.lower()))
        tokens2 = set(word_tokenize(text2.lower()))

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def clear_history(self):
        """Clear search history."""
        self.query_history = []
        self.result_history = []

# --- Query Understanding ---
class QueryUnderstanding:
    """Understand and analyze search queries."""

    def __init__(self):
        """Initialize the query understanding system."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a search query.

        Args:
            query: Search query

        Returns:
            Query analysis
        """
        # Tokenize query
        tokens = word_tokenize(query.lower())

        # Extract key terms (non-stopwords)
        key_terms = [token for token in tokens if token not in self.stop_words
                    and token not in string.punctuation
                    and len(token) > 2]

        # Lemmatize terms
        lemmatized_terms = [self.lemmatizer.lemmatize(term) for term in key_terms]

        # Detect query type
        query_type = self._detect_query_type(query, tokens)

        # Extract entities
        entities = self._extract_entities(query)

        # Detect filters
        filters = self._detect_filters(query)

        return {
            "original_query": query,
            "tokens": tokens,
            "key_terms": key_terms,
            "lemmatized_terms": lemmatized_terms,
            "query_type": query_type,
            "entities": entities,
            "filters": filters
        }

    def _detect_query_type(self, query: str, tokens: List[str]) -> str:
        """
        Detect the type of query.

        Args:
            query: Search query
            tokens: Tokenized query

        Returns:
            Query type
        """
        # Check for question
        question_words = {"what", "who", "where", "when", "why", "how"}
        if tokens and tokens[0].lower() in question_words:
            return "question"

        # Check for command
        command_words = {"find", "search", "get", "show", "list", "display"}
        if tokens and tokens[0].lower() in command_words:
            return "command"

        # Check for boolean query
        boolean_operators = {"and", "or", "not"}
        if any(op in tokens for op in boolean_operators):
            return "boolean"

        # Default to keyword query
        return "keyword"

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from a query.

        Args:
            query: Search query

        Returns:
            Dictionary of entity types to entities
        """
        entities = {
            "dates": [],
            "numbers": [],
            "proper_nouns": []
        }

        # Extract dates
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',    # YYYY-MM-DD
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',    # DD-MM-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["dates"].extend(matches)

        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities["numbers"] = re.findall(number_pattern, query)

        # Extract proper nouns (simple heuristic)
        tokens = word_tokenize(query)
        for i, token in enumerate(tokens):
            if token[0].isupper() and (i == 0 or tokens[i-1] in {'.', '!', '?', ';'}):
                entities["proper_nouns"].append(token)

        return entities

    def _detect_filters(self, query: str) -> Dict[str, Any]:
        """
        Detect filters in a query.

        Args:
            query: Search query

        Returns:
            Dictionary of filters
        """
        filters = {}

        # Detect date filters
        date_filters = {
            "before": re.findall(r'before\s+(\d{4}(?:-\d{1,2}-\d{1,2})?)', query, re.IGNORECASE),
            "after": re.findall(r'after\s+(\d{4}(?:-\d{1,2}-\d{1,2})?)', query, re.IGNORECASE),
            "between": re.findall(r'between\s+(\d{4}(?:-\d{1,2}-\d{1,2})?)\s+and\s+(\d{4}(?:-\d{1,2}-\d{1,2})?)', query, re.IGNORECASE)
        }

        if any(date_filters.values()):
            filters["date"] = {k: v for k, v in date_filters.items() if v}

        # Detect type filters
        type_patterns = [
            (r'\b(pdf|document)s?\b', "pdf"),
            (r'\b(docx?|word)s?\b', "doc"),
            (r'\b(xlsx?|excel|spreadsheet)s?\b', "spreadsheet"),
            (r'\b(pptx?|powerpoint|presentation)s?\b', "presentation"),
            (r'\b(txt|text)s?\b', "text")
        ]

        doc_types = []
        for pattern, doc_type in type_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                doc_types.append(doc_type)

        if doc_types:
            filters["document_type"] = doc_types

        return filters

# --- Factory Function ---
def create_advanced_search(vector_store: Any, documents: List[Dict[str, Any]] = None) -> ContextAwareSearch:
    """
    Create an advanced search system.

    Args:
        vector_store: Vector store for embedding-based search
        documents: Optional list of documents to index

    Returns:
        Context-aware search system
    """
    # Create semantic search
    semantic_search = SemanticSearch(documents)

    # Create hybrid search
    hybrid_search = HybridSearch(vector_store)

    # Add documents if provided
    if documents:
        hybrid_search.add_documents(documents)

    # Create context-aware search
    context_aware_search = ContextAwareSearch(vector_store)

    return context_aware_search