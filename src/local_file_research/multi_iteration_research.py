"""
Multi-iteration research for Local File Deep Research.

This module provides functionality for conducting multi-iteration research
with query expansion and follow-up questions based on collected context.
"""

import logging
import time
from typing import List, Dict, Any, Union

from .vector_store import VectorStore
from .document_analysis import DocumentAnalyzer
from .advanced_search import QueryExpander
from .dspy_config import DSPyAgentRegistry, DSPY_CONFIGURED
from .llamaindex_vector_store import LlamaIndexVectorStore
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Import safe_str from logging_config if available
try:
    from .logging_config import safe_str
except ImportError:
    # Fallback implementation if logging_config is not available
    def safe_str(obj):
        """Convert any object to a string that can be safely printed/logged."""
        if obj is None:
            return 'None'
        try:
            # First try normal string conversion
            return str(obj)
        except UnicodeEncodeError:
            # If that fails, try with explicit encoding/decoding
            try:
                if isinstance(obj, str):
                    # For strings, encode with replacement
                    return obj.encode('utf-8', errors='backslashreplace').decode('utf-8')
                else:
                    # For other objects, convert to string with replacement
                    return str(obj).encode('utf-8', errors='backslashreplace').decode('utf-8')
            except Exception:
                # Last resort - return a placeholder
                return "[Unicode conversion error]"

class MultiIterationResearch:
    """
    Multi-iteration research system that uses query expansion and follow-up questions.
    """

    def __init__(self, vector_store: Union[VectorStore, LlamaIndexVectorStore], max_iterations: int = 3,
                 questions_per_iteration: int = 3, max_context_size: int = 10000,
                 max_k: int = 20, relevance_threshold: float = 0.5):
        """
        Initialize the multi-iteration research system.

        Args:
            vector_store: Vector store for searching documents (VectorStore or LlamaIndexVectorStore)
            max_iterations: Maximum number of research iterations
            questions_per_iteration: Number of follow-up questions per iteration
            max_context_size: Maximum size of the context in characters (default: 10000)
            max_k: Maximum limit for results to retrieve before deduplication (default: 20).
                 Actual retrieval count is dynamically calculated based on already collected sources.
            relevance_threshold: Minimum relevance score for results (default: 0.5)
        """
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        self.questions_per_iteration = questions_per_iteration
        self.max_context_size = max_context_size
        self.max_k = max_k
        self.relevance_threshold = relevance_threshold
        self.accumulated_unique_sources = set()  # Track all unique sources found so far
        self.accumulated_unique_chunks = set()  # Track all unique chunks found so far
        self.seen_chunks = set()  # For deduplicating chunks across searches
        self.document_analyzer = DocumentAnalyzer()
        self.query_expander = QueryExpander()



        # Performance tracking
        self.performance_metrics = {
            "search_time": 0,
            "analysis_time": 0,
            "question_generation_time": 0,
            "total_time": 0,
            "source_diversity": {},  # Track source diversity across iterations
            "verification_time": 0   # Track verification time
        }

        # State tracking
        self.questions_by_iteration = {}
        self.all_findings = []
        self.current_knowledge = ""

        # Enhanced tracking for multi-iteration research
        self.findings_by_question = {}  # Dictionary to track findings by question
        self.context_by_question = {}   # Dictionary to track context by question

        # Tracking for deduplication
        self.seen_chunks = set()  # Set of chunk identifiers we've already processed

    def run(self, query: str, top_k: int = 5, context_filter: str = None,
            max_k: int = None, relevance_threshold: float = None) -> Dict[str, Any]:
        """
        Run multi-iteration research.

        Args:
            query: Original research query
            top_k: Number of results to retrieve per question
            context_filter: Optional filter for results
            max_k: Maximum number of results to retrieve before deduplication (overrides instance value if provided)
            relevance_threshold: Minimum relevance score for results (overrides instance value if provided)

        Returns:
            Dictionary with research findings and metadata
        """
        # Override instance values if provided
        if max_k is not None:
            self.max_k = max_k
        if relevance_threshold is not None:
            self.relevance_threshold = relevance_threshold
        start_time = time.time()
        logger.info(f"Starting multi-iteration research for query: '{query}'")

        # Reset state
        self.questions_by_iteration = {}
        self.all_findings = []
        self.current_knowledge = ""
        self.seen_chunks = set()
        self.accumulated_unique_sources = set()  # Reset accumulated unique sources
        self.accumulated_unique_chunks = set()  # Reset accumulated unique chunks
        self.findings_by_question = {}
        self.context_by_question = {}
        self.performance_metrics = {
            "search_time": 0,
            "analysis_time": 0,
            "question_generation_time": 0,
            "verification_time": 0,
            "source_diversity": {},
            "total_time": 0
        } # Reset timings

        # Initial query expansion
        expanded_query = self.query_expander.expand_query(query)
        # Store expanded query as instance variable for use in _perform_initial_search
        self.expanded_query = expanded_query
        logger.info(f"Expanded query: '{expanded_query}'")

        # Log if the expanded query is different from the original
        if expanded_query != query:
            logger.info(f"Using expanded query: '{expanded_query}' (different from original)")
        else:
            logger.info("Expanded query is the same as original, will only search once")

        # Perform initial search with both original and expanded queries
        self._perform_initial_search(query, top_k, context_filter)

        # Log the questions_by_iteration to verify it contains the expected data
        logger.info(f"Initial questions_by_iteration: {self.questions_by_iteration}")

        # Track all questions asked so far to avoid duplicates
        all_questions = set()
        for q in self.questions_by_iteration.get('0', []):
            all_questions.add(q.lower())

        # Subsequent iterations with follow-up questions
        for iteration in range(1, self.max_iterations):
            logger.info(f"Starting research iteration {iteration+1}/{self.max_iterations}")

            # Generate follow-up questions, ensuring they're unique
            follow_up_questions = self._generate_follow_up_questions(query, iteration=iteration)

            # Filter out questions we've already asked
            unique_questions = []
            for q in follow_up_questions:
                if q.lower() not in all_questions:
                    unique_questions.append(q)
                    all_questions.add(q.lower())

            # If we don't have any unique questions, generate some generic ones
            if not unique_questions:
                logger.info("No unique questions generated, adding generic follow-ups")
                generic_questions = [
                    f"What are additional aspects of {query} not covered so far?",
                    f"What are alternative perspectives on {query}?"
                ]
                for q in generic_questions:
                    if q.lower() not in all_questions:
                        unique_questions.append(q)
                        all_questions.add(q.lower())

            # Store the unique questions for this iteration
            self.questions_by_iteration[str(iteration)] = unique_questions
            logger.info(f"Iteration {iteration+1}: Using {len(unique_questions)} unique follow-up questions")

            # Search for each follow-up question
            iteration_findings = []
            for question in unique_questions:
                # Calculate dynamic max_k using accumulated unique chunks
                dynamic_max_k = len(self.accumulated_unique_chunks) + top_k
                search_top_k = min(dynamic_max_k, self.max_k)
                logger.info(f"Dynamic max_k for follow-up question '{question}': {len(self.accumulated_unique_chunks)} accumulated unique chunks + {top_k} top_k = {dynamic_max_k}, limited by max_k={self.max_k}")

                results = self._search_and_analyze(
                    question,
                    top_k=top_k,
                    context_filter=context_filter,
                    is_follow_up=True,  # This is a follow-up question
                    pre_calculated_max_k=search_top_k  # Use dynamic max_k
                )

                # Update accumulated unique sources and chunks with new findings
                new_sources = set()
                new_chunks = set()

                # Log current accumulated unique sources and chunks
                logger.info(f"Current accumulated unique sources before processing follow-up question results: {self.accumulated_unique_sources}")
                logger.info(f"Current accumulated unique chunks before processing follow-up question results: {self.accumulated_unique_chunks}")

                for i, result in enumerate(results):
                    source = result.get("file_path", "")
                    chunk_id = self._get_chunk_identifier(result)

                    # Log more details about the source and chunk
                    # Log normalized score
                    logger.info(f"Follow-up result {i+1}: chunk_id={chunk_id}, score={result.get('score', 'N/A')}, file_path={source}")

                    # Track unique sources
                    if source:
                        is_new_source = source not in self.accumulated_unique_sources
                        logger.info(f"Source '{source}' is {'new' if is_new_source else 'already in accumulated sources'}")

                        if is_new_source:
                            new_sources.add(source)

                        # Add to accumulated sources regardless
                        self.accumulated_unique_sources.add(source)

                    # Track unique chunks
                    is_new_chunk = chunk_id not in self.accumulated_unique_chunks
                    logger.info(f"Chunk '{chunk_id}' is {'new' if is_new_chunk else 'already in accumulated chunks'}")

                    if is_new_chunk:
                        new_chunks.add(chunk_id)

                    # Add to accumulated chunks regardless
                    self.accumulated_unique_chunks.add(chunk_id)

                if new_sources or new_chunks:
                    logger.info(f"Found {len(new_sources)} new unique sources from follow-up question '{question}'")
                    logger.info(f"Found {len(new_chunks)} new unique chunks from follow-up question '{question}'")
                    logger.info(f"Accumulated unique sources now: {len(self.accumulated_unique_sources)}")
                    logger.info(f"Accumulated unique chunks now: {len(self.accumulated_unique_chunks)}")

                # Add results to iteration_findings
                iteration_findings.extend(results)

                # Add results to all_findings immediately
                self.all_findings.extend(results)
                logger.info(f"Added {len(results)} results from follow-up question '{question}' to all_findings")
                logger.info(f"all_findings now contains {len(self.all_findings)} items")

            # Since we've already added results to all_findings, we only need to update knowledge
            self._update_current_knowledge()
            logger.info(f"After iteration {iteration+1}, all_findings contains {len(self.all_findings)} items")

            # Check if we have enough information
            if self._check_sufficient_information():
                logger.info(f"Sufficient information gathered after iteration {iteration+1}")
                break

        # Limit the final context if it's too large
        max_final_context_size = 100000  # Maximum size for the final context
        if sum(len(f.get("content", "")) for f in self.all_findings) > max_final_context_size:
            logger.info(f"Limiting final context to approximately {max_final_context_size} characters")
            self._limit_final_context(max_final_context_size)

        # Pool all context from all questions for enhanced reporting
        all_pooled_context = self._pool_all_context()

        # Perform verification AFTER pooling to avoid duplicate verification
        # This ensures we only verify each unique finding once
        self._verify_pooled_findings(all_pooled_context["findings"])

        # Calculate total time
        total_time = time.time() - start_time
        self.performance_metrics["total_time"] = total_time

        logger.info(f"Multi-iteration research complete in {total_time:.2f} seconds")

        # --- Return data for ReportGenerator, NOT the report itself ---
        return {
            "findings": self.all_findings, # Raw findings from all iterations
            "pooled_context": all_pooled_context, # Pooled/deduplicated context and findings
            "questions_by_iteration": self.questions_by_iteration,
            "iterations": len(self.questions_by_iteration),
            "performance_metrics": self.performance_metrics,
            # Removed report generation from here
        }

    def _search_and_analyze(self, query: str, top_k: int, context_filter: str = None,
                           is_follow_up: bool = False, pre_calculated_max_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for documents and analyze them.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            context_filter: Optional filter for results
            is_follow_up: Whether this is a follow-up question (affects search behavior)
            pre_calculated_max_k: Optional pre-calculated max_k to avoid recalculation

        Returns:
            List of analyzed search results
        """
        # Search
        search_start_time = time.time()

        # Log with safe_str to handle Unicode characters
        logger.info(f"Searching for query: '{safe_str(query)}'")

        # Use pre-calculated max_k if provided, otherwise calculate
        if pre_calculated_max_k is not None:
            search_top_k = pre_calculated_max_k
            logger.info(f"Using pre-calculated max_k: {search_top_k}")
        else:
            search_top_k = top_k
            logger.info(f"Using standard top_k: {search_top_k}")

        # Check if we have a LlamaIndexVectorStore with a valid index for query engine approach
        if isinstance(self.vector_store, LlamaIndexVectorStore) and hasattr(self.vector_store, 'index') and self.vector_store.index:
            logger.info(f"Using LlamaIndex query engine for improved search with top_k={search_top_k}...")
            try:
                # Create a query engine with the specified top_k
                query_engine = self.vector_store.index.as_query_engine(similarity_top_k=search_top_k)

                # Execute the query
                response = query_engine.query(query)

                # Extract source nodes
                results = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    logger.info(f"Found {len(response.source_nodes)} source nodes using query engine")
                    for i, node in enumerate(response.source_nodes):
                        # Convert node to dictionary format
                        result = {
                            "node_id": node.node_id,
                            "content": node.text,
                            "score": node.score if hasattr(node, 'score') else 0.0
                        }

                        # Add metadata if available
                        if hasattr(node, 'metadata') and node.metadata:
                            result.update(node.metadata)

                        results.append(result)
                else:
                    logger.warning("Query engine did not return source nodes, falling back to traditional search")
                    # Fall back to traditional search
                    from .pipeline import embed_text
                    query_embedding = embed_text(query)
                    results = self.vector_store.search(query_embedding, top_k=search_top_k)
            except Exception as e:
                logger.error(f"Error using query engine: {e}. Falling back to traditional search.")
                # Fall back to traditional search
                from .pipeline import embed_text
                query_embedding = embed_text(query)
                results = self.vector_store.search(query_embedding, top_k=search_top_k)
        else:
            # Use traditional search if not a LlamaIndexVectorStore or no index
            from .pipeline import embed_text
            query_embedding = embed_text(query)

            # Log embedding details
            logger.info(f"Generated embedding for query: '{safe_str(query)}' with dimension {len(query_embedding)}")

            # Log a few values from the embedding to help diagnose issues
            if query_embedding:
                embedding_preview = [f"{val:.4f}" for val in query_embedding[:5]]
                logger.info(f"Embedding preview (first 5 values): {embedding_preview}")

            # Search vector store with max_k for follow-up questions or top_k for original query
            results = self.vector_store.search(query_embedding, top_k=search_top_k)

        # Log search results
        logger.info(f"Search returned {len(results)} results for query: '{safe_str(query)}'")

        # Apply relevance threshold filtering
        if is_follow_up:
            # For follow-up questions, apply relevance threshold
            results = self._filter_by_relevance(results, self.relevance_threshold)

            # Create a copy of seen_chunks to avoid deduplicating against chunks we're about to add
            # This prevents the double deduplication issue
            seen_chunks_copy = set(self.seen_chunks)

            # Deduplicate results against already seen chunks (using the copy)
            results = self._deduplicate_results(results, seen_chunks_copy)

            # Log the state after deduplication
            logger.info(f"After deduplication with seen_chunks_copy, {len(results)} results remain")

            # Limit to top_k after deduplication
            if len(results) > top_k:
                logger.info(f"Limiting {len(results)} deduplicated results to top {top_k}")
                results = results[:top_k]
        else:
            # For original query, just apply relevance threshold
            results = self._filter_by_relevance(results, self.relevance_threshold)

        # Track source diversity
        if results:
            # Get current iteration
            current_iteration = None
            for iter_num, questions in self.questions_by_iteration.items():
                if query in questions:
                    current_iteration = iter_num
                    break

            if current_iteration is None:
                current_iteration = "0"  # Default to first iteration if not found

            # Initialize source diversity tracking for this iteration if needed
            if current_iteration not in self.performance_metrics["source_diversity"]:
                self.performance_metrics["source_diversity"][current_iteration] = {
                    "sources": {},
                    "total_results": 0,
                    "unique_sources": 0,
                    "queries": {}  # Track which sources came from which queries
                }

            # Track sources for this iteration
            iteration_sources = set()
            for result in results:
                source_path = result.get("file_path", "unknown")
                if source_path not in self.performance_metrics["source_diversity"][current_iteration]["sources"]:
                    self.performance_metrics["source_diversity"][current_iteration]["sources"][source_path] = 0
                self.performance_metrics["source_diversity"][current_iteration]["sources"][source_path] += 1

                # Track which query found this source
                if query not in self.performance_metrics["source_diversity"][current_iteration]["queries"]:
                    self.performance_metrics["source_diversity"][current_iteration]["queries"][query] = set()
                self.performance_metrics["source_diversity"][current_iteration]["queries"][query].add(source_path)

                iteration_sources.add(source_path)

            # Update metrics
            self.performance_metrics["source_diversity"][current_iteration]["total_results"] += len(results)
            self.performance_metrics["source_diversity"][current_iteration]["unique_sources"] = len(self.performance_metrics["source_diversity"][current_iteration]["sources"])

            # Check for repeated sources across iterations
            if current_iteration != "0":  # Skip for first iteration
                previous_iterations = [i for i in self.performance_metrics["source_diversity"].keys() if i != current_iteration]
                for prev_iter in previous_iterations:
                    prev_sources = set(self.performance_metrics["source_diversity"][prev_iter]["sources"].keys())
                    common_sources = iteration_sources.intersection(prev_sources)
                    if common_sources:
                        # Calculate percentage of overlap
                        overlap_percentage = len(common_sources) / len(iteration_sources) * 100
                        if overlap_percentage > 75:  # If more than 75% of sources are the same
                            logger.warning(f"WARNING: High source overlap ({overlap_percentage:.1f}%) between iteration {prev_iter} and {current_iteration}")
                            logger.warning(f"Common sources: {common_sources}")

            # Log the first result to see what it contains
            if results:
                first_result = results[0]
                # Log normalized score
                logger.info(f"First result: score={first_result.get('score', 'N/A')}, file_path={first_result.get('file_path', 'N/A')}")
                logger.info(f"First result content preview: {safe_str(first_result.get('content', 'No content'))[:100]}...")
        else:
            logger.warning(f"No results found for query: '{safe_str(query)}'. This might indicate an issue with the vector store or embeddings.")

            # Check if this is a direct query from the user (not a follow-up)
            is_original_query = query in self.questions_by_iteration.get('0', []) if hasattr(self, 'questions_by_iteration') else False

            if is_original_query:
                logger.error(f"No results found for original user query: '{safe_str(query)}'. This is a critical issue.")
                logger.error("Please check that the vector database contains relevant content and is properly indexed.")
                logger.error("The vector database may be empty or the query may not match any content.")

        # Apply context filter if provided
        if context_filter:
            logger.info(f"Applying context filter: {safe_str(context_filter)}")
            filtered_results = [r for r in results if context_filter.lower() in safe_str(r.get("file_path", "")).lower()]
            logger.info(f"After filtering, {len(filtered_results)} results remain")
            results = filtered_results

        search_time = time.time() - search_start_time
        self.performance_metrics["search_time"] += search_time

        # Analyze results
        analysis_start_time = time.time()
        analyzed_results = []

        # Log the number of results to analyze
        logger.info(f"Analyzing {len(results)} search results for query: '{safe_str(query)}'")
        if not results:
            logger.warning(f"No search results to analyze for query: '{safe_str(query)}'")

            # Check if this is a direct query from the user (not a follow-up)
            is_original_query = query in self.questions_by_iteration.get('0', []) if hasattr(self, 'questions_by_iteration') else False

            if is_original_query:
                logger.error(f"No results found for original user query: '{safe_str(query)}'. This is a critical issue.")
                logger.error("Please check that the vector database contains relevant content and is properly indexed.")
                logger.error("The vector database may be empty or the query may not match any content.")

            return []

        # Log detailed information about each result before analysis
        logger.info("=== DETAILED RESULTS BEFORE ANALYSIS ===")
        for i, result in enumerate(results):
            chunk_id = self._get_chunk_identifier(result)
            logger.info(f"Result {i+1} before analysis: chunk_id={chunk_id}, score={result.get('score', 'N/A')}, file_path={result.get('file_path', 'N/A')}")
            logger.info(f"Result {i+1} content preview: {safe_str(result.get('content', 'No content'))[:100]}...")
            logger.info(f"Result {i+1} already in seen_chunks: {chunk_id in self.seen_chunks}")
        logger.info("=== END DETAILED RESULTS ===")

        for i, result in enumerate(results):
            try:
                # Log the result being analyzed
                logger.info(f"Analyzing result {i+1}/{len(results)}")

                # Check if result has content
                if not result.get("content"):
                    logger.warning(f"Result {i+1} has no content, skipping")
                    continue

                # Get source information for chunk-level deduplication
                source_path = result.get("file_path", "")

                # Create a unique chunk identifier using the helper method
                chunk_id = self._get_chunk_identifier(result)

                # We've already deduplicated against seen_chunks in the previous step,
                # so we don't need to check again here. This avoids the double deduplication issue.
                # However, we still need to check if this chunk is in all_findings to update related_questions.

                found_in_all_findings = False
                for existing_finding in self.all_findings:
                    existing_chunk_id = self._get_chunk_identifier(existing_finding)
                    if existing_chunk_id == chunk_id:
                        # Add this query to the related_questions list
                        if "related_questions" not in existing_finding:
                            existing_finding["related_questions"] = [existing_finding.get("query", "")]
                        if query not in existing_finding["related_questions"]:
                            existing_finding["related_questions"].append(query)
                        logger.info(f"Added query '{query}' to related_questions for chunk {chunk_id} in all_findings")
                        found_in_all_findings = True
                        break

                if found_in_all_findings:
                    logger.info(f"Chunk {chunk_id} found in all_findings, but still processing it for this query")

                # Mark this chunk as seen
                logger.info(f"Adding chunk {chunk_id} to seen_chunks")
                self.seen_chunks.add(chunk_id)

                # Initialize related_questions with the current query
                result["related_questions"] = [query]

                # Check if this document has already been analyzed
                existing_doc = next((d for d in self.all_findings if d.get("file_path") == source_path), None)

                if existing_doc and "analysis" in existing_doc:
                    # Use existing analysis
                    result["analysis"] = existing_doc["analysis"]
                    result["summary"] = existing_doc.get("summary", "")
                else:
                    # Perform new analysis
                    analysis = self.document_analyzer.analyze_document(
                        result.get("content", ""),
                        source_path,
                        query
                    )
                    result["analysis"] = analysis
                    result["summary"] = analysis.get("summary", "")

                # Add citation metadata
                result["citation"] = {
                    "source_path": result.get("source_path", source_path),
                    "source_name": result.get("source_name"),
                    "source_type": result.get("source_type"),
                    "start": result.get("start"),
                    "end": result.get("end"),
                    "score": result.get("score", 0.0),  # Add score to citation for confidence
                }

                # Add query information to track which question led to this finding
                result["query"] = query

                # Try to determine which iteration this is from
                current_iteration = None
                for iter_num, questions in self.questions_by_iteration.items():
                    if query in questions:
                        current_iteration = int(iter_num) if iter_num.isdigit() else 0
                        break

                # Add query info with iteration
                result["query_info"] = {
                    "query": query,
                    "iteration": current_iteration if current_iteration is not None else 0
                }

                # Add confidence level based on score
                if "score" in result:
                    result["confidence"] = result["score"]

                # Make sure content field exists
                if "content" not in result or not result["content"]:
                    result["content"] = "No content available"

                # Add query information
                if "query" not in result:
                    result["query"] = query

                # Track findings by question for enhanced reporting
                if query not in self.findings_by_question:
                    self.findings_by_question[query] = []
                self.findings_by_question[query].append(result)

                # Track context by question for enhanced reporting
                if query not in self.context_by_question:
                    self.context_by_question[query] = []
                self.context_by_question[query].append(result.get("content", ""))

                analyzed_results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing document: {e}", exc_info=True)

        analysis_time = time.time() - analysis_start_time
        self.performance_metrics["analysis_time"] += analysis_time

        # Log detailed information about the final results being returned
        logger.info(f"Final results from _search_and_analyze for query '{safe_str(query)}': {len(analyzed_results)} results")
        for i, result in enumerate(analyzed_results):
            chunk_id = self._get_chunk_identifier(result)
            logger.info(f"Final result {i+1}: chunk_id={chunk_id}, score={result.get('score', 'N/A')}, file_path={result.get('file_path', 'N/A')}")

        return analyzed_results

    def _update_current_knowledge(self):
        """Update the current knowledge based on all findings."""
        # Extract summaries from all findings
        summaries = []
        for finding in self.all_findings:
            # Make sure content field exists
            if "content" not in finding or not finding["content"]:
                finding["content"] = "No content available"

            summary = finding.get("summary", "")
            if summary:
                summaries.append(summary)

        # Combine summaries
        combined_knowledge = "\n\n".join(summaries)

        # Limit the context size if it exceeds the maximum
        if len(combined_knowledge) > self.max_context_size:
            # Truncate the knowledge to the maximum size
            # We keep the most recent findings by truncating from the beginning
            combined_knowledge = combined_knowledge[-self.max_context_size:]
            # Make sure we don't cut in the middle of a sentence
            first_period = combined_knowledge.find(".")
            if first_period > 0:
                combined_knowledge = combined_knowledge[first_period + 1:].strip()

        self.current_knowledge = combined_knowledge

    def _generate_follow_up_questions(self, original_query: str, iteration: int = 1) -> List[str]:
        """
        Generate follow-up questions based on current knowledge.
        Relies on dspy_config._ensure_required_fields for handling agent inputs.
        """
        question_start_time = time.time()
        questions = []
        agent_used = "None" # Track which method succeeded

        # Try DSPy query_refinement agent first
        if DSPY_CONFIGURED: # Check if DSPy is ready
            try:
                query_refinement_agent = DSPyAgentRegistry.get_agent("query_refinement")
                if query_refinement_agent:
                    logger.info("Attempting question generation with DSPy QueryRefinementAgent...")
                    context_val = self.current_knowledge if self.current_knowledge else "No knowledge gathered yet."
                    agent_inputs = {
                        "query": original_query,
                        "context": context_val,
                        # --- Ensure these are integers as per corrected signature ---
                        "num_queries": int(self.questions_per_iteration),
                        "iteration": int(iteration),
                        # ------------------------------------------------------------
                    }
                    # The helper function will add defaults for 'content', 'document' if needed
                    prepared_data = DSPyAgentRegistry._ensure_required_fields("query_refinement", query_refinement_agent, agent_inputs)

                    # --- Execute the agent ---
                    logger.debug(f"Calling query_refinement with: {prepared_data}")
                    result = query_refinement_agent(**prepared_data)
                    # -------------------------

                    raw_questions = result.get("related_queries", []) if result and hasattr(result, 'get') else []

                    # Process raw_questions (handle string or list)
                    if isinstance(raw_questions, str):
                         lines = [line.strip() for line in raw_questions.split('\n') if line.strip()]
                         for line in lines: q = line.lstrip('0123456789.- ').strip(); q = q + '?' if q and not q.endswith('?') else q; questions.append(q)
                    elif isinstance(raw_questions, list):
                         for q in raw_questions:
                              if isinstance(q, str): q_strip = q.strip(); q_strip = q_strip + '?' if q_strip and not q_strip.endswith('?') else q_strip; questions.append(q_strip)

                    if questions:
                        agent_used = "QueryRefinementAgent"
                        logger.info(f"Generated {len(questions)} questions via {agent_used}.")
                    else:
                        logger.warning("QueryRefinementAgent returned no questions, attempting fallback.")
                else:
                     logger.warning("Query refinement agent not found, attempting fallback.")
            except Exception as e:
                logger.error(f"Error generating questions with DSPy QueryRefinementAgent: {e}", exc_info=True)
                logger.error(f"Traceback: {traceback.format_exc()}") # Log full traceback
                # Continue to fallback

        # Fallback 1: Use TextGeneratorAgent if first agent failed or wasn't found/configured
        if not questions and DSPY_CONFIGURED:
            logger.info("Attempting fallback question generation with TextGeneratorAgent...")
            try:
                text_generator = DSPyAgentRegistry.get_agent("text_generator")
                if text_generator:
                     prompt = f"""Based on the original query "{original_query}" and the current knowledge context, generate {self.questions_per_iteration} specific follow-up questions to explore the topic further. Focus on gaps or related areas. Current knowledge: {self.current_knowledge[:1500]}...
Output ONLY the questions, one per line, numbered."""
                     agent_inputs = {"prompt": prompt, "query": original_query}
                     # Add context for the text generator if needed
                     if self.current_knowledge: agent_inputs["context"] = self.current_knowledge[:2000]
                     prepared_data = DSPyAgentRegistry._ensure_required_fields("text_generator", text_generator, agent_inputs)

                     # --- EXECUTE THE AGENT ---
                     result = text_generator(**prepared_data)
                     # -------------------------

                     response = result.get("text", "") if result and hasattr(result, 'get') else ""

                     # Extract questions from response
                     lines = [line.strip() for line in response.split('\n') if line.strip()]
                     for line in lines:
                         q = line.lstrip('0123456789.- ').strip()
                         q = q + '?' if q and not q.endswith('?') else q
                         questions.append(q)

                     if questions:
                         agent_used = "TextGeneratorAgent"
                         logger.info(f"Generated {len(questions)} questions via {agent_used}.")
                     else:
                         logger.warning("TextGeneratorAgent returned no questions.")
                else:
                     logger.error("Fallback TextGenerator agent also not found.")
            except Exception as e:
                logger.error(f"Error generating questions with fallback TextGenerator: {e}", exc_info=True)

        # Fallback 2: Use DocumentAnalyzer's basic analysis if other methods fail
        if not questions:
             logger.warning("Using DocumentAnalyzer basic analysis as final fallback for question generation.")
             agent_used = "DocumentAnalyzer(basic)"
             try:
                  prompt = f"Based on the original query '{original_query}' and context '{self.current_knowledge[:500]}...', generate {self.questions_per_iteration} follow-up questions."
                  # NOTE: This calls the *basic analysis* which is non-LLM
                  analysis = self.document_analyzer._basic_analysis(prompt)
                  summary = analysis.get("summary", "")
                  potential_questions = [line.strip() for line in summary.split('\n') if '?' in line]
                  if potential_questions:
                       questions = potential_questions[:self.questions_per_iteration]
                       logger.info(f"Extracted {len(questions)} questions from basic analysis.")
                  else:
                       logger.warning("Basic analysis did not yield questions.")

             except Exception as e:
                  logger.error(f"Error generating questions with basic analysis fallback: {e}", exc_info=True)

        # If still no questions, generate generic ones
        if not questions:
            logger.warning("No questions generated by any method, using generic questions.")
            agent_used = "GenericFallback"
            questions = [
                f"What are the key aspects of {original_query}?",
                f"What additional information is available about {original_query}?"
            ][:self.questions_per_iteration]

        question_time = time.time() - question_start_time
        self.performance_metrics["question_generation_time"] += question_time
        logger.info(f"Generated {len(questions)} follow-up questions using {agent_used} in {question_time:.2f}s")

        # Ensure only unique and correct number are returned
        final_questions = []
        seen_q = set()
        for q in questions:
             # Basic cleaning and lowercasing for uniqueness check
             q_clean = q.strip().lower()
             if q_clean and q_clean not in seen_q:
                  final_questions.append(q.strip()) # Store original stripped version
                  seen_q.add(q_clean)
             if len(final_questions) >= self.questions_per_iteration:
                  break

        # If still not enough questions after filtering, add generic ones back if needed
        while len(final_questions) < self.questions_per_iteration:
            generic_q = f"What else is important about {original_query}?"
            if generic_q.lower() not in seen_q:
                final_questions.append(generic_q)
                seen_q.add(generic_q.lower())
            else: # Avoid infinite loop if generic question already added
                break

        return final_questions

    def _check_sufficient_information(self) -> bool:
        """
        Check if we have sufficient information to answer the query.

        Returns:
            True if sufficient information has been gathered
        """
        # Check if we have reached the maximum context size
        if len(self.current_knowledge) >= self.max_context_size * 0.9:  # 90% of max size
            return True

        # Simple heuristic: check if we have enough content
        if len(self.current_knowledge) > 5000:
            return True

        # Check if we have a diverse set of sources
        sources = set()
        for finding in self.all_findings:
            source = finding.get("file_path", "")
            if source:
                sources.add(source)

        # If we have at least 5 different sources, consider it sufficient
        if len(sources) >= 5:
            return True

        return False

    def _has_sufficient_information(self) -> bool:
        """Alias for _check_sufficient_information for backward compatibility."""
        return self._check_sufficient_information("")

    def _perform_initial_search(self, query: str, top_k: int = 5, context_filter: str = None) -> None:
        """
        Perform initial search with both original and expanded queries.

        Args:
            query: The original query
            top_k: Number of results to retrieve
            context_filter: Optional filter for results
        """
        # First iteration: search with original and expanded query
        iteration_findings = []

        # Store first iteration questions before searching
        self.questions_by_iteration['0'] = [query]

        # Check if we have an expanded query that's different from the original
        has_expanded_query = hasattr(self, 'expanded_query') and self.expanded_query != query

        # Add expanded query to questions list if it exists and is different
        if has_expanded_query:
            self.questions_by_iteration['0'].append(self.expanded_query)
            logger.info(f"Will search with both original query and expanded query: '{self.expanded_query}'")

        # Search with original query
        logger.info(f"Searching with original query: '{query}'")
        original_results = self._search_and_analyze(
            query,
            top_k=top_k,
            context_filter=context_filter,
            is_follow_up=False,  # Ensure this is False for original query
            pre_calculated_max_k=None  # No pre-calculation needed for original query
        )
        logger.info(f"Found {len(original_results)} results with original query")

        # Add original results to iteration_findings
        iteration_findings.extend(original_results)

        # Add original results to all_findings immediately
        self.all_findings.extend(original_results)
        logger.info(f"Added {len(original_results)} results from original query to all_findings")
        logger.info(f"all_findings now contains {len(self.all_findings)} items")

        # Extract unique sources and chunks from original results
        new_unique_sources = set()
        new_unique_chunks = set()

        # Log current accumulated unique sources and chunks (should be empty at this point)
        logger.info(f"Current accumulated unique sources before processing original query results: {self.accumulated_unique_sources}")
        logger.info(f"Current accumulated unique chunks before processing original query results: {self.accumulated_unique_chunks}")

        for i, finding in enumerate(original_results):
            source = finding.get("file_path", "")
            chunk_id = self._get_chunk_identifier(finding)

            # Log normalized score
            logger.info(f"Original result {i+1}: chunk_id={chunk_id}, score={finding.get('score', 'N/A')}, file_path={source}")

            # Track unique sources
            if source:
                is_new_source = source not in self.accumulated_unique_sources
                logger.info(f"Source '{source}' is {'new' if is_new_source else 'already in accumulated sources'}")

                if is_new_source:
                    new_unique_sources.add(source)

                # Add to accumulated sources regardless
                self.accumulated_unique_sources.add(source)

            # Track unique chunks
            is_new_chunk = chunk_id not in self.accumulated_unique_chunks
            logger.info(f"Chunk '{chunk_id}' is {'new' if is_new_chunk else 'already in accumulated chunks'}")

            if is_new_chunk:
                new_unique_chunks.add(chunk_id)

            # Add to accumulated chunks regardless
            self.accumulated_unique_chunks.add(chunk_id)

        logger.info(f"Found {len(new_unique_sources)} unique sources from original query")
        logger.info(f"Found {len(new_unique_chunks)} unique chunks from original query")
        logger.info(f"Accumulated unique sources so far: {len(self.accumulated_unique_sources)}")
        logger.info(f"Accumulated unique chunks so far: {len(self.accumulated_unique_chunks)}")

        # Only proceed with expanded query if it exists and is different
        if has_expanded_query and self.expanded_query != query:
            logger.info(f"Searching with expanded query: '{self.expanded_query}'")

            # Calculate dynamic max_k using the accumulated unique chunks
            dynamic_max_k = len(self.accumulated_unique_chunks) + top_k
            search_top_k = min(dynamic_max_k, self.max_k)
            logger.info(f"Dynamic max_k for expanded query: {len(self.accumulated_unique_chunks)} accumulated unique chunks + {top_k} top_k = {dynamic_max_k}, limited by max_k={self.max_k}")

            # Log the current state of seen_chunks before expanded query search
            logger.info(f"Before expanded query search: seen_chunks contains {len(self.seen_chunks)} items")

            # Store the current seen_chunks for comparison
            seen_chunks_before = set(self.seen_chunks)

            expanded_results = self._search_and_analyze(
                self.expanded_query,
                top_k=top_k,
                context_filter=context_filter,
                is_follow_up=True,  # This should be True only for expanded query
                pre_calculated_max_k=search_top_k
            )

            # Calculate new chunks added during expanded query search
            new_chunks = self.seen_chunks - seen_chunks_before
            logger.info(f"Expanded query search added {len(new_chunks)} new chunks to seen_chunks")
            if new_chunks:
                logger.info(f"New chunks added: {new_chunks}")

            # Log the state of all_findings before adding expanded results
            logger.info(f"Before adding expanded results, all_findings contains {len(self.all_findings)} items")

            # Check if any chunks in new_chunks are already in all_findings
            for chunk_id in new_chunks:
                found = False
                for finding in self.all_findings:
                    finding_chunk_id = self._get_chunk_identifier(finding)
                    if finding_chunk_id == chunk_id:
                        found = True
                        logger.info(f"Chunk {chunk_id} is already in all_findings")
                        break
                if not found:
                    logger.info(f"Chunk {chunk_id} is NOT in all_findings - this is unexpected if it's in seen_chunks")

            if expanded_results:
                logger.info(f"Found {len(expanded_results)} additional unique results from expanded query")
                # Log details of the expanded results and update accumulated unique sources and chunks
                new_sources_from_expanded = set()
                new_chunks_from_expanded = set()

                # First, log the current accumulated unique sources and chunks
                logger.info(f"Current accumulated unique sources before processing expanded results: {self.accumulated_unique_sources}")
                logger.info(f"Current accumulated unique chunks before processing expanded results: {self.accumulated_unique_chunks}")

                for i, result in enumerate(expanded_results):
                    chunk_id = self._get_chunk_identifier(result)
                    source = result.get("file_path", "")

                    # Log normalized score
                    logger.info(f"Expanded result {i+1}: chunk_id={chunk_id}, score={result.get('score', 'N/A')}, file_path={source}")
                    logger.info(f"Expanded result {i+1} content preview: {safe_str(result.get('content', 'No content'))[:100]}...")

                    # Track unique sources
                    if source:
                        is_new_source = source not in self.accumulated_unique_sources
                        logger.info(f"Source '{source}' is {'new' if is_new_source else 'already in accumulated sources'}")

                        if is_new_source:
                            new_sources_from_expanded.add(source)

                        # Add to accumulated sources regardless
                        self.accumulated_unique_sources.add(source)

                    # Track unique chunks
                    is_new_chunk = chunk_id not in self.accumulated_unique_chunks
                    logger.info(f"Chunk '{chunk_id}' is {'new' if is_new_chunk else 'already in accumulated chunks'}")

                    if is_new_chunk:
                        new_chunks_from_expanded.add(chunk_id)

                    # Add to accumulated chunks regardless
                    self.accumulated_unique_chunks.add(chunk_id)

                logger.info(f"Found {len(new_sources_from_expanded)} new unique sources from expanded query")
                logger.info(f"Found {len(new_chunks_from_expanded)} new unique chunks from expanded query")
                logger.info(f"Accumulated unique sources after expanded query: {len(self.accumulated_unique_sources)}")
                logger.info(f"Accumulated unique chunks after expanded query: {len(self.accumulated_unique_chunks)}")

                # Add expanded results to iteration_findings
                iteration_findings.extend(expanded_results)

                # Add expanded results to all_findings immediately
                self.all_findings.extend(expanded_results)
                logger.info(f"Added {len(expanded_results)} results from expanded query to all_findings")
            else:
                logger.info("No additional unique results found from expanded query")

            # Log the state of all_findings after adding expanded results
            logger.info(f"After adding expanded results, all_findings contains {len(self.all_findings)} items")

        # Since we've already added original results to all_findings, we only need to update knowledge
        self._update_current_knowledge()

        # Log the total findings from initial search
        logger.info(f"Total findings from initial search: {len(iteration_findings)}")
        logger.info(f"Final all_findings count after initial search: {len(self.all_findings)}")

    # --- DEPRECATED: Kept for backward compatibility ---
    def _verify_findings(self):
        """
        DEPRECATED: Use _verify_pooled_findings instead.
        This method is kept for backward compatibility.
        """
        logger.warning("_verify_findings is deprecated. Use _verify_pooled_findings instead.")
        # No-op - verification now happens after pooling
        pass

    def _verify_pooled_findings(self, pooled_findings):
        """
        Verify pooled findings for consistency and accuracy using fact_verification agent.
        This method verifies only unique findings after deduplication to avoid redundant verification.

        Args:
            pooled_findings: List of deduplicated findings to verify
        """
        verification_start_time = time.time()

        if not DSPY_CONFIGURED:
            logger.warning("Skipping finding verification: DSPy not configured.")
            return

        # Log the number of findings to verify
        logger.info(f"Verifying {len(pooled_findings)} deduplicated findings")
        if not pooled_findings:
            logger.warning("No pooled findings to verify. This might indicate an issue with the search or analysis.")
            return

        try:
            fact_verification_agent = DSPyAgentRegistry.get_agent("fact_verification")
            if fact_verification_agent:
                logger.info("Verifying deduplicated findings using DSPy FactVerificationAgent...")
                verified_count = 0
                for i, finding in enumerate(pooled_findings):
                    try:
                        content = finding.get("content", "")
                        summary = finding.get("summary", "")
                        if not content or not summary:
                            logger.warning(f"Finding {i} missing content or summary, skipping verification")
                            continue # Skip

                        # Check if this finding has already been verified
                        if "verification" in finding:
                            logger.debug(f"Finding {i} already verified, skipping")
                            verified_count += 1
                            continue

                        agent_inputs = {"content": content, "summary": summary}
                        # Use helper
                        prepared_data = DSPyAgentRegistry._ensure_required_fields("fact_verification", fact_verification_agent, agent_inputs)
                        result = fact_verification_agent(**prepared_data)

                        # Access attributes safely
                        is_consistent = getattr(result, "is_consistent", True)
                        confidence = getattr(result, "confidence", 0.5)
                        notes = getattr(result, "notes", "")

                        finding["verification"] = {
                            "is_consistent": is_consistent,
                            "confidence": confidence,
                            "notes": notes
                        }
                        verified_count += 1
                    except Exception as verify_e:
                        logger.warning(f"Error verifying finding {finding.get('citation', {}).get('source_path', '?')}: {verify_e}")

                verification_time = time.time() - verification_start_time
                self.performance_metrics["verification_time"] = verification_time

                logger.info(f"Verified {verified_count}/{len(pooled_findings)} deduplicated findings in {verification_time:.2f}s")

                # Log efficiency metrics
                if hasattr(self, 'all_findings') and self.all_findings:
                    total_findings = len(self.all_findings)
                    savings = total_findings - len(pooled_findings)
                    if savings > 0:
                        savings_percent = (savings / total_findings) * 100
                        logger.info(f"Verification efficiency: Avoided {savings} duplicate verifications ({savings_percent:.1f}% reduction)")
            else:
                logger.warning("Fact verification agent not found. Skipping verification.")
        except Exception as e:
            logger.error(f"Error setting up finding verification: {e}", exc_info=True)

    # --- MODIFY _expand_query ---
    def _expand_query(self, query: str) -> str:
        """Expand the query using the query expander."""
        try:
            # Use the QueryExpander instance directly
            expanded = self.query_expander.expand_query(query)
            self.expanded_query = expanded # Store for potential use
            return expanded
        except Exception as e:
            logger.error(f"Error expanding query: {e}", exc_info=True)
            self.expanded_query = query # Store original if expansion fails
            return query

    def _pool_all_context(self) -> Dict[str, Any]:
        """
        Pool and deduplicate context from all questions across all iterations.

        Returns:
            Dictionary with pooled context information
        """
        logger.info("Pooling and deduplicating context from all questions")

        # Get all unique questions
        all_questions = set()
        for iteration_questions in self.questions_by_iteration.values():
            for question in iteration_questions:
                all_questions.add(question)

        # Pool all findings
        pooled_findings = []
        seen_chunk_ids = set()

        # First, collect all findings from all questions
        for question in all_questions:
            if question in self.findings_by_question:
                for finding in self.findings_by_question[question]:
                    # Create a unique identifier for this chunk
                    source_path = finding.get("file_path", "")
                    source_name = finding.get("source_name", "")
                    start_pos = finding.get("start", 0)
                    end_pos = finding.get("end", 0)
                    chunk_id = f"{source_path}|{source_name}|{start_pos}|{end_pos}"

                    # Only add if we haven't seen this chunk before
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        pooled_findings.append(finding)

        # Sort pooled findings by score (descending)
        pooled_findings = sorted(pooled_findings, key=lambda x: x.get("score", 0), reverse=True)

        # Pool all context
        pooled_context = []
        for finding in pooled_findings:
            content = finding.get("content", "")
            if content:
                pooled_context.append(content)

        # Create a summary of the pooling with enhanced metrics
        total_findings_before = sum(len(findings) for findings in self.findings_by_question.values())
        total_findings_after = len(pooled_findings)
        duplication_rate = ((total_findings_before - total_findings_after) / total_findings_before * 100) if total_findings_before > 0 else 0

        # Track sources and chunks by question for detailed analysis
        sources_by_question = {}
        chunks_by_question = {}
        for question, findings in self.findings_by_question.items():
            # Track sources (file paths)
            sources_by_question[question] = set(finding.get("file_path", "") for finding in findings)

            # Track chunks (using chunk identifiers)
            chunks_by_question[question] = set(self._get_chunk_identifier(finding) for finding in findings)

        # Calculate source and chunk overlap between questions
        question_overlap = {}
        all_questions_list = list(all_questions)
        for i, q1 in enumerate(all_questions_list):
            if q1 not in sources_by_question or q1 not in chunks_by_question:
                continue
            for q2 in all_questions_list[i+1:]:
                if q2 not in sources_by_question or q2 not in chunks_by_question:
                    continue

                # Calculate source overlap
                common_sources = sources_by_question[q1].intersection(sources_by_question[q2])
                source_overlap_percent = len(common_sources) / max(len(sources_by_question[q1]), len(sources_by_question[q2])) * 100 if sources_by_question[q1] and sources_by_question[q2] else 0

                # Calculate chunk overlap
                common_chunks = chunks_by_question[q1].intersection(chunks_by_question[q2])
                chunk_overlap_percent = len(common_chunks) / max(len(chunks_by_question[q1]), len(chunks_by_question[q2])) * 100 if chunks_by_question[q1] and chunks_by_question[q2] else 0

                if common_sources or common_chunks:
                    overlap_key = f"{q1} ↔ {q2}"
                    question_overlap[overlap_key] = {
                        "common_sources": list(common_sources),
                        "source_overlap_percent": source_overlap_percent,
                        "common_chunks": list(common_chunks),
                        "chunk_overlap_percent": chunk_overlap_percent
                    }

                    # Log high overlap as a warning
                    if source_overlap_percent > 75:
                        logger.warning(f"High source overlap ({source_overlap_percent:.1f}%) between questions: '{q1}' and '{q2}'")
                    if chunk_overlap_percent > 75:
                        logger.warning(f"High chunk overlap ({chunk_overlap_percent:.1f}%) between questions: '{q1}' and '{q2}'")

        pooling_summary = {
            "total_questions": len(all_questions),
            "total_findings_before_deduplication": total_findings_before,
            "total_findings_after_deduplication": total_findings_after,
            "duplication_rate": duplication_rate,
            "total_context_chunks": len(pooled_context),
            "questions": list(all_questions),
            "overlap_analysis": question_overlap,
            "sources_by_question": {q: list(sources) for q, sources in sources_by_question.items()},
            "chunks_by_question": {q: list(chunks) for q, chunks in chunks_by_question.items()}
        }

        logger.info(f"Pooled context summary: {pooling_summary['total_findings_after_deduplication']} unique findings from {pooling_summary['total_questions']} questions (duplication rate: {duplication_rate:.1f}%)")

        # Ensure we have at least one item in pooled_context to avoid empty context errors
        if not pooled_context:
            pooled_context = ["No specific content found for the questions."]

        # Create a combined context string for DSPy agents
        combined_context = "\n\n".join(pooled_context[:5])  # Limit to first 5 chunks to avoid token limits

        # Create a dictionary with essential fields
        result = {
            "findings": pooled_findings,
            "context": pooled_context,
            "summary": pooling_summary,
            "content": combined_context,
            "questions_by_iteration": self.questions_by_iteration  # Ensure questions_by_iteration is included
        }

        # Log the questions_by_iteration to verify it contains the expected data
        logger.info(f"Pooled context questions_by_iteration: {self.questions_by_iteration}")

        # Make sure each finding has content
        for finding in pooled_findings:
            if "content" not in finding or not finding["content"]:
                finding["content"] = "No content available"

        return result

    def _get_chunk_identifier(self, result: Dict) -> str:
        """
        Create a unique identifier for a chunk based on its source information.

        Args:
            result: The result dictionary containing source information

        Returns:
            A string identifier for the chunk
        """
        source_path = result.get("file_path", "")
        source_name = result.get("source_name", "")
        start_pos = result.get("start", 0)
        end_pos = result.get("end", 0)
        node_id = result.get("node_id", "")

        # Create a unique identifier - prefer node_id if available
        if node_id:
            return f"node:{node_id}"
        return f"{source_path}|{source_name}|{start_pos}|{end_pos}"

    def _deduplicate_results(self, results: List[Dict], seen_chunks: set = None) -> List[Dict]:
        """
        Deduplicate search results based on chunk identifiers.

        Args:
            results: List of search results to deduplicate
            seen_chunks: Optional set of already seen chunk identifiers

        Returns:
            Deduplicated list of search results
        """
        if seen_chunks is None:
            seen_chunks = set()

        # Log the current state of seen_chunks before deduplication
        logger.info(f"Before deduplication: seen_chunks contains {len(seen_chunks)} items")

        # Log all chunk IDs that will be processed
        all_chunk_ids = []
        for result in results:
            chunk_id = self._get_chunk_identifier(result)
            all_chunk_ids.append(chunk_id)

        logger.info(f"Processing {len(all_chunk_ids)} chunk IDs for deduplication: {all_chunk_ids}")

        deduplicated = []
        duplicate_count = 0
        for result in results:
            chunk_id = self._get_chunk_identifier(result)
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                deduplicated.append(result)
                # Log normalized score
                logger.info(f"Added unique chunk: {chunk_id}, score={result.get('score', 'N/A')}, file_path={result.get('file_path', 'N/A')}")
            else:
                duplicate_count += 1
                # Log normalized score
                logger.info(f"Skipping duplicate chunk: {chunk_id}, score={result.get('score', 'N/A')}, file_path={result.get('file_path', 'N/A')}")

        logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)} unique results (skipped {duplicate_count} duplicates)")
        return deduplicated

    def _filter_by_relevance(self, results: List[Dict], threshold: float) -> List[Dict]:
        """
        Filter search results by relevance score.

        Args:
            results: List of search results to filter
            threshold: Minimum relevance score threshold

        Returns:
            Filtered list of search results
        """
        if not results:
            return []

        # Log all scores before filtering
        all_scores = [(i, r.get("score", 0), r.get("file_path", "N/A")) for i, r in enumerate(results)]
        logger.info(f"Before relevance filtering, scores (idx, score, path): {all_scores}")

        # Filter results
        filtered = []
        filtered_out = []

        for r in results:
            # Use the normalized score (should be between 0 and 1)
            score = r.get("score", 0)

            if score >= threshold:
                filtered.append(r)
                logger.debug(f"Keeping result with score={score} (>= threshold {threshold}): {r.get('file_path', 'N/A')}")
            else:
                filtered_out.append((score, r.get("file_path", "N/A")))
                logger.debug(f"Filtering out result with score={score} (< threshold {threshold}): {r.get('file_path', 'N/A')}")

        logger.info(f"Filtered {len(results)} results to {len(filtered)} results with score >= {threshold}")

        # Log details about filtered out results
        if filtered_out:
            logger.info(f"Filtered out {len(filtered_out)} results with scores below threshold: {filtered_out}")

        return filtered

    def _limit_final_context(self, max_size: int) -> None:
        """
        Limit the final context to a maximum size while preserving the most relevant findings.

        Args:
            max_size: Maximum size in characters for the final context
        """
        if not self.all_findings:
            return

        # Sort findings by relevance (if available) or by iteration (newer is better)
        def get_finding_score(finding):
            # Check for score directly in the finding
            if "score" in finding:
                return finding["score"]

            # If the finding has a relevance score, use it
            if "relevance" in finding:
                return finding["relevance"]

            # If the finding has a citation with a score, use it
            citation = finding.get("citation", {})
            if "score" in citation:
                return citation["score"]

            # Otherwise, use the position in the list (newer findings are better)
            return 0

        # Sort findings by score (higher is better)
        sorted_findings = sorted(self.all_findings, key=get_finding_score, reverse=True)

        # Calculate the total size
        total_size = sum(len(f.get("content", "")) for f in sorted_findings)

        if total_size <= max_size:
            # No need to limit
            return

        # Keep adding findings until we reach the limit
        kept_findings = []
        current_size = 0

        for finding in sorted_findings:
            content_size = len(finding.get("content", ""))

            # If adding this finding would exceed the limit, skip it
            if current_size + content_size > max_size:
                continue

            # Add the finding
            kept_findings.append(finding)
            current_size += content_size

            # If we've reached the limit, stop
            if current_size >= max_size * 0.9:  # 90% of max size
                break

        # Update the findings list
        self.all_findings = kept_findings

        # Update the current knowledge
        self._update_current_knowledge()

        logger.info(f"Limited context to {len(kept_findings)} findings ({current_size} characters)")
