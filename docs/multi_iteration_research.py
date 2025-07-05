"""
Multi-iteration research for Local File Deep Research.

This module provides functionality for conducting multi-iteration research
with query expansion and follow-up questions based on collected context.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from collections import Counter

from .vector_store import VectorStore
from .document_analysis import DocumentAnalyzer
from .advanced_search import QueryExpander
from .dspy_config import DSPyAgentRegistry, DSPY_CONFIGURED
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

    def __init__(self, vector_store: VectorStore, max_iterations: int = 3, questions_per_iteration: int = 2, max_context_size: int = 10000):
        """
        Initialize the multi-iteration research system.

        Args:
            vector_store: Vector store for searching documents
            max_iterations: Maximum number of research iterations
            questions_per_iteration: Number of follow-up questions per iteration
            max_context_size: Maximum size of the context in characters (default: 10000)
        """
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        self.questions_per_iteration = questions_per_iteration
        self.max_context_size = max_context_size
        self.document_analyzer = DocumentAnalyzer()
        self.query_expander = QueryExpander()



        # Performance tracking
        self.performance_metrics = {
            "search_time": 0,
            "analysis_time": 0,
            "question_generation_time": 0,
            "total_time": 0
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

    def run(self, query: str, top_k: int = 5, context_filter: str = None) -> Dict[str, Any]:
        """
        Run multi-iteration research.

        Args:
            query: Original research query
            top_k: Number of results to retrieve per question
            context_filter: Optional filter for results
            session_id: Optional session ID for loading vector store

        Returns:
            Dictionary with research findings and metadata
        """
        start_time = time.time()
        logger.info(f"Starting multi-iteration research for query: '{query}'")

        # Reset state
        self.questions_by_iteration = {}
        self.all_findings = []
        self.current_knowledge = ""
        self.seen_chunks = set()
        self.findings_by_question = {}
        self.context_by_question = {}
        self.performance_metrics = {"search_time": 0, "analysis_time": 0, "question_generation_time": 0} # Reset timings

        # Initial query expansion
        expanded_query = self.query_expander.expand_query(query)
        logger.info(f"Expanded query: '{expanded_query}'")
        self._perform_initial_search(query, top_k, context_filter) # Uses expanded_query internally


        # First iteration: search with original and expanded query
        iteration_findings = []

        # Search with original query
        original_results = self._search_and_analyze(
            query,
            top_k=top_k,
            context_filter=context_filter
        )
        iteration_findings.extend(original_results)

        # Search with expanded query if different
        if expanded_query != query:
            expanded_results = self._search_and_analyze(
                expanded_query,
                top_k=top_k,
                context_filter=context_filter
            )
            # Results from expanded_results are already deduplicated in _search_and_analyze
            # Just add them to the findings
            iteration_findings.extend(expanded_results)

        # Update findings and knowledge
        self.all_findings.extend(iteration_findings)
        self._update_current_knowledge()

        # Store first iteration questions
        self.questions_by_iteration['0'] = [query, expanded_query]

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
                results = self._search_and_analyze(
                    question,
                    top_k=top_k,
                    context_filter=context_filter
                )
                iteration_findings.extend(results)

            # Update findings and knowledge
            self.all_findings.extend(iteration_findings)
            self._update_current_knowledge()

            # Check if we have enough information
            if self._check_sufficient_information():
                logger.info(f"Sufficient information gathered after iteration {iteration+1}")
                break

        # Final verification and fact-checking
        self._verify_findings()

        # Limit the final context if it's too large
        max_final_context_size = 64000  # Maximum size for the final context
        if sum(len(f.get("content", "")) for f in self.all_findings) > max_final_context_size:
            logger.info(f"Limiting final context to approximately {max_final_context_size} characters")
            self._limit_final_context(max_final_context_size)

        # Calculate total time
        total_time = time.time() - start_time
        self.performance_metrics["total_time"] = total_time

        logger.info(f"Multi-iteration research complete in {total_time:.2f} seconds")

        # Pool all context from all questions for enhanced reporting
        all_pooled_context = self._pool_all_context()
        # Calculate total time
        total_time = time.time() - start_time
        self.performance_metrics["total_time"] = total_time
        logger.info(f"Multi-iteration research phase complete in {total_time:.2f} seconds")

        # --- Return data for ReportGenerator, NOT the report itself ---
        return {
            "findings": self.all_findings, # Raw findings from all iterations
            "pooled_context": all_pooled_context, # Pooled/deduplicated context and findings
            "questions_by_iteration": self.questions_by_iteration,
            "iterations": len(self.questions_by_iteration),
            "performance_metrics": self.performance_metrics,
            # Removed report generation from here
        }

    def _search_and_analyze(self, query: str, top_k: int, context_filter: str = None) -> List[Dict[str, Any]]:
        """
        Search for documents and analyze them.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            context_filter: Optional filter for results
            session_id: Optional session ID for loading vector store

        Returns:
            List of analyzed search results
        """
        # Search
        search_start_time = time.time()

        # Get query embedding
        from .pipeline import embed_text
        query_embedding = embed_text(query)

        # Log with safe_str to handle Unicode characters
        logger.info(f"Searching for query: {safe_str(query)}")

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Apply context filter if provided
        if context_filter:
            logger.info(f"Applying context filter: {safe_str(context_filter)}")
            results = [r for r in results if context_filter.lower() in safe_str(r.get("file_path", "")).lower()]

        search_time = time.time() - search_start_time
        self.performance_metrics["search_time"] += search_time

        # Analyze results
        analysis_start_time = time.time()
        analyzed_results = []

        for result in results:
            try:
                # Get source information for chunk-level deduplication
                source_path = result.get("file_path", "")
                source_name = result.get("source_name", "")
                start_pos = result.get("start", 0)
                end_pos = result.get("end", 0)

                # Create a unique chunk identifier
                chunk_id = f"{source_path}|{source_name}|{start_pos}|{end_pos}"

                # Skip if we've already seen this exact chunk
                if chunk_id in self.seen_chunks:
                    continue

                # Mark this chunk as seen
                self.seen_chunks.add(chunk_id)

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
            session_id: Optional session ID for loading vector store
        """
        # First iteration: search with original and expanded query
        iteration_findings = []

        # Search with original query
        original_results = self._search_and_analyze(
            query,
            top_k=top_k,
            context_filter=context_filter
        )
        iteration_findings.extend(original_results)

        # Search with expanded query if different
        if hasattr(self, 'expanded_query') and self.expanded_query != query:
            expanded_results = self._search_and_analyze(
                self.expanded_query,
                top_k=top_k,
                context_filter=context_filter
            )
            # Results from expanded_results are already deduplicated in _search_and_analyze
            # Just add them to the findings
            iteration_findings.extend(expanded_results)

        # Update findings and knowledge
        self.all_findings.extend(iteration_findings)
        self._update_current_knowledge()

        # Store first iteration questions
        self.questions_by_iteration['0'] = [query]
        if hasattr(self, 'expanded_query') and self.expanded_query != query:
            self.questions_by_iteration['0'].append(self.expanded_query)

    # --- MODIFY _verify_findings ---
    def _verify_findings(self):
        """Verify findings for consistency and accuracy using fact_verification agent."""
        # +++ ADD CHECK +++
        if not DSPY_CONFIGURED:
             logger.warning("Skipping finding verification: DSPy not configured.")
             return

        try:
            fact_verification_agent = DSPyAgentRegistry.get_agent("fact_verification")
            if fact_verification_agent:
                logger.info("Verifying findings using DSPy FactVerificationAgent...")
                verified_count = 0
                for finding in self.all_findings:
                    try:
                         content = finding.get("content", "")
                         summary = finding.get("summary", "")
                         if not content or not summary: continue # Skip

                         agent_inputs = {"content": content, "summary": summary}
                         # Use helper
                         prepared_data = DSPyAgentRegistry._ensure_required_fields("fact_verification", fact_verification_agent, agent_inputs)
                         result = fact_verification_agent(**prepared_data)

                         # +++ MODIFY: Access attributes safely +++
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
                logger.info(f"Verified {verified_count}/{len(self.all_findings)} findings.")
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

        # Create a summary of the pooling
        pooling_summary = {
            "total_questions": len(all_questions),
            "total_findings_before_deduplication": sum(len(findings) for findings in self.findings_by_question.values()),
            "total_findings_after_deduplication": len(pooled_findings),
            "total_context_chunks": len(pooled_context),
            "questions": list(all_questions)
        }

        logger.info(f"Pooled context summary: {pooling_summary['total_findings_after_deduplication']} unique findings from {pooling_summary['total_questions']} questions")

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
            "content": combined_context
        }

        # Make sure each finding has content
        for finding in pooled_findings:
            if "content" not in finding or not finding["content"]:
                finding["content"] = "No content available"

        return result

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