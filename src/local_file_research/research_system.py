"""
ResearchSystem for standalone deep research, RAG, and report generation.
Supports user-selectable RAG or multi-iteration research, and report modes (normal, chain_of_thought, enhanced).
"""

from .pipeline import search_index
from .document_analysis import DocumentAnalyzer
from .dspy_config import DSPyAgentRegistry, DSPY_CONFIGURED #initialize_dspy
from .vector_store import VectorStore
from .advanced_reporting import AdvancedReportGenerator
from .multi_iteration_research import MultiIterationResearch # Import MultiIterationResearch
from .serialization_utils import make_json_serializable
from .llamaindex_vector_store import LlamaIndexVectorStore
from typing import List, Dict, Any, Optional, Union
import logging
import time
import json
import traceback # Import traceback
import socket # Import socket for timeout

# Get logger instance
logger = logging.getLogger(__name__)

class CitationHandler:
    @staticmethod
    def format_sources(results: List[Dict]) -> List[Dict]:
        """Formats sources for citation with deduplication."""
        seen_chunks = {}
        deduplicated_sources = []
        ref_counter = 1

        # Filter out None values from results
        filtered_results = [r for r in results if r is not None]
        if len(filtered_results) < len(results):
            logger.warning(f"Filtered out {len(results) - len(filtered_results)} None values from results list")

        # Sort results by score if available, to prioritize more relevant citations
        try:
             results_sorted = sorted(filtered_results, key=lambda x: x.get("score", 0.0), reverse=True)
        except Exception as e:
             logger.warning(f"Error sorting results: {e}. Using unsorted results.")
             results_sorted = filtered_results # Fallback if sorting fails

        for r in results_sorted:
            try:
                citation = r.get("citation", {}) or {}  # Ensure citation is at least an empty dict
                if citation is None:
                    citation = {}

                source_path = citation.get("source_path")
                source_name = citation.get("source_name")
                start_pos = citation.get("start", 0) or 0  # Ensure we have a number
                end_pos = citation.get("end", 0) or 0  # Ensure we have a number
                score = r.get("score", citation.get("score", 0.0)) # Prefer score from result itself

                # Create a unique key for this chunk - handle potential None values
                path_part = str(source_path) if source_path is not None else ""
                name_part = str(source_name) if source_name is not None else ""
                chunk_key = f"{path_part}|{name_part}|{start_pos}|{end_pos}"

                # Skip if we've already seen this exact chunk
                if chunk_key in seen_chunks:
                    # Add reference number back to the original finding dict if duplicate
                    if "reference_number" not in r and chunk_key in seen_chunks:
                         r["reference_number"] = seen_chunks[chunk_key]["reference_number"]
                    continue

                # Estimate page/line (simple estimation)
                page_num = (start_pos // 3000) + 1 if start_pos is not None and start_pos > 0 else 1
                line_num = ((start_pos % 3000) // 80) + 1 if start_pos is not None and start_pos > 0 else 1

                source_info = {
                    "source_path": source_path,
                    "source_name": source_name,
                    "source_type": citation.get("source_type"),
                    "start": start_pos,
                    "end": end_pos,
                    "score": score,
                    "reference_number": ref_counter, # Use incremental reference number
                    "page": page_num,
                    "line": line_num
                }
                deduplicated_sources.append(source_info)
                seen_chunks[chunk_key] = source_info # Store the added source info
                # Add reference number back to the original finding dict for linking
                r["reference_number"] = ref_counter
                ref_counter += 1
            except Exception as e:
                logger.error(f"Error processing citation in format_sources: {e}")
                # Still increment counter to maintain consistent numbering
                ref_counter += 1


        return deduplicated_sources

    @staticmethod
    def inject_citations(text: str, sources: List[Dict]) -> str:
        """Injects reference numbers into the text (simple placeholder)."""
        # This is a placeholder. Robust implementation is complex.
        # We currently add references at the end of summaries/conclusions.
        refs = ", ".join(sorted([f"[{src['reference_number']}]" for src in sources]))
        if refs:
            # Avoid modifying the core text for now, just indicate sources were used.
            # Return text + f" (Sources: {refs})"
            return text # Keep text unmodified for now
        return text

    @staticmethod
    def format_finding_for_report(finding: Dict, index: int, agent_chain_name: str = None) -> str:
        """Formats a single finding for inclusion in the report body."""
        report_part = ""
        ref = finding.get('reference_number', index) # Use assigned ref num or index
        # Safely get summary with type checking
        analysis = finding.get('analysis', {})
        analysis_summary = ""
        if isinstance(analysis, dict):
            analysis_summary = analysis.get('summary', "")

        summary = (
            analysis_summary or
            finding.get('summary') or
            finding.get('answer') or
            finding.get('info') or
            finding.get('content', '')[:200] + "..."
        )

        report_part += f"### Finding {ref}\n" # Use reference number
        report_part += f"{summary}\n\n"

        # Add analysis details if available
        analysis = finding.get('analysis', {})
        if analysis:
            # Handle both dictionary and string analysis
            if isinstance(analysis, dict):
                key_points = analysis.get('key_points', [])
                if key_points:
                    report_part += "**Key Points:**\n"
                    for point in key_points[:5]:  # Limit points
                        report_part += f"- {point}\n"
                    report_part += "\n"
            elif isinstance(analysis, str) and analysis.strip():
                # If analysis is a non-empty string, display it directly
                report_part += f"**Analysis:** {analysis}\n\n"

        # Add agent chain results if available
        # These are the fields that were previously in the Agent Chain Results section
        agent_fields = {
            "source_type": "Source Type",
            "chain_status": "Chain Status",
            "interpretation": "Interpretation",
            "insights": "Insights",
            "limitations": "Limitations",
            "recommendations": "Recommendations",
            "next_steps": "Next Steps",
            "alternatives": "Alternatives",
            "rationale": "Rationale",
            "details": "Details",
            "challenges": "Challenges",
            "solutions": "Solutions"
        }

        for field, title in agent_fields.items():
            if field in finding and finding[field]:
                value = finding[field]
                report_part += f"**{title}:**\n"

                if isinstance(value, list):
                    # Filter out single-character items (likely from DSPy agent)
                    filtered_items = []
                    for item in value:
                        if isinstance(item, str) and len(item.strip()) <= 1:
                            # Skip single-character items
                            continue
                        filtered_items.append(item)

                    for item in filtered_items:
                        report_part += f"- {item}\n"
                elif isinstance(value, str):
                    # If it's a string, check if it contains newlines or commas
                    if '\n' in value:
                        for line in value.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('-'):
                                report_part += f"- {line}\n"
                            elif line:
                                report_part += f"{line}\n"
                    else:
                        report_part += f"{value}\n"
                else:
                    report_part += f"{value}\n"

                report_part += "\n"

        # Add article information if available and we're not using comprehensive_analysis agent chain
        # (which will generate a comprehensive article at the end)
        if "article" in finding and finding["article"] and not (agent_chain_name == "comprehensive_analysis"):
            report_part += "**Article:**\n"
            report_part += f"{finding['article'][:300]}...\n\n"

            # Add article metadata
            if "article_type" in finding:
                report_part += f"**Article Type:** {finding['article_type']}\n"

            # Handle key themes with proper filtering
            if "key_themes" in finding and finding["key_themes"]:
                report_part += "**Key Themes:**\n"
                themes = finding["key_themes"]

                if isinstance(themes, list):
                    # Filter out single-character themes
                    filtered_themes = [theme for theme in themes if not (isinstance(theme, str) and len(theme.strip()) <= 1)]
                    for theme in filtered_themes:
                        report_part += f"- {theme}\n"
                elif isinstance(themes, str):
                    # If it's a string, split by newlines or commas
                    if '\n' in themes:
                        for line in themes.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('-'):
                                report_part += f"- {line}\n"
                            elif line:
                                report_part += f"{line}\n"
                    else:
                        for theme in themes.split(','):
                            theme = theme.strip()
                            if theme:
                                report_part += f"- {theme}\n"

            if "word_count" in finding:
                report_part += f"**Word Count:** {finding['word_count']}\n"
            report_part += "\n"

        # Add verification information if available
        verification = finding.get('verification', {})
        if verification:
            confidence = verification.get('confidence', 0.5)
            # Ensure confidence is float
            try: confidence = float(confidence)
            except (ValueError, TypeError): confidence = 0.5

            is_consistent = verification.get('is_consistent', True)
            notes = verification.get('notes', '')

            if is_consistent:
                report_part += f"**Verification:** ✅ Consistent (Confidence: {confidence:.2f})\n"
            else:
                report_part += f"**Verification:** ⚠️ Potential inconsistency (Confidence: {confidence:.2f})\n"

            if notes:
                report_part += f"**Notes:** {notes}\n"
            report_part += "\n"

        # Add Source Info (using reference number)
        source_name = finding.get('citation', {}).get('source_name', finding.get('citation', {}).get('source_path', 'Unknown Source'))
        report_part += f"**Source:** {source_name} [Ref: {ref}]\n"

        # Add Related Questions if available
        if "related_questions" in finding and finding["related_questions"]:
            report_part += "**Related Questions:**\n"
            for related_query in finding["related_questions"]:
                report_part += f"- {related_query}\n"
            report_part += "\n"
        else:
            report_part += "\n"

        return report_part

class ReportGenerator:
    def __init__(self, mode: str = "normal"):
        """Initialize the report generator."""
        self.mode = mode
        self.advanced_reporter = AdvancedReportGenerator()
        logger.info(f"Initialized ReportGenerator with mode: {mode}")

    def generate_report(self, findings: List[Dict], query: str, research_mode: str = "rag", pooled_context: Optional[Dict] = None, agent_chain_name: Optional[str] = None) -> Dict:
        """Generate a report from the findings."""
        logger.info(f"Generating report. Mode: {self.mode}, Research Mode: {research_mode}, Agent Chain: {agent_chain_name}, Findings: {len(findings)}, Pooled Context Provided: {pooled_context is not None}")

        # Determine the primary set of findings to use for the report
        report_findings = findings
        if pooled_context and research_mode == "multi_iteration":
            pooled_findings_list = pooled_context.get("findings", [])
            if pooled_findings_list:
                logger.info(f"Using {len(pooled_findings_list)} pooled/deduplicated findings for report generation.")
                report_findings = pooled_findings_list
            else:
                 logger.warning("Pooled context provided but contains no findings. Using original findings (if any).")

        if not report_findings:
            logger.warning("No findings available to generate report.")
            return {
                "report": f"# Report for: '{query}'\n\nNo findings available to generate a report.",
                "sources": [],
                "original_findings": findings # Return original findings even if report fails
            }

        final_findings = self._deduplicate_findings(report_findings)
        logger.info(f"Findings count for report generation after deduplication: {len(final_findings)}")

        if not final_findings:
             logger.warning("No unique findings remaining after deduplication.")
             return {
                 "report": f"# Report for: '{query}'\n\nNo unique findings available to generate a report.",
                 "sources": [],
                 "original_findings": findings # Return original findings
             }

        sources = CitationHandler.format_sources(final_findings) # Assigns reference_number

        report_content = ""
        sections_added = []

        # --- Collect Agent Chain Outputs (If Any) ---
        # This part remains largely the same, but we'll integrate it differently below
        agent_chain_outputs = {}
        if agent_chain_name:
            logger.info(f"Collecting outputs from agent chain: {agent_chain_name}")
            for finding in final_findings:
                for key, value in finding.items():
                    if key in ["content", "score", "citation", "reference_number", "node_id", "source_name", "file_path", "summary", "analysis", "query", "query_info", "confidence", "verification"]:
                        continue
                    if key not in agent_chain_outputs: agent_chain_outputs[key] = []
                    if value:
                        if isinstance(value, (str, list, dict)):
                            agent_chain_outputs[key].append(value)
            logger.info(f"Found agent chain outputs: {list(agent_chain_outputs.keys())}")


        # --- Build Report Content ---
        try:
            # Basic report header
            report_header = f"# Report for: '{query}'\n\n## Query\n{query}\n\n"
            if self.mode == "enhanced": report_header = f"# Enhanced Report for: '{query}'\n\n## Query\n{query}\n\n"
            elif self.mode == "chain_of_thought": report_header = f"# Chain-of-Thought Report for: '{query}'\n\n## Query\n{query}\n\n"

            # Add expanded query and follow-up questions from multi-iteration research if available
            if pooled_context and research_mode == "multi_iteration":
                questions_by_iteration = pooled_context.get("questions_by_iteration", {})
                logger.info(f"Report generation - questions_by_iteration from pooled_context: {questions_by_iteration}")

                if questions_by_iteration:  # If we have any questions
                    # Find sources for each query to show reference numbers
                    query_to_sources = {}
                    for finding in final_findings:
                        query = finding.get("query", "")
                        ref_num = finding.get("reference_number", "")
                        if query and ref_num:
                            if query not in query_to_sources:
                                query_to_sources[query] = []
                            if ref_num not in query_to_sources[query]:
                                query_to_sources[query].append(ref_num)

                        # Also check related_questions
                        related_questions = finding.get("related_questions", [])
                        if related_questions:
                            for related_q in related_questions:
                                if related_q and related_q != query:  # Don't double-count the primary query
                                    if related_q not in query_to_sources:
                                        query_to_sources[related_q] = []
                                    if ref_num not in query_to_sources[related_q]:
                                        query_to_sources[related_q].append(ref_num)

                    # Log the mapping for debugging
                    logger.info(f"Query to sources mapping: {query_to_sources}")

                    # Consolidate original and expanded queries into a bullet point list
                    report_header += "### Queries\n"

                    # Add original query with source references
                    original_query = questions_by_iteration['0'][0]
                    source_refs = query_to_sources.get(original_query, [])
                    source_refs_str = f" [Sources: {', '.join(map(str, sorted(source_refs)))}]" if source_refs else ""
                    report_header += f"- {original_query}{source_refs_str}\n"

                    # Add expanded query from iteration 0 if it exists
                    if '0' in questions_by_iteration and len(questions_by_iteration['0']) > 1:
                        logger.info(f"Adding expanded queries to report: {questions_by_iteration['0'][1:]}")
                        # Skip the first question which is the original query
                        for q in questions_by_iteration['0'][1:]:
                            source_refs = query_to_sources.get(q, [])
                            source_refs_str = f" [Sources: {', '.join(map(str, sorted(source_refs)))}]" if source_refs else ""
                            report_header += f"- {q}{source_refs_str}\n"
                    else:
                        logger.warning(f"No expanded queries found in iteration 0: {questions_by_iteration.get('0', [])}")

                    report_header += "\n"

                    # Then show follow-up questions from subsequent iterations
                    if len(questions_by_iteration) > 1:  # If we have iterations beyond the initial one
                        logger.info(f"Adding follow-up questions to report from {len(questions_by_iteration)-1} iterations")
                        report_header += "### Follow-up Questions\n"
                        for iteration, questions in sorted(questions_by_iteration.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                            if iteration == '0':  # Skip iteration 0 as we've already handled it
                                continue

                            report_header += f"**Iteration {int(iteration)+1}:**\n"
                            for q in questions:
                                source_refs = query_to_sources.get(q, [])
                                source_refs_str = f" [Sources: {', '.join(map(str, sorted(source_refs)))}]" if source_refs else ""
                                report_header += f"- {q}{source_refs_str}\n"
                            report_header += "\n"
                    else:
                        logger.warning("No follow-up questions found in questions_by_iteration")

            report_content += report_header

            # --- Integrate Article Section from Agent Chain Results (if applicable) ---
            # Only include the article and its subsections, other fields are moved to detailed findings
            # Skip this section if we're going to generate a comprehensive article later
            # (which happens in enhanced mode with comprehensive_analysis agent chain)
            if agent_chain_name and agent_chain_outputs and not (self.mode == "enhanced" and agent_chain_name == "comprehensive_analysis"):
                # Check if there's an article to display
                if 'article' in agent_chain_outputs:
                    report_content += f"## Article from {agent_chain_name}\n\n"

                    # Process article
                    article_values = agent_chain_outputs['article']
                    if article_values and isinstance(article_values[0], str):
                        combined_article = "\n\n".join(article_values)
                        report_content += f"{combined_article}\n\n"

                    # Process article metadata
                    if 'article_type' in agent_chain_outputs:
                        report_content += "### Article Type\n"
                        article_types = agent_chain_outputs['article_type']
                        if article_types and isinstance(article_types[0], str):
                            report_content += f"{article_types[0]}\n\n"

                    if 'key_themes' in agent_chain_outputs:
                        report_content += "### Key Themes\n"
                        themes_lists = agent_chain_outputs['key_themes']
                        all_themes = []
                        for theme_list in themes_lists:
                            if isinstance(theme_list, list):
                                # Filter out single-character themes (likely from DSPy agent)
                                filtered_themes = [theme for theme in theme_list if not (isinstance(theme, str) and len(theme.strip()) <= 1)]
                                all_themes.extend(filtered_themes)
                            elif isinstance(theme_list, str):
                                # If it's a string, split by newlines or commas
                                if '\n' in theme_list:
                                    for line in theme_list.split('\n'):
                                        line = line.strip()
                                        if line and not line.startswith('-'):
                                            all_themes.append(line)
                                else:
                                    for theme in theme_list.split(','):
                                        theme = theme.strip()
                                        if theme:
                                            all_themes.append(theme)
                        if all_themes:
                            for theme in all_themes:
                                report_content += f"- {theme}\n"
                            report_content += "\n"

                    if 'word_count' in agent_chain_outputs:
                        report_content += "### Word Count\n"
                        word_counts = agent_chain_outputs['word_count']
                        if word_counts and isinstance(word_counts[0], (int, str)):
                            report_content += f"{word_counts[0]}\n\n"

                    sections_added.append(f"Article from {agent_chain_name}")
                    report_content += "---\n\n" # Add separator

            # --- Generate Sections Based on Report Mode ---
            if self.mode == "enhanced":
                # Interpretations
                try:
                    interpretations = self.advanced_reporter.generate_interpretations(final_findings, query)
                    if interpretations and interpretations.get("main_interpretation"):
                        report_content += "## Interpretations\n\n"
                        report_content += f"{interpretations.get('main_interpretation', 'N/A')}\n\n"
                        key_insights = interpretations.get('key_insights', [])
                        if key_insights: report_content += self._format_list_section("Key Insights", key_insights)
                        limitations = interpretations.get('limitations', [])
                        if limitations: report_content += self._format_list_section("Limitations", limitations)
                        confidence = interpretations.get('confidence', 0.5)
                        try: confidence_val = float(confidence)
                        except (ValueError, TypeError): confidence_val = 0.5
                        report_content += f"**Confidence:** {confidence_val:.2f}\n\n"
                        sections_added.append("Interpretations")
                    else: logger.warning("Interpretations generation returned empty/invalid result.")
                except Exception as interp_e: logger.error(f"Error during Interpretations generation: {interp_e}", exc_info=True)

                # Proposals
                try:
                    proposals = self.advanced_reporter.generate_proposals(final_findings, query)
                    if proposals and proposals.get("recommendations"):
                        report_content += "## Proposals and Recommendations\n\n"
                        recommendations = proposals.get('recommendations', [])
                        if recommendations: report_content += self._format_list_section("Recommendations", recommendations)
                        next_steps = proposals.get('next_steps', [])
                        if next_steps: report_content += self._format_list_section("Next Steps", next_steps)
                        alternatives = proposals.get('alternatives', [])
                        if alternatives: report_content += self._format_list_section("Alternatives", alternatives)
                        rationale = proposals.get('rationale', '')
                        if rationale: report_content += f"### Rationale\n{rationale}\n\n"
                        sections_added.append("Proposals")
                    else: logger.warning("Proposals generation returned empty/invalid result.")
                except Exception as prop_e: logger.error(f"Error during Proposals generation: {prop_e}", exc_info=True)

                # Technical View
                try:
                    technical_view = self.advanced_reporter.generate_technical_view(final_findings, query)
                    if technical_view and technical_view.get("technical_analysis"):
                        report_content += "## Technical Analysis\n\n"
                        report_content += f"{technical_view.get('technical_analysis', 'N/A')}\n\n"
                        details = technical_view.get('technical_details', [])
                        if details: report_content += self._format_list_section("Technical Details", details)
                        challenges = technical_view.get('technical_challenges', [])
                        if challenges: report_content += self._format_list_section("Technical Challenges", challenges)
                        solutions = technical_view.get('technical_solutions', [])
                        if solutions: report_content += self._format_list_section("Technical Solutions", solutions)
                        sections_added.append("Technical Analysis")
                    else: logger.warning("Technical Analysis generation returned empty/invalid result.")
                except Exception as tech_e: logger.error(f"Error during Technical Analysis generation: {tech_e}", exc_info=True)

                # --- Add Detailed Findings Section (Enhanced Mode) ---
                report_content += "## Detailed Findings\n\n"
                if final_findings:
                    for i, finding in enumerate(final_findings):
                        report_content += CitationHandler.format_finding_for_report(finding, i + 1, agent_chain_name)
                    sections_added.append("Detailed Findings")
                else:
                    report_content += "No detailed findings to display.\n\n"

                # Comprehensive Article (comes after detailed findings)
                try:
                    # Extract all queries for better article type determination
                    all_queries = [query]
                    if pooled_context and research_mode == "multi_iteration":
                        questions_by_iteration = pooled_context.get("questions_by_iteration", {})
                        for iteration, questions in questions_by_iteration.items():
                            all_queries.extend(questions)

                    # Generate comprehensive article using all available information
                    comprehensive_article = self.advanced_reporter.generate_comprehensive_article(
                        findings=final_findings,
                        query=query,
                        all_queries=all_queries,
                        interpretations=interpretations if 'interpretations' in locals() else None,
                        proposals=proposals if 'proposals' in locals() else None,
                        technical_view=technical_view if 'technical_view' in locals() else None
                    )

                    if comprehensive_article and comprehensive_article.get("article"):
                        article = comprehensive_article.get("article", "")
                        article_type = comprehensive_article.get("article_type", "general").title()
                        word_count = comprehensive_article.get("word_count", 0)
                        word_count_display = f"{word_count:,}" if word_count else "N/A"

                        # Add article to report
                        report_content += f"## Comprehensive {article_type} ({word_count_display} words)\n\n"

                        # Add key themes
                        key_themes = comprehensive_article.get("key_themes", [])
                        # Filter out single-character themes (likely from DSPy agent)
                        if isinstance(key_themes, list):
                            filtered_themes = [theme for theme in key_themes if not (isinstance(theme, str) and len(theme.strip()) <= 1)]
                            if filtered_themes:
                                report_content += self._format_list_section("Key Themes", filtered_themes, prefix="-")
                        else:
                            if key_themes:
                                report_content += self._format_list_section("Key Themes", key_themes, prefix="-")

                        # Add article content
                        has_headings = any(line.strip().startswith('#') for line in article.split('\n'))
                        if not has_headings:
                            report_content += "### " + query.title() + "\n\n"

                        # Add the article with citations already included
                        report_content += article + "\n\n"
                        sections_added.append("Comprehensive Article")
                    else:
                        logger.warning("Comprehensive article generation returned empty/invalid result. Falling back to synthesis.")
                        # Fallback to regular synthesis if comprehensive article fails
                        synthesis = self.advanced_reporter.generate_comprehensive_synthesis(final_findings, query)
                        if synthesis and synthesis.get("article"):
                            article = synthesis.get("article", "")
                            article_type = synthesis.get("article_type", "general").title()
                            word_count = synthesis.get("word_count", 0)
                            word_count_display = f"{word_count:,}" if word_count else "N/A"
                            report_content += f"## Comprehensive {article_type} ({word_count_display} words)\n\n"
                            key_themes = synthesis.get("key_themes", [])
                            # Filter out single-character themes (likely from DSPy agent)
                            if isinstance(key_themes, list):
                                filtered_themes = [theme for theme in key_themes if not (isinstance(theme, str) and len(theme.strip()) <= 1)]
                                if filtered_themes:
                                    report_content += self._format_list_section("Key Themes", filtered_themes, prefix="-")
                            else:
                                if key_themes:
                                    report_content += self._format_list_section("Key Themes", key_themes, prefix="-")
                            has_headings = any(line.strip().startswith('#') for line in article.split('\n'))
                            if not has_headings: report_content += "### " + query.title() + "\n\n"
                            report_content += article + "\n\n"
                            sections_added.append("Synthesis")
                        else:
                            logger.warning("Synthesis generation returned empty/invalid result.")
                except Exception as article_e:
                    logger.error(f"Error during Comprehensive Article generation: {article_e}", exc_info=True)

                # If no specific sections added, fallback to basic summary
                #if "Interpretations" not in sections_added and "Proposals" not in sections_added and "Technical Analysis" not in sections_added and "Synthesis" not in sections_added and "Detailed Findings" not in sections_added:
                #    logger.warning("Enhanced report generation failed to produce specific sections. Adding basic summary.")
                #    report_content += self._generate_basic_summary(final_findings) # Basic summary doesn't need sources arg now
                #    sections_added.append("Basic Summary Fallback")


            elif self.mode == "chain_of_thought":
                 report_content += "## Reasoning Process\n\n"
                 try:
                     combined_context = self._create_context_for_cot(final_findings)
                     if not combined_context:
                          report_content += "Insufficient context for Chain-of-Thought analysis.\n"
                     else:
                          cot_agent = DSPyAgentRegistry.get_agent("chain_of_thought")
                          if cot_agent:
                              cot_result = self.advanced_reporter._call_dspy_agent(
                                  "chain_of_thought", {"content": combined_context, "query": query}
                              )
                              if cot_result and cot_result.get("cot"):
                                  reasoning = cot_result.get('cot', 'N/A')
                                  conclusion = cot_result.get('conclusion', 'N/A')
                                  # Format reasoning and conclusion with citations
                                  report_content += f"**Reasoning:**\n{self._add_refs_to_text(reasoning, final_findings)}\n\n" # Use helper
                                  if conclusion and conclusion != 'N/A':
                                      report_content += f"**Conclusion:**\n{self._add_refs_to_text(conclusion, final_findings)}\n\n" # Use helper
                                  else: logger.warning("CoT agent did not return a conclusion.")
                                  sections_added.append("CoT Reasoning")
                              else:
                                  logger.warning("ChainOfThought agent returned empty/invalid reasoning result. Using manual CoT.")
                                  report_content += self._generate_manual_cot(final_findings) # Use finding refs
                          else:
                              logger.warning("ChainOfThought agent not found. Using manual CoT.")
                              report_content += self._generate_manual_cot(final_findings) # Use finding refs
                 except Exception as cot_e:
                     logger.error(f"Error during CoT generation: {cot_e}", exc_info=True)
                     report_content += "Error occurred during Chain-of-Thought generation.\n\n"
                     report_content += self._generate_basic_summary(final_findings, agent_chain_name) # Fallback
                     sections_added.append("CoT Error Fallback")
                 report_content += "---\n\n" # Separator after CoT

                 # --- Add Detailed Findings Section (CoT Mode) ---
                 report_content += "## Supporting Findings\n\n"
                 if final_findings:
                     for i, finding in enumerate(final_findings):
                         report_content += CitationHandler.format_finding_for_report(finding, i + 1, agent_chain_name)
                     sections_added.append("Detailed Findings")
                 else:
                     report_content += "No detailed findings to display.\n\n"


            else: # Normal mode
                 report_content += self._generate_basic_summary(final_findings, agent_chain_name) # Use finding refs
                 sections_added.append("Basic Summary")

            # --- Add Sources Section ---
            report_content += "\n## Sources\n\n"
            if sources:
                report_content += "| Ref | Source | Type | Page | Line | Relevance |\n"
                report_content += "|-----|--------|------|------|------|-----------|\n"
                for source in sources:
                    if source is None:
                        logger.warning("Encountered None source in sources list, skipping")
                        continue

                    try:
                        ref = source.get("reference_number", "?")
                        # Fix the potential None issue by ensuring we have a string before slicing
                        source_name = source.get("source_name", None)
                        source_path = source.get("source_path", None)
                        name_str = source_name if source_name is not None else (source_path if source_path is not None else "Unknown")
                        name = name_str[:60] if name_str else "Unknown"  # Truncate if not empty

                        type_ = source.get("source_type", "N/A")
                        page = source.get("page", "-")
                        line = source.get("line", "-")
                        # Get the normalized score (should be between 0 and 1)
                        score_val = source.get("score", 0.0)

                        # Format the score as a percentage (0-100%)
                        score_str = f"{score_val * 100:.1f}%" if isinstance(score_val, (float, int)) else "N/A"
                        report_content += f"| [{ref}] | {name} | {type_} | {page} | {line} | {score_str} |\n"
                    except Exception as src_err:
                        logger.error(f"Error processing source: {src_err}")
                        # Add a placeholder row for the problematic source
                        report_content += f"| ? | Error processing source | N/A | - | - | N/A |\n"
            else:
                 report_content += "No sources cited.\n"

        except Exception as e:
            logger.error(f"Critical Error during report content generation: {e}", exc_info=True)
            if not report_content.strip().startswith("#"):
                 report_content = f"# Report Generation Error for: '{query}'\n\n"
            else: report_content += "\n\n---\n\n"
            report_content += f"A critical error occurred during report generation:\n```\n{traceback.format_exc()}\n```"

        if len(report_content) < len(report_header) + 50:
             report_content = f"# Report for: '{query}'\n\nNo detailed report could be generated. Please review findings or logs."
             logger.warning("Generated report content was empty or very short after processing. Providing fallback.")

        # Return the report string and the list of source dictionaries
        # Also return the final findings list used for the report (with reference numbers)
        result = {
            "report": report_content,
            "sources": sources,
            "final_findings_for_report": final_findings
        }
        logger.info(f"Report generated. Final length: {len(report_content)}. Sections added: {sections_added}")
        return result

    def _create_context_for_cot(self, findings: List[Dict]) -> str:
         """Creates a limited context string suitable for CoT analysis."""
         context_parts = []
         MAX_COT_FINDINGS = 10
         MAX_COT_CONTEXT_LEN = 8000
         current_len = 0

         for finding in findings[:MAX_COT_FINDINGS]:
             ref = finding.get('reference_number', '?') # Use assigned ref num
             summary = finding.get('summary', finding.get('content', '')[:300] + "...")
             finding_str = f"Finding [{ref}]: {summary}" # Include ref num
             if current_len + len(finding_str) < MAX_COT_CONTEXT_LEN:
                  context_parts.append(finding_str)
                  current_len += len(finding_str) + 2
             else:
                  logger.warning(f"Stopping CoT context generation at finding {ref} due to length limit.")
                  break
         return "\n\n".join(context_parts) if context_parts else ""

    def _generate_basic_summary(self, findings: List[Dict], agent_chain_name: str = None) -> str:
        """Generates a simple summary section with explicit finding details."""
        content = "## Summary of Findings\n\n"
        if not findings: return content + "No findings to summarize.\n"
        for i, finding in enumerate(findings):
            content += CitationHandler.format_finding_for_report(finding, i + 1, agent_chain_name) # Use helper
        return content

    def _generate_manual_cot(self, findings: List[Dict]) -> str:
         """Generates a structured manual CoT using finding details."""
         content = "**Manual Reasoning:**\n"
         if not findings: return content + "No findings to analyze.\n"
         ref_numbers = []
         for i, finding in enumerate(findings):
              ref = finding.get('reference_number', i+1) # Use assigned ref number
              ref_numbers.append(str(ref))
              snippet = finding.get('content', 'N/A')[:150] + "..."
              summary = finding.get('summary', finding.get('analysis', {}).get('summary', 'No summary.'))
              content += f"- **Step {i+1} (Source [{ref}]):** Analyzing snippet: \"{snippet}\". Key Point: {summary}\n"
         # Update conclusion to list the actual reference numbers used
         content += f"\n**Overall Conclusion:** Based on the analysis of sources [{', '.join(ref_numbers)}], the key takeaways are...\n\n" # Placeholder conclusion
         return content

    def _format_list_section(self, title: str, items: list, prefix: str = "-") -> str:
        """Formats a list into a markdown section, handling strings."""
        if not items: return ""
        processed_items = []

        # Process string items
        if isinstance(items, str):
            # If it's a string, split by newlines or commas
            if '\n' in items:
                processed_items = [line.strip().lstrip(prefix).strip() for line in items.split('\n') if line.strip()]
            else:
                processed_items = [item.strip() for item in items.split(',') if item.strip()]

        # Process list items
        elif isinstance(items, list):
            # Filter out single-character items (likely from DSPy agent)
            processed_items = []
            for item in items:
                if isinstance(item, str) and len(item.strip()) <= 1:
                    # Skip single-character items
                    continue
                processed_items.append(item)
        else:
            logger.warning(f"Unexpected item type '{type(items)}' in _format_list_section for '{title}'. Converting to string.")
            processed_items = [str(items)]

        content = f"### {title}\n"
        if processed_items:
            for item in processed_items:
                item_str = str(item).strip()
                if item_str:
                    content += f"{prefix} {item_str}\n"
        else:
            content += "N/A\n"
        return content + "\n"

    def _deduplicate_findings(self, findings: List[Dict]) -> List[Dict]:
        """Deduplicate findings based on source chunk ID."""
        seen_keys = set()
        deduplicated = []
        for finding in findings:
            citation = finding.get("citation", {})
            source_key = citation.get("source_path") or citation.get("source_name", "")
            start_pos = citation.get("start", 0) or 0
            end_pos = citation.get("end", 0) or 0
            unique_key = f"{source_key}::{start_pos}-{end_pos}"

            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduplicated.append(finding)
            else:
                logger.debug(f"Skipping duplicate finding chunk: {unique_key}")
        return deduplicated

    def _add_refs_to_text(self, text: str, findings: List[Dict]) -> str:
        """Adds reference numbers to text based on associated findings (placeholder)."""
        # This is a simplified version. Real implementation is hard.
        # For now, just append all reference numbers found.
        refs = sorted(list(set(f.get('reference_number') for f in findings if f.get('reference_number') is not None)))
        if refs:
            ref_str = ", ".join(f"[{r}]" for r in refs)
            return f"{text} (Based on Sources: {ref_str})"
        return text


class ResearchSystem:
    def __init__(self, vector_store: Union[VectorStore, LlamaIndexVectorStore]):
        self.vector_store = vector_store
        self.document_analyzer = DocumentAnalyzer()

    def _add_citation_to_result(self, result: Dict) -> Dict:
        """
        Add citation information to a result dictionary if it doesn't already have it.
        This ensures consistent citation metadata across different search methods.

        Args:
            result: The result dictionary to add citation to

        Returns:
            The result dictionary with citation added
        """
        if "citation" not in result:
            result["citation"] = {
                "source_path": result.get("source_path", result.get("file_path")),
                "source_name": result.get("source_name"),
                "source_type": result.get("source_type"),
                "start": result.get("start"),
                "end": result.get("end"),
                "score": result.get("score", 0.0)  # Add score to citation
            }
        return result

    def run(
        self,
        query: str,
        research_mode: str = "rag",
        report_mode: str = "normal",
        agent_chain_name: str = None,
        top_k: int = 5,
        context_filter: str = None,
        session_id: str = None,
        max_iterations: int = 3,
        max_k: int = 20,
        relevance_threshold: float = 0.5
    ) -> Dict:
        """
        Runs the research process and generates a report.

        Args:
            query: The research query
            research_mode: Research mode ("rag" or "multi_iteration")
            report_mode: Report mode ("normal", "chain_of_thought", or "enhanced")
            agent_chain_name: Optional agent chain to use
            top_k: Number of results to retrieve per question
            context_filter: Optional filter for results
            session_id: Optional session ID for loading vector store
            max_iterations: Maximum number of research iterations for multi_iteration mode
            max_k: Maximum limit for results to retrieve before deduplication for multi_iteration mode.
                 Actual retrieval count is dynamically calculated based on already collected sources.
            relevance_threshold: Minimum relevance score for results in multi_iteration mode

        Returns:
            Dictionary with research results and report
        """
        start_time = time.time()
        original_timeout = socket.getdefaulttimeout()
        complex_op = research_mode == "multi_iteration" or report_mode in ["chain_of_thought", "enhanced"]
        timeout_duration = 300 if complex_op else 120

        research_findings = []
        pooled_context_data = None
        questions_by_iteration = None
        final_response = {}

        try:
            socket.setdefaulttimeout(timeout_duration)
            logger.info(f"Set socket timeout to {timeout_duration} seconds for operation")
            logger.info(f"Starting research for query: '{query}' with research_mode='{research_mode}', report_mode='{report_mode}', agent_chain='{agent_chain_name}'")

            project_id = getattr(self.vector_store, 'project_id', None)

            # --- Phase 1: Retrieval / Multi-Iteration ---
            if research_mode == "multi_iteration":
                logger.info("Running Multi-Iteration Research...")
                multi_research = MultiIterationResearch(
                    self.vector_store,
                    max_iterations=max_iterations,
                    max_k=max_k,
                    relevance_threshold=relevance_threshold
                )
                multi_results = multi_research.run(
                    query,
                    top_k=top_k,
                    context_filter=context_filter,
                    max_k=max_k,
                    relevance_threshold=relevance_threshold
                )
                research_findings = multi_results.get("findings", [])
                pooled_context_data = multi_results.get("pooled_context")
                questions_by_iteration = multi_results.get("questions_by_iteration")
                logger.info(f"Multi-iteration complete. Raw Findings: {len(research_findings)}, Pooled Findings: {len(pooled_context_data.get('findings', [])) if pooled_context_data else 'N/A'}")

            elif research_mode == "rag":
                logger.info("Running RAG Search...")
                if isinstance(self.vector_store, LlamaIndexVectorStore) and hasattr(self.vector_store, 'index') and self.vector_store.index:
                    logger.info("Using LlamaIndex query engine for improved search...")
                    try:
                        query_engine = self.vector_store.index.as_query_engine(similarity_top_k=top_k)
                        response = query_engine.query(query)
                        research_findings = []
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            logger.info(f"Found {len(response.source_nodes)} source nodes using query engine")
                            for i, node in enumerate(response.source_nodes):
                                result = {"node_id": node.node_id, "content": node.text, "score": node.score if hasattr(node, 'score') else 0.0}
                                if hasattr(node, 'metadata') and node.metadata: result.update(node.metadata)
                                research_findings.append(result)
                        else:
                            logger.warning("Query engine did not return source nodes, falling back to traditional search")
                            research_findings = search_index(vector_store=self.vector_store, query=query, top_k=top_k, context_filter=context_filter, session_id=session_id, project_id=project_id)
                    except Exception as e:
                        logger.error(f"Error using query engine: {e}. Falling back to traditional search.")
                        research_findings = search_index(vector_store=self.vector_store, query=query, top_k=top_k, context_filter=context_filter, session_id=session_id, project_id=project_id)
                else:
                    research_findings = search_index(vector_store=self.vector_store, query=query, top_k=top_k, context_filter=context_filter, session_id=session_id, project_id=project_id)
                logger.info(f"RAG search complete. Findings: {len(research_findings)}")
            else:
                 logger.warning(f"Unknown research mode: {research_mode}. Defaulting to RAG.")
                 if isinstance(self.vector_store, LlamaIndexVectorStore) and hasattr(self.vector_store, 'index') and self.vector_store.index:
                    logger.info("Using LlamaIndex query engine for improved search...")
                    try:
                        query_engine = self.vector_store.index.as_query_engine(similarity_top_k=top_k)
                        response = query_engine.query(query)
                        research_findings = []
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            logger.info(f"Found {len(response.source_nodes)} source nodes using query engine")
                            for i, node in enumerate(response.source_nodes):
                                result = {"node_id": node.node_id, "content": node.text, "score": node.score if hasattr(node, 'score') else 0.0}
                                if hasattr(node, 'metadata') and node.metadata: result.update(node.metadata)
                                research_findings.append(result)
                        else:
                            logger.warning("Query engine did not return source nodes, falling back to traditional search")
                            research_findings = search_index(vector_store=self.vector_store, query=query, top_k=top_k, context_filter=context_filter, session_id=session_id, project_id=project_id)
                    except Exception as e:
                        logger.error(f"Error using query engine: {e}. Falling back to traditional search.")
                        research_findings = search_index(vector_store=self.vector_store, query=query, top_k=top_k, context_filter=context_filter, session_id=session_id, project_id=project_id)
                 else:
                    research_findings = search_index(vector_store=self.vector_store, query=query, top_k=top_k, context_filter=context_filter, session_id=session_id, project_id=project_id)

            # Ensure all findings have citation information regardless of how they were retrieved
            if research_mode == "rag":  # Only process for RAG mode, multi_iteration already handles this
                for finding in research_findings:
                    self._add_citation_to_result(finding)
                logger.info("Added citation information to all RAG findings")

            # Check if any findings were retrieved
            findings_available = bool(research_findings or (pooled_context_data and pooled_context_data.get("findings")))
            if not findings_available:
                logger.warning("No findings retrieved in any research mode.")
                elapsed_time = time.time() - start_time
                final_response = {
                    "report": f"# Report for: '{query}'\n\nNo relevant information found.",
                    "sources": [], "findings": [], "elapsed_time": f"{elapsed_time:.2f} seconds",
                    "research_mode": research_mode, "report_mode": report_mode, "questions_by_iteration": None
                }
                return make_json_serializable(final_response)

            # Determine which findings list to use for report generation
            findings_for_report = pooled_context_data['findings'] if pooled_context_data and pooled_context_data.get('findings') else research_findings

            # --- Phase 2: Agentic Processing (If Chain Specified) ---
            # Run the chain *before* report generation if specified
            if agent_chain_name and findings_for_report:
                logger.info(f"Running specified agent chain: {agent_chain_name} on {len(findings_for_report)} findings before report generation")
                processed_findings_chain = []
                for idx, finding in enumerate(findings_for_report):
                     logger.debug(f"Processing finding {idx+1}/{len(findings_for_report)} with chain '{agent_chain_name}'")
                     try:
                         chain_input = finding.copy()
                         chain_input['query'] = query
                         result_data = DSPyAgentRegistry.run_chain(agent_chain_name, chain_input)
                         # Update the finding dictionary *in place* within findings_for_report
                         finding.update(result_data)
                         processed_findings_chain.append(finding) # Collect potentially modified findings
                     except Exception as e:
                         logger.error(f"Error running agent chain '{agent_chain_name}' on finding {idx+1}: {e}", exc_info=True)
                         finding["agent_chain_error"] = str(e)
                         processed_findings_chain.append(finding) # Still include finding with error
                # Update findings_for_report with potentially modified findings
                findings_for_report = processed_findings_chain
                logger.info(f"Agent chain pre-processing complete.")


            # --- Phase 3: Report Generation ---
            logger.info(f"Generating final report. Mode: {report_mode}, Agent Chain Used: {agent_chain_name}")

            # Log the pooled_context_data to verify it contains questions_by_iteration
            if pooled_context_data:
                logger.info(f"pooled_context_data contains questions_by_iteration: {pooled_context_data.get('questions_by_iteration') is not None}")
                if 'questions_by_iteration' in pooled_context_data:
                    logger.info(f"questions_by_iteration in pooled_context_data: {pooled_context_data['questions_by_iteration']}")

            report_generator = ReportGenerator(mode=report_mode)
            report_data = report_generator.generate_report(
                findings_for_report, query, research_mode,
                pooled_context=pooled_context_data,
                agent_chain_name=agent_chain_name # Pass chain name so generator knows it ran
            )

            elapsed_time = time.time() - start_time
            logger.info(f"Research and report generation complete in {elapsed_time:.2f} seconds")

            # --- Phase 4: Prepare Response ---
            # report_data contains: {"report": str, "sources": list, "final_findings_for_report": list}
            final_response = {
                "report": report_data.get("report", "Error: Report generation failed."),
                "sources": report_data.get("sources", []),
                # Return the final list of findings used by the report (with ref numbers)
                "findings": report_data.get("final_findings_for_report", []),
                # Also include findings as "results" for backward compatibility with UI
                "results": report_data.get("final_findings_for_report", []),
                "elapsed_time": f"{elapsed_time:.2f} seconds",
                "research_mode": research_mode,
                "report_mode": report_mode,
                "questions_by_iteration": questions_by_iteration,
                # Include agent chain outputs explicitly if needed by the API caller
                #"agent_outputs": agent_chain_outputs if agent_chain_name else None
            }

            logger.info(f"Final report length: {len(final_response['report'])}")
            if len(final_response['report']) < 200:
                 logger.warning(f"Generated report seems short: {final_response['report'][:150]}...")

            return make_json_serializable(final_response) # Ensure serializable

        except Exception as e:
             logger.error(f"Critical error in ResearchSystem.run: {e}", exc_info=True)
             elapsed_time = time.time() - start_time
             final_response = {
                  "report": f"# Research System Error\n\nAn unexpected error occurred: {e}\n\nTraceback:\n{traceback.format_exc()}",
                  "sources": [],
                  "findings": research_findings if not pooled_context_data else pooled_context_data.get('findings', []),
                  "elapsed_time": f"{elapsed_time:.2f} seconds",
                  "research_mode": research_mode, "report_mode": report_mode, "questions_by_iteration": questions_by_iteration,
                  "error": str(e)
             }
             return make_json_serializable(final_response)

        finally:
             socket.setdefaulttimeout(original_timeout)
             logger.info(f"Reset socket timeout to default ({original_timeout} seconds)")
