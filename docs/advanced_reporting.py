# filename: src/local_file_research/advanced_reporting.py
"""
Advanced reporting module for Local File Deep Research.

This module provides enhanced reporting capabilities with interpretations,
proposals, and technical views.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import json # Import json for robust parsing in fallbacks
import traceback # +++ ADD THIS +++
from .dspy_config import DSPyAgentRegistry, DSPY_CONFIGURED # NEW

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedReportGenerator:
    """
    Advanced report generator that provides interpretations, proposals, and technical views.
    """

    def __init__(self):
        """Initialize the advanced report generator."""
        pass # Init can be empty now or do other setup

    def _call_dspy_agent(self, agent_name: str, required_fields: Dict[str, Any]) -> Optional[Any]:
        """Helper function to call a DSPy agent with robust error handling and input checking."""
        
        if not DSPY_CONFIGURED:
             logger.error(f"Cannot call agent '{agent_name}': DSPy is not configured.")
             return None        
        
        agent = DSPyAgentRegistry.get_agent(agent_name)
        if not agent:
            logger.warning(f"DSPy agent '{agent_name}' not found. Cannot generate this report section.")
            return None

        logger.info(f"Using DSPy {agent_name}Agent")

        # Ensure all required fields are present using the registry's helper
        # This will add fallbacks if necessary based on the signature
        try:
            prepared_data = DSPyAgentRegistry._ensure_required_fields(agent_name, agent, required_fields)
        except Exception as prep_e:
            logger.error(f"Error preparing data for agent '{agent_name}': {prep_e}", exc_info=True)
            return None # Cannot proceed if preparation fails

        try:
            # Log the exact fields being passed
            logger.debug(f"Calling {agent_name} with fields: {list(prepared_data.keys())}")
            result = agent(**prepared_data)
            logger.info(f"Successfully called DSPy agent '{agent_name}'. Result type: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"Error calling DSPy agent '{agent_name}': {e}", exc_info=True)
            # Log the inputs that caused the error for debugging
            logger.error(f"Inputs provided to {agent_name}: {prepared_data}")
            return None # Return None on error

    def generate_interpretations(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate interpretations of the findings."""
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Falling back for interpretations.")
                return self._generate_interpretations_with_text_generator(findings, query)

            agent_inputs = {"query": query if query else "Interpret these findings", "context": context}
            result = self._call_dspy_agent("interpreter", agent_inputs) # Use helper

            # +++ MODIFY RESULT CHECKING +++
            if result and hasattr(result, 'get'):
                interpretation = result.get("interpretation", "")
                if not interpretation or len(interpretation) < 10: # Adjusted check
                     logger.warning(f"InterpreterAgent returned short/empty interpretation: '{str(interpretation)[:50]}...'. Falling back.")
                     return self._generate_interpretations_with_text_generator(findings, query)
                # Success case
                return {
                    "main_interpretation": interpretation,
                    "key_insights": result.get("insights", []),
                    "limitations": result.get("limitations", []),
                    "confidence": result.get("confidence", 0.5)
                }
            else:
                # Log the actual problematic result
                logger.warning(f"InterpreterAgent returned None or invalid type ({type(result)}): {result}. Falling back.")
                return self._generate_interpretations_with_text_generator(findings, query)
        except Exception as e:
            logger.error(f"Error generating interpretations: {e}", exc_info=True)
            return self._generate_interpretations_with_text_generator(findings, query)

    def generate_proposals(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate proposals based on the findings."""
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Falling back for proposals.")
                return self._generate_proposals_with_text_generator(findings, query)

            agent_inputs = {"query": query if query else "Generate proposals based on these findings", "context": context}
            result = self._call_dspy_agent("proposal_generator", agent_inputs) # Use helper

            # +++ MODIFY RESULT CHECKING +++
            if result and hasattr(result, 'get'):
                 recommendations = result.get("recommendations", [])
                 meaningful_recs = [rec for rec in recommendations if rec] if isinstance(recommendations, list) else []
                 if not meaningful_recs:
                      logger.warning("ProposalGeneratorAgent returned no meaningful recommendations. Falling back.")
                      return self._generate_proposals_with_text_generator(findings, query)
                 # Success case
                 return {
                    "recommendations": recommendations, "next_steps": result.get("next_steps", []),
                    "alternatives": result.get("alternatives", []), "rationale": result.get("rationale", "")
                 }
            else:
                logger.warning(f"ProposalGeneratorAgent returned None or invalid type ({type(result)}): {result}. Falling back.")
                return self._generate_proposals_with_text_generator(findings, query)
        except Exception as e:
            logger.error(f"Error generating proposals: {e}", exc_info=True)
            return self._generate_proposals_with_text_generator(findings, query)

    def generate_technical_view(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate technical view of the findings."""
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Cannot generate technical view.")
                return self._generate_technical_view_with_text_generator(findings, query) # Fallback

            agent_inputs = {
                "query": query if query else "Generate technical view based on these findings",
                "context": context,
            }

            result = self._call_dspy_agent("technical_analyzer", agent_inputs)

            if result and hasattr(result, 'get'):
                analysis = result.get("analysis", "")
                if not analysis or len(analysis) < 10: logger.warning(f"TechAnalyzer short/empty. Falling back."); return self._generate_technical_view_with_text_generator(findings, query)
                return {"technical_analysis": analysis, "technical_details": result.get("details",[]), "technical_challenges": result.get("challenges",[]), "technical_solutions": result.get("solutions",[])}
            else: logger.warning(f"TechAnalyzer invalid result ({type(result)}): {result}. Falling back."); return self._generate_technical_view_with_text_generator(findings, query)
        except Exception as e: logger.error(f"Error generating tech view: {e}", exc_info=True); return self._generate_technical_view_with_text_generator(findings, query)

    def generate_comprehensive_synthesis(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate a comprehensive synthesis of the findings."""
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Cannot generate synthesis.")
                context = f"Query: {query}\nNo specific findings available." # Minimal context

            agent_inputs = {
                "query": query if query else "Synthesize these findings",
                "context": context,
            }

            result = self._call_dspy_agent("content_synthesizer", agent_inputs)

            if result and hasattr(result, 'get'):
                article = result.get("article", "")
                if not article or len(article) < 20: logger.warning(f"Synthesizer short/empty. Falling back."); return self._generate_comprehensive_synthesis_with_text_generator(findings, query)
                word_count = result.get("word_count", len(article.split()))
                return {"article": article, "article_type": result.get("article_type", "general"), "key_themes": result.get("key_themes", []), "word_count": word_count}
            else: logger.warning(f"Synthesizer invalid result ({type(result)}): {result}. Falling back."); return self._generate_comprehensive_synthesis_with_text_generator(findings, query)
        except Exception as e: logger.error(f"Error generating synthesis: {e}", exc_info=True); return self._generate_comprehensive_synthesis_with_text_generator(findings, query)

    def _prepare_context_from_findings(self, findings: List[Dict]) -> str:
        """Prepare context string from findings for agent input."""
        context_parts = []
        logger.info(f"Preparing context from {len(findings)} findings for report generation.")

        # Limit number of findings used for context to avoid excessive length
        MAX_FINDINGS_FOR_CONTEXT = 50 # Adjust as needed
        findings_to_use = findings[:MAX_FINDINGS_FOR_CONTEXT]
        if len(findings) > MAX_FINDINGS_FOR_CONTEXT:
             logger.warning(f"Using top {MAX_FINDINGS_FOR_CONTEXT} findings (out of {len(findings)}) for context generation.")

        for i, finding in enumerate(findings_to_use, 1):
            if not isinstance(finding, dict):
                logger.warning(f"Skipping finding at index {i-1} as it is not a dictionary: {finding}")
                continue

            # Get the most informative content
            summary = finding.get('summary', '')
            analysis = finding.get('analysis', {})
            content = finding.get('content', '') # Get raw content as fallback
            key_points = analysis.get('key_points', []) if isinstance(analysis, dict) else []

            # Prioritize summary, then key points, then raw content snippet
            finding_text = ""
            if summary:
                finding_text = summary
            elif key_points:
                 # Handle case where key_points might be a string instead of list
                 if isinstance(key_points, str):
                     finding_text = "Key points:\n" + key_points # Assume string is pre-formatted
                 elif isinstance(key_points, list):
                      finding_text = "Key points:\n" + "\n".join([f"- {point}" for point in key_points])
            elif content:
                 finding_text = content[:500] + ("..." if len(content) > 500 else "") # Use snippet of content

            if finding_text:
                 context_parts.append(f"--- Finding {i} ---")
                 # Add source information
                 citation = finding.get('citation', {})
                 source_name = citation.get('source_name', citation.get('source_path', 'Unknown Source'))
                 context_parts.append(f"Source: {source_name}")
                 # Add query info if available
                 query_info = finding.get('query_info', {})
                 if query_info and query_info.get('query'):
                      context_parts.append(f"Related Question: {query_info['query']}")

                 context_parts.append(f"Content: {finding_text}")
                 context_parts.append("------\n")

        # Add a clear separator and instruction
        generated_context = "Context based on research findings:\n\n" + "\n".join(context_parts).strip()

        # Limit total context length further if needed (e.g., 8k chars)
        MAX_CONTEXT_LEN = 30000
        if len(generated_context) > MAX_CONTEXT_LEN:
             generated_context = generated_context[:MAX_CONTEXT_LEN] + "\n... [Context Truncated]"
             logger.warning(f"Truncated prepared context to {MAX_CONTEXT_LEN} characters.")

        logger.info(f"Generated context string length: {len(generated_context)}")
        if not generated_context or len(generated_context) < 100:
             logger.warning("Generated context is very short or empty.")
             # Provide a minimal context if completely empty
             if not generated_context.strip():
                  return "No findings available to generate context."
        return generated_context


    # --- Fallback methods using Text Generator ---

    def _generate_interpretations_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate interpretations using text generator (Fallback)."""
        logger.info("Attempting fallback: interpretations with text_generator.")
        try:
            context = self._prepare_context_from_findings(findings)
            if not context: 
                logger.warning("Context is empty for fallback interpretation generation.")
                context = "No findings available to interpret."

            prompt = f"""Based on the following research findings, provide an interpretation for the query: "{query}"

Findings Context:
{context}

Please provide ONLY the following sections, clearly marked:
Main Interpretation: [Your interpretation here (1-2 paragraphs)]
Key Insights: [List 3-5 key insights, one per line starting with '-']
Limitations: [List 2-3 limitations, one per line starting with '-']
Confidence: [A single number between 0.0 and 1.0]
"""
            # Use helper to call agent
            agent_inputs = {"prompt": prompt, "query": query, "context": context}
            result = self._call_dspy_agent("text_generator", agent_inputs) # Use helper
            response = result.get("text", "") if result and hasattr(result, 'get') else ""
            if not response:
                 logger.warning("Fallback text_generator for interpretations returned no response.")
                 return {"main_interpretation": "Fallback failed: No response.", "key_insights": [], "limitations": [], "confidence": 0.0}

            logger.debug(f"Fallback Interpretation Response:\n{response[:200]}...")


            # Robust parsing
            main_interpretation = ""
            key_insights = []
            limitations = []
            confidence = 0.5
            lines = response.splitlines()
            current_section = None
            for line in lines:
                line_strip = line.strip()
                if line_strip.startswith("Main Interpretation:"):
                    current_section = "interpretation"
                    main_interpretation += line_strip[len("Main Interpretation:"):].strip()
                elif line_strip.startswith("Key Insights:"):
                    current_section = "insights"
                elif line_strip.startswith("Limitations:"):
                    current_section = "limitations"
                elif line_strip.startswith("Confidence:"):
                    current_section = "confidence"
                    try:
                        confidence_str = line_strip[len("Confidence:"):].strip()
                        confidence = float(confidence_str)
                    except ValueError:
                        confidence = 0.5 # Default on parsing error
                elif current_section == "interpretation" and not line_strip.startswith(("Key Insights:", "Limitations:", "Confidence:")):
                     main_interpretation += "\n" + line_strip if main_interpretation else line_strip
                elif current_section == "insights" and line_strip.startswith("-"):
                    key_insights.append(line_strip[1:].strip())
                elif current_section == "limitations" and line_strip.startswith("-"):
                    limitations.append(line_strip[1:].strip())

            # Ensure fields have default values if parsing fails
            if not main_interpretation: main_interpretation = "Interpretation generation failed via fallback."
            if not key_insights: key_insights = ["No insights extracted via fallback."]
            if not limitations: limitations = ["Analysis limitations unclear via fallback."]


            return {
                "main_interpretation": main_interpretation.strip(),
                "key_insights": key_insights,
                "limitations": limitations,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error generating interpretations with text generator: {e}", exc_info=True)
            # Return structure indicating error
            return {
                "main_interpretation": "Error during fallback interpretation generation.",
                "key_insights": [], "limitations": [], "confidence": 0.1
            }

    def _generate_proposals_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate proposals using text generator (Fallback)."""
        logger.info("Attempting to generate proposals using fallback text generator.")
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                 context = "No findings available to generate proposals."

            prompt = f"""Based on the research findings context below, provide proposals for the query: "{query}"

Findings Context:
{context}

Please provide ONLY the following sections, clearly marked:
Recommendations: [List 3-5 recommendations, one per line starting with '-']
Next Steps: [List 2-3 next steps, one per line starting with '-']
Alternatives: [List 1-2 alternatives, one per line starting with '-']
Rationale: [Your rationale here (1-2 paragraphs)]
"""
            # Use helper to call agent
            agent_inputs = {"prompt": prompt, "query": query, "context": context}
            result = self._call_dspy_agent("text_generator", agent_inputs)
            response = result.get("text", "") if result and hasattr(result, 'get') else ""
            
            if not response: 
                 logger.warning("Fallback text_generator proposals empty.")
                 return {"recommendations": ["Fallback failed."], "next_steps": [], "alternatives": [], "rationale": ""}
            
            logger.debug(f"Fallback Proposals Response:\n{response[:200]}...")

            # Robust parsing
            recommendations, next_steps, alternatives = [], [], []
            rationale = ""
            lines = response.splitlines()
            current_section = None
            for line in lines:
                line_strip = line.strip()
                if line_strip.startswith("Recommendations:"): current_section = "recommendations"
                elif line_strip.startswith("Next Steps:"): current_section = "next_steps"
                elif line_strip.startswith("Alternatives:"): current_section = "alternatives"
                elif line_strip.startswith("Rationale:"):
                    current_section = "rationale"
                    rationale += line_strip[len("Rationale:"):].strip()
                elif current_section == "rationale" and not line_strip.startswith(("Recommendations:", "Next Steps:", "Alternatives:")) :
                    rationale += "\n" + line_strip if rationale else line_strip
                elif current_section == "recommendations" and line_strip.startswith("-"): recommendations.append(line_strip[1:].strip())
                elif current_section == "next_steps" and line_strip.startswith("-"): next_steps.append(line_strip[1:].strip())
                elif current_section == "alternatives" and line_strip.startswith("-"): alternatives.append(line_strip[1:].strip())

            # Defaults on failure
            if not recommendations: recommendations = ["Proposal generation failed via fallback."]
            if not next_steps: next_steps = ["No next steps extracted via fallback."]
            if not alternatives: alternatives = ["No alternatives extracted via fallback."]
            if not rationale: rationale = "Rationale generation failed via fallback."

            return {
                "recommendations": recommendations, "next_steps": next_steps,
                "alternatives": alternatives, "rationale": rationale.strip()
            }
        except Exception as e:
            logger.error(f"Error generating proposals with text generator: {e}", exc_info=True)
            return {
                "recommendations": ["Error during fallback proposal generation."],
                "next_steps": [], "alternatives": [], "rationale": "Error."
            }

    def _generate_technical_view_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate technical view using text generator (Fallback)."""
        logger.info("Attempting to generate technical view using fallback text generator.")
        try:
            context = self._prepare_context_from_findings(findings)
            if not context: context = "No findings available for technical analysis."

            prompt = f"""Based on the research findings context below, provide a technical analysis for the query: "{query}"

Findings Context:
{context}

Please provide ONLY the following sections, clearly marked:
Technical Analysis: [Your analysis here (1-2 paragraphs)]
Technical Details: [List 3-5 details, one per line starting with '-']
Technical Challenges: [List 2-3 challenges, one per line starting with '-']
Technical Solutions: [List 2-3 solutions, one per line starting with '-']
"""
            # Use helper to call agent
            agent_inputs = {"prompt": prompt, "query": query, "context": context}
            result = self._call_dspy_agent("text_generator", agent_inputs)
            response = result.get("text", "") if result else ""
            logger.debug(f"Fallback Technical View Response:\n{response[:500]}...")
            if not response: 
                logger.warning("Fallback Technical View  empty.")
                return {"article": "Fallback failed: No response.", "key_insights": [], "limitations": [], "confidence": 0.0}

            logger.debug(f"Fallback Interpretation Response:\n{response[:200]}...")

            # Robust parsing
            technical_analysis = ""
            technical_details, technical_challenges, technical_solutions = [], [], []
            lines = response.splitlines()
            current_section = None
            for line in lines:
                line_strip = line.strip()
                if line_strip.startswith("Technical Analysis:"):
                     current_section = "analysis"; technical_analysis += line_strip[len("Technical Analysis:"):].strip()
                elif line_strip.startswith("Technical Details:"): current_section = "details"
                elif line_strip.startswith("Technical Challenges:"): current_section = "challenges"
                elif line_strip.startswith("Technical Solutions:"): current_section = "solutions"
                elif current_section == "analysis" and not line_strip.startswith(("Technical Details:", "Technical Challenges:", "Technical Solutions:")):
                     technical_analysis += "\n" + line_strip if technical_analysis else line_strip
                elif current_section == "details" and line_strip.startswith("-"): technical_details.append(line_strip[1:].strip())
                elif current_section == "challenges" and line_strip.startswith("-"): technical_challenges.append(line_strip[1:].strip())
                elif current_section == "solutions" and line_strip.startswith("-"): technical_solutions.append(line_strip[1:].strip())

            # Defaults on failure
            if not technical_analysis: technical_analysis = "Technical analysis generation failed via fallback."
            if not technical_details: technical_details = ["No details extracted."]
            if not technical_challenges: technical_challenges = ["No challenges identified."]
            if not technical_solutions: technical_solutions = ["No solutions proposed."]


            return {
                "technical_analysis": technical_analysis.strip(), "technical_details": technical_details,
                "technical_challenges": technical_challenges, "technical_solutions": technical_solutions
            }
        except Exception as e:
            logger.error(f"Error generating technical view with text generator: {e}", exc_info=True)
            return {
                "technical_analysis": "Error during fallback technical analysis generation.",
                "technical_details": [], "technical_challenges": [], "technical_solutions": []
            }

    def _generate_comprehensive_synthesis_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate comprehensive synthesis using text generator (Fallback)."""
        logger.info("Attempting to generate synthesis using fallback text generator.")
        try:
            context = self._prepare_context_from_findings(findings)
            if not context: context = "No findings available to synthesize."

            article_type = self._determine_article_type(query, findings)
            prompt = f"""Based on the research findings context below, write a comprehensive {article_type} about: "{query}"

Findings Context:
{context}

Write a well-structured, informative {article_type}. Include an introduction, key themes (use markdown headings like ## Theme), and a conclusion.
"""
            # Use helper to call agent
            agent_inputs = {"prompt": prompt, "query": query, "context": context}
            result = self._call_dspy_agent("text_generator", agent_inputs)
            article = result.get("text", "") if result else ""
            logger.debug(f"Fallback Synthesis Response:\n{article[:500]}...")


            # Basic metadata extraction
            actual_word_count = len(article.split())
            # Extract themes based on markdown headings
            key_themes = [line.strip().lstrip('#').strip() for line in article.splitlines() if line.strip().startswith("#")]
            if not key_themes: key_themes = ["Main Topic", "Key Findings", "Conclusion"] # Fallback themes

            # Defaults on failure
            if not article: article = "Comprehensive synthesis generation failed via fallback."

            return {
                "article": article, "article_type": article_type,
                "key_themes": key_themes[:5], "word_count": actual_word_count
            }
        except Exception as e:
            logger.error(f"Error generating comprehensive synthesis with text generator: {e}", exc_info=True)
            return {
                "article": "Error during fallback synthesis generation.",
                "article_type": "general", "key_themes": [], "word_count": 0
            }

    def _determine_article_type(self, query: str, findings: List[Dict]) -> str:
        """Determine the appropriate article type based on the query and findings."""
        query_lower = query.lower()

        # Keywords for different types
        scientific_keywords = ["research", "study", "experiment", "scientific", "academic", "journal", "hypothesis", "theory", "data analysis", "methodology", "findings", "results", "conclusion", "physics", "chemistry", "biology", "medicine", "engineering", "technology", "algorithm", "computation"]
        news_keywords = ["current events", "breaking", "latest", "today", "yesterday", "this week", "this month", "recent", "update", "development", "announcement", "press release", "politics", "economy", "government", "election", "policy", "legislation", "regulation", "crisis", "conflict"]
        blog_keywords = ["opinion", "perspective", "view", "experience", "personal", "lifestyle", "trend", "popular", "culture", "entertainment", "media", "social media", "influencer", "celebrity", "fashion", "food", "travel", "hobby", "review"]
        technical_keywords = ["how-to", "tutorial", "guide", "walkthrough", "step-by-step", "instruction", "manual", "documentation", "code", "programming", "software", "hardware", "system", "network", "database", "security", "implementation", "configuration", "setup", "installation"]
        philosophical_keywords = ["philosophy", "spiritual", "spirituality", "religion", "belief", "faith", "meaning", "purpose", "existence", "consciousness", "mindfulness", "meditation", "wisdom", "enlightenment", "soul", "divine", "sacred", "ritual", "practice", "tradition", "teaching", "doctrine", "theology", "metaphysics", "ethics", "moral", "value", "virtue", "transcendence", "immanence", "being", "reality"]

        keywords_map = {
            "scientific article": scientific_keywords, "news article": news_keywords,
            "blog post": blog_keywords, "technical guide": technical_keywords,
            "philosophical essay": philosophical_keywords
        }

        counts = {type_name: 0 for type_name in keywords_map}

        # Check query
        for type_name, keywords in keywords_map.items():
            counts[type_name] += sum(1 for kw in keywords if kw in query_lower)

        # Check findings content (simplified)
        all_content = " ".join([f.get("content", "")[:200] for f in findings[:5]]).lower() # Check snippets
        for type_name, keywords in keywords_map.items():
            counts[type_name] += sum(1 for kw in keywords if kw in all_content) * 0.5

        # Determine max
        if not any(counts.values()): # If all counts are zero
            return "informative article" # Default

        max_type = max(counts, key=counts.get)
        return max_type