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

        # Add special parameters for text_generator to handle large responses
        if agent_name == "text_generator":
            # Add a parameter to indicate this is a long-form response
            prepared_data["long_form"] = True
            # Add a parameter to indicate the minimum expected length
            if "prompt" in prepared_data and "2000 words" in prepared_data["prompt"]:
                prepared_data["min_length"] = 2000
                logger.info("Added min_length parameter for comprehensive article generation")

        try:
            # Log the exact fields being passed
            logger.debug(f"Calling {agent_name} with fields: {list(prepared_data.keys())}")

            # Log a preview of each field's content
            for key, value in prepared_data.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    logger.debug(f"Field '{key}' content preview: {preview}")
                elif isinstance(value, list):
                    logger.debug(f"Field '{key}' is a list with {len(value)} items.")
                elif isinstance(value, dict):
                    logger.debug(f"Field '{key}' is a dict with keys: {list(value.keys())}")
                else:
                    logger.debug(f"Field '{key}' type: {type(value)}")

            # Call the agent
            result = agent(**prepared_data)

            # Log the result
            logger.info(f"Successfully called DSPy agent '{agent_name}'. Result type: {type(result)}")

            # For text_generator, ensure we handle the response properly
            if agent_name == "text_generator":
                # If the result is a string, wrap it in a dict
                if isinstance(result, str):
                    logger.info("Converting string result to dict for text_generator")
                    result = {"text": result}
                # Special handling for DSPy Prediction objects
                elif result.__class__.__name__ == 'Prediction':
                    logger.info(f"Processing DSPy Prediction object for text_generator: {dir(result)}")
                    # Check if it has a text attribute
                    if hasattr(result, 'text') and result.text:
                        logger.info(f"Found text attribute in Prediction object, length: {len(result.text)}")
                        # Keep the original object but ensure text is accessible
                        # Don't convert to dict as we might lose access to other attributes
                    else:
                        # If no text attribute, try to extract content from other attributes
                        logger.info("No text attribute found in Prediction object, checking other attributes")
                        text_content = ""
                        # Check for other text-like attributes
                        for attr_name in ['content', 'article', 'response', 'output']:
                            if hasattr(result, attr_name) and getattr(result, attr_name):
                                attr_value = getattr(result, attr_name)
                                if isinstance(attr_value, str) and attr_value.strip():
                                    text_content = attr_value
                                    logger.info(f"Found content in {attr_name} attribute, length: {len(text_content)}")
                                    break

                        # If we found content in another attribute, add it as 'text'
                        if text_content:
                            # Add the text attribute to the Prediction object
                            result.text = text_content
                            logger.info(f"Added text attribute to Prediction object with content from {attr_name}")
                        else:
                            # Last resort: check if there's a prompt_response attribute
                            # This is specific to how DSPy might structure its Prediction objects
                            if hasattr(result, 'prompt_response') and result.prompt_response:
                                logger.info("Checking prompt_response attribute for content")
                                if isinstance(result.prompt_response, str) and result.prompt_response.strip():
                                    result.text = result.prompt_response
                                    logger.info(f"Added text attribute with content from prompt_response, length: {len(result.prompt_response)}")
                                elif hasattr(result.prompt_response, 'completion') and result.prompt_response.completion:
                                    result.text = result.prompt_response.completion
                                    logger.info(f"Added text attribute with content from prompt_response.completion, length: {len(result.prompt_response.completion)}")
                                elif hasattr(result.prompt_response, 'content') and result.prompt_response.content:
                                    result.text = result.prompt_response.content
                                    logger.info(f"Added text attribute with content from prompt_response.content, length: {len(result.prompt_response.content)}")
                                elif hasattr(result.prompt_response, '__dict__'):
                                    logger.info(f"prompt_response attributes: {result.prompt_response.__dict__}")
                                    # Try to find any string attribute with substantial content
                                    for pr_attr, pr_value in result.prompt_response.__dict__.items():
                                        if isinstance(pr_value, str) and len(pr_value) > 100:
                                            result.text = pr_value
                                            logger.info(f"Added text attribute with content from prompt_response.{pr_attr}, length: {len(pr_value)}")
                                            break
                # If the result has a __dict__ attribute, convert it to a regular dict
                elif hasattr(result, '__dict__'):
                    logger.info("Converting object to dict for text_generator")
                    result = {k: v for k, v in result.__dict__.items() if not k.startswith('_')}

            # Log the result structure and preview
            if hasattr(result, '__dict__'):
                logger.debug(f"Result attributes: {list(result.__dict__.keys())}")
                result_preview = {}
                for k, v in result.__dict__.items():
                    if k.startswith('_'):
                        continue
                    if isinstance(v, str):
                        # For long text fields, log more of the content
                        if len(v) > 1000 and (k == 'text' or k == 'article' or k == 'content'):
                            logger.debug(f"Long text field '{k}' length: {len(v)}")
                            logger.debug(f"Field '{k}' start: {v[:200]}...")
                            logger.debug(f"Field '{k}' end: ...{v[-200:]}")
                            result_preview[k] = f"{v[:100]}... [truncated {len(v)} chars] ...{v[-100:]}"
                        else:
                            result_preview[k] = f"{v[:100]}..." if len(v) > 100 else v
                    else:
                        result_preview[k] = v
                logger.debug(f"Result preview (dict): {result_preview}")
            elif isinstance(result, dict):
                logger.debug(f"Result keys: {list(result.keys())}")
                result_preview = {}
                for k, v in result.items():
                    if isinstance(v, str):
                        # For long text fields, log more of the content
                        if len(v) > 1000 and (k == 'text' or k == 'article' or k == 'content'):
                            logger.debug(f"Long text field '{k}' length: {len(v)}")
                            logger.debug(f"Field '{k}' start: {v[:200]}...")
                            logger.debug(f"Field '{k}' end: ...{v[-200:]}")
                            result_preview[k] = f"{v[:100]}... [truncated {len(v)} chars] ...{v[-100:]}"
                        else:
                            result_preview[k] = f"{v[:100]}..." if len(v) > 100 else v
                    else:
                        result_preview[k] = v
                logger.debug(f"Result preview (dict): {result_preview}")
            elif isinstance(result, str):
                 logger.debug(f"Result length (str): {len(result)}")
                 logger.debug(f"Result preview (str): {result[:200]}...")
                 if len(result) > 1000:
                     logger.debug(f"Result end (str): ...{result[-200:]}")
            else:
                 logger.debug(f"Result preview (other type): {result}")

            return result
        except Exception as e:
            logger.error(f"Error calling DSPy agent '{agent_name}': {e}", exc_info=True)
            # Log the inputs that caused the error for debugging
            logger.error(f"Inputs provided to {agent_name}: {prepared_data}")
            return None # Return None on error

    def generate_interpretations(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate interpretations of the findings."""
        # --- Start Logging ---
        logger.info("Attempting to generate interpretations...")
        # --- End Logging ---
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty for interpretations. Falling back.")
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
                logger.info("Successfully generated interpretations using InterpreterAgent.") # <<< Success Log
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
        # --- Start Logging ---
        logger.info("Attempting to generate proposals...")
        # --- End Logging ---
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty for proposals. Falling back.")
                return self._generate_proposals_with_text_generator(findings, query)

            agent_inputs = {"query": query if query else "Generate proposals based on these findings", "context": context}
            # Log the inputs we're sending to the agent
            logger.info(f"Calling proposal_generator with query: '{query[:50]}...' and context length: {len(context)}")

            # Call the agent
            result = self._call_dspy_agent("proposal_generator", agent_inputs) # Use helper

            # Log the result type and structure
            if result:
                logger.info(f"ProposalGeneratorAgent returned result of type: {type(result)}")
                if hasattr(result, 'get'):
                    for key in ['recommendations', 'next_steps', 'alternatives', 'rationale']:
                        value = result.get(key, None)
                        value_type = type(value)
                        value_preview = str(value)[:50] + "..." if value and len(str(value)) > 50 else value
                        logger.info(f"ProposalGeneratorAgent result['{key}']: {value_type} = {value_preview}")

            # +++ MODIFY RESULT CHECKING +++
            if result and hasattr(result, 'get'):
                 # Get recommendations and handle both string and list formats
                 recommendations = result.get("recommendations", "")

                 # If recommendations is a string, split it into a list by lines
                 if isinstance(recommendations, str):
                     # Split by lines and clean up
                     rec_list = [line.strip() for line in recommendations.split('\n') if line.strip()]
                     # Remove bullet points or numbering if present
                     rec_list = [rec.lstrip('-•*0123456789. ') for rec in rec_list]
                     recommendations = rec_list

                 # Filter out empty recommendations
                 meaningful_recs = [rec for rec in recommendations if rec]

                 if not meaningful_recs:
                      logger.warning("ProposalGeneratorAgent returned no meaningful recommendations. Falling back.")
                      return self._generate_proposals_with_text_generator(findings, query)

                 # Process next_steps in the same way
                 next_steps = result.get("next_steps", "")
                 if isinstance(next_steps, str):
                     # Split by lines and clean up
                     next_steps_list = [line.strip() for line in next_steps.split('\n') if line.strip()]
                     # Remove bullet points or numbering if present
                     next_steps_list = [step.lstrip('-•*0123456789. ') for step in next_steps_list]
                     next_steps = next_steps_list

                 # Process alternatives in the same way
                 alternatives = result.get("alternatives", "")
                 if isinstance(alternatives, str):
                     # Split by lines and clean up
                     alternatives_list = [line.strip() for line in alternatives.split('\n') if line.strip()]
                     # Remove bullet points or numbering if present
                     alternatives_list = [alt.lstrip('-•*0123456789. ') for alt in alternatives_list]
                     alternatives = alternatives_list

                 # Success case
                 logger.info("Successfully generated proposals using ProposalGeneratorAgent.") # <<< Success Log
                 return {
                    "recommendations": meaningful_recs,
                    "next_steps": next_steps if isinstance(next_steps, list) else [],
                    "alternatives": alternatives if isinstance(alternatives, list) else [],
                    "rationale": result.get("rationale", "")
                 }
            else:
                logger.warning(f"ProposalGeneratorAgent returned None or invalid type ({type(result)}): {result}. Falling back.")
                return self._generate_proposals_with_text_generator(findings, query)
        except Exception as e:
            logger.error(f"Error generating proposals: {e}", exc_info=True)
            return self._generate_proposals_with_text_generator(findings, query)

    def generate_technical_view(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate technical view of the findings."""
        # --- Start Logging ---
        logger.info("Attempting to generate technical view...")
        # --- End Logging ---
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty for technical view. Falling back.")
                return self._generate_technical_view_with_text_generator(findings, query) # Fallback

            agent_inputs = {
                "query": query if query else "Generate technical view based on these findings",
                "context": context,
            }

            result = self._call_dspy_agent("technical_analyzer", agent_inputs)

            if result and hasattr(result, 'get'):
                analysis = result.get("analysis", "")
                if not analysis or len(analysis) < 10:
                     logger.warning(f"TechnicalAnalyzerAgent returned short/empty analysis: '{str(analysis)[:50]}...'. Falling back.")
                     return self._generate_technical_view_with_text_generator(findings, query)
                # Success case
                logger.info("Successfully generated technical view using TechnicalAnalyzerAgent.") # <<< Success Log
                return {"technical_analysis": analysis, "technical_details": result.get("details",[]), "technical_challenges": result.get("challenges",[]), "technical_solutions": result.get("solutions",[])}
            else:
                 logger.warning(f"TechnicalAnalyzerAgent returned invalid result ({type(result)}): {result}. Falling back.");
                 return self._generate_technical_view_with_text_generator(findings, query)
        except Exception as e: logger.error(f"Error generating tech view: {e}", exc_info=True); return self._generate_technical_view_with_text_generator(findings, query)

    # +++ ADD DETAILED LOGGING +++
    def generate_comprehensive_synthesis(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """Generate a comprehensive synthesis of the findings."""
        # --- Start Logging ---
        logger.info("Attempting to generate comprehensive synthesis...")
        # --- End Logging ---
        try:
            context = self._prepare_context_from_findings(findings)
            if not context:
                logger.warning("Prepared context is empty. Cannot generate synthesis. Falling back.")
                return self._generate_comprehensive_synthesis_with_text_generator(findings, query)

            agent_inputs = {
                "query": query if query else "Synthesize these findings",
                "context": context,
            }

            # --- Log Agent Call Start ---
            logger.info(f"Calling content_synthesizer for query: '{query[:50]}...' with context length {len(context)}")
            # --- End Log ---

            result = self._call_dspy_agent("content_synthesizer", agent_inputs)

            # --- Log Agent Call Result ---
            if result:
                 logger.info(f"ContentSynthesizerAgent returned result of type: {type(result)}")
                 if hasattr(result, 'get'):
                      article_preview = str(result.get('article', ''))[:100] + "..."
                      article_type = result.get('article_type', 'N/A')
                      key_themes = result.get('key_themes', 'N/A')
                      word_count = result.get('word_count', 'N/A')
                      logger.info(f"  Article Preview: {article_preview}")
                      logger.info(f"  Article Type: {article_type}, Key Themes: {key_themes}, Word Count: {word_count}")
            else:
                 logger.warning("ContentSynthesizerAgent returned None.")
            # --- End Log ---

            if result and hasattr(result, 'get'):
                article = result.get("article", "")
                # --- ADJUSTED LENGTH CHECK ---
                if not article or len(article.split()) < 10: # Check word count instead of char length
                    logger.warning(f"ContentSynthesizerAgent returned short/empty article (words: {len(article.split())}). Falling back.")
                    return self._generate_comprehensive_synthesis_with_text_generator(findings, query)
                # --- END ADJUSTED CHECK ---
                word_count = result.get("word_count", len(article.split()))
                # Success case
                logger.info("Successfully generated synthesis using ContentSynthesizerAgent.") # <<< Success Log
                return {"article": article, "article_type": result.get("article_type", "general"), "key_themes": result.get("key_themes", []), "word_count": word_count}
            else:
                 logger.warning(f"ContentSynthesizerAgent returned invalid result ({type(result)}): {result}. Falling back.")
                 return self._generate_comprehensive_synthesis_with_text_generator(findings, query)
        except Exception as e:
             logger.error(f"Error generating synthesis: {e}", exc_info=True)
             return self._generate_comprehensive_synthesis_with_text_generator(findings, query)

    def generate_comprehensive_article(self, findings: List[Dict], query: str, all_queries: List[str] = None,
                                      interpretations: Dict = None, proposals: Dict = None,
                                      technical_view: Dict = None) -> Dict[str, Any]:
        """
        Generate a comprehensive article that covers all findings and report sections.

        Args:
            findings: List of research findings
            query: The main research query
            all_queries: List of all queries including expanded and follow-up questions
            interpretations: Optional interpretations section data
            proposals: Optional proposals section data
            technical_view: Optional technical view section data

        Returns:
            Dictionary with article content and metadata
        """
        # --- Start Logging ---
        logger.info("Attempting to generate comprehensive article...")
        # --- End Logging ---

        try:
            # Determine the appropriate article type based on all queries and findings
            article_type = self._determine_article_type(query, findings, all_queries)
            logger.info(f"Determined article type for comprehensive article: {article_type}")

            # Prepare context from findings with reference numbers
            context_parts = []
            logger.info(f"Preparing context from {len(findings)} findings for comprehensive article.")

            # Add a header explaining the reference format
            context_parts.append("# Research Findings with Reference Numbers")
            context_parts.append("The following findings are available for citation in your article. Use the reference numbers [Ref: X] when citing specific information.")
            context_parts.append("")

            # Add findings with reference numbers
            for i, finding in enumerate(findings, 1):
                ref = finding.get('reference_number', i)

                # Get the most informative content
                summary = ""
                analysis = finding.get('analysis', {})
                if isinstance(analysis, dict) and analysis.get('summary'):
                    summary = analysis.get('summary')
                elif finding.get('summary'):
                    summary = finding.get('summary')

                content = finding.get('content', '')

                # Add source information
                citation = finding.get('citation', {})
                source_name = citation.get('source_name', citation.get('source_path', 'Unknown Source'))

                context_parts.append(f"## Finding [Ref: {ref}]")
                context_parts.append(f"Source: {source_name}")

                if summary:
                    context_parts.append(f"Summary: {summary}")

                # Add content snippet
                content_snippet = content[:5000] + ("..." if len(content) > 5000 else "")
                context_parts.append(f"Content: {content_snippet}")

                # Add any additional analysis or insights
                if isinstance(analysis, dict):
                    key_points = analysis.get('key_points', [])
                    if key_points:
                        context_parts.append("Key Points:")
                        for point in key_points:
                            context_parts.append(f"- {point}")

                # Add agent chain results if available
                agent_fields = {
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
                        context_parts.append(f"{title}:")
                        if isinstance(value, list):
                            for item in value:
                                context_parts.append(f"- {item}")
                        else:
                            context_parts.append(str(value))

                context_parts.append("")  # Add blank line between findings

            # Add interpretations if available
            if interpretations:
                context_parts.append("# Interpretations")
                if interpretations.get('main_interpretation'):
                    context_parts.append(interpretations.get('main_interpretation'))

                key_insights = interpretations.get('key_insights', [])
                if key_insights:
                    context_parts.append("Key Insights:")
                    for insight in key_insights:
                        context_parts.append(f"- {insight}")

                limitations = interpretations.get('limitations', [])
                if limitations:
                    context_parts.append("Limitations:")
                    for limitation in limitations:
                        context_parts.append(f"- {limitation}")

                context_parts.append("")

            # Add proposals if available
            if proposals:
                context_parts.append("# Proposals and Recommendations")

                recommendations = proposals.get('recommendations', [])
                if recommendations:
                    context_parts.append("Recommendations:")
                    for rec in recommendations:
                        context_parts.append(f"- {rec}")

                next_steps = proposals.get('next_steps', [])
                if next_steps:
                    context_parts.append("Next Steps:")
                    for step in next_steps:
                        context_parts.append(f"- {step}")

                alternatives = proposals.get('alternatives', [])
                if alternatives:
                    context_parts.append("Alternatives:")
                    for alt in alternatives:
                        context_parts.append(f"- {alt}")

                rationale = proposals.get('rationale', '')
                if rationale:
                    context_parts.append(f"Rationale: {rationale}")

                context_parts.append("")

            # Add technical view if available
            if technical_view:
                context_parts.append("# Technical Analysis")
                if technical_view.get('technical_analysis'):
                    context_parts.append(technical_view.get('technical_analysis'))

                details = technical_view.get('technical_details', [])
                if details:
                    context_parts.append("Technical Details:")
                    for detail in details:
                        context_parts.append(f"- {detail}")

                challenges = technical_view.get('technical_challenges', [])
                if challenges:
                    context_parts.append("Technical Challenges:")
                    for challenge in challenges:
                        context_parts.append(f"- {challenge}")

                solutions = technical_view.get('technical_solutions', [])
                if solutions:
                    context_parts.append("Technical Solutions:")
                    for solution in solutions:
                        context_parts.append(f"- {solution}")

                context_parts.append("")

            # Combine all context parts
            context = "\n".join(context_parts)

            # Limit total context length if needed
            MAX_CONTEXT_LEN = 2000000  # Increased for comprehensive article
            if len(context) > MAX_CONTEXT_LEN:
                logger.warning(f"Context for comprehensive article exceeds {MAX_CONTEXT_LEN} characters. Truncating.")
                context = context[:MAX_CONTEXT_LEN] + "\n... [Context Truncated]"

            # Create a prompt for the article generation
            all_queries_text = ""
            if all_queries and len(all_queries) > 1:
                all_queries_text = "\n".join([f"- {q}" for q in all_queries])
                all_queries_text = f"\n\nAll related queries:\n{all_queries_text}"

            prompt = f"""Based on the comprehensive research findings and analysis provided, write a detailed {article_type} about: "{query}"{all_queries_text}

The article should:
1. Be at least 2000 words in length
2. Include proper citations using the reference numbers provided in the format [Ref: X]
3. Cover all major aspects of the topic thoroughly
4. Be well-structured with appropriate headings and subheadings
5. Include an introduction, body sections, and conclusion
6. Synthesize information from multiple sources
7. Provide balanced coverage of different perspectives
8. Be written in a professional, authoritative tone appropriate for a {article_type}

Research Findings and Analysis:
{context}

IMPORTANT INSTRUCTIONS:
- Use the exact article type specified: {article_type}
- Include at least 10 citations to specific findings using the reference numbers [Ref: X]
- Ensure the article is at least 2000 words
- After the article, provide the metadata as specified below
- Do not use separators "---" in the article

Format your response as follows:

[Write your article here with proper formatting, paragraphs, section headings, and citations]

---

Article Type: {article_type}

Key Themes:
- [Theme 1]
- [Theme 2]
- [Theme 3]
- [Theme 4]
- [Theme 5]

Word Count: [Approximate word count]
"""

            # Call the text generator with the comprehensive prompt
            logger.info(f"Calling text_generator for comprehensive article. Prompt length: {len(prompt)}")
            agent_inputs = {"prompt": prompt, "query": query, "context": context}

            # Use a special parameter to ensure we get the full response
            # This is critical for long articles
            result = self._call_dspy_agent("text_generator", agent_inputs)

            # Log detailed information about the result
            if result:
                logger.info(f"Text generator result type: {type(result)}")
                if hasattr(result, '__dict__'):
                    logger.info(f"Result attributes: {list(result.__dict__.keys())}")
                elif isinstance(result, dict):
                    logger.info(f"Result keys: {list(result.keys())}")

            # Get the text from the result, ensuring we handle all possible formats
            response = ""
            if result:
                # Log more detailed information about the result type
                logger.info(f"Processing result of type: {type(result)}, class name: {result.__class__.__name__}")

                # Special handling for DSPy Prediction objects
                if result.__class__.__name__ == 'Prediction':
                    logger.info(f"Handling DSPy Prediction object with attributes: {dir(result)}")
                    # Try to access the text attribute directly
                    if hasattr(result, 'text') and result.text:
                        response = result.text
                        logger.info(f"Extracted text from Prediction.text attribute, length: {len(response)}")
                    # Check for prompt_response attribute (common in DSPy)
                    elif hasattr(result, 'prompt_response') and result.prompt_response:
                        logger.info("Found prompt_response attribute in Prediction object")
                        if isinstance(result.prompt_response, str) and result.prompt_response.strip():
                            response = result.prompt_response
                            logger.info(f"Extracted text from Prediction.prompt_response, length: {len(response)}")
                        elif hasattr(result.prompt_response, 'completion') and result.prompt_response.completion:
                            response = result.prompt_response.completion
                            logger.info(f"Extracted text from Prediction.prompt_response.completion, length: {len(response)}")
                        elif hasattr(result.prompt_response, 'content') and result.prompt_response.content:
                            response = result.prompt_response.content
                            logger.info(f"Extracted text from Prediction.prompt_response.content, length: {len(response)}")
                        elif hasattr(result.prompt_response, '__dict__'):
                            logger.info(f"prompt_response attributes: {result.prompt_response.__dict__}")
                            # Try to find any string attribute with substantial content
                            for pr_attr, pr_value in result.prompt_response.__dict__.items():
                                if isinstance(pr_value, str) and len(pr_value) > 100:
                                    response = pr_value
                                    logger.info(f"Extracted text from Prediction.prompt_response.{pr_attr}, length: {len(response)}")
                                    break
                    # If no text attribute or it's empty, try to get all attributes
                    elif hasattr(result, '__dict__'):
                        logger.info(f"Prediction object attributes: {result.__dict__}")
                        # Look for text-like attributes
                        for attr_name in ['text', 'content', 'article', 'response', 'output']:
                            if hasattr(result, attr_name) and getattr(result, attr_name):
                                attr_value = getattr(result, attr_name)
                                if isinstance(attr_value, str) and attr_value.strip():
                                    response = attr_value
                                    logger.info(f"Found text in Prediction.{attr_name}, length: {len(response)}")
                                    break

                # Handle dictionary-like objects
                elif isinstance(result, dict) or hasattr(result, 'get'):
                    # Try to get text from dictionary-like object
                    response = result.get("text", "")
                    logger.info(f"Extracted text from dictionary 'text' key, length: {len(response)}")

                    # Check if there's a 'full_text' or 'complete_text' field that might contain the full article
                    if not response or len(response.split()) < 500:  # If text is short, look for alternatives
                        for alt_field in ['full_text', 'complete_text', 'content', 'article', 'response', 'output']:
                            if alt_field in result and result[alt_field]:
                                alt_text = result[alt_field]
                                if isinstance(alt_text, str) and len(alt_text.split()) > len(response.split()):
                                    logger.info(f"Using alternative field '{alt_field}' which has more content, length: {len(alt_text)}")
                                    response = alt_text
                                    break

                # Handle objects with text attribute
                elif hasattr(result, 'text'):
                    # Try to get text from object with text attribute
                    response = result.text
                    logger.info(f"Extracted text from object.text attribute, length: {len(response)}")

                # Handle string results
                elif isinstance(result, str):
                    # If result is already a string, use it directly
                    response = result
                    logger.info(f"Result is already a string, length: {len(response)}")

                # Last resort: try to convert the result to a string
                else:
                    try:
                        # Try to extract __str__ representation
                        response = str(result)
                        logger.info(f"Converted result to string using str(), length: {len(response)}")

                        # If the string representation is too short or looks like an object reference,
                        # try to extract attributes
                        if len(response) < 100 and ('<' in response and '>' in response):
                            logger.info("String representation looks like an object reference, trying to extract attributes")
                            if hasattr(result, '__dict__'):
                                attrs = result.__dict__
                                logger.info(f"Object attributes: {list(attrs.keys())}")

                                # Look for text-like attributes
                                for attr_name, attr_value in attrs.items():
                                    if isinstance(attr_value, str) and len(attr_value) > 100:
                                        response = attr_value
                                        logger.info(f"Found text in attribute '{attr_name}', length: {len(response)}")
                                        break
                    except Exception as e:
                        logger.error(f"Failed to convert result to string: {e}")
                        response = ""

            # Log the response
            logger.info(f"Text generator response length for comprehensive article: {len(response)}")
            if response:
                # Log more of the response to help with debugging
                preview_length = min(1000, len(response))
                logger.debug(f"Text generator response preview:\n{response[:preview_length]}...")
                # Also log the end of the response to see if it's being truncated
                if len(response) > 1000:
                    logger.debug(f"End of response:\n...{response[-500:]}")
            else:
                logger.warning("Text generator returned no response for comprehensive article.")
                return {
                    "article": "Failed to generate comprehensive article: No response from text generator.",
                    "article_type": article_type,
                    "key_themes": [],
                    "word_count": 0
                }

            # Parse response
            article = ""
            metadata_str = ""

            # Log the full response structure to help with debugging
            logger.info(f"Response length: {len(response)}")
            logger.info(f"Response word count: {len(response.split())}")

            # Safety check - if response is empty or not a string, handle gracefully
            if not response or not isinstance(response, str):
                logger.error(f"Invalid response type: {type(response)}. Cannot parse article.")
                return {
                    "article": "Failed to generate comprehensive article: Invalid response format.",
                    "article_type": "general",
                    "key_themes": [],
                    "word_count": 0
                }

            try:
                # Initialize variables
                separator_found = False
                article = ""
                metadata_str = ""

                # PRIORITY CHECK: First check for "---" separator as this is the most common format
                if "---" in response:
                    logger.info("Found '---' separator in response, attempting to split")

                    # Split at the first occurrence of "---"
                    parts = response.split("---", 1)
                    if len(parts) > 1:
                        # Check if what follows looks like metadata
                        second_part = parts[1].strip()

                        # If the second part contains any metadata markers, use this split
                        if any(marker in second_part.lower() for marker in ["article type:", "key themes:", "word count:"]):
                            article = parts[0].strip()
                            metadata_str = "---" + second_part  # Include the separator in metadata
                            separator_found = True
                            logger.info("Successfully split at '---' separator with metadata following")
                        else:
                            # Even if no standard metadata markers, if it's short and comes after a substantial article,
                            # it's likely metadata
                            if len(parts[0].split()) > 100 and len(second_part.split()) < 50:
                                article = parts[0].strip()
                                metadata_str = "---" + second_part
                                separator_found = True
                                logger.info("Split at '---' separator based on content length patterns")

                    # If we couldn't determine a clean split, log this for debugging
                    if not separator_found:
                        logger.info("Found '---' but couldn't determine a clean split. First part length: " +
                                   f"{len(parts[0].split())} words, Second part length: {len(parts[1].split())} words")

                # If "---" separator didn't work, check for other separator formats
                if not separator_found:
                    separators = ["***", "___", "\n\n\n", "Article Type:", "Key Themes:"]

                    for separator in separators:
                        if separator in response:
                            logger.info(f"Found separator: '{separator}'")

                            if separator in ["Article Type:", "Key Themes:"]:
                                # These are part of the metadata, so we need to find where the article ends
                                # Look for a section with multiple newlines before these markers
                                lines = response.split("\n")
                                article_end_idx = 0

                                for i in range(len(lines) - 3):
                                    if i + 2 < len(lines) and (separator in lines[i] or
                                        separator in lines[i+1] or
                                        separator in lines[i+2]):
                                        # Look for empty lines before this point
                                        for j in range(i-1, max(0, i-10), -1):
                                            if j > 0 and j-1 >= 0 and not lines[j].strip() and not lines[j-1].strip():
                                                article_end_idx = j
                                                break
                                        if article_end_idx > 0:
                                            break

                                if article_end_idx > 0:
                                    article = "\n".join(lines[:article_end_idx]).strip()
                                    metadata_str = "\n".join(lines[article_end_idx:]).strip()
                                    separator_found = True
                                    logger.info(f"Split at line {article_end_idx} based on metadata markers")
                                    break
                            else:
                                # Standard separator
                                parts = response.split(separator, 1)
                                article = parts[0].strip()
                                metadata_str = parts[1].strip() if len(parts) > 1 else ""
                                separator_found = True
                                break

                # Check if the response contains a proper article with headings
                if not separator_found:
                    lines = response.split("\n")

                    # Look for markdown headings (# Title) which indicate article structure
                    heading_lines = []
                    for i, line in enumerate(lines):
                        if line.strip().startswith('#') and not line.strip().startswith('##'):
                            heading_lines.append(i)

                    # If we found headings, this is likely an article
                    if heading_lines and len(heading_lines) >= 2:
                        logger.info(f"Found {len(heading_lines)} main headings in the response, treating as article")

                        # Look for metadata markers after the last heading section
                        metadata_start = -1
                        for i in range(len(lines) - 1, heading_lines[-1], -1):
                            if i < len(lines) and ("---" in lines[i] or
                                                "Article Type:" in lines[i] or
                                                "Key Themes:" in lines[i] or
                                                "Word Count:" in lines[i]):
                                metadata_start = i
                                # Look for a clean break point before this metadata
                                for j in range(i-1, max(0, i-5), -1):
                                    if j >= 0 and not lines[j].strip():
                                        metadata_start = j
                                        break
                                break

                        if metadata_start > 0:
                            article = "\n".join(lines[:metadata_start]).strip()
                            metadata_str = "\n".join(lines[metadata_start:]).strip()
                            separator_found = True
                            logger.info(f"Split at line {metadata_start} based on article structure and metadata markers")
                        else:
                            # No metadata found, but we have article structure
                            article = response.strip()
                            logger.info("Found article structure but no metadata section")

                    # If we still haven't found a separator, try to find a natural break point
                    if not separator_found:
                        # Try to find a natural break point - look for "References" or "Conclusion" followed by empty lines
                        markers = ["References", "Bibliography", "Conclusion", "Final Thoughts", "Summary"]

                        for marker in markers:
                            for i in range(len(lines) - 5):
                                if i + 3 < len(lines) and marker in lines[i] and lines[i].strip().startswith(("#", "##", "###")) and not lines[i+3].strip():
                                    # Found a section header followed by content and then empty lines
                                    # This might be the end of the article content
                                    for j in range(i+3, min(len(lines), i+10)):
                                        if j < len(lines) and ("Article Type:" in lines[j] or "Key Themes:" in lines[j] or "Word Count:" in lines[j]):
                                            article = "\n".join(lines[:j]).strip()
                                            metadata_str = "\n".join(lines[j:]).strip()
                                            separator_found = True
                                            logger.info(f"Split at line {j} based on content structure")
                                            break
                                    if separator_found:
                                        break
                            if separator_found:
                                break

                # Last resort: assume whole response is article if separator missing
                if not separator_found:
                    article = response.strip()
                    logger.warning("No metadata separator found in text_generator response. Assuming entire output is the article.")

                    # Try to extract metadata from the end if it looks like metadata
                    lines = article.split("\n")
                    for i in range(max(0, len(lines) - 15), len(lines)):
                        if i < len(lines) and ("Article Type:" in lines[i] or "Key Themes:" in lines[i] or "Word Count:" in lines[i]):
                            # Found metadata at the end
                            for j in range(i-1, max(0, i-5), -1):
                                if j >= 0 and not lines[j].strip():
                                    article = "\n".join(lines[:j]).strip()
                                    metadata_str = "\n".join(lines[j:]).strip()
                                    logger.info(f"Extracted metadata from end of response at line {j}")
                                    break
                            break
            except Exception as e:
                logger.error(f"Error parsing article response: {e}", exc_info=True)
                # If parsing fails, use the entire response as the article
                article = response.strip()
                metadata_str = ""

            # Calculate actual word count
            actual_word_count = len(article.split())
            logger.info(f"Comprehensive article generated. Actual word count: {actual_word_count}")

            # Check if word count meets minimum requirement
            if actual_word_count < 2000:
                logger.warning(f"Generated article is shorter than required (got {actual_word_count} words, needed 2000+)")
                # We'll still return it, but log the warning

            # Initialize default values
            parsed_article_type = article_type
            key_themes = []

            # Parse metadata if available
            if metadata_str:
                logger.info(f"Parsing metadata: {metadata_str}")

                # Check if metadata starts with "---" and remove it
                if metadata_str.startswith("---"):
                    metadata_str = metadata_str[3:].strip()
                    logger.info("Removed leading '---' from metadata")

                lines = metadata_str.splitlines()
                current_section = None
                in_themes_section = False

                # First pass: look for standard metadata markers
                for line in lines:
                    line_strip = line.strip()
                    if not line_strip:
                        continue
                    line_lower = line_strip.lower()

                    # Check for article type
                    if "article type:" in line_lower:
                        # Extract everything after "Article Type:"
                        type_text = line_strip[line_lower.find("article type:") + len("article type:"):].strip()
                        if type_text:
                            parsed_article_type = type_text
                            logger.info(f"Found article type: {parsed_article_type}")

                    # Check for key themes section
                    elif "key themes:" in line_lower:
                        current_section = "themes"
                        in_themes_section = True
                        logger.info("Found key themes section")

                    # Check for word count
                    elif "word count:" in line_lower:
                        # Try to extract the word count for logging purposes
                        count_text = line_strip[line_lower.find("word count:") + len("word count:"):].strip()
                        try:
                            reported_count = int(''.join(c for c in count_text if c.isdigit()))
                            logger.info(f"Reported word count in metadata: {reported_count}")
                        except:
                            logger.info(f"Could not parse reported word count: {count_text}")

                    # Process theme items
                    elif current_section == "themes" or in_themes_section:
                        # Look for theme items (could be with various bullet formats)
                        if line_strip.startswith(("-", "*", "•", "○", "◦", "▪", "▫", "■", "□", "1.", "2.", "3.")):
                            # Extract the theme text after the bullet
                            theme_text = line_strip.lstrip("-*•○◦▪▫■□0123456789. ").strip()
                            if theme_text:
                                key_themes.append(theme_text)
                                logger.info(f"Found theme: {theme_text}")
                        elif not line_strip.lower().startswith(("article", "word count")):
                            # If it's not a bullet but also not a new section, it might be a theme without a bullet
                            if theme_text := line_strip:
                                key_themes.append(theme_text)
                                logger.info(f"Found theme without bullet: {theme_text}")

                # If we didn't find any themes but have metadata, try to extract them differently
                if not key_themes and metadata_str:
                    # Look for themes in the metadata text
                    theme_section = False
                    for line in lines:
                        line_strip = line.strip()
                        if not line_strip:
                            continue

                        if "key themes" in line_strip.lower():
                            theme_section = True
                            continue

                        if theme_section and not line_strip.lower().startswith(("article type", "word count")):
                            # This might be a theme
                            theme_text = line_strip.lstrip("-*•○◦▪▫■□0123456789. ").strip()
                            if theme_text:
                                key_themes.append(theme_text)
                                logger.info(f"Found theme in second pass: {theme_text}")

                # If we still don't have themes, look for any lines that might be themes
                if not key_themes:
                    logger.info("No themes found in standard format, looking for any potential theme lines")
                    for line in lines:
                        line_strip = line.strip()
                        # Skip empty lines, headers, and metadata markers
                        if (not line_strip or
                            line_strip.lower().startswith(("article type", "word count", "key themes")) or
                            line_strip == "---"):
                            continue

                        # If it looks like a bullet point or numbered item
                        if (line_strip.startswith(("-", "*", "•", "○", "◦", "▪", "▫", "■", "□")) or
                            (len(line_strip) > 2 and line_strip[0].isdigit() and line_strip[1] == '.')):
                            theme_text = line_strip.lstrip("-*•○◦▪▫■□0123456789. ").strip()
                            if theme_text and len(theme_text) > 3:  # Ensure it's not just a short marker
                                key_themes.append(theme_text)
                                logger.info(f"Found potential theme: {theme_text}")
                                # Limit to 5 themes
                                if len(key_themes) >= 5:
                                    break

                # If we still don't have an article type, try to extract it from the article content
                if parsed_article_type == article_type:
                    # Look for phrases like "In this [type]" or "This [type] explores"
                    article_lower = article.lower()
                    for type_name in ["article", "essay", "analysis", "guide", "report", "review", "overview", "summary", "paper"]:
                        if f"this {type_name}" in article_lower or f"in this {type_name}" in article_lower:
                            parsed_article_type = f"{article_type.split()[0]} {type_name}"
                            logger.info(f"Extracted article type from content: {parsed_article_type}")
                            break

            # If no themes were parsed, try to extract from article
            if not key_themes:
                logger.warning("Could not parse key themes from metadata. Extracting from article headings.")
                lines = article.split("\n")
                potential_headings = [line.strip() for line in lines if line.strip() and line.strip().startswith(('#', '*'))]
                # Extract text after heading marker
                key_themes = [h.lstrip('#* ').strip() for h in potential_headings if h.lstrip('#* ').strip()][:5]

                if not key_themes:
                    key_themes = ["Main topic analysis", "Key findings", "Important considerations"]

            key_themes = key_themes[:5]  # Limit to 5 themes

            # Check if article generation was successful
            if not article or actual_word_count < 100:
                logger.error("Comprehensive article generation failed: Generated article was too short.")
                return {
                    "article": "Failed to generate comprehensive article: Generated content was too short.",
                    "article_type": parsed_article_type,
                    "key_themes": key_themes,
                    "word_count": actual_word_count
                }

            logger.info("Successfully generated comprehensive article.")
            return {
                "article": article,
                "article_type": parsed_article_type,
                "key_themes": key_themes,
                "word_count": actual_word_count
            }

        except Exception as e:
            logger.error(f"Error generating comprehensive article: {e}", exc_info=True)
            return {
                "article": f"Error generating comprehensive article: {str(e)}",
                "article_type": "general",
                "key_themes": ["Error in article generation process."],
                "word_count": 0
            }
    # +++ END ADDED LOGGING +++

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
                 finding_text = content[:1500] + ("..." if len(content) > 1500 else "") # Use snippet of content

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
        MAX_CONTEXT_LEN = 150000
        if len(generated_context) > MAX_CONTEXT_LEN:
             generated_context = generated_context[:MAX_CONTEXT_LEN] + "\n... [Context Truncated]"
             logger.warning(f"Truncated prepared context to {MAX_CONTEXT_LEN} characters.")

        # --- ADD CONTEXT LOGGING ---
        logger.info(f"Generated context string length: {len(generated_context)}")
        if not generated_context or len(generated_context) < 100:
             logger.warning("Generated context is very short or empty.")
             # Provide a minimal context if completely empty
             if not generated_context.strip():
                  logger.error("Context generation resulted in an empty string!") # <<< ERROR Log
                  return "No findings available to generate context."
        logger.debug(f"Prepared context preview: {generated_context[:500]}...") # Log context preview
        # --- END CONTEXT LOGGING ---

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

            logger.info("Successfully generated interpretations using text_generator fallback.") # <<< Success Log
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

Please provide ONLY the following sections, clearly marked with section headers:

Recommendations:
- [First recommendation]
- [Second recommendation]
- [Third recommendation]
- [Fourth recommendation]
- [Fifth recommendation]

Next Steps:
- [First next step]
- [Second next step]
- [Third next step]

Alternatives:
- [First alternative]
- [Second alternative]

Rationale:
[Your rationale here (1-2 paragraphs)]

IMPORTANT: Format each recommendation, next step, and alternative as a separate line with a dash (-) at the beginning. Do not use numbering.
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
                if not line_strip: continue

                # Check for section headers (more robust detection)
                lower_line = line_strip.lower()
                if "recommendation" in lower_line and (":" in line_strip or line_strip.endswith("recommendations")):
                    current_section = "recommendations"
                    continue
                elif ("next step" in lower_line or "next steps" in lower_line) and (":" in line_strip or line_strip.endswith("steps")):
                    current_section = "next_steps"
                    continue
                elif "alternative" in lower_line and (":" in line_strip or line_strip.endswith("alternatives")):
                    current_section = "alternatives"
                    continue
                elif "rationale" in lower_line and (":" in line_strip or line_strip.endswith("rationale")):
                    current_section = "rationale"
                    rationale += line_strip[line_strip.find(":") + 1:].strip() if ":" in line_strip else ""
                    continue

                # Add content to appropriate section
                if current_section == "recommendations":
                    # Check if line starts with a bullet point, number, or other marker
                    clean_line = line_strip.lstrip('-•*0123456789.) ').strip()
                    if clean_line:  # Only add non-empty lines
                        recommendations.append(clean_line)
                elif current_section == "next_steps":
                    clean_line = line_strip.lstrip('-•*0123456789.) ').strip()
                    if clean_line:
                        next_steps.append(clean_line)
                elif current_section == "alternatives":
                    clean_line = line_strip.lstrip('-•*0123456789.) ').strip()
                    if clean_line:
                        alternatives.append(clean_line)
                elif current_section == "rationale" and not any(header in lower_line for header in ["recommendation", "next step", "alternative"]):
                    rationale += "\n" + line_strip if rationale else line_strip

            # Defaults on failure
            if not recommendations: recommendations = ["Proposal generation failed via fallback."]
            if not next_steps: next_steps = ["No next steps extracted via fallback."]
            if not alternatives: alternatives = ["No alternatives extracted via fallback."]
            if not rationale: rationale = "Rationale generation failed via fallback."

            logger.info("Successfully generated proposals using text_generator fallback.") # <<< Success Log
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
                logger.warning("Fallback Technical View is empty.")
                return {"technical_analysis": "Fallback failed: No response.", "technical_details": [], "technical_challenges": [], "technical_solutions": []}

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

            logger.info("Successfully generated technical view using text_generator fallback.") # <<< Success Log
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

    # +++ ADD DETAILED LOGGING +++
    def _generate_comprehensive_synthesis_with_text_generator(self, findings: List[Dict], query: str) -> Dict[str, Any]:
        """
        Generate comprehensive synthesis using text generator (Fallback).

        Args:
            findings: List of findings from research
            query: Original research query

        Returns:
            Dictionary with synthesis content and metadata
        """
        # --- Start Logging ---
        logger.info("Attempting fallback: synthesis with text_generator.")
        # --- End Logging ---
        try:
            text_generator = DSPyAgentRegistry.get_agent("text_generator")
            if text_generator:
                # Prepare context
                context = self._prepare_context_from_findings(findings)
                if not context:
                     logger.warning("Context is empty for fallback synthesis generation.")
                     context = "No findings available to synthesize."

                # Determine the appropriate article type based on the query and findings
                article_type = self._determine_article_type(query, findings)
                logger.info(f"Determined article type for fallback synthesis: {article_type}")

                # --- MODIFY PROMPT WORD COUNT (removed exact target) ---
                prompt = f"""Based on the following research findings, write a comprehensive {article_type} (around 2000 words) about: "{query}"

Findings:
{context}

Please write a well-structured, informative, and engaging {article_type} that synthesizes all the information above.
The {article_type} should include:

1. An attention-grabbing introduction
2. Well-organized sections with appropriate headings
3. In-depth analysis of key themes and concepts
4. Relevant examples, data, or case studies from the research findings
5. A thoughtful conclusion

IMPORTANT INSTRUCTIONS:
- Use the exact article type I've specified: {article_type}
- Do not label or prefix your article with "Article:" - just write the content directly
- After the article, provide the metadata as specified below
- Do not use seperators "---" in the article

Format your response as follows:

[Write your article here with proper formatting, paragraphs, and section headings]

---

Article Type: {article_type}

Key Themes:
- [Theme 1]
- [Theme 2]
- [Theme 3]
- [Theme 4]
- [Theme 5]

Word Count: [Approximate word count]
"""
                # --- END MODIFIED PROMPT ---

                # --- Log Agent Call ---
                logger.info(f"Calling text_generator for synthesis fallback. Prompt length: {len(prompt)}")
                agent_inputs = {"prompt": prompt, "query": query, "context": context} # Added query/context
                # --- End Log ---

                result = self._call_dspy_agent("text_generator", agent_inputs) # Use helper
                response = result.get("text", "") if result and hasattr(result, 'get') else ""

                # --- Log Agent Response ---
                logger.info(f"TextGeneratorAgent fallback response length: {len(response)}")
                if response:
                    logger.debug(f"TextGeneratorAgent fallback response preview:\n{response[:500]}...")
                else:
                    logger.warning("TextGeneratorAgent fallback for synthesis returned no response.")
                    return {
                        "article": "Fallback synthesis failed: No response from text generator.",
                        "article_type": article_type, "key_themes": [], "word_count": 0
                    }
                # --- End Log ---

                # Parse response
                article = ""
                metadata_str = ""
                # --- IMPROVED PARSING ---
                if "---" in response:
                    parts = response.split("---", 1)
                    article = parts[0].strip()
                    metadata_str = parts[1].strip()
                else: # Assume whole response is article if separator missing
                    article = response.strip()
                    logger.warning("Metadata separator '---' not found in text_generator response. Assuming entire output is the article.")

                # Calculate actual word count
                actual_word_count = len(article.split())
                logger.info(f"Fallback article generated. Actual word count: {actual_word_count}")

                # Initialize default values
                parsed_article_type = article_type
                key_themes = []

                # --- ROBUST METADATA PARSING ---
                if metadata_str:
                    logger.info(f"Parsing metadata: {metadata_str}")
                    lines = metadata_str.splitlines()
                    current_section = None
                    for line in lines:
                        line_strip = line.strip()
                        if not line_strip: continue
                        line_lower = line_strip.lower()

                        if line_lower.startswith("article type:"):
                             parsed_article_type = line_strip[len("Article Type:"):].strip() or article_type
                        elif line_lower.startswith("key themes:"):
                             current_section = "themes"
                        elif line_lower.startswith("word count:"):
                             # Use the actual calculated word count, ignore the LLM's count
                             pass
                        elif current_section == "themes" and line_strip.startswith("-"):
                             theme = line_strip[1:].strip()
                             if theme: key_themes.append(theme)

                # If no themes were parsed, try to extract from article
                if not key_themes:
                    logger.warning("Could not parse key themes from metadata. Extracting from article headings.")
                    lines = article.split("\n")
                    potential_headings = [line.strip() for line in lines if line.strip() and line.strip().startswith(('#', '*'))]
                    # Extract text after heading marker
                    key_themes = [h.lstrip('#* ').strip() for h in potential_headings if h.lstrip('#* ').strip()][:5]

                    if not key_themes:
                        key_themes = ["Main topic analysis", "Key findings", "Important considerations"]
                # --- END ROBUST PARSING ---

                key_themes = key_themes[:5] # Limit to 5 themes

                # --- Check generated article length ---
                if not article or actual_word_count < 10:
                     logger.error("Fallback synthesis generated an empty or very short article. Returning error.")
                     return {
                        "article": "Fallback synthesis failed: Generated article was too short.",
                        "article_type": parsed_article_type, "key_themes": key_themes, "word_count": actual_word_count
                     }
                # --- End Check ---

                logger.info("Successfully generated synthesis using text_generator fallback.") # <<< Success Log
                return {
                    "article": article,
                    "article_type": parsed_article_type,
                    "key_themes": key_themes,
                    "word_count": actual_word_count
                }
            else:
                # Default response if no text generator
                logger.error("Text generator agent not found for fallback synthesis.")
                return {
                    "article": "No comprehensive synthesis available due to missing text generation capabilities.",
                    "article_type": "general", "key_themes": ["No themes available."], "word_count": 0
                }
        except Exception as e:
            logger.error(f"Error generating comprehensive synthesis with text generator: {e}", exc_info=True)
            return {
                "article": "Error generating comprehensive synthesis via fallback.",
                "article_type": "general", "key_themes": ["Error in synthesis generation process."], "word_count": 0
            }
    # +++ END ADDED LOGGING +++

    def _determine_article_type(self, query: str, findings: List[Dict], all_queries: List[str] = None) -> str:
        """
        Determine the appropriate article type based on the query, findings, and all related queries.

        Args:
            query: The main research query
            findings: List of research findings
            all_queries: Optional list of all queries including expanded and follow-up questions

        Returns:
            The determined article type
        """
        # --- Start Logging ---
        logger.info(f"Determining article type for query: '{query[:50]}...'")
        # --- End Logging ---

        # Ensure all_queries is a list
        if all_queries is None:
            all_queries = [query]

        # Convert all queries to lowercase for matching
        all_queries_lower = [q.lower() for q in all_queries]

        # Keywords for different types (expanded with more keywords)
        scientific_keywords = [
            "research", "study", "experiment", "scientific", "academic", "journal", "hypothesis",
            "theory", "data analysis", "methodology", "findings", "results", "conclusion",
            "physics", "chemistry", "biology", "medicine", "engineering", "technology",
            "algorithm", "computation", "evidence", "empirical", "statistical", "quantitative",
            "qualitative", "laboratory", "clinical", "peer-reviewed", "publication", "thesis",
            "dissertation", "analysis", "synthesis", "investigation", "observation", "measurement"
        ]

        news_keywords = [
            "current events", "breaking", "latest", "today", "yesterday", "this week", "this month",
            "recent", "update", "development", "announcement", "press release", "politics",
            "economy", "government", "election", "policy", "legislation", "regulation", "crisis",
            "conflict", "report", "incident", "event", "situation", "headline", "news", "journalist",
            "media coverage", "press", "broadcast", "publication", "bulletin", "alert", "flash",
            "briefing", "statement", "declaration", "communiqué", "dispatch"
        ]

        blog_keywords = [
            "opinion", "perspective", "view", "experience", "personal", "lifestyle", "trend",
            "popular", "culture", "entertainment", "media", "social media", "influencer",
            "celebrity", "fashion", "food", "travel", "hobby", "review", "blog", "post",
            "article", "commentary", "reflection", "thought", "musing", "diary", "journal",
            "log", "chronicle", "narrative", "story", "anecdote", "impression", "reaction",
            "response", "feedback", "critique", "criticism", "evaluation", "assessment"
        ]

        technical_keywords = [
            "how-to", "tutorial", "guide", "walkthrough", "step-by-step", "instruction", "manual",
            "documentation", "code", "programming", "software", "hardware", "system", "network",
            "database", "security", "implementation", "configuration", "setup", "installation",
            "technical", "technology", "tool", "application", "platform", "framework", "library",
            "API", "interface", "protocol", "standard", "specification", "architecture", "design",
            "pattern", "best practice", "recommendation", "solution", "troubleshooting", "debugging"
        ]

        philosophical_keywords = [
            "philosophy", "spiritual", "spirituality", "religion", "belief", "faith", "meaning",
            "purpose", "existence", "consciousness", "mindfulness", "meditation", "wisdom",
            "enlightenment", "soul", "divine", "sacred", "ritual", "practice", "tradition",
            "teaching", "doctrine", "theology", "metaphysics", "ethics", "moral", "value",
            "virtue", "transcendence", "immanence", "being", "reality", "ontology", "epistemology",
            "logic", "reason", "rationality", "intuition", "insight", "reflection", "contemplation",
            "introspection", "phenomenology", "existentialism", "idealism", "materialism", "dualism"
        ]

        educational_keywords = [
            "education", "learning", "teaching", "instruction", "curriculum", "course", "lesson",
            "study", "school", "college", "university", "academy", "training", "workshop", "seminar",
            "lecture", "class", "classroom", "student", "teacher", "professor", "instructor",
            "educator", "pedagogy", "andragogy", "didactic", "educational", "academic", "scholarly",
            "intellectual", "cognitive", "knowledge", "skill", "competence", "ability", "capability"
        ]

        business_keywords = [
            "business", "company", "corporation", "enterprise", "organization", "firm", "startup",
            "industry", "sector", "market", "economy", "finance", "investment", "profit", "revenue",
            "sales", "marketing", "advertising", "branding", "product", "service", "customer",
            "client", "consumer", "management", "leadership", "strategy", "planning", "operations",
            "logistics", "supply chain", "human resources", "HR", "recruitment", "hiring", "talent"
        ]

        keywords_map = {
            "scientific article": scientific_keywords,
            "news article": news_keywords,
            "blog post": blog_keywords,
            "technical guide": technical_keywords,
            "philosophical essay": philosophical_keywords,
            "educational resource": educational_keywords,
            "business analysis": business_keywords
        }

        counts = {type_name: 0 for type_name in keywords_map}

        # Check all queries with higher weight for the main query
        for type_name, keywords in keywords_map.items():
            # Check main query with higher weight (2.0)
            main_query_lower = all_queries_lower[0]
            counts[type_name] += sum(2.0 for kw in keywords if kw in main_query_lower)

            # Check all other queries with normal weight (1.0)
            for query_lower in all_queries_lower[1:]:
                counts[type_name] += sum(1.0 for kw in keywords if kw in query_lower)

        # Check findings content more comprehensively
        # First, gather all content and summaries
        all_content = []
        all_summaries = []

        for finding in findings:
            if finding.get("content"):
                all_content.append(finding.get("content", "")[:1000].lower())  # Use more content

            # Include summaries which often contain key concepts
            if finding.get("summary"):
                all_summaries.append(finding.get("summary", "").lower())

            # Include analysis summaries if available
            analysis = finding.get("analysis", {})
            if isinstance(analysis, dict) and analysis.get("summary"):
                all_summaries.append(analysis.get("summary", "").lower())

        # Combine all content and summaries
        combined_text = " ".join(all_content + all_summaries)

        # Check combined text against keywords
        for type_name, keywords in keywords_map.items():
            # Use a higher weight for findings content (1.5) as it's more indicative
            counts[type_name] += sum(1.5 for kw in keywords if kw in combined_text)

        # Determine max with a preference for more substantial article types
        if not any(counts.values()):  # If all counts are zero
            determined_type = "informative article"  # Default
        else:
            # Get the top 2 types
            sorted_types = sorted(counts.items(), key=lambda x: x[1], reverse=True)

            # If scientific or technical are close to the top, prefer them
            if len(sorted_types) >= 2:
                top_type, top_score = sorted_types[0]
                second_type, second_score = sorted_types[1]

                # If the second type is scientific or technical and close to the top score,
                # prefer it over blog posts or news articles
                if (second_type in ["scientific article", "technical guide"] and
                    top_type in ["blog post", "news article"] and
                    second_score >= top_score * 0.8):
                    determined_type = second_type
                else:
                    determined_type = top_type
            else:
                determined_type = sorted_types[0][0]

        # --- Log Result ---
        logger.info(f"Determined article type: {determined_type} (scores: {counts})")
        # --- End Log ---
        return determined_type
