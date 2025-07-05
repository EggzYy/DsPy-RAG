"""
Document Analysis Module for Local File Deep Research.

This module provides specialized document analysis capabilities for different file types.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from .dspy_config import DSPyAgentRegistry, DSPY_CONFIGURED
import re
# Configure logging
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """
    Document analyzer that uses DSPy agents to analyze different types of documents.
    """

    def __init__(self):
        pass # Init should be empty or handle non-DSPy setup

    def analyze_document(self, content: str, file_path: str = None, query: str = None) -> Dict[str, Any]:
        """
        Analyze a document using the appropriate analyzer based on file type.

        Args:
            content: The document content to analyze
            file_path: Optional path to the file (used to determine file type)
            query: Optional query to focus the analysis

        Returns:
            Dictionary with analysis results
        """
        try:
            # Determine file type from extension
            file_type = "text"
            if file_path:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".pdf", ".doc", ".docx"]:
                    file_type = "document"
                elif ext in [".py", ".js", ".java", ".c", ".cpp", ".h", ".cs", ".php"]:
                    file_type = "code"
                elif ext in [".csv", ".xls", ".xlsx", ".json"]:
                    file_type = "data"
                elif ext in [".md", ".txt"]:
                    file_type = "text"
            
            # +++ ADD CHECK FOR DSPY CONFIGURATION +++
            if not DSPY_CONFIGURED:
                logger.warning("DSPy not configured. Using basic analysis for analyze_document.")
                return self._basic_analysis(content, query)
            
            # Get the document analyzer agent
            analyzer = DSPyAgentRegistry.get_agent("document_analyzer")
            if not analyzer:
                logger.warning("Document analyzer agent not found. Using basic analysis.")
                return self._basic_analysis(content, query)

            # Analyze the document
            context = f"File type: {file_type}"
            if query:
                context += f"\nQuery: {query}"

            # Provide all required fields for document_analyzer
            result = analyzer(
                document=content[:10000],  # Limit to 10000 chars to avoid token limits
                context=context,
                query=query if query else "Analyze this document",  # Use query or default
                content=content[:10000]  # Add content field to avoid missing field warning
            )

            # Process result
            # +++ MODIFY RESULT CHECKING +++
            if result and hasattr(result, 'get'):
                 analysis = {
                     "summary": result.get("summary", ""),
                     "key_points": self._ensure_list(result.get("key_points", [])),
                     "entities": self._ensure_list(result.get("entities", [])),
                     "sentiment": result.get("sentiment", "neutral")
                 }
                 # (Keep file type specific analysis calls)
                 if file_type == "code": analysis["code_analysis"] = self._analyze_code(content, ext[1:] if ext else "")
                 elif file_type == "data": analysis["data_analysis"] = self._analyze_data(content)
                 return analysis
            else:
                 logger.warning(f"DocumentAnalyzer agent returned unexpected result type: {type(result)}. Falling back.")
                 return self._basic_analysis(content, query)
        
        except Exception as e:
            logger.error(f"Error analyzing document: {e}", exc_info=True)
            return self._basic_analysis(content, query) # Fallback on error

    def _ensure_list(self, value):
        """Ensure the value is a list."""
        if isinstance(value, str):
            # Improved splitting for lists that might be newline or comma-separated
            items = re.split(r'\n|,', value) # Use regex to split by newline OR comma
            # Clean up items: remove leading/trailing whitespace, list markers, periods
            return [item.strip().lstrip('-* ').rstrip('.') for item in items if item.strip()]
        elif isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()] # Ensure items are strings and stripped
        else:
            return [] # Return empty list for other types

    def _basic_analysis(self, content: str, query: str = None) -> Dict[str, Any]:
        """
        Perform basic analysis without DSPy agents.

        Args:
            content: The document content to analyze
            query: Optional query to focus the analysis

        Returns:
            Dictionary with basic analysis results
        """
        # Simple word count and keyword extraction
        words = content.split()
        word_count = len(words)

        # Extract potential keywords (simple approach)
        word_freq = {}
        for word in words:
            word = word.lower().strip(".,!?;:()")
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        # Create a more meaningful summary
        summary = f"This document contains {word_count} words."

        # Add query-focused summary if query is provided
        if query:
            # Find sentences that might contain the query terms
            query_terms = set(query.lower().split())
            sentences = content.split('.')
            relevant_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_words = set(sentence.lower().split())
                if any(term in sentence_words for term in query_terms):
                    relevant_sentences.append(sentence)

            if relevant_sentences:
                summary += f" Based on the query '{query}', the following information was found:\n"
                summary += "\n".join(relevant_sentences[:3]) + "..."

        # Create more meaningful key points
        key_points = []

        # Add document statistics
        key_points.append(f"Document contains {word_count} words")

        # Add top keywords as key points
        key_points.append(f"Top keywords: {', '.join([word for word, _ in keywords[:5]])}")

        # Add query-related key point if query is provided
        if query:
            key_points.append(f"Analysis focused on: {query}")

        # Add document structure key point
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        key_points.append(f"Document contains approximately {len(paragraphs)} paragraphs")

        return {
            "summary": summary,
            "key_points": key_points,
            "entities": [word for word, _ in keywords[:5]],
            "sentiment": "neutral"
        }

    def _analyze_code(self, content: str, language: str) -> Dict[str, Any]:
        """
        Analyze code content.

        Args:
            content: The code content to analyze
            language: The programming language

        Returns:
            Dictionary with code analysis results
        """
        if not DSPY_CONFIGURED: return {"analysis": "Code analysis requires DSPy."}
        try:
            # Try to use the chain_of_thought agent for code analysis
            cot_agent = DSPyAgentRegistry.get_agent("chain_of_thought")
            if cot_agent:
                # Provide all required fields for chain_of_thought
                code_content = f"Code in {language}:\n{content[:5000]}"
                result = cot_agent(
                    content=code_content,
                    document=code_content,  # Duplicate content in document field
                    context=code_content,   # Use content as context
                    query=f"Analyze this {language} code"  # Default query
                )
                return {"analysis": result.get("cot", "")}
            else:
                # Basic code analysis
                lines = content.split("\n")
                line_count = len(lines)
                return {
                    "analysis": f"Code file with {line_count} lines of {language} code."
                }
        except Exception as e:
            logger.error(f"Error analyzing code: {e}", exc_info=True)
            return {"analysis": f"Error analyzing code: {e}"}

    def _analyze_data(self, content: str) -> Dict[str, Any]:
        """
        Analyze data content (CSV, Excel, etc.).

        Args:
            content: The data content to analyze

        Returns:
            Dictionary with data analysis results
        """
        if not DSPY_CONFIGURED: return {"analysis": "Data analysis requires DSPy."}
        try:
            # Try to use the extractor agent for data analysis
            extractor = DSPyAgentRegistry.get_agent("extractor")
            if extractor:
                # Provide all required fields for extractor
                result = extractor(
                    content=content[:5000],
                    document=content[:5000],  # Duplicate content in document field
                    context=content[:5000],   # Use content as context
                    query="Extract key information from this data"  # Default query
                )
                return {"analysis": result.get("info", "")}
            else:
                # Basic data analysis
                lines = content.split("\n")
                line_count = len(lines)
                return {
                    "analysis": f"Data file with {line_count} rows."
                }
        except Exception as e:
            logger.error(f"Error analyzing data: {e}", exc_info=True)
            return {"analysis": f"Error analyzing data: {e}"}
