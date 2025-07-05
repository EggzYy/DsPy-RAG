# filename: src/local_file_research/dspy_agents.py
"""
Custom DSPy agents for Local File Deep Research.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
import json

# Configure logging
logger = logging.getLogger(__name__)

# Initialize DSPy
try:
    import dspy
    from .dspy_config import DSPyAgentRegistry
    DSPY_AVAILABLE = True
except ImportError:
    logger.warning("DSPy not available. Custom agents will not be functional.")
    DSPY_AVAILABLE = False
    DSPY_CONFIGURED = False # Ensure this is False if import fails

class DocumentTypeError(Exception):
    """Exception raised when a document type is not supported."""
    pass

def analyze_document(content: str, document_type: str, query: str = None) -> Dict[str, Any]:
    """
    Analyze a document using the appropriate DSPy agent based on document type.

    Args:
        content: Document content
        document_type: Type of document (code, spreadsheet, pdf, tech_doc, research_paper)
        query: Optional query to focus the analysis

    Returns:
        Analysis results

    Raises:
        DocumentTypeError: If the document type is not supported
    """
    if not DSPY_AVAILABLE or not DSPY_CONFIGURED:
        logger.warning("DSPy not available or configured. Returning basic analysis for analyze_document.")
        from .document_analysis import DocumentAnalyzer; analyzer = DocumentAnalyzer()
        return analyzer._basic_analysis(content, query) # Use fallback


    # Determine the appropriate agent based on document type
    agent = None
    if document_type == "code":
        agent = DSPyAgentRegistry.get_agent("code_analyzer")
        # Detect language from content or file extension
        language = _detect_language(content)
        # Add all required fields with non-empty values
        result = agent(
            code=content[:10000],
            language=language,
            query=query or "Analyze this code",
            # Add all required fields with non-empty values
            document=content[:10000],  # Duplicate content in document field
            content=content[:10000],   # Duplicate content in content field
            context=content[:10000]    # Use content as context
        )
    elif document_type == "spreadsheet":
        agent = DSPyAgentRegistry.get_agent("spreadsheet_analyzer")
        # Add all required fields with non-empty values
        result = agent(
            data=content[:10000],
            query=query or "Analyze this spreadsheet data",
            # Add all required fields with non-empty values
            document=content[:10000],  # Duplicate content in document field
            content=content[:10000],   # Duplicate content in content field
            context=content[:10000]    # Use content as context
        )
    elif document_type == "pdf":
        agent = DSPyAgentRegistry.get_agent("pdf_analyzer")
        # Add all required fields with non-empty values
        result = agent(
            content=content[:10000],
            query=query or "Analyze this PDF document",
            # Add all required fields with non-empty values
            document=content[:10000],  # Duplicate content in document field
            context=content[:10000]    # Use content as context
        )
    elif document_type == "tech_doc":
        agent = DSPyAgentRegistry.get_agent("tech_doc_analyzer")
        # Add all required fields with non-empty values
        result = agent(
            content=content[:10000],
            document_type="technical document",
            query=query or "Analyze this technical document",
            # Add all required fields with non-empty values
            document=content[:10000],  # Duplicate content in document field
            context=content[:10000]    # Use content as context
        )
    elif document_type == "research_paper":
        agent = DSPyAgentRegistry.get_agent("research_paper_analyzer")
        # Add all required fields with non-empty values
        result = agent(
            content=content[:10000],
            query=query or "Analyze this research paper",
            # Add all required fields with non-empty values
            document=content[:10000],  # Duplicate content in document field
            context=content[:10000]    # Use content as context
        )
    else:
        raise DocumentTypeError(f"Unsupported document type: {document_type}")

    # Convert result to dictionary
    if hasattr(result, "__dict__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    else:
        return result

def synthesize_documents(documents: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]:
    """
    Synthesize information from multiple documents.

    Args:
        documents: List of document contents and metadata
        query: Optional query to focus the synthesis

    Returns:
        Synthesis results
    """
    if not DSPY_AVAILABLE or not DSPY_CONFIGURED:
        return {"error": "DSPy not available or configured."}


    # Get the multi-document synthesis agent
    agent = DSPyAgentRegistry.get_agent("multi_doc_synthesizer")
    if not agent:
        return {"error": "Multi-document synthesis agent not available"}

    # Format documents for the agent
    formatted_docs = []
    for doc in documents:
        formatted_docs.append({
            "content": doc.get("content", "")[:5000],  # Limit content size
            "title": doc.get("title", "Untitled"),
            "type": doc.get("type", "unknown")
        })

    # Convert to JSON string for the agent
    docs_json = json.dumps(formatted_docs)

    # Run the agent with all required fields
    # Use the first document's content as a fallback for document and content fields
    first_doc_content = formatted_docs[0]["content"] if formatted_docs else ""

    result = agent(
        documents=docs_json,
        query=query or "Synthesize these documents",
        # Add any other required fields with non-empty values
        document=first_doc_content,  # Use first document's content
        content=first_doc_content     # Use first document's content
    )

    # Convert result to dictionary
    if hasattr(result, "__dict__"):
        return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
    else:
        return result

def _detect_language(content: str) -> str:
    """
    Detect the programming language from code content.

    Args:
        content: Code content

    Returns:
        Detected language
    """
    # Simple language detection based on keywords and syntax
    content_lower = content.lower()

    if "def " in content and "import " in content and ("self" in content or ":" in content):
        return "python"
    elif "function " in content and ("{" in content and "}" in content):
        return "javascript"
    elif "public class " in content or "private class " in content:
        return "java"
    elif "#include" in content and ("int main" in content or "void main" in content):
        return "c/c++"
    elif "using namespace" in content or "std::" in content:
        return "c++"
    elif "<?php" in content:
        return "php"
    elif "<html" in content or "<!doctype html" in content_lower:
        return "html"
    elif "@interface" in content or "@implementation" in content:
        return "objective-c"
    elif "func " in content and ("let " in content or "var " in content):
        return "swift"
    elif "package " in content and "import " in content and "{" in content:
        return "go"
    else:
        return "unknown"