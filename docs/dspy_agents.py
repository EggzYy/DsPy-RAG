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

# --- Document Type Signatures ---

if DSPY_AVAILABLE:
    # Code Document Analysis Signature
    class CodeAnalysisSignature(dspy.Signature):
        """Analyze code documents and extract key information."""
        code = dspy.InputField(desc="The code content to analyze")
        language = dspy.InputField(desc="The programming language of the code")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        content = dspy.InputField(desc="Alternative content source (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of what the code does")
        functions = dspy.OutputField(desc="List of key functions/methods in the code")
        classes = dspy.OutputField(desc="List of key classes in the code")
        dependencies = dspy.OutputField(desc="List of external dependencies or imports")
        complexity = dspy.OutputField(desc="Assessment of code complexity")
        issues = dspy.OutputField(desc="Potential issues or bugs in the code")
        suggestions = dspy.OutputField(desc="Suggestions for improvement")

    # Spreadsheet Analysis Signature
    class SpreadsheetAnalysisSignature(dspy.Signature):
        """Analyze spreadsheet data and extract key information."""
        data = dspy.InputField(desc="The spreadsheet content in text format")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        content = dspy.InputField(desc="Alternative content source (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of the data")
        structure = dspy.OutputField(desc="Description of the data structure (columns, sheets)")
        key_metrics = dspy.OutputField(desc="Key metrics or statistics from the data")
        patterns = dspy.OutputField(desc="Patterns or trends identified in the data")
        anomalies = dspy.OutputField(desc="Anomalies or outliers in the data")
        insights = dspy.OutputField(desc="Key insights derived from the data")

    # PDF Document Analysis Signature
    class PDFAnalysisSignature(dspy.Signature):
        """Analyze PDF documents and extract key information."""
        content = dspy.InputField(desc="The PDF content in text format")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of the document")
        key_points = dspy.OutputField(desc="Key points from the document")
        entities = dspy.OutputField(desc="Important entities mentioned in the document")
        topics = dspy.OutputField(desc="Main topics covered in the document")
        structure = dspy.OutputField(desc="Document structure (sections, headings)")
        citations = dspy.OutputField(desc="Citations or references in the document")

    # Technical Document Analysis Signature
    class TechnicalDocAnalysisSignature(dspy.Signature):
        """Analyze technical documents and extract key information."""
        content = dspy.InputField(desc="The technical document content")
        document_type = dspy.InputField(desc="The type of technical document (e.g., API doc, whitepaper)")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of the technical document")
        key_concepts = dspy.OutputField(desc="Key technical concepts explained in the document")
        technical_details = dspy.OutputField(desc="Important technical details or specifications")
        requirements = dspy.OutputField(desc="Requirements or prerequisites mentioned")
        examples = dspy.OutputField(desc="Code examples or usage examples")
        limitations = dspy.OutputField(desc="Limitations or constraints mentioned")

    # Research Paper Analysis Signature
    class ResearchPaperAnalysisSignature(dspy.Signature):
        """Analyze research papers and extract key information."""
        content = dspy.InputField(desc="The research paper content")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary of the research paper")
        research_question = dspy.OutputField(desc="The main research question or objective")
        methodology = dspy.OutputField(desc="The methodology used in the research")
        findings = dspy.OutputField(desc="Key findings or results")
        limitations = dspy.OutputField(desc="Limitations of the research")
        implications = dspy.OutputField(desc="Implications or applications of the research")
        future_work = dspy.OutputField(desc="Suggested future work")

    # Chain-of-Thought Document Analysis
    class ChainOfThoughtAnalysisSignature(dspy.Signature):
        """Perform step-by-step reasoning on a document."""
        content = dspy.InputField(desc="The document content to analyze")
        query = dspy.InputField(desc="The query or task to focus the analysis on")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Additional context for analysis (optional)", default="")

        reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
        key_insights = dspy.OutputField(desc="Key insights derived from the reasoning")
        conclusion = dspy.OutputField(desc="Conclusion based on the reasoning")

    # Multi-Document Synthesis
    class MultiDocumentSynthesisSignature(dspy.Signature):
        """Synthesize information from multiple documents."""
        documents = dspy.InputField(desc="List of document contents and their metadata")
        query = dspy.InputField(desc="The query or task to focus the synthesis on")
        document = dspy.InputField(desc="Single document content (optional)", default="")
        content = dspy.InputField(desc="Alternative content source (optional)", default="")

        synthesis = dspy.OutputField(desc="Synthesized information from all documents")
        common_themes = dspy.OutputField(desc="Common themes across documents")
        contradictions = dspy.OutputField(desc="Contradictions or disagreements between documents")
        unique_insights = dspy.OutputField(desc="Unique insights from specific documents")
        integrated_view = dspy.OutputField(desc="Integrated view of the information")

    # --- ADDED MISSING SIGNATURES FROM dspy_config.py ---
    class SummarizerSignature(dspy.Signature):
        """Summarize the provided content."""
        content = dspy.InputField(desc="The text content to summarize")
        query = dspy.InputField(desc="Optional query to focus the summary", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        context = dspy.InputField(desc="Alternative context (optional)", default="")

        summary = dspy.OutputField(desc="A concise summary")

    class AnswererSignature(dspy.Signature):
        """Answer a question based on the provided context."""
        context = dspy.InputField(desc="Context relevant to the question")
        query = dspy.InputField(desc="The question to answer")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        content = dspy.InputField(desc="Alternative content source (optional)", default="")

        answer = dspy.OutputField(desc="The answer derived from the context")

    class ExtractorSignature(dspy.Signature):
        """Extract specific information (e.g., entities, keywords) from content."""
        content = dspy.InputField(desc="The text content to extract from")
        query = dspy.InputField(desc="Description of the information to extract (e.g., 'names of people')")
        context = dspy.InputField(desc="Context relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        info = dspy.OutputField(desc="The extracted information")

    class FactCheckerSignature(dspy.Signature):
        """Assess the factual consistency of a statement against provided context."""
        statement = dspy.InputField(desc="The statement to fact-check")
        context = dspy.InputField(desc="The context to check against")
        query = dspy.InputField(desc="Optional focus for fact-checking", default="")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        fact_check = dspy.OutputField(desc="Assessment of factual consistency (e.g., 'Consistent', 'Inconsistent', 'Needs More Info') with explanation")

    class DocumentAnalysisSignature(dspy.Signature):
        """Provide a structured analysis of a document."""
        document = dspy.InputField(desc="The document text")
        query = dspy.InputField(desc="Optional query to guide analysis", default="Analyze this document")
        context = dspy.InputField(desc="Optional additional context", default="")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")

        summary = dspy.OutputField(desc="Overall summary")
        key_points = dspy.OutputField(desc="List of key points")
        entities = dspy.OutputField(desc="List of important entities")
        sentiment = dspy.OutputField(desc="Overall sentiment (e.g., Positive, Negative, Neutral)")

    class InterpreterSignature(dspy.Signature):
        """Interpret research findings contextually."""
        query = dspy.InputField(desc="The original research query")
        context = dspy.InputField(desc="Aggregated research findings")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        interpretation = dspy.OutputField(desc="Main interpretation")
        insights = dspy.OutputField(desc="List of key insights")
        limitations = dspy.OutputField(desc="List of limitations")
        confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")

    class ProposalGeneratorSignature(dspy.Signature):
        """Generate actionable proposals from findings."""
        query = dspy.InputField(desc="Original research query")
        context = dspy.InputField(desc="Research findings context")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        recommendations = dspy.OutputField(desc="List of specific recommendations")
        next_steps = dspy.OutputField(desc="List of concrete next steps")
        alternatives = dspy.OutputField(desc="List of alternative approaches")
        rationale = dspy.OutputField(desc="Justification for proposals")

    class TechnicalAnalyzerSignature(dspy.Signature):
        """Provide technical analysis of findings."""
        query = dspy.InputField(desc="Original research query")
        context = dspy.InputField(desc="Research findings context")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        analysis = dspy.OutputField(desc="Technical analysis summary")
        details = dspy.OutputField(desc="List of technical details")
        challenges = dspy.OutputField(desc="List of technical challenges")
        solutions = dspy.OutputField(desc="List of potential solutions")

    # --- CORRECTED QueryRefinementSignature ---
    class QueryRefinementSignature(dspy.Signature):
        """Generate refined/follow-up queries."""
        instructions = 'Generate refined/follow-up queries.' # Moved instructions here

        query = dspy.InputField(desc="Original query")
        context = dspy.InputField(desc="Current research context/knowledge")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        num_queries: int = dspy.InputField(desc="Number of queries to generate") # Corrected type hint
        iteration: int = dspy.InputField(desc="Current research iteration") # Corrected type hint

        related_queries = dspy.OutputField(desc="List of refined/follow-up queries")
    # --- END CORRECTION ---

    class FactVerificationSignature(dspy.Signature):
        """Verify content consistency against a summary."""
        content = dspy.InputField(desc="Content snippet to verify")
        summary = dspy.InputField(desc="Overall summary to verify against")
        context = dspy.InputField(desc="Context relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        query = dspy.InputField(desc="Optional focus for verification", default="Verify facts")

        is_consistent = dspy.OutputField(desc="Boolean indicating consistency")
        confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")
        notes = dspy.OutputField(desc="Explanation for consistency assessment")

    class TextGeneratorSignature(dspy.Signature):
        """Generic text generation based on a prompt."""
        prompt = dspy.InputField(desc="The input prompt")
        context = dspy.InputField(desc="Optional context", default="")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")
        query = dspy.InputField(desc="Optional query focus", default="")

        text = dspy.OutputField(desc="Generated text")

    class ContentSynthesizerSignature(dspy.Signature):
        """Synthesize findings into a structured article."""
        query = dspy.InputField(desc="Original research query")
        context = dspy.InputField(desc="Research findings context")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        article = dspy.OutputField(desc="Synthesized article content")
        article_type = dspy.OutputField(desc="Type of article (e.g., summary, analysis, blog post)")
        key_themes = dspy.OutputField(desc="List of key themes discussed")
        word_count = dspy.OutputField(desc="Approximate word count")

    class QueryExpansionSignature(dspy.Signature):
        """Expand a query with related terms/concepts."""
        query = dspy.InputField(desc="Query to expand")
        context = dspy.InputField(desc="Optional context", default="")
        content = dspy.InputField(desc="Content relevant to the question (optional)", default="")
        document = dspy.InputField(desc="Alternative document content (optional)", default="")

        expanded_query = dspy.OutputField(desc="Expanded query string")

# --- Agent Implementation ---

def register_custom_agents():
    """Register custom DSPy agents for different document types."""
    if not DSPY_AVAILABLE or not DSPY_CONFIGURED:
        logger.warning("DSPy not available or configured. Cannot register custom agents.")
        return


    # Register code analysis agent
    code_analyzer = dspy.Predict(CodeAnalysisSignature)
    DSPyAgentRegistry.register_agent("code_analyzer", code_analyzer)

    # Register spreadsheet analysis agent
    spreadsheet_analyzer = dspy.Predict(SpreadsheetAnalysisSignature)
    DSPyAgentRegistry.register_agent("spreadsheet_analyzer", spreadsheet_analyzer)

    # Register PDF analysis agent
    pdf_analyzer = dspy.Predict(PDFAnalysisSignature)
    DSPyAgentRegistry.register_agent("pdf_analyzer", pdf_analyzer)

    # Register technical document analysis agent
    tech_doc_analyzer = dspy.Predict(TechnicalDocAnalysisSignature)
    DSPyAgentRegistry.register_agent("tech_doc_analyzer", tech_doc_analyzer)

    # Register research paper analysis agent
    research_paper_analyzer = dspy.Predict(ResearchPaperAnalysisSignature)
    DSPyAgentRegistry.register_agent("research_paper_analyzer", research_paper_analyzer)

    # Register chain-of-thought analysis agent
    cot_analyzer = dspy.Predict(ChainOfThoughtAnalysisSignature)
    DSPyAgentRegistry.register_agent("cot_analyzer", cot_analyzer)

    # Register multi-document synthesis agent
    multi_doc_synthesizer = dspy.Predict(MultiDocumentSynthesisSignature)
    DSPyAgentRegistry.register_agent("multi_doc_synthesizer", multi_doc_synthesizer)

    # Register agent chains
    DSPyAgentRegistry.register_chain("code_analysis_chain", ["code_analyzer"])
    DSPyAgentRegistry.register_chain("spreadsheet_analysis_chain", ["spreadsheet_analyzer"])
    DSPyAgentRegistry.register_chain("pdf_analysis_chain", ["pdf_analyzer"])
    DSPyAgentRegistry.register_chain("tech_doc_analysis_chain", ["tech_doc_analyzer"])
    DSPyAgentRegistry.register_chain("research_paper_analysis_chain", ["research_paper_analyzer"])
    DSPyAgentRegistry.register_chain("cot_analysis_chain", ["cot_analyzer"])
    DSPyAgentRegistry.register_chain("multi_doc_synthesis_chain", ["multi_doc_synthesizer"])

    # Register combined chains
    DSPyAgentRegistry.register_chain("code_review_chain", ["code_analyzer", "cot_analyzer"])
    DSPyAgentRegistry.register_chain("data_analysis_chain", ["spreadsheet_analyzer", "cot_analyzer"])
    DSPyAgentRegistry.register_chain("research_review_chain", ["research_paper_analyzer", "cot_analyzer"])

    logger.info("Custom DSPy agents registered successfully")

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