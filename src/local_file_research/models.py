"""
API models for the local file research system.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ResearchRequest(BaseModel):
    """
    Request model for research endpoint.
    """
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results to return")
    session_id: Optional[str] = Field(None, description="Session ID")
    research_mode: str = Field("rag", description="Research mode (rag, multi_iteration)")
    report_mode: str = Field("normal", description="Report format (normal, chain_of_thought, enhanced)")
    # Note: analysis_focus is currently not used in the backend but kept for future implementation
    analysis_focus: str = Field("summarize", description="Analysis focus (summarize, answer, extract, analyze) - Not currently used")
    context_filter: Optional[str] = Field(None, description="Context filter")
    project_filter: Optional[str] = Field(None, description="Project filter")
    chain: Optional[str] = Field(None, description="Chain to use")

class ResearchResponse(BaseModel):
    """
    Response model for research endpoint.
    """
    query: str = Field(..., description="Search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    session_id: str = Field(..., description="Session ID")
    timestamp: str = Field(..., description="Timestamp")
    user: str = Field(..., description="User")
    research_mode: str = Field(..., description="Research mode")
    report_mode: str = Field(..., description="Report mode")
    report: Optional[str] = Field(None, description="Generated research report")

class IndexingRequest(BaseModel):
    """
    Request model for indexing endpoint.
    """
    session_id: Optional[str] = Field(None, description="Session ID")
    project_id: Optional[str] = Field(None, description="Project ID")
    context_filter: Optional[str] = Field(None, description="Context filter")
    external_sources: Optional[List[Dict[str, Any]]] = Field(None, description="External sources")
    chunk_size: int = Field(1024, description="Chunk size")
    chunk_overlap: int = Field(0, description="Chunk overlap")
    root_dir: Optional[str] = Field(".", description="Root directory to scan for files")

class IndexingResponse(BaseModel):
    """
    Response model for indexing endpoint.
    """
    session_id: str = Field(..., description="Session ID")
    timestamp: str = Field(..., description="Timestamp")
    user: str = Field(..., description="User")
    status: str = Field(..., description="Status")
    message: str = Field(..., description="Message")

class DocumentUploadRequest(BaseModel):
    """
    Request model for document upload endpoint.
    """
    project_id: Optional[str] = Field(None, description="Project ID")
    title: Optional[str] = Field(None, description="Document title")
    document_type: Optional[str] = Field(None, description="Document type")
    content: Optional[str] = Field(None, description="Document content")

class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload endpoint.
    """
    document_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    document_type: str = Field(..., description="Document type")
    timestamp: str = Field(..., description="Timestamp")
    user: str = Field(..., description="User")
    status: str = Field(..., description="Status")
    message: str = Field(..., description="Message")

class ProjectCreateRequest(BaseModel):
    """
    Request model for project creation endpoint.
    """
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")

class ProjectCreateResponse(BaseModel):
    """
    Response model for project creation endpoint.
    """
    project_id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    timestamp: str = Field(..., description="Timestamp")
    user: str = Field(..., description="User")
    status: str = Field(..., description="Status")
    message: str = Field(..., description="Message")
