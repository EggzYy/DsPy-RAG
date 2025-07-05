"""
API endpoints for document management.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import os
import json
import uuid
import time
from datetime import datetime

# Import local modules
from src.local_file_research.document_manager import save_file_to_database, get_document_by_id, get_documents
from src.local_file_research.storage_manager import read_document_content
from src.local_file_research.auth import get_current_user

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents(
    project_id: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    List all documents, optionally filtered by project_id.
    """
    try:
        # Get documents, filtered by project_id if provided
        documents = get_documents(project_id=project_id)

        # If we have a project_id but no documents were found directly,
        # try to get documents from the registry
        if project_id and not documents:
            try:
                from .document_registry import get_project_documents
                registry_docs = get_project_documents(project_id)

                # If we found documents in the registry, convert them to the expected format
                if registry_docs:
                    # Get the full document details for each document
                    documents = []
                    for doc_id in registry_docs:
                        doc = get_document_by_id(doc_id)
                        if doc:
                            documents.append(doc)

                    logger.info(f"Found {len(documents)} documents for project {project_id} in registry")
            except Exception as e:
                logger.warning(f"Error getting documents from registry: {e}")

        return documents
    except Exception as e:
        logger.error(f"Error in list_documents endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/documents/{document_id}", response_model=Dict[str, Any])
async def get_document(
    document_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get a document by ID.
    """
    try:
        document = get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        return document
    except Exception as e:
        logger.error(f"Error in get_document endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@router.post("/documents/upload", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    current_user: str = Depends(get_current_user)
):
    """
    Upload a document.
    """
    try:
        # Read file content
        file_content = await file.read()

        # Use original filename if title not provided
        if not title:
            title = file.filename

        # Save file to database
        file_path, file_size = save_file_to_database(file_content, project_id, current_user, file.filename)

        # Create document
        from .document_manager import create_document
        document = create_document(
            title=title,
            file=file_content,
            owner=current_user,
            project_id=project_id,
            original_filename=file.filename
        )

        return {
            "document_id": document["document_id"],
            "title": document["title"],
            "document_type": document["document_type"],
            "file_path": file_path,
            "file_size": file_size,
            "message": f"Document uploaded successfully: {title}"
        }
    except Exception as e:
        logger.error(f"Error in upload_document endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.delete("/documents/{document_id}", response_model=Dict[str, Any])
async def delete_document(
    document_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Delete a document.
    """
    try:
        from .document_manager import delete_document as delete_doc
        success = delete_doc(document_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        return {
            "document_id": document_id,
            "message": f"Document {document_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error in delete_document endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/documents/{document_id}/content", response_model=Dict[str, Any])
async def get_document_content(
    document_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get document content.
    """
    try:
        content = read_document_content(document_id)

        if not content:
            raise HTTPException(status_code=404, detail=f"Content for document {document_id} not found")

        return {
            "document_id": document_id,
            "content": content
        }
    except Exception as e:
        logger.error(f"Error in get_document_content endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document content: {str(e)}")

@router.get("/documents/{document_id}/registry", response_model=Dict[str, Any])
async def get_document_registry(
    document_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get document registry information.
    """
    try:
        from .document_registry import get_document_registry as get_registry
        registry_info = get_registry(document_id)

        if not registry_info:
            raise HTTPException(status_code=404, detail=f"Registry info for document {document_id} not found")

        return registry_info
    except Exception as e:
        logger.error(f"Error in get_document_registry endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document registry info: {str(e)}")

def setup_document_routes(app):
    """
    Set up document routes for the FastAPI app.
    """
    app.include_router(router, prefix="/documents", tags=["Documents"])
    logger.info("Set up document routes")
