"""
API endpoints for project management.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import os
import json
import uuid
import time
from datetime import datetime

# Import local modules
from src.local_file_research.document_registry import get_projects, get_project_by_id, create_project as create_project_registry, delete_project as delete_project_registry
from src.local_file_research.auth import get_current_user
from src.local_file_research.models import ProjectCreateRequest, ProjectCreateResponse

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/projects", response_model=List[Dict[str, Any]])
async def list_projects(
    current_user: str = Depends(get_current_user)
):
    """
    List all projects.
    """
    try:
        projects = get_projects()
        return projects
    except Exception as e:
        logger.error(f"Error in list_projects endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")

@router.get("/projects/{project_id}", response_model=Dict[str, Any])
async def get_project(
    project_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get a project by ID.
    """
    try:
        project = get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        return project
    except Exception as e:
        logger.error(f"Error in get_project endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")

@router.post("/projects", response_model=Dict[str, Any])
async def create_project(
    request: ProjectCreateRequest = Body(...),
    current_user: str = Depends(get_current_user)
):
    """
    Create a new project.
    """
    try:
        project = create_project_registry(
            name=request.name,
            description=request.description or "",
            owner=current_user
        )

        return {
            "project_id": project["project_id"],
            "name": project["name"],
            "description": project["description"],
            "created": project["created"],
            "owner": project["owner"],
            "status": "success",
            "message": f"Project created successfully: {project['name']}"
        }
    except Exception as e:
        logger.error(f"Error in create_project endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@router.delete("/projects/{project_id}", response_model=Dict[str, Any])
async def delete_project(
    project_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Delete a project.
    """
    try:
        success = delete_project_registry(project_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

        return {
            "project_id": project_id,
            "status": "success",
            "message": f"Project {project_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error in delete_project endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

@router.get("/projects/{project_id}/documents", response_model=List[Dict[str, Any]])
async def get_project_documents(
    project_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get all documents in a project.
    """
    try:
        project = get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

        # Get documents from project
        documents = project.get("documents", [])

        return documents
    except Exception as e:
        logger.error(f"Error in get_project_documents endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get project documents: {str(e)}")

def setup_project_routes(app):
    """
    Set up project routes for the FastAPI app.
    """
    app.include_router(router, prefix="/projects", tags=["Projects"])
    logger.info("Set up project routes")
