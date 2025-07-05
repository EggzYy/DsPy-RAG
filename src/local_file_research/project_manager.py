"""
Project management for Local File Deep Research.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
PROJECTS_DIR = os.environ.get("PROJECTS_DIR", "projects")

# --- Helper Functions ---
def _ensure_projects_dir():
    """Ensure the projects directory exists."""
    os.makedirs(PROJECTS_DIR, exist_ok=True)

    # Create projects file if it doesn't exist
    projects_file = os.path.join(PROJECTS_DIR, "projects.json")
    if not os.path.exists(projects_file):
        with open(projects_file, "w") as f:
            json.dump([], f)

def _load_projects() -> List[Dict[str, Any]]:
    """Load projects from file."""
    _ensure_projects_dir()
    try:
        with open(os.path.join(PROJECTS_DIR, "projects.json"), "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def _save_projects(projects: List[Dict[str, Any]]):
    """Save projects to file."""
    _ensure_projects_dir()
    with open(os.path.join(PROJECTS_DIR, "projects.json"), "w") as f:
        json.dump(projects, f, indent=2)

def _get_project(project_id: str) -> Optional[Dict[str, Any]]:
    """Get a project by ID."""
    projects = _load_projects()
    for project in projects:
        if project["project_id"] == project_id:
            return project
    return None

# --- Project Functions ---
def create_project(name: str, description: str, owner: str) -> Dict[str, Any]:
    """
    Create a new project.

    Args:
        name: Project name
        description: Project description
        owner: Username of the project owner

    Returns:
        Project dictionary
    """
    if not name or not owner:
        raise ValueError("Project name and owner are required")

    # Create project
    project_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    project = {
        "project_id": project_id,
        "name": name,
        "description": description,
        "owner": owner,
        "members": [owner],
        "documents": [],
        "created_at": now,
        "updated_at": now,
        "access_level": "owner"
    }

    # Save project
    projects = _load_projects()
    projects.append(project)
    _save_projects(projects)

    return project

def get_projects(username: str = None) -> List[Dict[str, Any]]:
    """
    Get all projects or projects for a specific user.

    Args:
        username: Optional username to filter projects by

    Returns:
        List of project dictionaries
    """
    projects = _load_projects()

    if username:
        # Filter projects by username (either owner or member)
        return [p for p in projects if username in p["members"]]

    return projects

def get_project_by_id(project_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a project by ID.

    Args:
        project_id: Project ID

    Returns:
        Project dictionary or None if not found
    """
    return _get_project(project_id)

def update_project(project_id: str, name: str = None, description: str = None) -> Optional[Dict[str, Any]]:
    """
    Update a project.

    Args:
        project_id: Project ID
        name: New project name (optional)
        description: New project description (optional)

    Returns:
        Updated project dictionary or None if not found
    """
    projects = _load_projects()

    for i, project in enumerate(projects):
        if project["project_id"] == project_id:
            if name:
                project["name"] = name
            if description:
                project["description"] = description

            project["updated_at"] = datetime.now().isoformat()
            _save_projects(projects)
            return project

    return None

def delete_project(project_id: str) -> bool:
    """
    Delete a project.

    Args:
        project_id: Project ID

    Returns:
        True if the project was deleted, False otherwise
    """
    # First, get the project to get its documents
    project = get_project_by_id(project_id)
    document_ids = []
    if project:
        document_ids = [doc["document_id"] for doc in project.get("documents", [])]

    projects = _load_projects()
    original_count = len(projects)

    projects = [p for p in projects if p["project_id"] != project_id]

    if len(projects) < original_count:
        _save_projects(projects)

        # Clean up project indices
        try:
            # Import here to avoid circular imports
            from .project_index_cleanup import remove_project_indices, thoroughly_clean_document_files

            # First clean up all documents in the project
            for doc_id in document_ids:
                try:
                    cleanup_result = thoroughly_clean_document_files(doc_id)
                    logger.info(f"Thorough document cleanup for {doc_id} in deleted project {project_id}: {cleanup_result['status']}")
                    logger.info(f"  - Files removed/updated: {cleanup_result['storage_files_removed'] + cleanup_result['project_indices_updated'] + cleanup_result['embedding_files_removed'] + cleanup_result['session_files_updated']}")
                except Exception as e:
                    logger.error(f"Error cleaning up document {doc_id} during project deletion: {e}")

            # Then clean up the project indices
            cleanup_result = remove_project_indices(project_id)
            logger.info(f"Project index cleanup for project {project_id}: {cleanup_result['status']}, removed {cleanup_result['files_removed']} files")
        except Exception as e:
            logger.error(f"Error cleaning up project indices: {e}")

        return True

    return False

def add_member(project_id: str, username: str) -> Optional[Dict[str, Any]]:
    """
    Add a member to a project.

    Args:
        project_id: Project ID
        username: Username to add

    Returns:
        Updated project dictionary or None if not found
    """
    projects = _load_projects()

    for i, project in enumerate(projects):
        if project["project_id"] == project_id:
            if username not in project["members"]:
                project["members"].append(username)
                project["updated_at"] = datetime.now().isoformat()
                _save_projects(projects)
            return project

    return None

def remove_member(project_id: str, username: str) -> Optional[Dict[str, Any]]:
    """
    Remove a member from a project.

    Args:
        project_id: Project ID
        username: Username to remove

    Returns:
        Updated project dictionary or None if not found
    """
    projects = _load_projects()

    for i, project in enumerate(projects):
        if project["project_id"] == project_id:
            if username in project["members"] and username != project["owner"]:
                project["members"].remove(username)
                project["updated_at"] = datetime.now().isoformat()
                _save_projects(projects)
            return project

    return None

def add_document(project_id: str, document_id: str, title: str) -> Optional[Dict[str, Any]]:
    """
    Add a document to a project.

    Args:
        project_id: Project ID
        document_id: Document ID
        title: Document title

    Returns:
        Updated project dictionary or None if not found
    """
    projects = _load_projects()

    for i, project in enumerate(projects):
        if project["project_id"] == project_id:
            document = {
                "document_id": document_id,
                "title": title,
                "added_at": datetime.now().isoformat()
            }

            project["documents"].append(document)
            project["updated_at"] = datetime.now().isoformat()
            _save_projects(projects)

            # Register document-project association in registry if available
            try:
                from .document_registry import register_document_project
                register_document_project(document_id, project_id)
                logger.info(f"Registered document {document_id} in project {project_id} in registry")
            except ImportError:
                logger.warning("Document registry not available. Document-project association not registered.")

            return project

    return None

def remove_document(project_id: str, document_id: str) -> Optional[Dict[str, Any]]:
    """
    Remove a document from a project.

    Args:
        project_id: Project ID
        document_id: Document ID

    Returns:
        Updated project dictionary or None if not found
    """
    projects = _load_projects()

    for i, project in enumerate(projects):
        if project["project_id"] == project_id:
            # Check if document exists in project before removing
            doc_exists = any(d["document_id"] == document_id for d in project["documents"])

            # Remove document from project
            project["documents"] = [d for d in project["documents"] if d["document_id"] != document_id]
            project["updated_at"] = datetime.now().isoformat()
            _save_projects(projects)

            # If document was in project, update registry and perform thorough cleanup
            if doc_exists:
                # First try to update the document registry if available
                try:
                    from .document_registry import unregister_document_project
                    unregister_document_project(document_id, project_id)
                    logger.info(f"Unregistered document {document_id} from project {project_id} in registry")
                except ImportError:
                    logger.warning("Document registry not available. Document-project association not unregistered.")

                # Then perform thorough cleanup
                try:
                    # Import here to avoid circular imports
                    from .project_index_cleanup import thoroughly_clean_document_files
                    cleanup_result = thoroughly_clean_document_files(document_id)
                    logger.info(f"Thorough document cleanup for {document_id} from project {project_id}: {cleanup_result['status']}")
                    logger.info(f"  - Storage files removed: {cleanup_result['storage_files_removed']}")
                    logger.info(f"  - Project indices updated: {cleanup_result['project_indices_updated']}")
                    logger.info(f"  - Embedding files removed: {cleanup_result['embedding_files_removed']}")
                    logger.info(f"  - Session files updated: {cleanup_result['session_files_updated']}")
                    if cleanup_result['errors']:
                        logger.warning(f"  - Errors during cleanup: {len(cleanup_result['errors'])}")
                        for error in cleanup_result['errors'][:5]:  # Log first 5 errors
                            logger.warning(f"    - {error}")
                except Exception as e:
                    logger.error(f"Error during thorough document cleanup: {e}")

            return project

    return None

# Initialize
_ensure_projects_dir()
