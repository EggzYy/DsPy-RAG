"""
Document Registry for Local File Deep Research.
This module provides a centralized registry for tracking all files and resources related to documents.
It serves as the single source of truth for all file locations and resource management.
"""

import os
import json
import logging
import glob
import shutil
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Import constants
from .storage_manager import STORAGE_DIR, LEGACY_METADATA_FILE
from .embedding import EMBEDDING_CACHE_DIR as EMBEDDINGS_DIR
from .project_indexer import PROJECT_INDEX_DIR

# Registry file path
REGISTRY_DIR = os.path.join(STORAGE_DIR, "registry")
DOCUMENT_REGISTRY_PATH = os.path.join(REGISTRY_DIR, "document_registry.json")

# Session directory
SESSION_DIR = os.path.join(STORAGE_DIR, "sessions")

def ensure_registry_dir():
    """Ensure the registry directory exists."""
    os.makedirs(REGISTRY_DIR, exist_ok=True)

def initialize_document_registry():
    """
    Initialize the document registry.
    This function should be called at application startup.
    """
    logger.info("Initializing document registry...")

    # Ensure registry directory exists
    ensure_registry_dir()

    # Create registry file if it doesn't exist
    if not os.path.exists(DOCUMENT_REGISTRY_PATH):
        logger.info("Creating new document registry...")
        registry = {
            "documents": {},
            "projects": {},
            "sessions": {},
            "last_updated": datetime.now().isoformat()
        }
        _save_registry(registry)

    # Load registry to verify it's valid
    try:
        registry = _load_registry()
        logger.info(f"Document registry initialized with {len(registry.get('documents', {}))} documents and {len(registry.get('projects', {}))} projects")
    except Exception as e:
        logger.error(f"Error initializing document registry: {e}")
        logger.info("Rebuilding document registry...")
        scan_and_rebuild_registry()

def _load_registry() -> Dict[str, Any]:
    """
    Load the document registry.

    Returns:
        Dictionary containing the document registry
    """
    ensure_registry_dir()

    if not os.path.exists(DOCUMENT_REGISTRY_PATH):
        return {"documents": {}, "last_updated": datetime.now().isoformat()}

    try:
        with open(DOCUMENT_REGISTRY_PATH, "r") as f:
            registry = json.load(f)
        return registry
    except Exception as e:
        logger.error(f"Error loading document registry: {e}")
        return {"documents": {}, "last_updated": datetime.now().isoformat()}

def _save_registry(registry: Dict[str, Any]):
    """
    Save the document registry.

    Args:
        registry: Dictionary containing the document registry
    """
    ensure_registry_dir()

    registry["last_updated"] = datetime.now().isoformat()

    try:
        with open(DOCUMENT_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving document registry: {e}")

def register_document(document_id: str, metadata: Dict[str, Any]):
    """
    Register a document in the registry.

    Args:
        document_id: Document ID
        metadata: Document metadata
    """
    registry = _load_registry()

    # Create or update document entry
    if document_id not in registry["documents"]:
        # Initialize with ref_count
        registry["documents"][document_id] = {
            "document_id": document_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata,
            "files": [],
            "projects": [],
            "embeddings": [],
            "indices": [],
            "ref_count": 0
        }
    else:
        registry["documents"][document_id]["updated_at"] = datetime.now().isoformat()

        # Get existing metadata
        existing_metadata = registry["documents"][document_id].get("metadata", {})

        # Merge metadata - preserve existing fields if they're not in the new metadata
        merged_metadata = {}

        # First, add all existing metadata
        for key, value in existing_metadata.items():
            merged_metadata[key] = value

        # Then, update with new metadata, preserving content if needed
        if isinstance(metadata, dict):
            # Direct metadata dictionary
            for key, value in metadata.items():
                if key != "content" or value:  # Don't overwrite content with empty value
                    merged_metadata[key] = value
        elif "metadata" in metadata:
            # Nested metadata dictionary
            for key, value in metadata["metadata"].items():
                if key != "content" or value:  # Don't overwrite content with empty value
                    merged_metadata[key] = value

            # Also copy top-level fields that aren't 'metadata'
            for key, value in metadata.items():
                if key != "metadata":
                    registry["documents"][document_id][key] = value

        # Update the registry with merged metadata
        registry["documents"][document_id]["metadata"] = merged_metadata

        # If content is provided in the metadata, also save it to a content file
        if "content" in merged_metadata and merged_metadata["content"]:
            try:
                from .storage_manager import save_document_content
                content_path = save_document_content(document_id, merged_metadata["content"])
                logger.info(f"Saved content from registry to dedicated content file at {content_path}")
            except Exception as e:
                logger.warning(f"Could not save registry content to content file: {str(e)}")

    # Log the metadata being saved
    logger.info(f"Saving document {document_id} to registry with metadata keys: {list(registry['documents'][document_id]['metadata'].keys())}")

    _save_registry(registry)

def register_document_file(document_id: str, file_path: str, file_type: str):
    """
    Register a file associated with a document.

    Args:
        document_id: Document ID
        file_path: Path to the file
        file_type: Type of file (storage, version, embedding, index)
    """
    registry = _load_registry()

    # Ensure document exists in registry
    if document_id not in registry["documents"]:
        logger.warning(f"Document {document_id} not found in registry. Creating entry.")
        registry["documents"][document_id] = {
            "document_id": document_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {},
            "files": [],
            "projects": [],
            "embeddings": [],
            "indices": []
        }

    # Add file to document
    file_entry = {
        "path": file_path,
        "type": file_type,
        "registered_at": datetime.now().isoformat()
    }

    # Check if file already exists
    for i, existing_file in enumerate(registry["documents"][document_id]["files"]):
        if existing_file["path"] == file_path:
            # Update existing entry
            registry["documents"][document_id]["files"][i] = file_entry
            break
    else:
        # Add new entry
        registry["documents"][document_id]["files"].append(file_entry)

    registry["documents"][document_id]["updated_at"] = datetime.now().isoformat()
    _save_registry(registry)

def register_document_project(document_id: str, project_id: str):
    """
    Register a project association for a document.
    This updates both the document's entry (adding the project to its "projects" list)
    and the project's entry (adding the document to its "documents" list).

    Args:
        document_id: Document ID
        project_id: Project ID
    """
    registry = _load_registry()

    # Ensure document exists in registry
    if document_id not in registry["documents"]:
        logger.warning(f"Document {document_id} not found in registry. Creating entry.")
        registry["documents"][document_id] = {
            "document_id": document_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {},
            "files": [],
            "projects": [],
            "embeddings": [],
            "indices": [],
            "ref_count": 0
        }

    # Add project to document if not already present
    if project_id not in registry["documents"][document_id]["projects"]:
        registry["documents"][document_id]["projects"].append(project_id)
        registry["documents"][document_id]["updated_at"] = datetime.now().isoformat()
        registry["documents"][document_id]["ref_count"] += 1
        logger.info(f"Added project {project_id} to document {document_id}")

    # Ensure project exists in registry
    if "projects" not in registry:
        registry["projects"] = {}

    if project_id not in registry["projects"]:
        logger.warning(f"Project {project_id} not found in registry. Creating entry.")
        registry["projects"][project_id] = {
            "project_id": project_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "documents": [],
            "sessions": []
        }

    # Add document to project if not already present
    if document_id not in registry["projects"][project_id]["documents"]:
        registry["projects"][project_id]["documents"].append(document_id)
        registry["projects"][project_id]["updated_at"] = datetime.now().isoformat()
        logger.info(f"Added document {document_id} to project {project_id}")

    # Save the updated registry
    _save_registry(registry)

def unregister_document_project(document_id: str, project_id: str):
    """
    Remove a project association for a document.
    This updates both the document's entry (removing the project from its "projects" list)
    and the project's entry (removing the document from its "documents" list).

    Args:
        document_id: Document ID
        project_id: Project ID
    """
    registry = _load_registry()

    # Ensure document exists in registry
    if document_id not in registry["documents"]:
        logger.warning(f"Document {document_id} not found in registry.")
        return

    # Remove project from document if present
    if project_id in registry["documents"][document_id]["projects"]:
        registry["documents"][document_id]["projects"].remove(project_id)
        registry["documents"][document_id]["updated_at"] = datetime.now().isoformat()

        # Decrement reference count
        if "ref_count" in registry["documents"][document_id]:
            registry["documents"][document_id]["ref_count"] = max(0, registry["documents"][document_id]["ref_count"] - 1)

        logger.info(f"Removed project {project_id} from document {document_id}")

    # Ensure projects exist in registry
    if "projects" in registry and project_id in registry["projects"]:
        # Remove document from project if present
        if document_id in registry["projects"][project_id]["documents"]:
            registry["projects"][project_id]["documents"].remove(document_id)
            registry["projects"][project_id]["updated_at"] = datetime.now().isoformat()
            logger.info(f"Removed document {document_id} from project {project_id}")

    # Save the updated registry
    _save_registry(registry)

def register_document_embedding(document_id: str, embedding_path: str):
    """
    Register an embedding file for a document.

    Args:
        document_id: Document ID
        embedding_path: Path to the embedding file
    """
    registry = _load_registry()

    # Ensure document exists in registry
    if document_id not in registry["documents"]:
        logger.warning(f"Document {document_id} not found in registry. Creating entry.")
        registry["documents"][document_id] = {
            "document_id": document_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {},
            "files": [],
            "projects": [],
            "embeddings": [],
            "indices": []
        }

    # Add embedding to document if not already present
    if embedding_path not in registry["documents"][document_id]["embeddings"]:
        registry["documents"][document_id]["embeddings"].append(embedding_path)
        registry["documents"][document_id]["updated_at"] = datetime.now().isoformat()
        _save_registry(registry)

def register_document_index(document_id: str, index_path: str):
    """
    Register an index file for a document.

    Args:
        document_id: Document ID
        index_path: Path to the index file
    """
    registry = _load_registry()

    # Ensure document exists in registry
    if document_id not in registry["documents"]:
        logger.warning(f"Document {document_id} not found in registry. Creating entry.")
        registry["documents"][document_id] = {
            "document_id": document_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {},
            "files": [],
            "projects": [],
            "embeddings": [],
            "indices": []
        }

    # Add index to document if not already present
    if index_path not in registry["documents"][document_id]["indices"]:
        registry["documents"][document_id]["indices"].append(index_path)
        registry["documents"][document_id]["updated_at"] = datetime.now().isoformat()
        _save_registry(registry)

def unregister_document_file(document_id: str, file_path: str) -> bool:
    """
    Remove a file association from a document in the registry.

    Args:
        document_id: Document ID
        file_path: Path to the file to unregister

    Returns:
        True if the file was unregistered, False otherwise
    """
    registry = _load_registry()

    # Check if document exists in registry
    if document_id not in registry["documents"]:
        logger.warning(f"Document {document_id} not found in registry.")
        return False

    # Find and remove the file entry
    files = registry["documents"][document_id].get("files", [])
    for i, file_entry in enumerate(files):
        if file_entry["path"] == file_path:
            # Remove the file entry
            registry["documents"][document_id]["files"].pop(i)
            registry["documents"][document_id]["updated_at"] = datetime.now().isoformat()
            _save_registry(registry)
            logger.info(f"Unregistered file {file_path} from document {document_id}")
            return True

    logger.warning(f"File {file_path} not found in registry for document {document_id}")
    return False

def unregister_document(document_id: str) -> Dict[str, Any]:
    """
    Remove a document from the registry and return all its registered files and associations.

    Args:
        document_id: Document ID

    Returns:
        Dictionary containing all registered files and associations for the document
    """
    registry = _load_registry()

    # Check if document exists in registry
    if document_id not in registry["documents"]:
        logger.warning(f"Document {document_id} not found in registry.")
        return {
            "document_id": document_id,
            "files": [],
            "projects": [],
            "embeddings": [],
            "indices": []
        }

    # Get document data
    document_data = registry["documents"][document_id]

    # Remove document from registry
    del registry["documents"][document_id]
    _save_registry(registry)

    return {
        "document_id": document_id,
        "files": document_data.get("files", []),
        "projects": document_data.get("projects", []),
        "embeddings": document_data.get("embeddings", []),
        "indices": document_data.get("indices", [])
    }

def get_document_registry(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get registry information for a document.

    Args:
        document_id: Document ID

    Returns:
        Dictionary containing registry information for the document, or None if not found
    """
    registry = _load_registry()

    # Check if document exists in registry
    if document_id not in registry["documents"]:
        return None

    return registry["documents"][document_id]

def get_projects() -> List[Dict[str, Any]]:
    """
    Get all projects.

    Returns:
        List of projects
    """
    registry = _load_registry()

    # Check if projects exist in registry
    if "projects" not in registry:
        return []

    # Convert projects dictionary to list
    projects_list = []
    for project_id, project in registry["projects"].items():
        projects_list.append(project)

    return projects_list

def get_project_by_id(project_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a project by ID.

    Args:
        project_id: Project ID

    Returns:
        Project dictionary or None if not found
    """
    registry = _load_registry()

    # Check if projects exist in registry
    if "projects" not in registry:
        return None

    # Check if project exists in registry
    if project_id not in registry["projects"]:
        return None

    return registry["projects"][project_id]

def create_project(name: str, description: str = "", owner: str = "default_user") -> Dict[str, Any]:
    """
    Create a new project.

    Args:
        name: Project name
        description: Project description
        owner: Project owner

    Returns:
        Project dictionary
    """
    registry = _load_registry()

    # Initialize projects dictionary if it doesn't exist
    if "projects" not in registry:
        registry["projects"] = {}

    # Generate project ID
    import uuid
    project_id = str(uuid.uuid4())

    # Create project entry
    project = {
        "project_id": project_id,
        "name": name,
        "description": description,
        "owner": owner,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "documents": []
    }

    # Add project to registry
    registry["projects"][project_id] = project

    # Save registry
    _save_registry(registry)

    return project

def get_project_directory(project_id: str) -> str:
    """
    Get the directory for a project.

    Args:
        project_id: Project ID

    Returns:
        Path to the project directory
    """
    # Create project directory path
    project_dir = os.path.join(STORAGE_DIR, "projects", project_id)

    # Ensure the directory exists
    os.makedirs(project_dir, exist_ok=True)

    return project_dir

def delete_project(project_id: str) -> bool:
    """
    Delete a project.

    Args:
        project_id: Project ID

    Returns:
        True if the project was deleted, False otherwise
    """
    registry = _load_registry()

    # Check if projects exist in registry
    if "projects" not in registry:
        return False

    # Check if project exists in registry
    if project_id not in registry["projects"]:
        return False

    # Remove project from registry
    del registry["projects"][project_id]

    # Save registry
    _save_registry(registry)

    return True

def get_project_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a project by name.

    Args:
        name: Project name

    Returns:
        Project dictionary or None if not found
    """
    registry = _load_registry()

    # Check if projects exist in registry
    if "projects" not in registry:
        return None

    # Find project by name
    for project_id, project in registry["projects"].items():
        if project.get("name") == name:
            return project

    return None

def get_project_documents(project_id: str) -> List[str]:
    """
    Get all document IDs associated with a project.
    This checks both the project's "documents" list and all documents' "projects" lists
    to ensure the most accurate and complete list is returned.

    Args:
        project_id: Project ID

    Returns:
        List of document IDs
    """
    registry = _load_registry()

    # Check if projects exist in registry
    if "projects" not in registry:
        logger.warning("No projects found in registry")
        return []

    # Check if project exists
    if project_id not in registry["projects"]:
        logger.warning(f"Project {project_id} not found in registry")
        return []

    # Get documents from project
    project_docs = set(registry["projects"][project_id].get("documents", []))
    logger.debug(f"Found {len(project_docs)} documents in project {project_id}'s documents list")

    # Also check all documents for this project
    docs_from_document_entries = set()
    for doc_id, doc in registry["documents"].items():
        if "projects" in doc and project_id in doc["projects"]:
            docs_from_document_entries.add(doc_id)

    logger.debug(f"Found {len(docs_from_document_entries)} documents with project {project_id} in their projects list")

    # Combine both sets
    all_docs = list(project_docs.union(docs_from_document_entries))

    # If there's a discrepancy, log a warning and fix the registry
    if project_docs != docs_from_document_entries:
        missing_in_project = docs_from_document_entries - project_docs
        missing_in_docs = project_docs - docs_from_document_entries

        if missing_in_project:
            logger.warning(f"Found {len(missing_in_project)} documents that reference project {project_id} but are not in the project's documents list")
            # Add missing documents to project's documents list
            for doc_id in missing_in_project:
                if doc_id not in registry["projects"][project_id]["documents"]:
                    registry["projects"][project_id]["documents"].append(doc_id)
                    logger.info(f"Added document {doc_id} to project {project_id}'s documents list")

        if missing_in_docs:
            logger.warning(f"Found {len(missing_in_docs)} documents in project {project_id}'s documents list that don't reference the project")
            # Add project to documents' projects list
            for doc_id in missing_in_docs:
                if doc_id in registry["documents"]:
                    if project_id not in registry["documents"][doc_id]["projects"]:
                        registry["documents"][doc_id]["projects"].append(project_id)
                        logger.info(f"Added project {project_id} to document {doc_id}'s projects list")

                        # Increment reference count
                        if "ref_count" in registry["documents"][doc_id]:
                            registry["documents"][doc_id]["ref_count"] += 1
                else:
                    logger.warning(f"Document {doc_id} referenced in project {project_id} not found in registry")
                    # Remove from project's documents list
                    if doc_id in registry["projects"][project_id]["documents"]:
                        registry["projects"][project_id]["documents"].remove(doc_id)
                        logger.info(f"Removed non-existent document {doc_id} from project {project_id}'s documents list")

        # Save the updated registry
        _save_registry(registry)

    logger.info(f"Found {len(all_docs)} total documents for project {project_id}")
    return all_docs

def scan_and_rebuild_registry():
    """
    Scan all directories and rebuild the document registry.
    This is useful for recovering from registry corruption or initializing the registry.
    """
    logger.info("Rebuilding document registry...")

    # Start with a fresh registry
    registry = {
        "documents": {},
        "last_updated": datetime.now().isoformat(),
        "projects": {},
        "sessions": {},
        "embeddings": {},
        "indices": {}
    }

    # Scan storage directory
    if os.path.exists(STORAGE_DIR):
        logger.info(f"Scanning storage directory: {STORAGE_DIR}")

        # Scan main document files
        for filename in os.listdir(STORAGE_DIR):
            if filename.startswith("doc_"):
                document_id = filename.replace("doc_", "")

                # Initialize document entry if needed
                if document_id not in registry["documents"]:
                    registry["documents"][document_id] = {
                        "document_id": document_id,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "metadata": {},
                        "files": [],
                        "projects": [],
                        "embeddings": [],
                        "indices": [],
                        "ref_count": 0  # Reference count for tracking usage
                    }

                # Add file to document
                file_path = os.path.join(STORAGE_DIR, filename)
                registry["documents"][document_id]["files"].append({
                    "path": file_path,
                    "type": "storage",
                    "registered_at": datetime.now().isoformat()
                })

        # Scan versions directory
        versions_dir = os.path.join(STORAGE_DIR, "versions")
        if os.path.exists(versions_dir):
            logger.info(f"Scanning versions directory: {versions_dir}")

            for filename in os.listdir(versions_dir):
                if filename.startswith("ver_"):
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        document_id = parts[1]

                        # Initialize document entry if needed
                        if document_id not in registry["documents"]:
                            registry["documents"][document_id] = {
                                "document_id": document_id,
                                "created_at": datetime.now().isoformat(),
                                "updated_at": datetime.now().isoformat(),
                                "metadata": {},
                                "files": [],
                                "projects": [],
                                "embeddings": [],
                                "indices": [],
                                "ref_count": 0  # Reference count for tracking usage
                            }

                        # Add file to document
                        file_path = os.path.join(versions_dir, filename)
                        registry["documents"][document_id]["files"].append({
                            "path": file_path,
                            "type": "version",
                            "registered_at": datetime.now().isoformat()
                        })

    # Scan project folders
    project_folders = [d for d in os.listdir(STORAGE_DIR)
                      if os.path.isdir(os.path.join(STORAGE_DIR, d))
                      and not d in ["registry", "versions", "sessions"]]

    for project_id in project_folders:
        project_path = os.path.join(STORAGE_DIR, project_id)

        # Initialize project entry
        if project_id not in registry["projects"]:
            registry["projects"][project_id] = {
                "project_id": project_id,
                "path": project_path,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "documents": [],
                "sessions": []
            }

        # Scan session folders
        session_folders = [d for d in os.listdir(project_path)
                          if os.path.isdir(os.path.join(project_path, d))
                          and d.startswith("session_")]

        for session_id in session_folders:
            session_path = os.path.join(project_path, session_id)

            # Initialize session entry
            if session_id not in registry["sessions"]:
                registry["sessions"][session_id] = {
                    "session_id": session_id,
                    "project_id": project_id,
                    "path": session_path,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "documents": []
                }

            # Add session to project
            if session_id not in registry["projects"][project_id]["sessions"]:
                registry["projects"][project_id]["sessions"].append(session_id)

    # Scan embeddings directory
    if os.path.exists(EMBEDDINGS_DIR):
        logger.info(f"Scanning embeddings directory: {EMBEDDINGS_DIR}")

        for filename in os.listdir(EMBEDDINGS_DIR):
            embedding_path = os.path.join(EMBEDDINGS_DIR, filename)

            # Initialize embedding entry
            if filename not in registry["embeddings"]:
                registry["embeddings"][filename] = {
                    "path": embedding_path,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "documents": [],
                    "ref_count": 0  # Reference count for tracking usage
                }

            # Try to associate with documents
            for document_id in registry["documents"]:
                if document_id in filename:
                    # Add embedding to document
                    if embedding_path not in registry["documents"][document_id]["embeddings"]:
                        registry["documents"][document_id]["embeddings"].append(embedding_path)

                    # Add document to embedding
                    if document_id not in registry["embeddings"][filename]["documents"]:
                        registry["embeddings"][filename]["documents"].append(document_id)
                        registry["embeddings"][filename]["ref_count"] += 1

    # Scan project indices
    if os.path.exists(PROJECT_INDEX_DIR):
        logger.info(f"Scanning project indices directory: {PROJECT_INDEX_DIR}")

        for filename in os.listdir(PROJECT_INDEX_DIR):
            if filename.endswith("_index_info.json"):
                index_path = os.path.join(PROJECT_INDEX_DIR, filename)

                # Initialize index entry
                if filename not in registry["indices"]:
                    registry["indices"][filename] = {
                        "path": index_path,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "documents": [],
                        "project_id": None
                    }

                try:
                    with open(index_path, "r") as f:
                        index_info = json.load(f)

                    if "document_ids" in index_info:
                        project_id = filename.split("_index_info.json")[0]
                        registry["indices"][filename]["project_id"] = project_id

                        # Initialize project if needed
                        if project_id not in registry["projects"]:
                            registry["projects"][project_id] = {
                                "project_id": project_id,
                                "created_at": datetime.now().isoformat(),
                                "updated_at": datetime.now().isoformat(),
                                "documents": [],
                                "sessions": []
                            }

                        for document_id in index_info["document_ids"]:
                            # Initialize document entry if needed
                            if document_id not in registry["documents"]:
                                registry["documents"][document_id] = {
                                    "document_id": document_id,
                                    "created_at": datetime.now().isoformat(),
                                    "updated_at": datetime.now().isoformat(),
                                    "metadata": {},
                                    "files": [],
                                    "projects": [],
                                    "embeddings": [],
                                    "indices": [],
                                    "ref_count": 0  # Reference count for tracking usage
                                }

                            # Add project to document
                            if project_id not in registry["documents"][document_id]["projects"]:
                                registry["documents"][document_id]["projects"].append(project_id)
                                registry["documents"][document_id]["ref_count"] += 1

                            # Add document to project
                            if document_id not in registry["projects"][project_id]["documents"]:
                                registry["projects"][project_id]["documents"].append(document_id)

                            # Add index to document
                            if index_path not in registry["documents"][document_id]["indices"]:
                                registry["documents"][document_id]["indices"].append(index_path)

                            # Add document to index
                            if document_id not in registry["indices"][filename]["documents"]:
                                registry["indices"][filename]["documents"].append(document_id)
                except Exception as e:
                    logger.error(f"Error reading index file {index_path}: {e}")

    # Save the rebuilt registry
    _save_registry(registry)

    logger.info(f"Registry rebuild complete. Found {len(registry['documents'])} documents, {len(registry['projects'])} projects, {len(registry['sessions'])} sessions, {len(registry['embeddings'])} embeddings, {len(registry['indices'])} indices.")
    return registry

def get_all_document_files(document_id: str) -> Dict[str, List[str]]:
    """
    Get all files associated with a document.

    Args:
        document_id: Document ID

    Returns:
        Dictionary containing lists of files by type
    """
    registry = _load_registry()

    # Check if document exists in registry
    if document_id not in registry["documents"]:
        logger.warning(f"Document {document_id} not found in registry.")
        return {
            "storage_files": [],
            "version_files": [],
            "embedding_files": [],
            "index_files": [],
            "projects": []
        }

    document_data = registry["documents"][document_id]

    # Organize files by type
    storage_files = []
    version_files = []

    for file_entry in document_data.get("files", []):
        if file_entry["type"] == "storage":
            storage_files.append(file_entry["path"])
        elif file_entry["type"] == "version":
            version_files.append(file_entry["path"])

    return {
        "storage_files": storage_files,
        "version_files": version_files,
        "embedding_files": document_data.get("embeddings", []),
        "index_files": document_data.get("indices", []),
        "projects": document_data.get("projects", []),
        "ref_count": document_data.get("ref_count", 0)
    }

def cleanup_embeddings(document_ids: List[str] = None, force_cleanup: bool = True) -> Dict[str, Any]:
    """
    Clean up embedding files that are no longer needed.
    After indexing, we can remove all embedding files since they're now in the vector store.

    Args:
        document_ids: Optional list of document IDs to clean up embeddings for.
                     If None, clean up all unreferenced embeddings.
        force_cleanup: If True, remove all embedding files regardless of reference count.
                      This is useful after indexing when embeddings are stored in the vector store.

    Returns:
        Dictionary with cleanup statistics
    """
    registry = _load_registry()

    stats = {
        "status": "success",
        "total_files_removed": 0,
        "total_bytes_freed": 0,
        "errors": []
    }

    logger.info(f"Starting embedding cleanup for document IDs: {document_ids if document_ids else 'all'}")
    logger.info(f"Force cleanup mode: {force_cleanup}")

    # If force_cleanup is True, remove all embedding files regardless of reference count
    if force_cleanup:
        logger.info("Force cleanup mode: Removing all embedding files regardless of reference count")

        # Get all embedding files to remove
        embedding_files_to_remove = []

        if document_ids:
            # Only remove embeddings for specific documents
            for document_id in document_ids:
                if document_id in registry["documents"]:
                    document = registry["documents"][document_id]
                    embedding_files = document.get("embeddings", [])
                    embedding_files_to_remove.extend(embedding_files)

                    # Clear embeddings list for this document
                    document["embeddings"] = []
        else:
            # Remove all embedding files
            for filename, embedding in list(registry["embeddings"].items()):
                embedding_files_to_remove.append(embedding["path"])

            # Clear all embeddings from registry
            registry["embeddings"] = {}

            # Clear embeddings lists for all documents
            for document_id in registry["documents"]:
                registry["documents"][document_id]["embeddings"] = []

        # Remove all collected embedding files
        for embedding_path in set(embedding_files_to_remove):
            try:
                if os.path.exists(embedding_path):
                    file_size = os.path.getsize(embedding_path)
                    os.remove(embedding_path)
                    stats["total_files_removed"] += 1
                    stats["total_bytes_freed"] += file_size
                    logger.info(f"Removed embedding file: {embedding_path} ({file_size} bytes)")
            except Exception as e:
                error_msg = f"Error cleaning up embedding {embedding_path}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        # Also clean up any remaining files in the embeddings directory
        if os.path.exists(EMBEDDINGS_DIR):
            for filename in os.listdir(EMBEDDINGS_DIR):
                embedding_path = os.path.join(EMBEDDINGS_DIR, filename)
                try:
                    if os.path.isfile(embedding_path):
                        file_size = os.path.getsize(embedding_path)
                        os.remove(embedding_path)
                        stats["total_files_removed"] += 1
                        stats["total_bytes_freed"] += file_size
                        logger.info(f"Removed additional embedding file: {embedding_path} ({file_size} bytes)")
                except Exception as e:
                    error_msg = f"Error cleaning up additional embedding {embedding_path}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
    else:
        # Standard cleanup - only remove unreferenced embeddings
        logger.info("Standard cleanup mode: Only removing unreferenced embedding files")

        # If document_ids is provided, only clean up embeddings for those documents
        if document_ids:
            for document_id in document_ids:
                if document_id in registry["documents"]:
                    document = registry["documents"][document_id]

                    # Get all embedding files for this document
                    embedding_files = document.get("embeddings", [])

                    for embedding_path in embedding_files:
                        try:
                            # Get filename from path
                            filename = os.path.basename(embedding_path)

                            # Check if this embedding is used by other documents
                            if filename in registry["embeddings"]:
                                embedding_entry = registry["embeddings"][filename]

                                # Remove document from embedding's document list
                                if document_id in embedding_entry["documents"]:
                                    embedding_entry["documents"].remove(document_id)
                                    embedding_entry["ref_count"] -= 1

                                    # If no more references, delete the embedding file
                                    if embedding_entry["ref_count"] <= 0:
                                        if os.path.exists(embedding_path):
                                            file_size = os.path.getsize(embedding_path)
                                            os.remove(embedding_path)
                                            stats["total_files_removed"] += 1
                                            stats["total_bytes_freed"] += file_size
                                            logger.info(f"Removed embedding file: {embedding_path} ({file_size} bytes)")

                                            # Remove from registry
                                            del registry["embeddings"][filename]
                        except Exception as e:
                            error_msg = f"Error cleaning up embedding {embedding_path}: {str(e)}"
                            logger.error(error_msg)
                            stats["errors"].append(error_msg)

                    # Clear embeddings list for this document
                    document["embeddings"] = []
        else:
            # Clean up all unreferenced embeddings
            for filename, embedding in list(registry["embeddings"].items()):
                if embedding["ref_count"] <= 0:
                    embedding_path = embedding["path"]
                    try:
                        if os.path.exists(embedding_path):
                            file_size = os.path.getsize(embedding_path)
                            os.remove(embedding_path)
                            stats["total_files_removed"] += 1
                            stats["total_bytes_freed"] += file_size
                            logger.info(f"Removed unreferenced embedding file: {embedding_path} ({file_size} bytes)")

                            # Remove from registry
                            del registry["embeddings"][filename]
                    except Exception as e:
                        error_msg = f"Error cleaning up embedding {embedding_path}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

    # Save registry changes
    _save_registry(registry)

    # Add human-readable size
    from .document_cleanup import _format_bytes
    stats["human_readable_bytes_freed"] = _format_bytes(stats["total_bytes_freed"])

    return stats

def cleanup_session_files(session_id: str = None) -> Dict[str, Any]:
    """
    Clean up session files that are no longer needed.

    Args:
        session_id: Optional session ID to clean up. If None, clean up all sessions.

    Returns:
        Dictionary with cleanup statistics
    """
    registry = _load_registry()

    stats = {
        "status": "success",
        "total_files_removed": 0,
        "total_bytes_freed": 0,
        "errors": []
    }

    # If session_id is provided, only clean up that session
    if session_id:
        if session_id in registry["sessions"]:
            session = registry["sessions"][session_id]
            project_id = session.get("project_id")
            session_path = session.get("path")

            if session_path and os.path.exists(session_path):
                try:
                    # Walk through session directory and remove all files
                    for root, dirs, files in os.walk(session_path, topdown=False):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                stats["total_files_removed"] += 1
                                stats["total_bytes_freed"] += file_size
                                logger.info(f"Removed session file: {file_path} ({file_size} bytes)")
                            except Exception as e:
                                error_msg = f"Error removing session file {file_path}: {str(e)}"
                                logger.error(error_msg)
                                stats["errors"].append(error_msg)

                        # Remove empty directories
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            try:
                                os.rmdir(dir_path)
                                logger.info(f"Removed empty session directory: {dir_path}")
                            except Exception as e:
                                # Directory might not be empty
                                logger.debug(f"Could not remove directory {dir_path}: {str(e)}")
                except Exception as e:
                    error_msg = f"Error cleaning up session {session_id}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

            # Remove session from project
            if project_id and project_id in registry["projects"]:
                if session_id in registry["projects"][project_id]["sessions"]:
                    registry["projects"][project_id]["sessions"].remove(session_id)

            # Remove session from registry
            del registry["sessions"][session_id]
    else:
        # Clean up all sessions
        for session_id, session in list(registry["sessions"].items()):
            session_path = session.get("path")
            project_id = session.get("project_id")

            if session_path and os.path.exists(session_path):
                try:
                    # Walk through session directory and remove all files
                    for root, dirs, files in os.walk(session_path, topdown=False):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                stats["total_files_removed"] += 1
                                stats["total_bytes_freed"] += file_size
                                logger.info(f"Removed session file: {file_path} ({file_size} bytes)")
                            except Exception as e:
                                error_msg = f"Error removing session file {file_path}: {str(e)}"
                                logger.error(error_msg)
                                stats["errors"].append(error_msg)

                        # Remove empty directories
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            try:
                                os.rmdir(dir_path)
                                logger.info(f"Removed empty session directory: {dir_path}")
                            except Exception as e:
                                # Directory might not be empty
                                logger.debug(f"Could not remove directory {dir_path}: {str(e)}")
                except Exception as e:
                    error_msg = f"Error cleaning up session {session_id}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

            # Remove session from project
            if project_id and project_id in registry["projects"]:
                if session_id in registry["projects"][project_id]["sessions"]:
                    registry["projects"][project_id]["sessions"].remove(session_id)

            # Remove session from registry
            del registry["sessions"][session_id]

    # Save registry changes
    _save_registry(registry)

    # Add human-readable size
    from .document_cleanup import _format_bytes
    stats["human_readable_bytes_freed"] = _format_bytes(stats["total_bytes_freed"])

    return stats

def verify_registry_integrity() -> Dict[str, Any]:
    """
    Verify the integrity of the registry by checking that all referenced files exist
    and all existing files are properly registered.

    Returns:
        Dictionary with verification results
    """
    registry = _load_registry()

    results = {
        "status": "success",
        "missing_files": [],
        "unregistered_files": [],
        "inconsistencies": [],
        "fixed_issues": 0
    }

    # Check that all referenced files exist
    for document_id, document in registry["documents"].items():
        # Check document files
        for file_entry in document.get("files", []):
            file_path = file_entry["path"]
            if not os.path.exists(file_path):
                results["missing_files"].append({
                    "document_id": document_id,
                    "file_path": file_path,
                    "file_type": file_entry["type"]
                })

        # Check embedding files
        for embedding_path in document.get("embeddings", []):
            if not os.path.exists(embedding_path):
                results["missing_files"].append({
                    "document_id": document_id,
                    "file_path": embedding_path,
                    "file_type": "embedding"
                })

        # Check index files
        for index_path in document.get("indices", []):
            if not os.path.exists(index_path):
                results["missing_files"].append({
                    "document_id": document_id,
                    "file_path": index_path,
                    "file_type": "index"
                })

    # Check that all embedding files are registered
    if os.path.exists(EMBEDDINGS_DIR):
        for filename in os.listdir(EMBEDDINGS_DIR):
            embedding_path = os.path.join(EMBEDDINGS_DIR, filename)

            # Check if this embedding is in the registry
            if filename not in registry["embeddings"]:
                results["unregistered_files"].append({
                    "file_path": embedding_path,
                    "file_type": "embedding"
                })

    # Check that all index files are registered
    if os.path.exists(PROJECT_INDEX_DIR):
        for filename in os.listdir(PROJECT_INDEX_DIR):
            if filename.endswith("_index_info.json"):
                index_path = os.path.join(PROJECT_INDEX_DIR, filename)

                # Check if this index is in the registry
                if filename not in registry["indices"]:
                    results["unregistered_files"].append({
                        "file_path": index_path,
                        "file_type": "index"
                    })

    # Check reference counts
    for document_id, document in registry["documents"].items():
        calculated_ref_count = len(document.get("projects", []))
        stored_ref_count = document.get("ref_count", 0)

        if calculated_ref_count != stored_ref_count:
            results["inconsistencies"].append({
                "type": "document_ref_count",
                "document_id": document_id,
                "calculated": calculated_ref_count,
                "stored": stored_ref_count
            })

            # Fix the inconsistency
            document["ref_count"] = calculated_ref_count
            results["fixed_issues"] += 1

    for filename, embedding in registry["embeddings"].items():
        calculated_ref_count = len(embedding.get("documents", []))
        stored_ref_count = embedding.get("ref_count", 0)

        if calculated_ref_count != stored_ref_count:
            results["inconsistencies"].append({
                "type": "embedding_ref_count",
                "filename": filename,
                "calculated": calculated_ref_count,
                "stored": stored_ref_count
            })

            # Fix the inconsistency
            embedding["ref_count"] = calculated_ref_count
            results["fixed_issues"] += 1

    # Save registry if issues were fixed
    if results["fixed_issues"] > 0:
        _save_registry(registry)
        logger.info(f"Fixed {results['fixed_issues']} registry inconsistencies")

    # Update status if issues were found
    if results["missing_files"] or results["unregistered_files"] or results["inconsistencies"]:
        results["status"] = "issues_found"

    return results
