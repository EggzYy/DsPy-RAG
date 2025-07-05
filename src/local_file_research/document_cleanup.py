"""
Document cleanup utilities for Local File Deep Research.
This module provides functions to clean up document files after indexing.
It uses the document registry as the single source of truth for file management.
"""

import os
import logging
import json
import shutil
from typing import Dict, List, Any, Optional

from .storage_manager import STORAGE_DIR

# Define legacy directory for backward compatibility
BASE_DIR = os.environ.get("BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", os.path.join(BASE_DIR, "documents"))

# Configure logging
logger = logging.getLogger(__name__)

def cleanup_document_files(document_ids: List[str] = None) -> Dict[str, Any]:
    """
    Clean up document files after indexing.

    Args:
        document_ids: Optional list of document IDs to clean up. If None, clean up all.

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "status": "success",
        "total_files_removed": 0,
        "total_bytes_freed": 0,
        "errors": []
    }

    # Use the document registry to get file information
    try:
        from .document_registry import _load_registry, unregister_document_file

        # Load the registry
        registry = _load_registry()

        # If document_ids is provided, only clean up those documents
        if document_ids:
            for document_id in document_ids:
                if document_id in registry["documents"]:
                    document = registry["documents"][document_id]

                    # Get all storage files for this document
                    storage_files = [f["path"] for f in document.get("files", []) if f["type"] == "storage"]

                    # Remove each storage file
                    for file_path in storage_files:
                        try:
                            if os.path.exists(file_path):
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                stats["total_files_removed"] += 1
                                stats["total_bytes_freed"] += file_size
                                logger.info(f"Removed document file after indexing: {file_path} ({file_size} bytes)")

                                # Unregister the file from the document
                                unregister_document_file(document_id, file_path)
                                logger.info(f"Unregistered file {file_path} for document {document_id} from registry")
                        except Exception as e:
                            error_msg = f"Error removing file {file_path}: {str(e)}"
                            logger.error(error_msg)
                            stats["errors"].append(error_msg)
        else:
            # Clean up all documents
            for document_id, document in registry["documents"].items():
                # Get all storage files for this document
                storage_files = [f["path"] for f in document.get("files", []) if f["type"] == "storage"]

                # Remove each storage file
                for file_path in storage_files:
                    try:
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            stats["total_files_removed"] += 1
                            stats["total_bytes_freed"] += file_size
                            logger.info(f"Removed document file after indexing: {file_path} ({file_size} bytes)")

                            # Unregister the file from the document
                            unregister_document_file(document_id, file_path)
                            logger.info(f"Unregistered file {file_path} for document {document_id} from registry")
                    except Exception as e:
                        error_msg = f"Error removing file {file_path}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
    except ImportError:
        logger.warning("Document registry not available. Using legacy cleanup method.")

        # Clean up files in the storage directory using the legacy method
        if os.path.exists(STORAGE_DIR):
            # First, clean up main storage files
            try:
                # Get list of files in storage directory
                files = [f for f in os.listdir(STORAGE_DIR) if os.path.isfile(os.path.join(STORAGE_DIR, f))]

                # Filter out JSON files and registry directory
                files = [f for f in files if not f.endswith(".json") and f != "registry"]

                # Filter by document IDs if provided
                if document_ids:
                    # Match files that contain any of the document IDs
                    files = [f for f in files if any(doc_id in f for doc_id in document_ids)]

                # Remove files
                for file in files:
                    try:
                        file_path = os.path.join(STORAGE_DIR, file)
                        file_size = os.path.getsize(file_path)

                        # Extract document ID from filename
                        doc_id = None
                        if file.startswith("doc_"):
                            doc_id = file[4:]  # Remove "doc_" prefix

                        # Always remove the file after indexing
                        # The content is already extracted and stored in the document object
                        os.remove(file_path)
                        stats["total_files_removed"] += 1
                        stats["total_bytes_freed"] += file_size
                        logger.info(f"Removed document file after indexing: {file_path} ({file_size} bytes)")
                    except Exception as e:
                        error_msg = f"Error removing file {os.path.join(STORAGE_DIR, file)}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"Error during storage cleanup: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

            # Next, clean up files in project session folders
            # These are the original uploaded files that should be removed after indexing
            try:
                # Find all project folders
                project_folders = [d for d in os.listdir(STORAGE_DIR)
                                  if os.path.isdir(os.path.join(STORAGE_DIR, d))
                                  and not d in ["registry", "versions"]]

                logger.info(f"Found {len(project_folders)} project folders to check for cleanup")

                for project_folder in project_folders:
                    project_path = os.path.join(STORAGE_DIR, project_folder)
                    logger.info(f"Checking project folder: {project_path}")

                    # Find all session folders
                    try:
                        session_folders = [d for d in os.listdir(project_path)
                                          if os.path.isdir(os.path.join(project_path, d))
                                          and d.startswith("session_")]

                        logger.info(f"Found {len(session_folders)} session folders in project {project_folder}")

                        for session_folder in session_folders:
                            session_path = os.path.join(project_path, session_folder)
                            logger.info(f"Checking session folder: {session_path}")

                            # Find all user folders
                            try:
                                user_folders = [d for d in os.listdir(session_path)
                                               if os.path.isdir(os.path.join(session_path, d))]

                                logger.info(f"Found {len(user_folders)} user folders in session {session_folder}")

                                for user_folder in user_folders:
                                    user_path = os.path.join(session_path, user_folder)
                                    logger.info(f"Checking user folder: {user_path}")

                                    # Find all files in user folder
                                    try:
                                        user_files = [f for f in os.listdir(user_path)
                                                     if os.path.isfile(os.path.join(user_path, f))]

                                        logger.info(f"Found {len(user_files)} files in user folder {user_folder}")

                                        # If document_ids is provided, only remove files for those documents
                                        # Otherwise, remove all files
                                        for file in user_files:
                                            try:
                                                file_path = os.path.join(user_path, file)
                                                file_size = os.path.getsize(file_path)

                                                # Remove the file
                                                logger.info(f"Attempting to remove file: {file_path}")
                                                os.remove(file_path)
                                                stats["total_files_removed"] += 1
                                                stats["total_bytes_freed"] += file_size
                                                logger.info(f"Successfully removed session file after indexing: {file_path} ({file_size} bytes)")
                                            except Exception as e:
                                                error_msg = f"Error removing session file {file_path}: {str(e)}"
                                                logger.error(error_msg)
                                                stats["errors"].append(error_msg)
                                    except Exception as e:
                                        error_msg = f"Error listing files in user folder {user_path}: {str(e)}"
                                        logger.error(error_msg)
                                        stats["errors"].append(error_msg)
                            except Exception as e:
                                error_msg = f"Error listing user folders in session {session_path}: {str(e)}"
                                logger.error(error_msg)
                                stats["errors"].append(error_msg)
                    except Exception as e:
                        error_msg = f"Error listing session folders in project {project_path}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"Error during session folder cleanup: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

    # Also clean up legacy documents directory if it exists
    if os.path.exists(DOCUMENTS_DIR):
        try:
            # Load documents.json to preserve metadata
            documents_json_path = os.path.join(DOCUMENTS_DIR, "documents.json")
            if os.path.exists(documents_json_path):
                try:
                    with open(documents_json_path, "r") as f:
                        legacy_metadata = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading documents.json: {e}")
                    legacy_metadata = []
            else:
                legacy_metadata = []

            # Get list of files in documents directory
            files = [f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f))]

            # Filter out documents.json
            files = [f for f in files if f != "documents.json"]

            # Filter by document IDs if provided
            if document_ids:
                files = [f for f in files if f in document_ids]

            # Remove files
            for file in files:
                try:
                    file_path = os.path.join(DOCUMENTS_DIR, file)
                    file_size = os.path.getsize(file_path)

                    # Check if this document is in the metadata
                    document_in_metadata = any(doc.get("document_id") == file for doc in legacy_metadata)

                    if document_in_metadata:
                        # Remove the file but keep the metadata
                        os.remove(file_path)
                        stats["total_files_removed"] += 1
                        stats["total_bytes_freed"] += file_size
                        logger.info(f"Removed legacy document file: {file_path} ({file_size} bytes)")
                    else:
                        # Skip files not in metadata
                        logger.info(f"Skipping legacy file not in metadata: {file_path}")
                except Exception as e:
                    error_msg = f"Error removing legacy file {os.path.join(DOCUMENTS_DIR, file)}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
        except Exception as e:
            error_msg = f"Error during legacy document cleanup: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    # Add human-readable size
    stats["human_readable_bytes_freed"] = _format_bytes(stats["total_bytes_freed"])

    return stats

def cleanup_storage_files() -> Dict[str, Any]:
    """
    Clean up all document files in the storage directory, keeping only the metadata and registry.
    This function is called after indexing to remove original files that have been processed.

    Specifically cleans up:
    - Files in the root of the storage directory
    - Files in storage/content
    - Files in storage/projects
    - Files in project-specific folders

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "status": "success",
        "total_files_removed": 0,
        "total_bytes_freed": 0,
        "errors": []
    }

    if not os.path.exists(STORAGE_DIR):
        logger.warning(f"Storage directory {STORAGE_DIR} does not exist. Nothing to clean up.")
        return {
            "status": "skipped",
            "reason": "storage_dir_not_found",
            "total_files_removed": 0,
            "total_bytes_freed": 0,
            "human_readable_bytes_freed": "0B",
            "errors": []
        }

    try:
        # PART 1: Clean up files in the root of the storage directory
        logger.info(f"Cleaning up files in the root of the storage directory: {STORAGE_DIR}")
        files = [f for f in os.listdir(STORAGE_DIR) if os.path.isfile(os.path.join(STORAGE_DIR, f))]

        # Filter out metadata files and registry directory
        files = [f for f in files if not f.endswith(".json") and f != "registry"]

        # Remove files
        for file in files:
            try:
                file_path = os.path.join(STORAGE_DIR, file)
                file_size = os.path.getsize(file_path)

                # Extract document ID if possible
                doc_id = None
                if file.startswith("doc_"):
                    doc_id = file[4:]  # Remove "doc_" prefix

                # Remove the file
                os.remove(file_path)
                stats["total_files_removed"] += 1
                stats["total_bytes_freed"] += file_size
                logger.info(f"Removed storage file after indexing: {file_path} ({file_size} bytes)")

                # Update registry if possible
                if doc_id:
                    try:
                        from .document_registry import unregister_document_file
                        unregister_document_file(doc_id, file_path)
                        logger.info(f"Unregistered file {file_path} for document {doc_id} from registry")
                    except (ImportError, AttributeError):
                        # Registry not available or function not found
                        pass
            except Exception as e:
                error_msg = f"Error removing file {os.path.join(STORAGE_DIR, file)}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        # PART 2: Clean up files in storage/content directory
        content_dir = os.path.join(STORAGE_DIR, "content")
        if os.path.exists(content_dir):
            logger.info(f"Cleaning up content directory: {content_dir}")

            try:
                # First, try to remove all files in the directory
                for root, _, files in os.walk(content_dir):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            stats["total_files_removed"] += 1
                            stats["total_bytes_freed"] += file_size
                            logger.info(f"Removed content file: {file_path} ({file_size} bytes)")
                        except Exception as e:
                            error_msg = f"Error removing content file {file_path}: {str(e)}"
                            logger.error(error_msg)
                            stats["errors"].append(error_msg)

                # Then recreate the directory to ensure it's empty
                try:
                    # Remove the entire directory
                    shutil.rmtree(content_dir)
                    logger.info(f"Removed content directory: {content_dir}")

                    # Recreate the directory
                    os.makedirs(content_dir, exist_ok=True)
                    logger.info(f"Recreated empty content directory: {content_dir}")
                except Exception as e:
                    error_msg = f"Error recreating content directory: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"Error cleaning up content directory: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        # PART 3: Clean up files in storage/projects directory
        projects_dir = os.path.join(STORAGE_DIR, "projects")
        if os.path.exists(projects_dir):
            logger.info(f"Cleaning up projects directory: {projects_dir}")

            try:
                # First, try to remove all files in the directory
                for root, _, files in os.walk(projects_dir):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            stats["total_files_removed"] += 1
                            stats["total_bytes_freed"] += file_size
                            logger.info(f"Removed projects file: {file_path} ({file_size} bytes)")
                        except Exception as e:
                            error_msg = f"Error removing projects file {file_path}: {str(e)}"
                            logger.error(error_msg)
                            stats["errors"].append(error_msg)

                # Then recreate the directory to ensure it's empty
                try:
                    # Remove the entire directory
                    shutil.rmtree(projects_dir)
                    logger.info(f"Removed projects directory: {projects_dir}")

                    # Recreate the directory
                    os.makedirs(projects_dir, exist_ok=True)
                    logger.info(f"Recreated empty projects directory: {projects_dir}")
                except Exception as e:
                    error_msg = f"Error recreating projects directory: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"Error cleaning up projects directory: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        # PART 4: Clean up files in project-specific folders
        logger.info(f"Cleaning up files in project-specific folders")

        # Find all project folders (excluding special directories)
        project_folders = [d for d in os.listdir(STORAGE_DIR)
                          if os.path.isdir(os.path.join(STORAGE_DIR, d))
                          and d not in ["registry", "versions", "sessions", "embeddings", "content", "projects"]]

        logger.info(f"Found {len(project_folders)} project-specific folders to clean up")

        for project_folder in project_folders:
            project_path = os.path.join(STORAGE_DIR, project_folder)

            # Find all files in the project folder
            for root, _, files in os.walk(project_path):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)

                        # Skip metadata files
                        if file.endswith(".json"):
                            continue

                        file_size = os.path.getsize(file_path)

                        # Remove the file
                        os.remove(file_path)
                        stats["total_files_removed"] += 1
                        stats["total_bytes_freed"] += file_size
                        logger.info(f"Removed project file after indexing: {file_path} ({file_size} bytes)")
                    except Exception as e:
                        error_msg = f"Error removing project file {file_path}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            # Try to remove empty directories
            try:
                for root, dirs, files in os.walk(project_path, topdown=False):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            if not os.listdir(dir_path):  # Check if directory is empty
                                os.rmdir(dir_path)
                                logger.info(f"Removed empty directory: {dir_path}")
                        except Exception as e:
                            logger.debug(f"Could not remove directory {dir_path}: {str(e)}")
            except Exception as e:
                logger.debug(f"Error cleaning up empty directories in project {project_folder}: {str(e)}")

        # Add human-readable size
        stats["human_readable_bytes_freed"] = _format_bytes(stats["total_bytes_freed"])

        return stats
    except Exception as e:
        error_msg = f"Error during storage cleanup: {str(e)}"
        logger.error(error_msg)
        stats["status"] = "error"
        stats["errors"].append(error_msg)
        return stats

def cleanup_projects_folder(project_id: str = None) -> Dict[str, Any]:
    """
    Clean up files in the storage/projects folder after indexing.
    This function specifically targets the storage/projects folder, which contains
    temporary files uploaded to projects before indexing.

    Args:
        project_id: Optional project ID to clean up. If None, clean up all projects.

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "status": "success",
        "total_files_removed": 0,
        "total_bytes_freed": 0,
        "errors": []
    }

    # Get the projects directory path
    projects_dir = os.path.join(STORAGE_DIR, "projects")

    if not os.path.exists(projects_dir):
        logger.warning(f"Projects directory {projects_dir} does not exist. Creating it.")
        try:
            os.makedirs(projects_dir, exist_ok=True)
            return {
                "status": "created",
                "reason": "projects_dir_not_found",
                "total_files_removed": 0,
                "total_bytes_freed": 0,
                "human_readable_bytes_freed": "0B",
                "errors": []
            }
        except Exception as e:
            error_msg = f"Error creating projects directory: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "reason": "failed_to_create_projects_dir",
                "error": str(e),
                "total_files_removed": 0,
                "total_bytes_freed": 0,
                "human_readable_bytes_freed": "0B",
                "errors": [error_msg]
            }

    try:
        logger.info(f"Starting cleanup of storage/projects folder")

        # If project_id is provided, only clean up that project
        if project_id:
            project_path = os.path.join(projects_dir, project_id)
            if os.path.exists(project_path):
                logger.info(f"Cleaning up files for project {project_id}")

                try:
                    # Calculate total size before removal
                    total_size = 0
                    for root, _, files in os.walk(project_path):
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                total_size += os.path.getsize(file_path)
                            except Exception:
                                pass

                    # Remove the entire project directory
                    shutil.rmtree(project_path)
                    stats["total_files_removed"] += 1  # Count as one operation
                    stats["total_bytes_freed"] += total_size
                    logger.info(f"Removed project directory: {project_path} ({total_size} bytes)")

                    # Recreate the project directory
                    os.makedirs(project_path, exist_ok=True)
                    logger.info(f"Recreated empty project directory: {project_path}")
                except Exception as e:
                    error_msg = f"Error removing project directory {project_path}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
            else:
                logger.warning(f"Project directory {project_path} does not exist. Creating it.")
                try:
                    os.makedirs(project_path, exist_ok=True)
                    logger.info(f"Created project directory: {project_path}")
                except Exception as e:
                    error_msg = f"Error creating project directory {project_path}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
        else:
            # Clean up all projects by recreating the projects directory
            logger.info(f"Cleaning up all projects by recreating the projects directory")

            try:
                # Calculate total size before removal
                total_size = 0
                for root, _, files in os.walk(projects_dir):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path)
                        except Exception:
                            pass

                # Remove the entire projects directory
                shutil.rmtree(projects_dir)
                stats["total_files_removed"] += 1  # Count as one operation
                stats["total_bytes_freed"] += total_size
                logger.info(f"Removed projects directory: {projects_dir} ({total_size} bytes)")

                # Recreate the projects directory
                os.makedirs(projects_dir, exist_ok=True)
                logger.info(f"Recreated empty projects directory: {projects_dir}")
            except Exception as e:
                error_msg = f"Error recreating projects directory: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

                # No need to try to remove empty directories since we've already recreated the directory

        # Add human-readable size
        stats["human_readable_bytes_freed"] = _format_bytes(stats["total_bytes_freed"])

        return stats
    except Exception as e:
        error_msg = f"Error during projects folder cleanup: {str(e)}"
        logger.error(error_msg)
        stats["status"] = "error"
        stats["errors"].append(error_msg)
        stats["human_readable_bytes_freed"] = _format_bytes(stats["total_bytes_freed"])
        return stats

def _format_bytes(size_bytes: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable string
    """
    if size_bytes == 0:
        return "0B"

    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1

    return f"{size_bytes:.2f} {size_names[i]}"
