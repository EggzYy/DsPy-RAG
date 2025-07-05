"""
Project index cleanup utilities for Local File Deep Research.
This module provides functions to clean up project indices when documents are removed.
"""

import os
import json
import logging
import shutil
import glob
from typing import Dict, List, Any, Optional

from .project_indexer import (
    PROJECT_INDEX_DIR,
    get_project_index_path,
    get_project_vector_store_path,
    load_project_index_info,
    save_project_index_info
)
from .config import SESSION_PERSIST_DIR

# Import constants for other directories
from .embedding import EMBEDDINGS_DIR
from .storage_manager import STORAGE_DIR

# Configure logging
logger = logging.getLogger(__name__)

def cleanup_document_from_indices(document_id: str, project_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Clean up document references from project indices.

    This function:
    1. Finds all project indices that reference the document
    2. Updates the index metadata to remove the document reference
    3. Does NOT re-index the project (that would require a separate call)

    Args:
        document_id: Document ID to remove
        project_id: Optional project ID if known

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "status": "success",
        "projects_updated": 0,
        "errors": []
    }

    try:
        # If we know the project ID, only check that project
        if project_id:
            project_indices = [project_id]
        else:
            # Otherwise, scan all project indices
            project_indices = []
            if os.path.exists(PROJECT_INDEX_DIR):
                for filename in os.listdir(PROJECT_INDEX_DIR):
                    if filename.endswith("_index_info.json"):
                        project_id = filename.split("_index_info.json")[0]
                        project_indices.append(project_id)

        # Process each project index
        for pid in project_indices:
            index_info = load_project_index_info(pid)
            if not index_info:
                continue

            # Check if this document is in the index
            if "document_ids" in index_info and document_id in index_info["document_ids"]:
                # Remove the document from the list
                index_info["document_ids"].remove(document_id)

                # Update document count
                if "document_count" in index_info:
                    index_info["document_count"] = len(index_info["document_ids"])

                # Add a note about the removal
                index_info["document_removed"] = {
                    "document_id": document_id,
                    "removed_at": import_datetime().now().isoformat()
                }

                # Save the updated index info
                save_project_index_info(pid, index_info)
                stats["projects_updated"] += 1
                logger.info(f"Removed document {document_id} from project index {pid}")

    except Exception as e:
        logger.error(f"Error cleaning up document from indices: {e}")
        stats["status"] = "error"
        stats["errors"].append(str(e))

    return stats

def import_datetime():
    """Import datetime module on demand to avoid circular imports."""
    from datetime import datetime
    return datetime

def thoroughly_clean_document_files(document_id: str) -> Dict[str, Any]:
    """
    Thoroughly clean all files related to a document from all storage locations.

    This function:
    1. Checks the document registry for all associated files
    2. Removes document files from storage folder
    3. Removes document references from project indices
    4. Removes document embeddings from embeddings folder
    5. Removes document references from session files

    Args:
        document_id: Document ID to clean up

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "status": "success",
        "storage_files_removed": 0,
        "project_indices_updated": 0,
        "embedding_files_removed": 0,
        "session_files_updated": 0,
        "errors": []
    }

    try:
        # First, try to use the document registry if available
        try:
            from .document_registry import get_all_document_files, unregister_document

            # Get all files from registry
            logger.info(f"Getting document files from registry for {document_id}")
            registry_files = get_all_document_files(document_id)

            # Remove all files listed in registry
            for storage_file in registry_files["storage_files"]:
                if os.path.exists(storage_file):
                    try:
                        os.remove(storage_file)
                        stats["storage_files_removed"] += 1
                        logger.info(f"Removed storage file from registry: {storage_file}")
                    except Exception as e:
                        error_msg = f"Error removing {storage_file}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            for version_file in registry_files["version_files"]:
                if os.path.exists(version_file):
                    try:
                        os.remove(version_file)
                        stats["storage_files_removed"] += 1
                        logger.info(f"Removed version file from registry: {version_file}")
                    except Exception as e:
                        error_msg = f"Error removing {version_file}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            for embedding_file in registry_files["embedding_files"]:
                if os.path.exists(embedding_file):
                    try:
                        os.remove(embedding_file)
                        stats["embedding_files_removed"] += 1
                        logger.info(f"Removed embedding file from registry: {embedding_file}")
                    except Exception as e:
                        error_msg = f"Error removing {embedding_file}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            # Update project indices
            for project_id in registry_files["projects"]:
                try:
                    # Update project index
                    index_path = os.path.join(PROJECT_INDEX_DIR, f"{project_id}_index_info.json")
                    if os.path.exists(index_path):
                        with open(index_path, "r") as f:
                            index_info = json.load(f)

                        if "document_ids" in index_info and document_id in index_info["document_ids"]:
                            index_info["document_ids"].remove(document_id)

                            if "document_count" in index_info:
                                index_info["document_count"] = len(index_info["document_ids"])

                            index_info["document_removed"] = {
                                "document_id": document_id,
                                "removed_at": import_datetime().now().isoformat()
                            }

                            with open(index_path, "w") as f:
                                json.dump(index_info, f, indent=2)

                            stats["project_indices_updated"] += 1
                            logger.info(f"Updated project index from registry: {index_path}")
                except Exception as e:
                    error_msg = f"Error updating project index {project_id}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

            # Unregister document from registry
            unregister_document(document_id)
            logger.info(f"Unregistered document {document_id} from registry")

        except ImportError:
            logger.warning("Document registry not available. Falling back to manual cleanup.")
            # If registry is not available, fall back to manual cleanup

        # Perform manual cleanup as a fallback or additional measure

        # 1. Clean up storage files
        if os.path.exists(STORAGE_DIR):
            # Main document file
            doc_path = os.path.join(STORAGE_DIR, f"doc_{document_id}")
            if os.path.exists(doc_path):
                try:
                    os.remove(doc_path)
                    stats["storage_files_removed"] += 1
                    logger.info(f"Removed document file: {doc_path}")
                except Exception as e:
                    error_msg = f"Error removing {doc_path}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

            # Version files
            version_dir = os.path.join(STORAGE_DIR, "versions")
            if os.path.exists(version_dir):
                version_pattern = os.path.join(version_dir, f"ver_{document_id}_*")
                for version_file in glob.glob(version_pattern):
                    try:
                        os.remove(version_file)
                        stats["storage_files_removed"] += 1
                        logger.info(f"Removed version file: {version_file}")
                    except Exception as e:
                        error_msg = f"Error removing {version_file}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

        # 2. Clean up project indices
        if os.path.exists(PROJECT_INDEX_DIR):
            # Find all project index files
            for filename in os.listdir(PROJECT_INDEX_DIR):
                if filename.endswith("_index_info.json"):
                    index_path = os.path.join(PROJECT_INDEX_DIR, filename)
                    try:
                        # Load index info
                        with open(index_path, "r") as f:
                            index_info = json.load(f)

                        # Check if document is in index
                        if "document_ids" in index_info and document_id in index_info["document_ids"]:
                            # Remove document from list
                            index_info["document_ids"].remove(document_id)

                            # Update document count
                            if "document_count" in index_info:
                                index_info["document_count"] = len(index_info["document_ids"])

                            # Add a note about the removal
                            index_info["document_removed"] = {
                                "document_id": document_id,
                                "removed_at": import_datetime().now().isoformat()
                            }

                            # Save updated index info
                            with open(index_path, "w") as f:
                                json.dump(index_info, f, indent=2)

                            stats["project_indices_updated"] += 1
                            logger.info(f"Updated project index: {index_path}")

                            # Clean up vector store files that contain this document
                            project_id = filename.split("_index_info.json")[0]
                            session_id = index_info.get("session_id")
                            if session_id:
                                try:
                                    # Try to use FAISS-based document removal
                                    from .project_indexer import remove_documents_from_project

                                    # Remove document from project index using FAISS
                                    removal_result = remove_documents_from_project(project_id, [document_id])

                                    if removal_result["status"] == "success":
                                        logger.info(f"FAISS document removal for {document_id} from project {project_id}: removed {removal_result['removed_chunks']} chunks")
                                        stats["project_indices_updated"] += 1
                                    else:
                                        logger.warning(f"FAISS document removal failed: {removal_result.get('message', 'Unknown error')}")
                                        # Fall back to aggressive approach if FAISS removal fails
                                        vector_store_path = os.path.join(PROJECT_INDEX_DIR, f"{project_id}_{session_id}_vector_store")
                                        if os.path.exists(vector_store_path):
                                            try:
                                                os.remove(vector_store_path)
                                                stats["storage_files_removed"] += 1
                                                logger.info(f"Removed vector store: {vector_store_path}")
                                            except Exception as e:
                                                error_msg = f"Error removing {vector_store_path}: {e}"
                                                logger.error(error_msg)
                                                stats["errors"].append(error_msg)
                                except (ImportError, AttributeError):
                                    logger.warning("FAISS-based document removal not available. Using aggressive approach.")
                                    # Fall back to aggressive approach - removing the entire vector store
                                    vector_store_path = os.path.join(PROJECT_INDEX_DIR, f"{project_id}_{session_id}_vector_store")
                                    if os.path.exists(vector_store_path):
                                        try:
                                            os.remove(vector_store_path)
                                            stats["storage_files_removed"] += 1
                                            logger.info(f"Removed vector store: {vector_store_path}")
                                        except Exception as e:
                                            error_msg = f"Error removing {vector_store_path}: {e}"
                                            logger.error(error_msg)
                                            stats["errors"].append(error_msg)
                    except Exception as e:
                        error_msg = f"Error processing index file {index_path}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

        # 3. Clean up embeddings
        if os.path.exists(EMBEDDINGS_DIR):
            # Embeddings might be stored with document ID in the filename
            embedding_pattern = os.path.join(EMBEDDINGS_DIR, f"*{document_id}*")
            for embedding_file in glob.glob(embedding_pattern):
                try:
                    os.remove(embedding_file)
                    stats["embedding_files_removed"] += 1
                    logger.info(f"Removed embedding file: {embedding_file}")
                except Exception as e:
                    error_msg = f"Error removing {embedding_file}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

        # 4. Clean up session files
        if os.path.exists(SESSION_PERSIST_DIR):
            # Session vector stores
            for filename in os.listdir(SESSION_PERSIST_DIR):
                if filename.endswith("_vector_store"):
                    # This is a more aggressive approach - we're removing any vector store that might contain this document
                    # A more refined approach would be to load each vector store, check if it contains the document, remove those chunks, and save it
                    session_path = os.path.join(SESSION_PERSIST_DIR, filename)
                    try:
                        # Check if file contains document ID (this is a heuristic)
                        with open(session_path, "rb") as f:
                            content = f.read()
                            if document_id.encode() in content:
                                # File contains reference to document, remove it
                                os.remove(session_path)
                                stats["session_files_updated"] += 1
                                logger.info(f"Removed session file: {session_path}")
                    except Exception as e:
                        error_msg = f"Error processing session file {session_path}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

    except Exception as e:
        logger.error(f"Error in thorough document cleanup: {e}")
        stats["status"] = "error"
        stats["errors"].append(str(e))

    return stats

def remove_project_indices(project_id: str) -> Dict[str, Any]:
    """
    Remove all indices for a project.

    Args:
        project_id: Project ID

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "status": "success",
        "files_removed": 0,
        "chunks_removed": 0,
        "embeddings_removed": 0,
        "errors": []
    }

    try:
        # Try to use FAISS-based project removal first
        try:
            from .project_indexer import remove_project

            # Remove project using FAISS
            removal_result = remove_project(project_id)

            if removal_result["status"] == "success":
                logger.info(f"FAISS project removal for {project_id}: removed {removal_result['removed_chunks']} chunks, {removal_result['removed_files']} files")
                stats["chunks_removed"] = removal_result["removed_chunks"]
                stats["files_removed"] = removal_result["removed_files"]

                if "removed_embeddings" in removal_result:
                    stats["embeddings_removed"] = removal_result["removed_embeddings"]
                    logger.info(f"Removed {removal_result['removed_embeddings']} embedding files for unreferenced documents")

                # If there were errors, add them to our stats
                if removal_result.get("errors", []):
                    stats["errors"].extend(removal_result["errors"])
                    logger.warning(f"Some errors occurred during FAISS project removal: {removal_result['errors']}")

                # Return early since FAISS-based removal was successful
                return stats
            else:
                logger.warning(f"FAISS project removal failed: {removal_result.get('message', 'Unknown error')}")
                # Continue with legacy cleanup
        except (ImportError, AttributeError):
            logger.warning("FAISS-based project removal not available. Using legacy cleanup.")

        # Legacy cleanup approach
        # Find all files related to this project
        if os.path.exists(PROJECT_INDEX_DIR):
            for filename in os.listdir(PROJECT_INDEX_DIR):
                if filename.startswith(f"{project_id}_"):
                    file_path = os.path.join(PROJECT_INDEX_DIR, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        stats["files_removed"] += 1
                        logger.info(f"Removed project index file: {file_path}")
                    except Exception as e:
                        error_msg = f"Error removing {file_path}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

        # Also check session directory for project-specific files
        if os.path.exists(SESSION_PERSIST_DIR):
            for filename in os.listdir(SESSION_PERSIST_DIR):
                if filename.startswith(f"project_{project_id}_"):
                    file_path = os.path.join(SESSION_PERSIST_DIR, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        stats["files_removed"] += 1
                        logger.info(f"Removed project session file: {file_path}")
                    except Exception as e:
                        error_msg = f"Error removing {file_path}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

    except Exception as e:
        logger.error(f"Error removing project indices: {e}")
        stats["status"] = "error"
        stats["errors"].append(str(e))

    return stats
