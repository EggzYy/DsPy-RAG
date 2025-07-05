"""
Organization Manager for Local File Deep Research.
This module provides functions to manage folder organization and cleanup.
"""

import os
import logging
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

# Import constants
from .storage_manager import STORAGE_DIR
from .embedding import EMBEDDING_CACHE_DIR as EMBEDDINGS_DIR
from .project_indexer import PROJECT_INDEX_DIR

def ensure_organized_folders():
    """
    Ensure all folders are properly organized.
    Creates necessary folders if they don't exist.
    """
    # Ensure main folders exist
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(PROJECT_INDEX_DIR, exist_ok=True)

    # Ensure registry directory exists
    registry_dir = os.path.join(STORAGE_DIR, "registry")
    os.makedirs(registry_dir, exist_ok=True)

    # Ensure versions directory exists
    versions_dir = os.path.join(STORAGE_DIR, "versions")
    os.makedirs(versions_dir, exist_ok=True)

    logger.info(f"Ensured organized folder structure:")
    logger.info(f"  - Storage: {STORAGE_DIR}")
    logger.info(f"  - Embeddings: {EMBEDDINGS_DIR}")
    logger.info(f"  - Project Indices: {PROJECT_INDEX_DIR}")
    logger.info(f"  - Registry: {registry_dir}")
    logger.info(f"  - Versions: {versions_dir}")

    return {
        "storage_dir": STORAGE_DIR,
        "embeddings_dir": EMBEDDINGS_DIR,
        "project_indices_dir": PROJECT_INDEX_DIR,
        "registry_dir": registry_dir,
        "versions_dir": versions_dir
    }

def cleanup_after_indexing(document_ids: List[str] = None, project_id: str = None) -> Dict[str, Any]:
    """
    Perform a thorough cleanup after indexing documents.
    After indexing, we can remove all original files since the content is now in the vector store.

    Args:
        document_ids: Optional list of document IDs to clean up. If None, clean up all.
        project_id: Optional project ID to clean up. If provided, will clean up files in storage/projects for this project.

    Returns:
        Dictionary with cleanup statistics
    """
    from .document_cleanup import cleanup_document_files, cleanup_storage_files, cleanup_projects_folder
    from .database_cleanup import cleanup_database_files

    stats = {
        "status": "success",
        "document_files_removed": 0,
        "storage_files_removed": 0,
        "database_files_removed": 0,
        "embedding_files_removed": 0,
        "session_files_removed": 0,
        "projects_files_removed": 0,
        "total_bytes_freed": 0,
        "errors": []
    }

    logger.info(f"Starting thorough cleanup after indexing for document IDs: {document_ids if document_ids else 'all'}")
    if project_id:
        logger.info(f"Will also clean up files in storage/projects for project: {project_id}")
    logger.info(f"All original files will be removed since content is now in the vector store")

    # Clean up document files
    try:
        doc_cleanup_result = cleanup_document_files(document_ids)
        stats["document_files_removed"] = doc_cleanup_result["total_files_removed"]
        stats["total_bytes_freed"] += doc_cleanup_result["total_bytes_freed"]
        stats["errors"].extend(doc_cleanup_result["errors"])
        logger.info(f"Document cleanup: removed {doc_cleanup_result['total_files_removed']} files, freed {doc_cleanup_result['human_readable_bytes_freed']}")
    except Exception as e:
        error_msg = f"Error during document cleanup: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # Clean up storage files
    try:
        storage_cleanup_result = cleanup_storage_files()
        stats["storage_files_removed"] = storage_cleanup_result["total_files_removed"]
        stats["total_bytes_freed"] += storage_cleanup_result["total_bytes_freed"]
        stats["errors"].extend(storage_cleanup_result["errors"])
        logger.info(f"Storage cleanup: removed {storage_cleanup_result['total_files_removed']} files, freed {storage_cleanup_result['human_readable_bytes_freed']}")
    except Exception as e:
        error_msg = f"Error during storage cleanup: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # Clean up database files
    if document_ids:
        try:
            db_cleanup_result = cleanup_database_files(document_ids)
            stats["database_files_removed"] = db_cleanup_result["total_files_removed"]
            stats["total_bytes_freed"] += db_cleanup_result["total_bytes_freed"]
            stats["errors"].extend(db_cleanup_result["errors"])
            logger.info(f"Database cleanup: removed {db_cleanup_result['total_files_removed']} files, freed {db_cleanup_result['human_readable_bytes_freed']}")
        except Exception as e:
            error_msg = f"Error during database cleanup: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    # Clean up embedding files
    try:
        from .document_registry import cleanup_embeddings
        embedding_cleanup_result = cleanup_embeddings(document_ids)
        stats["embedding_files_removed"] = embedding_cleanup_result["total_files_removed"]
        stats["total_bytes_freed"] += embedding_cleanup_result["total_bytes_freed"]
        stats["errors"].extend(embedding_cleanup_result["errors"])
        logger.info(f"Embedding cleanup: removed {embedding_cleanup_result['total_files_removed']} files, freed {embedding_cleanup_result['human_readable_bytes_freed']}")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Embedding cleanup not available: {str(e)}")
    except Exception as e:
        error_msg = f"Error during embedding cleanup: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # Clean up session files
    try:
        from .document_registry import cleanup_session_files
        session_cleanup_result = cleanup_session_files()
        stats["session_files_removed"] = session_cleanup_result["total_files_removed"]
        stats["total_bytes_freed"] += session_cleanup_result["total_bytes_freed"]
        stats["errors"].extend(session_cleanup_result["errors"])
        logger.info(f"Session cleanup: removed {session_cleanup_result['total_files_removed']} files, freed {session_cleanup_result['human_readable_bytes_freed']}")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Session cleanup not available: {str(e)}")
    except Exception as e:
        error_msg = f"Error during session cleanup: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # Clean up projects folder
    try:
        projects_cleanup_result = cleanup_projects_folder(project_id)
        stats["projects_files_removed"] = projects_cleanup_result["total_files_removed"]
        stats["total_bytes_freed"] += projects_cleanup_result["total_bytes_freed"]
        stats["errors"].extend(projects_cleanup_result["errors"])
        logger.info(f"Projects folder cleanup: removed {projects_cleanup_result['total_files_removed']} files, freed {projects_cleanup_result['human_readable_bytes_freed']}")
    except Exception as e:
        error_msg = f"Error during projects folder cleanup: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # Verify registry integrity
    try:
        from .document_registry import verify_registry_integrity
        integrity_result = verify_registry_integrity()
        if integrity_result["status"] != "success":
            logger.warning(f"Registry integrity check found issues: {len(integrity_result['missing_files'])} missing files, {len(integrity_result['unregistered_files'])} unregistered files, {len(integrity_result['inconsistencies'])} inconsistencies")
            if integrity_result["fixed_issues"] > 0:
                logger.info(f"Fixed {integrity_result['fixed_issues']} registry inconsistencies")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Registry integrity check not available: {str(e)}")
    except Exception as e:
        error_msg = f"Error during registry integrity check: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # Format total bytes freed
    from .document_cleanup import _format_bytes
    stats["human_readable_bytes_freed"] = _format_bytes(stats["total_bytes_freed"])

    # Update status if there were errors
    if stats["errors"]:
        stats["status"] = "partial"

    return stats

def get_folder_stats() -> Dict[str, Any]:
    """
    Get statistics about the folders.

    Returns:
        Dictionary with folder statistics
    """
    stats = {
        "storage": {
            "path": STORAGE_DIR,
            "file_count": 0,
            "total_size": 0
        },
        "embeddings": {
            "path": EMBEDDINGS_DIR,
            "file_count": 0,
            "total_size": 0
        },
        "project_indices": {
            "path": PROJECT_INDEX_DIR,
            "file_count": 0,
            "total_size": 0
        }
    }

    # Get storage stats
    if os.path.exists(STORAGE_DIR):
        for root, _, files in os.walk(STORAGE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                stats["storage"]["file_count"] += 1
                stats["storage"]["total_size"] += os.path.getsize(file_path)

    # Get embeddings stats
    if os.path.exists(EMBEDDINGS_DIR):
        for root, _, files in os.walk(EMBEDDINGS_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                stats["embeddings"]["file_count"] += 1
                stats["embeddings"]["total_size"] += os.path.getsize(file_path)

    # Get project indices stats
    if os.path.exists(PROJECT_INDEX_DIR):
        for root, _, files in os.walk(PROJECT_INDEX_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                stats["project_indices"]["file_count"] += 1
                stats["project_indices"]["total_size"] += os.path.getsize(file_path)

    # Format sizes
    from .document_cleanup import _format_bytes
    stats["storage"]["human_readable_size"] = _format_bytes(stats["storage"]["total_size"])
    stats["embeddings"]["human_readable_size"] = _format_bytes(stats["embeddings"]["total_size"])
    stats["project_indices"]["human_readable_size"] = _format_bytes(stats["project_indices"]["total_size"])

    return stats
