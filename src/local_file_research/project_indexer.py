"""
Project indexing utilities for Local File Deep Research.
This module provides functions to manage project-specific indexing.
"""

import os
import json
import logging
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil

from .document_manager import get_documents, get_document_by_id
from .vector_store import get_vector_store, chunk_file_content
from .embedding import get_embeddings
from .database_cleanup import cleanup_database_files
from .document_cleanup import cleanup_document_files, cleanup_storage_files
from .metadata_optimizer import optimize_document_metadata, add_additional_metadata_fields
from .legacy_cleanup import remove_legacy_folders
from .config import SESSION_PERSIST_DIR, PROJECT_INDEX_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Ensure project index directory exists
os.makedirs(PROJECT_INDEX_DIR, exist_ok=True)

def get_project_index_path(project_id: str) -> str:
    """
    Get the path to the project index metadata file.

    Args:
        project_id: Project ID

    Returns:
        Path to the project index metadata file
    """
    return os.path.join(PROJECT_INDEX_DIR, f"{project_id}_index_info.json")

def get_project_vector_store_path(project_id: str, session_id: Optional[str] = None) -> str:
    """
    Get the path to the project vector store and ensure the directory exists.

    Args:
        project_id: Project ID
        session_id: Optional Session ID (not used in new implementation)

    Returns:
        Path to the project vector store
    """
    # Use a consistent path based only on project_id
    path = os.path.join(PROJECT_INDEX_DIR, f"{project_id}_vector_store")

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Also ensure the vector_store subdirectory exists
    vector_store_dir = os.path.join(path, "vector_store")
    os.makedirs(vector_store_dir, exist_ok=True)

    return path

def save_project_index_info(project_id: str, index_info: Dict[str, Any]) -> None:
    """
    Save project index information.

    Args:
        project_id: Project ID
        index_info: Index information dictionary
    """
    index_path = get_project_index_path(project_id)
    with open(index_path, "w") as f:
        json.dump(index_info, f, indent=2)
    logger.info(f"Saved project index info to {index_path}")

def load_project_index_info(project_id: str) -> Optional[Dict[str, Any]]:
    """
    Load project index information.

    Args:
        project_id: Project ID

    Returns:
        Index information dictionary or None if not found
    """
    index_path = get_project_index_path(project_id)
    if not os.path.exists(index_path):
        return None

    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading project index info: {e}")
        return None

def index_project_documents(project_id: str, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Index documents for a specific project.

    Args:
        project_id: Project ID
        document_ids: Optional list of document IDs to index. If None, index all documents in the project.

    Returns:
        Index information dictionary
    """
    # Create a new session ID
    session_id = f"project_{project_id}_{str(uuid.uuid4())}"

    # Get documents to index
    if document_ids:
        documents = [get_document_by_id(doc_id) for doc_id in document_ids]
        documents = [doc for doc in documents if doc is not None and doc.get("project_id") == project_id]
    else:
        documents = get_documents(project_id=project_id)

    if not documents:
        logger.warning(f"No documents found for project {project_id}")
        return {
            "session_id": session_id,
            "document_count": 0,
            "chunk_count": 0,
            "last_indexed": datetime.now().isoformat(),
            "status": "no_documents"
        }

    # Create vector store
    # Get the project vector store path
    project_vector_store_path = get_project_vector_store_path(project_id)
    logger.info(f"Creating vector store for project {project_id} at {project_vector_store_path}")

    # Create a fresh vector store
    try:
        # First, check if there's an existing vector store and try to load it
        vector_store = get_vector_store(use_faiss=True).load(project_vector_store_path)
        if vector_store:
            logger.info(f"Found existing vector store for project {project_id}, will add to it")
        else:
            # Create a new vector store
            logger.info(f"No existing vector store found, creating new one for project {project_id}")
            vector_store = get_vector_store(use_faiss=True)
    except Exception as e:
        logger.warning(f"Error loading existing vector store: {e}. Creating a new one.")
        # If there's an error loading, try to clean up the directory and create a fresh one
        try:
            import shutil
            if os.path.exists(project_vector_store_path):
                backup_dir = f"{project_vector_store_path}_backup_{int(time.time())}"
                logger.info(f"Backing up problematic vector store to {backup_dir}")
                shutil.copytree(project_vector_store_path, backup_dir)
                logger.info(f"Removing problematic vector store directory {project_vector_store_path}")
                shutil.rmtree(project_vector_store_path)
                os.makedirs(project_vector_store_path, exist_ok=True)
                logger.info(f"Created fresh directory at {project_vector_store_path}")
        except Exception as cleanup_err:
            logger.error(f"Error cleaning up problematic vector store directory: {cleanup_err}")

        # Create a new vector store
        vector_store = get_vector_store(use_faiss=True)

    # Process documents
    all_chunks = []
    for document in documents:
        # Create file record
        file_record = {
            "content": document["content"],
            "path": document.get("file_path", f"document_{document['document_id']}"),
            "name": document["title"],
            "source_type": document["document_type"],
            "document_id": document["document_id"],
            "project_id": document.get("project_id")
        }

        # Chunk document
        chunks = chunk_file_content(file_record)

        # Add document metadata to chunks
        for chunk in chunks:
            chunk["document_id"] = document["document_id"]
            chunk["document_title"] = document["title"]
            chunk["document_type"] = document["document_type"]
            chunk["project_id"] = document.get("project_id")
            chunk["owner"] = document.get("owner")

        all_chunks.extend(chunks)

    # Get embeddings for chunks
    if all_chunks:
        texts = [chunk["content"] for chunk in all_chunks]
        embeddings = get_embeddings(texts)

        # Add embeddings to chunks
        for i, chunk in enumerate(all_chunks):
            if i < len(embeddings):
                chunk["embedding"] = embeddings[i]

        # Add chunks to vector store
        vector_store.add_chunks(all_chunks)

        # Save vector store directly to the project folder
        project_vector_store_path = get_project_vector_store_path(project_id)
        vector_store.save(project_vector_store_path)

        # Log the save location
        logger.info(f"Saved vector store for project {project_id} to {project_vector_store_path}")

    # Create index info
    index_info = {
        "session_id": session_id,
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "document_ids": [doc["document_id"] for doc in documents],
        "last_indexed": datetime.now().isoformat(),
        "status": "success"
    }

    # Save index info
    save_project_index_info(project_id, index_info)

    # Clean up files after indexing
    if documents:
        document_ids = [doc["document_id"] for doc in documents]

        try:
            # Use the organization manager for comprehensive cleanup
            from .organization_manager import cleanup_after_indexing
            cleanup_result = cleanup_after_indexing(document_ids, project_id)

            logger.info(f"Cleanup after indexing project {project_id}: {cleanup_result['status']}")
            logger.info(f"  - Document files removed: {cleanup_result['document_files_removed']}")
            logger.info(f"  - Storage files removed: {cleanup_result['storage_files_removed']}")
            logger.info(f"  - Database files removed: {cleanup_result['database_files_removed']}")
            logger.info(f"  - Projects files removed: {cleanup_result['projects_files_removed']}")
            logger.info(f"  - Total space freed: {cleanup_result['human_readable_bytes_freed']}")

            if cleanup_result['errors']:
                logger.warning(f"  - Errors during cleanup: {len(cleanup_result['errors'])}")
                for error in cleanup_result['errors'][:5]:  # Log first 5 errors
                    logger.warning(f"    - {error}")

            # Add cleanup info to index info
            index_info["cleanup_result"] = cleanup_result
        except ImportError:
            logger.warning("Organization manager not available. Using legacy cleanup methods.")

            # Clean up database files
            db_cleanup_result = cleanup_database_files(document_ids)
            logger.info(f"Database cleanup after indexing project {project_id}: {db_cleanup_result['status']}, removed {db_cleanup_result['total_files_removed']} files, freed {db_cleanup_result['human_readable_bytes_freed']}")

            # Clean up document files
            doc_cleanup_result = cleanup_document_files(document_ids)
            logger.info(f"Document cleanup after indexing project {project_id}: {doc_cleanup_result['status']}, removed {doc_cleanup_result['total_files_removed']} files, freed {doc_cleanup_result['human_readable_bytes_freed']}")

            # Clean up storage files
            storage_cleanup_result = cleanup_storage_files()
            logger.info(f"Storage cleanup after indexing project {project_id}: {storage_cleanup_result['status']}, removed {storage_cleanup_result['total_files_removed']} files, freed {storage_cleanup_result['human_readable_bytes_freed']}")

            # Add cleanup info to index info
            index_info["database_cleanup_result"] = db_cleanup_result
            index_info["document_cleanup_result"] = doc_cleanup_result
            index_info["storage_cleanup_result"] = storage_cleanup_result

        # Force a direct cleanup of project session folders to ensure files are removed
        try:
            from .document_cleanup import cleanup_document_files
            logger.info(f"Performing direct cleanup of project session folders for project {project_id}")
            direct_cleanup_result = cleanup_document_files(document_ids)
            logger.info(f"Direct cleanup result: removed {direct_cleanup_result['total_files_removed']} files, freed {direct_cleanup_result['human_readable_bytes_freed']}")

            # Add direct cleanup info to index info
            index_info["direct_cleanup_result"] = direct_cleanup_result
        except Exception as e:
            logger.error(f"Error during direct cleanup of project session folders: {e}")
            index_info["direct_cleanup_error"] = str(e)

        # Optimize document metadata (keep excerpt by default)
        metadata_optimize_result = optimize_document_metadata(keep_excerpt=True, excerpt_length=500)
        logger.info(f"Metadata optimization after indexing project {project_id}: {metadata_optimize_result['status']}, processed {metadata_optimize_result.get('total_documents_processed', 0)} documents, saved {metadata_optimize_result.get('human_readable_bytes_saved', '0B')}")

        # Add additional metadata fields
        metadata_add_result = add_additional_metadata_fields()
        logger.info(f"Metadata enhancement after indexing project {project_id}: {metadata_add_result['status']}, processed {metadata_add_result.get('total_documents_processed', 0)} documents, added fields: {', '.join(metadata_add_result.get('fields_added', []))}")

        # Remove legacy folders (documents and database)
        legacy_cleanup_result = remove_legacy_folders()
        if legacy_cleanup_result["status"] == "success":
            logger.info(f"Legacy folder cleanup after indexing project {project_id}: removed folders: {', '.join(legacy_cleanup_result['folders_removed'])}")
        else:
            logger.info(f"Legacy folder cleanup after indexing project {project_id}: {legacy_cleanup_result['status']}, reason: {legacy_cleanup_result.get('reason', 'unknown')}")

        # Add metadata and legacy cleanup info to index info
        index_info["metadata_optimize_result"] = metadata_optimize_result
        index_info["metadata_add_result"] = metadata_add_result
        index_info["legacy_cleanup_result"] = legacy_cleanup_result
        save_project_index_info(project_id, index_info)

    return index_info

def get_latest_project_session_id(project_id: str) -> Optional[str]:
    """
    Get the latest session ID for a project.

    Args:
        project_id: Project ID

    Returns:
        Latest session ID or None if not found
    """
    index_info = load_project_index_info(project_id)
    if not index_info:
        return None

    return index_info.get("session_id")

def index_document(document_id: str) -> Dict[str, Any]:
    """
    Index a specific document.

    Args:
        document_id: Document ID

    Returns:
        Index information dictionary
    """
    document = get_document_by_id(document_id)
    if not document:
        logger.warning(f"Document {document_id} not found")
        return {
            "status": "error",
            "error": "document_not_found"
        }

    project_id = document.get("project_id")
    if not project_id:
        logger.warning(f"Document {document_id} has no project ID")
        return {
            "status": "error",
            "error": "no_project_id"
        }

    # Index the document within its project
    return index_project_documents(project_id, [document_id])

def get_project_vector_store(project_id: str) -> Optional[Any]:
    """
    Get the vector store for a project.

    Args:
        project_id: Project ID

    Returns:
        Vector store or None if not found
    """
    # Try to load directly from the project folder
    project_vector_store_path = get_project_vector_store_path(project_id)
    logger.info(f"Attempting to load vector store for project {project_id} from {project_vector_store_path}")

    vector_store = get_vector_store(use_faiss=True).load(project_vector_store_path)

    if vector_store:
        logger.info(f"Successfully loaded vector store for project {project_id}")
        return vector_store

    # If not found, try the legacy approach with session ID
    logger.info(f"Vector store not found at {project_vector_store_path}, trying legacy approach")
    session_id = get_latest_project_session_id(project_id)

    if not session_id:
        logger.warning(f"No session ID found for project {project_id}")
        return None

    # Try legacy project-specific location
    legacy_project_path = os.path.join(PROJECT_INDEX_DIR, f"{project_id}_{session_id}_vector_store")
    logger.info(f"Trying legacy project path: {legacy_project_path}")
    vector_store = get_vector_store(use_faiss=True).load(legacy_project_path)

    if vector_store:
        logger.info(f"Found vector store at legacy project path, copying to new location")
        # Save to the new location for future use
        vector_store.save(project_vector_store_path)
        return vector_store

    # If still not found, try session directory
    session_vector_store_path = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
    logger.info(f"Trying session path: {session_vector_store_path}")
    vector_store = get_vector_store(use_faiss=True).load(session_vector_store_path)

    if vector_store:
        logger.info(f"Found vector store at session path, copying to new location")
        # Save to the new location for future use
        vector_store.save(project_vector_store_path)
    else:
        logger.warning(f"No vector store found for project {project_id}")

    return vector_store

def remove_documents_from_project(project_id: str, document_ids: List[str]) -> Dict[str, Any]:
    """
    Remove specific documents from a project index.

    Args:
        project_id: Project ID
        document_ids: List of document IDs to remove

    Returns:
        Dictionary with removal results
    """
    if not document_ids:
        return {
            "status": "error",
            "error": "no_document_ids",
            "message": "No document IDs provided"
        }

    # Load project index info
    index_info = load_project_index_info(project_id)
    if not index_info:
        return {
            "status": "error",
            "error": "project_not_found",
            "message": f"Project {project_id} not found or not indexed"
        }

    # Get the vector store
    vector_store = get_project_vector_store(project_id)
    if not vector_store:
        return {
            "status": "error",
            "error": "vector_store_not_found",
            "message": f"Vector store for project {project_id} not found"
        }

    # Remove documents from vector store
    try:
        removed_count = vector_store.remove_chunks(document_ids=document_ids)
        logger.info(f"Removed {removed_count} chunks for documents {document_ids} from project {project_id}")

        # Update project index info
        current_doc_ids = index_info.get("document_ids", [])
        updated_doc_ids = [doc_id for doc_id in current_doc_ids if doc_id not in document_ids]

        index_info["document_ids"] = updated_doc_ids
        index_info["document_count"] = len(updated_doc_ids)
        index_info["last_updated"] = datetime.now().isoformat()

        # Save updated vector store directly to the project folder
        project_vector_store_path = get_project_vector_store_path(project_id)
        vector_store.save(project_vector_store_path)
        logger.info(f"Saved updated vector store for project {project_id} to {project_vector_store_path}")

        # Save updated index info
        save_project_index_info(project_id, index_info)

        # Update document registry
        try:
            from .document_registry import _load_registry, _save_registry

            registry = _load_registry()

            # Update document references in registry
            for doc_id in document_ids:
                if doc_id in registry["documents"]:
                    # Remove project from document's projects list
                    if project_id in registry["documents"][doc_id]["projects"]:
                        registry["documents"][doc_id]["projects"].remove(project_id)
                        # Decrement reference count
                        registry["documents"][doc_id]["ref_count"] -= 1
                        logger.info(f"Removed project {project_id} from document {doc_id} in registry")

            # Save registry changes
            _save_registry(registry)

            # Clean up unreferenced embeddings
            try:
                from .document_registry import cleanup_embeddings

                # Find documents with zero references
                zero_ref_docs = [doc_id for doc_id in document_ids
                               if doc_id in registry["documents"] and
                               registry["documents"][doc_id]["ref_count"] <= 0]

                if zero_ref_docs:
                    cleanup_result = cleanup_embeddings(zero_ref_docs)
                    logger.info(f"Cleaned up embeddings for {len(zero_ref_docs)} unreferenced documents: removed {cleanup_result['total_files_removed']} files, freed {cleanup_result['human_readable_bytes_freed']}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Embedding cleanup not available: {str(e)}")

        except ImportError:
            logger.warning("Document registry not available. Skipping registry updates.")

        return {
            "status": "success",
            "removed_chunks": removed_count,
            "removed_documents": len(document_ids),
            "remaining_documents": len(updated_doc_ids)
        }
    except Exception as e:
        logger.error(f"Error removing documents from project: {e}", exc_info=True)
        return {
            "status": "error",
            "error": "removal_failed",
            "message": str(e)
        }

def remove_project(project_id: str) -> Dict[str, Any]:
    """
    Remove an entire project and all its associated data.

    Args:
        project_id: Project ID

    Returns:
        Dictionary with removal results
    """
    # Load project index info
    index_info = load_project_index_info(project_id)
    if not index_info:
        return {
            "status": "error",
            "error": "project_not_found",
            "message": f"Project {project_id} not found or not indexed"
        }

    # Get document IDs from the project
    document_ids = index_info.get("document_ids", [])

    # Get the vector store
    vector_store = get_project_vector_store(project_id)

    results = {
        "status": "success",
        "project_id": project_id,
        "document_count": len(document_ids),
        "removed_chunks": 0,
        "removed_files": 0,
        "errors": []
    }

    # Remove all documents from vector store
    if vector_store:
        try:
            # Use project_id to remove all chunks for this project
            removed_count = vector_store.remove_chunks(project_id=project_id)
            results["removed_chunks"] = removed_count
            logger.info(f"Removed {removed_count} chunks from project {project_id}")
        except Exception as e:
            error_msg = f"Error removing chunks from vector store: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

    # Update document registry
    try:
        from .document_registry import _load_registry, _save_registry

        registry = _load_registry()
        zero_ref_docs = []

        # Update document references in registry
        for doc_id in document_ids:
            if doc_id in registry["documents"]:
                # Remove project from document's projects list
                if project_id in registry["documents"][doc_id]["projects"]:
                    registry["documents"][doc_id]["projects"].remove(project_id)
                    # Decrement reference count
                    registry["documents"][doc_id]["ref_count"] -= 1
                    logger.info(f"Removed project {project_id} from document {doc_id} in registry")

                    # Check if document has no more references
                    if registry["documents"][doc_id]["ref_count"] <= 0:
                        zero_ref_docs.append(doc_id)

        # Remove project from registry if it exists
        if "projects" in registry and project_id in registry["projects"]:
            del registry["projects"][project_id]
            logger.info(f"Removed project {project_id} from registry")

        # Save registry changes
        _save_registry(registry)

        # Clean up unreferenced embeddings
        if zero_ref_docs:
            try:
                from .document_registry import cleanup_embeddings

                cleanup_result = cleanup_embeddings(zero_ref_docs)
                logger.info(f"Cleaned up embeddings for {len(zero_ref_docs)} unreferenced documents: removed {cleanup_result['total_files_removed']} files, freed {cleanup_result['human_readable_bytes_freed']}")
                results["removed_embeddings"] = cleanup_result["total_files_removed"]
            except (ImportError, AttributeError) as e:
                logger.warning(f"Embedding cleanup not available: {str(e)}")
                results["errors"].append(f"Embedding cleanup not available: {str(e)}")
    except ImportError:
        logger.warning("Document registry not available. Skipping registry updates.")
        results["errors"].append("Document registry not available. Skipping registry updates.")

    # Remove project index files
    try:
        # Remove index info file
        index_path = get_project_index_path(project_id)
        if os.path.exists(index_path):
            os.remove(index_path)
            results["removed_files"] += 1
            logger.info(f"Removed project index file: {index_path}")

        # Remove vector store files
        session_id = index_info.get("session_id")
        if session_id:
            # Remove from project-specific location
            project_vector_store_path = get_project_vector_store_path(project_id, session_id)
            if os.path.exists(f"{project_vector_store_path}.index"):
                os.remove(f"{project_vector_store_path}.index")
                results["removed_files"] += 1
            if os.path.exists(f"{project_vector_store_path}.data"):
                os.remove(f"{project_vector_store_path}.data")
                results["removed_files"] += 1

            # Remove from session directory
            session_vector_store_path = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
            if os.path.exists(f"{session_vector_store_path}.index"):
                os.remove(f"{session_vector_store_path}.index")
                results["removed_files"] += 1
            if os.path.exists(f"{session_vector_store_path}.data"):
                os.remove(f"{session_vector_store_path}.data")
                results["removed_files"] += 1

            logger.info(f"Removed vector store files for project {project_id}")
    except Exception as e:
        error_msg = f"Error removing project files: {str(e)}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    # Update status if there were errors
    if results["errors"]:
        results["status"] = "partial"

    return results
