"""
Metadata optimizer for Local File Deep Research.
This module provides functions to optimize document metadata by removing unnecessary fields.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

from .storage_manager import STORAGE_DIR, LEGACY_METADATA_FILE

# Configure logging
logger = logging.getLogger(__name__)

def optimize_document_metadata(keep_excerpt: bool = True, excerpt_length: int = 500) -> Dict[str, Any]:
    """
    Optimize document metadata by removing unnecessary fields.

    Args:
        keep_excerpt: Whether to keep a small excerpt of the content
        excerpt_length: Length of the excerpt to keep (in characters)

    Returns:
        Dictionary with optimization statistics
    """
    print(f"Starting metadata optimization with keep_excerpt={keep_excerpt}, excerpt_length={excerpt_length}")

    stats = {
        "status": "success",
        "total_documents_processed": 0,
        "total_bytes_saved": 0,
        "fields_removed": [],
        "errors": []
    }

    if not os.path.exists(LEGACY_METADATA_FILE):
        print(f"Metadata file {LEGACY_METADATA_FILE} does not exist. Nothing to optimize.")
        logger.warning(f"Metadata file {LEGACY_METADATA_FILE} does not exist. Nothing to optimize.")
        return {"status": "skipped", "reason": "metadata_file_not_found"}

    try:
        print(f"Loading metadata from {LEGACY_METADATA_FILE}")
        # Load metadata
        with open(LEGACY_METADATA_FILE, "r") as f:
            documents = json.load(f)

        print(f"Loaded {len(documents)} documents from metadata file")
        original_size = os.path.getsize(LEGACY_METADATA_FILE)
        print(f"Original file size: {_format_bytes(original_size)}")

        # Fields to remove (no longer necessary)
        fields_to_remove = [
            "file_path",
            "db_file_path"
        ]

        # Process each document
        for i, doc in enumerate(documents):
            doc_id = doc.get("document_id", f"doc_{i}")
            print(f"Processing document {doc_id} ({i+1}/{len(documents)})")

            # Remove unnecessary fields
            for field in fields_to_remove:
                if field in doc:
                    print(f"  Removing field: {field}")
                    del doc[field]

            # Handle content field
            if "content" in doc:
                content_length = len(doc["content"])
                print(f"  Found content field with {content_length} characters")

                if keep_excerpt:
                    # Keep only a small excerpt of the content
                    content = doc["content"]
                    doc["content_excerpt"] = content[:excerpt_length] + ("..." if len(content) > excerpt_length else "")
                    print(f"  Created content_excerpt with {len(doc['content_excerpt'])} characters")
                    del doc["content"]
                    print(f"  Removed original content field")
                else:
                    # Remove content field entirely
                    print(f"  Removing content field entirely")
                    del doc["content"]

            stats["total_documents_processed"] += 1

        print(f"Saving optimized metadata to {LEGACY_METADATA_FILE}")
        # Save optimized metadata
        with open(LEGACY_METADATA_FILE, "w") as f:
            json.dump(documents, f, indent=2)

        # Calculate bytes saved
        new_size = os.path.getsize(LEGACY_METADATA_FILE)
        stats["total_bytes_saved"] = original_size - new_size
        stats["fields_removed"] = fields_to_remove + ["content"]
        stats["human_readable_bytes_saved"] = _format_bytes(stats["total_bytes_saved"])

        print(f"Metadata optimization complete: {stats['total_documents_processed']} documents processed, {stats['human_readable_bytes_saved']} saved")
        logger.info(f"Metadata optimization complete: {stats['total_documents_processed']} documents processed, {stats['human_readable_bytes_saved']} saved")

        return stats
    except Exception as e:
        error_msg = f"Error during metadata optimization: {str(e)}"
        print(f"ERROR: {error_msg}")
        logger.error(error_msg)
        stats["status"] = "error"
        stats["errors"].append(error_msg)
        return stats

def add_additional_metadata_fields() -> Dict[str, Any]:
    """
    Add additional useful metadata fields to documents.

    Returns:
        Dictionary with update statistics
    """
    stats = {
        "status": "success",
        "total_documents_processed": 0,
        "fields_added": [],
        "errors": []
    }

    if not os.path.exists(LEGACY_METADATA_FILE):
        logger.warning(f"Metadata file {LEGACY_METADATA_FILE} does not exist. Nothing to update.")
        return {"status": "skipped", "reason": "metadata_file_not_found"}

    try:
        # Load metadata
        with open(LEGACY_METADATA_FILE, "r") as f:
            documents = json.load(f)

        # Fields to add
        fields_to_add = [
            "indexed_at",
            "word_count",
            "language"
        ]

        stats["fields_added"] = fields_to_add

        # Process each document
        for doc in documents:
            # Add indexed_at field if not present
            if "indexed_at" not in doc:
                doc["indexed_at"] = doc.get("updated_at", doc.get("created_at"))

            # Add word_count field if not present
            if "word_count" not in doc and "content" in doc:
                doc["word_count"] = len(doc["content"].split())

            # Add language field if not present (default to English)
            if "language" not in doc:
                doc["language"] = "en"

            stats["total_documents_processed"] += 1

        # Save updated metadata
        with open(LEGACY_METADATA_FILE, "w") as f:
            json.dump(documents, f, indent=2)

        logger.info(f"Metadata update complete: {stats['total_documents_processed']} documents processed, fields added: {', '.join(stats['fields_added'])}")

        return stats
    except Exception as e:
        error_msg = f"Error during metadata update: {str(e)}"
        logger.error(error_msg)
        stats["status"] = "error"
        stats["errors"].append(error_msg)
        return stats

def get_document_content(document_id: str, project_id: str) -> Optional[str]:
    """
    Retrieve the full content of a document from the vector store.

    Args:
        document_id: Document ID
        project_id: Project ID

    Returns:
        Document content or None if not found
    """
    logger.info(f"Attempting to retrieve content for document {document_id} in project {project_id}")

    try:
        # First try to get content from the document registry
        try:
            from .document_registry import get_document_registry
            doc_registry = get_document_registry(document_id)
            if doc_registry and "metadata" in doc_registry and "content" in doc_registry["metadata"]:
                content = doc_registry["metadata"]["content"]
                logger.info(f"Retrieved content from registry for document {document_id}, length: {len(content)}")
                return content
        except (ImportError, Exception) as e:
            logger.warning(f"Could not retrieve content from registry: {str(e)}")

        # If not found in registry, try to get from vector store
        # Get the vector store for the project
        logger.info(f"Trying to get content from vector store for document {document_id} in project {project_id}")

        # Try with the provided project_id first
        vector_store = None
        if project_id:
            try:
                from .project_indexer import get_project_vector_store
                vector_store = get_project_vector_store(project_id)
                logger.info(f"Trying to get vector store for project {project_id}")
            except Exception as e:
                logger.warning(f"Error getting vector store for project {project_id}: {str(e)}")

        # If not found, try to get from the document's projects
        if vector_store is None:
            try:
                from .document_registry import get_document_registry
                doc_registry = get_document_registry(document_id)
                if doc_registry and "projects" in doc_registry:
                    for proj_id in doc_registry["projects"]:
                        logger.info(f"Trying project {proj_id} for document {document_id}")
                        try:
                            from .project_indexer import get_project_vector_store
                            vector_store = get_project_vector_store(proj_id)
                            if vector_store is not None:
                                logger.info(f"Found vector store for project {proj_id}")
                                project_id = proj_id  # Update project_id for later use
                                break
                        except Exception as e:
                            logger.warning(f"Error getting vector store for project {proj_id}: {str(e)}")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not retrieve projects from registry: {str(e)}")

        # Check if the vector store exists
        if vector_store is None:
            logger.warning(f"Vector store not found for document {document_id}")
            return None

        # Get all chunks for the document
        chunks = vector_store.get_chunks_by_document_id(document_id)

        if not chunks:
            logger.warning(f"No chunks found for document {document_id} in project {project_id}")
            return None

        # Log the number of chunks found
        logger.info(f"Found {len(chunks)} chunks for document {document_id} in project {project_id}")

        # Log the first chunk to debug
        if chunks:
            logger.info(f"First chunk keys: {list(chunks[0].keys())}")
            if "content" in chunks[0]:
                logger.info(f"First chunk content length: {len(chunks[0]['content'])}")
            else:
                logger.warning(f"First chunk has no content field")

        # Check if any chunks are missing content
        missing_content = [i for i, chunk in enumerate(chunks) if not chunk.get("content")]
        if missing_content:
            logger.warning(f"Chunks {missing_content} are missing content for document {document_id}")

        # Sort chunks by chunk_index to reconstruct the document
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        # Combine all chunk contents
        content = "".join([chunk.get("content", "") for chunk in chunks])

        # Log the content length
        logger.info(f"Retrieved content with length {len(content)} for document {document_id}")

        if not content:
            logger.warning(f"Empty content retrieved for document {document_id}")

            # Try to get content from the document file if it still exists
            try:
                from .storage_manager import read_document_file
                file_content = read_document_file(document_id)
                if file_content:
                    content = file_content.decode('utf-8', errors='replace')
                    logger.info(f"Retrieved content from document file for {document_id}, length: {len(content)}")
            except Exception as e:
                logger.warning(f"Could not retrieve content from document file: {str(e)}")

        return content
    except Exception as e:
        logger.error(f"Error retrieving content for document {document_id}: {str(e)}")
        return None

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
