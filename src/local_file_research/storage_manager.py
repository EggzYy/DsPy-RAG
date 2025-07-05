"""
Storage management for Local File Deep Research.
This module provides a consolidated storage system for documents.
"""

import os
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
BASE_DIR = os.environ.get("BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STORAGE_DIR = os.environ.get("STORAGE_DIR", os.path.join(BASE_DIR, "storage"))
# Legacy metadata file - now using document_registry.json instead
LEGACY_METADATA_FILE = os.path.join(STORAGE_DIR, "documents.json")

# --- Helper Functions ---
def ensure_storage_dir():
    """Ensure the storage directory exists."""
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # Create registry directory
    registry_dir = os.path.join(STORAGE_DIR, "registry")
    os.makedirs(registry_dir, exist_ok=True)

def get_document_path(document_id: str) -> str:
    """
    Get the path to a document file.

    Args:
        document_id: Document ID

    Returns:
        Path to the document file
    """
    return os.path.join(STORAGE_DIR, f"doc_{document_id}")

def get_version_path(document_id: str, version_id: str) -> str:
    """
    Get the path to a document version file.

    Args:
        document_id: Document ID
        version_id: Version ID

    Returns:
        Path to the version file
    """
    version_dir = os.path.join(STORAGE_DIR, "versions")
    os.makedirs(version_dir, exist_ok=True)
    return os.path.join(version_dir, f"ver_{document_id}_{version_id}")

def load_metadata() -> List[Dict[str, Any]]:
    """
    Load document metadata from file.

    Returns:
        List of document metadata dictionaries
    """
    ensure_storage_dir()

    # Try to load from document registry first
    try:
        from .document_registry import _load_registry
        registry = _load_registry()

        # Convert registry format to legacy format
        documents = []
        for doc_id, doc_data in registry["documents"].items():
            # Get the document content from storage if available
            content = None
            try:
                content_bytes = read_document_file(doc_id)
                if content_bytes:
                    content = content_bytes.decode('utf-8', errors='replace')
            except Exception as e:
                logger.warning(f"Error reading content for document {doc_id}: {e}")

            # Get project ID from registry
            project_id = None
            if doc_data.get("projects") and len(doc_data["projects"]) > 0:
                project_id = doc_data["projects"][0]

            # Create document object with all necessary fields
            document = {
                "document_id": doc_id,
                "title": doc_data.get("metadata", {}).get("title", "Untitled"),
                "document_type": doc_data.get("metadata", {}).get("document_type", "unknown"),
                "owner": doc_data.get("metadata", {}).get("owner", "unknown"),
                "project_id": project_id,
                "created_at": doc_data.get("created_at", datetime.now().isoformat()),
                "updated_at": doc_data.get("updated_at", datetime.now().isoformat()),
                "version_count": 1,
                "metadata": doc_data.get("metadata", {}),
                "file_path": get_document_path(doc_id),
                "content": content or "Content not loaded"
            }

            documents.append(document)

        return documents
    except ImportError:
        # Fall back to legacy metadata file
        try:
            if os.path.exists(LEGACY_METADATA_FILE):
                with open(LEGACY_METADATA_FILE, "r") as f:
                    return json.load(f)
        except json.JSONDecodeError:
            pass

        return []

def save_metadata(metadata: List[Dict[str, Any]]):
    """
    Save document metadata to file.

    Args:
        metadata: List of document metadata dictionaries
    """
    ensure_storage_dir()

    # Try to save to document registry first
    try:
        from .document_registry import _load_registry, _save_registry
        registry = _load_registry()

        # Update registry with metadata
        for doc in metadata:
            doc_id = doc.get("document_id")
            if doc_id and doc_id in registry["documents"]:
                registry["documents"][doc_id]["metadata"] = doc.get("metadata", {})
                registry["documents"][doc_id]["updated_at"] = doc.get("updated_at", datetime.now().isoformat())

        # Save registry
        _save_registry(registry)
    except ImportError:
        # Fall back to legacy metadata file
        with open(LEGACY_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

def get_document_metadata(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific document.

    Args:
        document_id: Document ID

    Returns:
        Document metadata dictionary or None if not found
    """
    metadata = load_metadata()
    for doc in metadata:
        if doc["document_id"] == document_id:
            return doc
    return None

def save_document_file(file_data: bytes, document_id: str, skip_file_creation: bool = False) -> str:
    """
    Save a document file to storage.

    Args:
        file_data: File content as bytes
        document_id: Document ID
        skip_file_creation: If True, don't actually create the file, just return the path.
                           This is useful when we're going to index the content directly.

    Returns:
        Path to the saved file (even if the file wasn't actually created)
    """
    ensure_storage_dir()
    file_path = get_document_path(document_id)

    if skip_file_creation:
        # Just return the path without creating the file
        logger.info(f"Skipping file creation for document {document_id} ({len(file_data)} bytes)")
        return file_path

    # Actually create the file
    with open(file_path, "wb") as f:
        f.write(file_data)

    file_size = len(file_data)
    logger.info(f"Saved document {document_id} ({file_size} bytes) to {file_path}")

    return file_path

def read_document_file(document_id: str) -> Optional[bytes]:
    """
    Read a document file from storage.

    Args:
        document_id: Document ID

    Returns:
        File content as bytes or None if not found
    """
    file_path = get_document_path(document_id)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return f.read()
    return None

def delete_document_file(document_id: str) -> bool:
    """
    Delete a document file from storage.

    Args:
        document_id: Document ID

    Returns:
        True if the file was deleted, False otherwise
    """
    file_path = get_document_path(document_id)
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted document file: {file_path}")
        return True
    return False

def migrate_from_old_structure():
    """
    Migrate documents from the old structure (documents and database folders) to the new storage system.
    """
    # Define old paths
    old_documents_dir = os.environ.get("DOCUMENTS_DIR", os.path.join(BASE_DIR, "documents"))
    old_database_dir = os.environ.get("DATABASE_DIR", os.path.join(BASE_DIR, "database"))
    old_metadata_file = os.path.join(old_documents_dir, "documents.json")

    # Check if old structure exists
    if not os.path.exists(old_metadata_file):
        logger.info("No old metadata file found, skipping migration")
        return

    # Load old metadata
    try:
        with open(old_metadata_file, "r") as f:
            old_metadata = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Could not parse old metadata file, skipping migration")
        return

    # Create new storage directory
    ensure_storage_dir()

    # Migrate each document
    migrated_count = 0
    for doc in old_metadata:
        document_id = doc.get("document_id")
        if not document_id:
            continue

        # Check if document file exists in old structure
        old_file_path = os.path.join(old_documents_dir, document_id)
        if os.path.exists(old_file_path):
            # Copy file to new location
            new_file_path = get_document_path(document_id)
            try:
                shutil.copy2(old_file_path, new_file_path)
                logger.info(f"Migrated document {document_id} from {old_file_path} to {new_file_path}")
                migrated_count += 1
            except Exception as e:
                logger.error(f"Error migrating document {document_id}: {e}")

    # Save metadata to new location
    save_metadata(old_metadata)

    logger.info(f"Migration complete: {migrated_count} documents migrated to new storage system")

def save_document_content(document_id: str, content: str, skip_file_creation: bool = False) -> str:
    """
    Save document content to a separate content file.
    This provides redundancy for document content.

    Args:
        document_id: Document ID
        content: Document content as string
        skip_file_creation: If True, don't actually create the file, just return the path.
                           This is useful when we're going to index the content directly.

    Returns:
        Path to the saved content file (even if the file wasn't actually created)
    """
    ensure_storage_dir()

    # Create content directory if it doesn't exist
    content_dir = os.path.join(STORAGE_DIR, "content")
    os.makedirs(content_dir, exist_ok=True)

    # Get the content path
    content_path = os.path.join(content_dir, f"content_{document_id}.txt")

    if skip_file_creation:
        # Just return the path without creating the file
        logger.info(f"Skipping content file creation for document {document_id} ({len(content)} chars)")
        return content_path

    # Actually create the file
    try:
        # First try with UTF-8 encoding
        try:
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(content)
        except UnicodeEncodeError:
            # If UTF-8 fails, try with UTF-8 and replace errors
            with open(content_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(content)
                logger.warning(f"Unicode encode errors in document {document_id}, using replacement characters")

        # Verify the file was created and is readable
        if os.path.exists(content_path) and os.path.getsize(content_path) > 0:
            logger.info(f"Saved document content for {document_id} to {content_path} ({os.path.getsize(content_path)} bytes)")

            # Try to read the file back to verify it's readable
            try:
                with open(content_path, "r", encoding="utf-8", errors="replace") as f:
                    test_read = f.read(100)  # Just read a small sample
                    logger.debug(f"Successfully verified content file is readable: {test_read[:20]}...")
            except Exception as read_error:
                logger.warning(f"File was created but could not be read back: {read_error}")
        else:
            logger.warning(f"File was not created or is empty: {content_path}")

        return content_path
    except Exception as e:
        logger.error(f"Error saving document content: {e}")

        # Try one more time with a different approach
        try:
            import codecs
            with codecs.open(content_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(content)
            logger.info(f"Successfully saved document content using codecs for {document_id}")
            return content_path
        except Exception as e2:
            logger.error(f"Second attempt to save document content failed: {e2}")
            raise e  # Raise the original error

def read_document_content(document_id: str) -> Optional[str]:
    """
    Read document content from the content file.

    Args:
        document_id: Document ID

    Returns:
        Document content as string or None if not found
    """
    ensure_storage_dir()

    # Get content file path
    content_path = os.path.join(STORAGE_DIR, "content", f"content_{document_id}.txt")

    if os.path.exists(content_path):
        try:
            # Try with UTF-8 first
            try:
                with open(content_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # If UTF-8 fails, try with UTF-8 and ignore errors
                with open(content_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    logger.warning(f"Unicode decode errors in document {document_id}, using replacement characters")

            logger.info(f"Read document content for {document_id} from {content_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading document content: {e}")

            # Try one more time with a different approach
            try:
                import codecs
                with codecs.open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                logger.info(f"Successfully read document content using codecs for {document_id}")
                return content
            except Exception as e2:
                logger.error(f"Second attempt to read document content failed: {e2}")

    return None

# Initialize
ensure_storage_dir()
