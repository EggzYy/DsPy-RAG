"""
Document versioning module for Local File Deep Research.
"""

import os
import json
import time
import hashlib
import logging
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Constants
VERSIONS_DIR = os.environ.get("VERSIONS_DIR", "versions")
VERSION_INDEX_FILE = os.path.join(VERSIONS_DIR, "version_index.json")

class VersioningError(Exception):
    """Base exception for versioning errors."""
    pass

def _ensure_versions_dir():
    """Ensure the versions directory exists."""
    os.makedirs(VERSIONS_DIR, exist_ok=True)
    
    # Create version index file if it doesn't exist
    if not os.path.exists(VERSION_INDEX_FILE):
        with open(VERSION_INDEX_FILE, 'w') as f:
            json.dump({}, f)

def _load_version_index() -> Dict[str, Dict[str, Any]]:
    """Load the version index."""
    _ensure_versions_dir()
    try:
        with open(VERSION_INDEX_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def _save_version_index(index: Dict[str, Dict[str, Any]]):
    """Save the version index."""
    _ensure_versions_dir()
    with open(VERSION_INDEX_FILE, 'w') as f:
        json.dump(index, f, indent=2)

def _compute_content_hash(content: str) -> str:
    """Compute a hash of the content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def _get_document_path(document_id: str, version_id: str) -> str:
    """Get the path to a document version."""
    return os.path.join(VERSIONS_DIR, f"{document_id}_{version_id}.json")

def create_document(content: str, metadata: Dict[str, Any] = None) -> str:
    """
    Create a new document with version tracking.
    
    Args:
        content: Document content
        metadata: Optional metadata
        
    Returns:
        Document ID
    """
    _ensure_versions_dir()
    
    # Generate document ID
    document_id = f"doc_{int(time.time())}_{os.urandom(4).hex()}"
    
    # Create first version
    version_id = create_version(document_id, content, metadata)
    
    return document_id

def create_version(document_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
    """
    Create a new version of a document.
    
    Args:
        document_id: Document ID
        content: Document content
        metadata: Optional metadata
        
    Returns:
        Version ID
    """
    _ensure_versions_dir()
    
    # Load version index
    index = _load_version_index()
    
    # Check if document exists
    if document_id not in index:
        index[document_id] = {
            "versions": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    # Compute content hash
    content_hash = _compute_content_hash(content)
    
    # Check if content is the same as the latest version
    if index[document_id]["versions"]:
        latest_version = index[document_id]["versions"][-1]
        if latest_version["content_hash"] == content_hash:
            return latest_version["version_id"]
    
    # Generate version ID
    version_id = f"v_{int(time.time())}_{os.urandom(4).hex()}"
    
    # Create version data
    version_data = {
        "version_id": version_id,
        "content_hash": content_hash,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    # Save version content
    document_path = _get_document_path(document_id, version_id)
    with open(document_path, 'w') as f:
        json.dump({
            "document_id": document_id,
            "version_id": version_id,
            "content": content,
            "metadata": metadata or {}
        }, f, indent=2)
    
    # Update version index
    index[document_id]["versions"].append(version_data)
    index[document_id]["updated_at"] = datetime.now().isoformat()
    _save_version_index(index)
    
    return version_id

def get_document(document_id: str, version_id: str = None) -> Dict[str, Any]:
    """
    Get a document version.
    
    Args:
        document_id: Document ID
        version_id: Optional version ID (defaults to latest)
        
    Returns:
        Document data
        
    Raises:
        VersioningError: If the document or version is not found
    """
    # Load version index
    index = _load_version_index()
    
    # Check if document exists
    if document_id not in index:
        raise VersioningError(f"Document '{document_id}' not found")
    
    # Get version ID
    if version_id is None:
        if not index[document_id]["versions"]:
            raise VersioningError(f"Document '{document_id}' has no versions")
        version_id = index[document_id]["versions"][-1]["version_id"]
    
    # Check if version exists
    version_exists = any(v["version_id"] == version_id for v in index[document_id]["versions"])
    if not version_exists:
        raise VersioningError(f"Version '{version_id}' not found for document '{document_id}'")
    
    # Load document content
    document_path = _get_document_path(document_id, version_id)
    try:
        with open(document_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        raise VersioningError(f"Failed to load document '{document_id}' version '{version_id}'")

def get_document_versions(document_id: str) -> List[Dict[str, Any]]:
    """
    Get all versions of a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        List of version data
        
    Raises:
        VersioningError: If the document is not found
    """
    # Load version index
    index = _load_version_index()
    
    # Check if document exists
    if document_id not in index:
        raise VersioningError(f"Document '{document_id}' not found")
    
    return index[document_id]["versions"]

def get_document_diff(document_id: str, version_id1: str, version_id2: str) -> List[str]:
    """
    Get the diff between two versions of a document.
    
    Args:
        document_id: Document ID
        version_id1: First version ID
        version_id2: Second version ID
        
    Returns:
        List of diff lines
        
    Raises:
        VersioningError: If the document or versions are not found
    """
    # Get document versions
    doc1 = get_document(document_id, version_id1)
    doc2 = get_document(document_id, version_id2)
    
    # Get content
    content1 = doc1["content"]
    content2 = doc2["content"]
    
    # Compute diff
    diff = difflib.unified_diff(
        content1.splitlines(),
        content2.splitlines(),
        fromfile=f"Version {version_id1}",
        tofile=f"Version {version_id2}",
        lineterm=""
    )
    
    return list(diff)

def delete_document(document_id: str) -> bool:
    """
    Delete a document and all its versions.
    
    Args:
        document_id: Document ID
        
    Returns:
        True if the document was deleted, False otherwise
    """
    # Load version index
    index = _load_version_index()
    
    # Check if document exists
    if document_id not in index:
        return False
    
    # Delete all version files
    for version in index[document_id]["versions"]:
        version_id = version["version_id"]
        document_path = _get_document_path(document_id, version_id)
        try:
            os.remove(document_path)
        except FileNotFoundError:
            pass
    
    # Remove document from index
    del index[document_id]
    _save_version_index(index)
    
    return True

def list_documents() -> List[Dict[str, Any]]:
    """
    List all documents.
    
    Returns:
        List of document data
    """
    # Load version index
    index = _load_version_index()
    
    # Format document data
    documents = []
    for document_id, document_data in index.items():
        documents.append({
            "document_id": document_id,
            "created_at": document_data["created_at"],
            "updated_at": document_data["updated_at"],
            "version_count": len(document_data["versions"]),
            "latest_version": document_data["versions"][-1]["version_id"] if document_data["versions"] else None
        })
    
    return documents

def search_documents(query: str) -> List[Dict[str, Any]]:
    """
    Search for documents containing the query.
    
    Args:
        query: Search query
        
    Returns:
        List of matching document data
    """
    # Load version index
    index = _load_version_index()
    
    # Search for documents
    results = []
    for document_id, document_data in index.items():
        # Get latest version
        if not document_data["versions"]:
            continue
        latest_version = document_data["versions"][-1]
        
        try:
            # Load document content
            document = get_document(document_id, latest_version["version_id"])
            
            # Check if query is in content
            if query.lower() in document["content"].lower():
                results.append({
                    "document_id": document_id,
                    "version_id": latest_version["version_id"],
                    "created_at": document_data["created_at"],
                    "updated_at": document_data["updated_at"],
                    "metadata": document["metadata"],
                    "snippet": _get_snippet(document["content"], query)
                })
        except VersioningError:
            continue
    
    return results

def _get_snippet(content: str, query: str, context_size: int = 100) -> str:
    """
    Get a snippet of content around the query.
    
    Args:
        content: Document content
        query: Search query
        context_size: Number of characters to include before and after the query
        
    Returns:
        Snippet of content
    """
    query_lower = query.lower()
    content_lower = content.lower()
    
    # Find the position of the query
    pos = content_lower.find(query_lower)
    if pos == -1:
        return content[:200] + "..."
    
    # Get the snippet
    start = max(0, pos - context_size)
    end = min(len(content), pos + len(query) + context_size)
    
    # Add ellipsis if needed
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    
    return prefix + content[start:end] + suffix
