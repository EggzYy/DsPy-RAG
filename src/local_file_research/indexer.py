"""
Indexer for Local File Deep Research.
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .document_manager import get_documents, get_document_by_id
from .vector_store import get_vector_store, chunk_file_content
from .embedding import get_embeddings
from .database_cleanup import cleanup_database_files

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
from .config import SESSION_PERSIST_DIR
INDEX_DIR = os.environ.get("INDEX_DIR", SESSION_PERSIST_DIR)
VECTOR_STORE_PATH = os.path.join(INDEX_DIR, "vector_store")
INDEX_INFO_PATH = os.path.join(INDEX_DIR, "index_info.json")

# --- Helper Functions ---
def _ensure_index_dir():
    """Ensure the index directory exists."""
    os.makedirs(INDEX_DIR, exist_ok=True)

def _save_index_info(index_info: Dict[str, Any]):
    """Save index info to file."""
    _ensure_index_dir()
    with open(INDEX_INFO_PATH, "w") as f:
        json.dump(index_info, f, indent=2)

def _load_index_info() -> Dict[str, Any]:
    """Load index info from file."""
    _ensure_index_dir()
    if not os.path.exists(INDEX_INFO_PATH):
        return {
            "last_indexed": None,
            "document_count": 0,
            "chunk_count": 0,
            "session_id": None
        }

    try:
        with open(INDEX_INFO_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {
            "last_indexed": None,
            "document_count": 0,
            "chunk_count": 0,
            "session_id": None
        }

# --- Indexing Functions ---
def index_documents(username: str, project_id: Optional[str] = None, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Index documents for search.

    Args:
        username: Username of the indexer
        project_id: Optional project ID to filter documents
        document_ids: Optional list of document IDs to index

    Returns:
        Index info dictionary
    """
    _ensure_index_dir()

    # Create a new session ID
    session_id = str(uuid.uuid4())

    # Get documents to index
    if document_ids:
        documents = [get_document_by_id(doc_id) for doc_id in document_ids]
        documents = [doc for doc in documents if doc is not None]
    else:
        documents = get_documents(username=username, project_id=project_id)

    if not documents:
        logger.warning(f"No documents found to index for user {username}")
        return {
            "session_id": session_id,
            "document_count": 0,
            "chunk_count": 0,
            "last_indexed": datetime.now().isoformat()
        }

    # Create vector store
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

        # Save vector store
        vector_store_path = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
        vector_store.save(vector_store_path)

    # Save index info
    index_info = {
        "session_id": session_id,
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "last_indexed": datetime.now().isoformat()
    }
    _save_index_info(index_info)

    # Clean up database files after indexing
    if documents:
        document_ids = [doc["document_id"] for doc in documents]
        cleanup_result = cleanup_database_files(document_ids)
        logger.info(f"Database cleanup after indexing: {cleanup_result['status']}, removed {cleanup_result['total_files_removed']} files, freed {cleanup_result['human_readable_bytes_freed']}")

        # Add cleanup info to index info
        index_info["cleanup_result"] = cleanup_result

    return index_info

def search_index(query: str, top_k: int = 5, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search the index for relevant chunks.

    Args:
        query: Search query
        top_k: Number of results to return
        session_id: Optional session ID to use a specific index

    Returns:
        List of relevant chunks
    """
    # Get index info
    index_info = _load_index_info()

    # Use provided session ID or the latest one
    if not session_id:
        session_id = index_info.get("session_id")

    if not session_id:
        logger.warning("No index found. Please index files first or provide a valid session ID.")
        raise ValueError("No index found. Please index files first or provide a valid session ID.")

    # Load vector store
    vector_store_path = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
    vector_store = get_vector_store(use_faiss=True).load(vector_store_path)

    if not vector_store:
        logger.warning(f"Could not load vector store for session {session_id}")
        raise ValueError(f"Could not load vector store for session {session_id}")

    # Get query embedding
    query_embedding = get_embeddings([query])[0]

    # Search
    results = vector_store.search(query_embedding, top_k=top_k)

    return results

def get_index_info() -> Dict[str, Any]:
    """
    Get information about the current index.

    Returns:
        Index info dictionary
    """
    return _load_index_info()

# Initialize
_ensure_index_dir()
