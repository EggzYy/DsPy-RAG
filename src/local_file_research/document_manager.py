"""
Document management for Local File Deep Research.
"""

import os
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the file processor module
from src.local_file_research.file_processor import process_file
from src.local_file_research.storage_manager import (
    save_document_file, read_document_file, delete_document_file,
    load_metadata as load_storage_metadata, save_metadata as save_storage_metadata
)

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
BASE_DIR = os.environ.get("BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VERSIONS_DIR = os.environ.get("VERSIONS_DIR", os.path.join(BASE_DIR, "versions"))

# --- Helper Functions ---
def _ensure_dirs():
    """Ensure all required directories exist."""
    # Use the storage manager to ensure storage directory exists
    from src.local_file_research.storage_manager import ensure_storage_dir
    ensure_storage_dir()

    # Ensure versions directory exists
    os.makedirs(VERSIONS_DIR, exist_ok=True)

def _get_database_path(project_id, username, filename):
    """Create a properly structured path for a file in the database directory.

    Args:
        project_id: The project ID
        username: The username of the uploader
        filename: The original filename

    Returns:
        Tuple of (directory_path, full_file_path)
    """
    # This function is deprecated and will be removed in a future version
    # It's kept for backward compatibility with code that might still use it
    logger.warning("_get_database_path is deprecated, use storage_manager functions instead")

    # Create a session ID based on timestamp
    session_id = f"session_{int(time.time())}"

    # Create a path in the storage directory instead
    dir_path = os.path.join("storage", project_id, session_id, username)
    os.makedirs(dir_path, exist_ok=True)

    # Create full file path
    file_path = os.path.join(dir_path, filename)

    return dir_path, file_path

def save_file_to_database(file_data, project_id, username, filename):
    """Save a file to the database directory with proper structure.

    Args:
        file_data: The binary content of the file
        project_id: The project ID
        username: The username of the uploader
        filename: The original filename

    Returns:
        Tuple of (saved_path, file_size)
    """
    _ensure_dirs()

    # Use the storage manager to save the file
    if project_id:
        # If we have a project ID, save to the project directory
        from .document_registry import get_project_directory
        project_dir = get_project_directory(project_id)
        file_path = os.path.join(project_dir, filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the file
        with open(file_path, "wb") as f:
            f.write(file_data)

        logger.info(f"Saved file {filename} ({len(file_data)} bytes) to project directory: {file_path}")
    else:
        # If no project ID, save to the storage directory
        file_path = os.path.join("storage", f"doc_{uuid.uuid4()}_{filename}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the file
        with open(file_path, "wb") as f:
            f.write(file_data)

        logger.info(f"Saved file {filename} ({len(file_data)} bytes) to storage: {file_path}")

    file_size = len(file_data)
    return file_path, file_size

def _load_documents() -> List[Dict[str, Any]]:
    """Load documents from file."""
    _ensure_dirs()

    # Use the storage manager to load metadata
    documents = load_storage_metadata()
    return documents

def _save_documents(documents: List[Dict[str, Any]]):
    """Save documents to file."""
    _ensure_dirs()

    # Save to storage
    save_storage_metadata(documents)

def _get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Get a document by ID."""
    documents = _load_documents()
    for document in documents:
        if document["document_id"] == document_id:
            return document
    return None

def _get_document_type(filename: str) -> str:
    """Get the document type from the filename."""
    if not filename:
        return "unknown"

    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".pdf"]:
            return "pdf"
        elif ext in [".docx", ".doc"]:
            return "docx"
        elif ext in [".xlsx", ".xls"]:
            return "xlsx"
        elif ext in [".pptx", ".ppt"]:
            return "pptx"
        elif ext in [".txt"]:
            return "txt"
        elif ext in [".md"]:
            return "md"
        elif ext in [".py"]:
            return "py"
        elif ext in [".js"]:
            return "js"
        elif ext in [".html", ".htm"]:
            return "html"
        elif ext in [".csv"]:
            return "csv"
        elif ext in [".json"]:
            return "json"
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"Error determining document type: {e}")
        return "unknown"

def _save_file(file, document_id: str, skip_file_creation: bool = False) -> str:
    """
    Save a file to disk.

    Args:
        file: File object (bytes, file-like object)
        document_id: Document ID
        skip_file_creation: If True, don't actually create the file, just return the path.
                           This is useful when we're going to index the content directly.

    Returns:
        Path to the saved file (even if the file wasn't actually created)
    """
    _ensure_dirs()

    # Handle different types of file objects
    try:
        # If we received bytes directly
        if isinstance(file, bytes):
            # Use storage manager to save the file
            file_path = save_document_file(file, document_id, skip_file_creation=skip_file_creation)
        else:
            # If it's a SpooledTemporaryFile or similar with a file attribute
            if hasattr(file, 'file'):
                file = file.file

            # Try to seek to the beginning if possible
            try:
                file.seek(0)
            except (AttributeError, IOError):
                # Some file objects might not support seek
                pass

            # Read the content in binary mode
            content = file.read()

            # Use storage manager to save the file
            file_path = save_document_file(content, document_id, skip_file_creation=skip_file_creation)

        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise

def _read_file(document_id: str) -> Optional[bytes]:
    """Read a file from disk."""
    # Read from storage
    content = read_document_file(document_id)
    return content

# --- Document Functions ---
def create_document(title: str, file, owner: str, project_id: Optional[str] = None, content: Optional[str] = None, document_type: Optional[str] = None, original_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new document.

    Args:
        title: Document title
        file: File object (bytes, file-like object, or path)
        owner: Username of the document owner
        project_id: Optional project ID
        content: Optional content string (for text documents)
        document_type: Optional document type
        original_filename: Optional original filename

    Returns:
        Document dictionary
    """
    if not title or not file or not owner:
        raise ValueError("Document title, file, and owner are required")

    # Create document
    document_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    try:
        # Check if we have direct content
        if content is not None and document_type is not None:
            # First save to database with proper structure
            if not original_filename:
                original_filename = f"{title}.{document_type}"

            # Convert content to bytes if it's a string
            if isinstance(content, str):
                file_data = content.encode('utf-8')
            else:
                file_data = content

            # Save to database with proper structure
            db_file_path, _ = save_file_to_database(file_data, project_id, owner, original_filename)
            logger.info(f"Saved content to database at {db_file_path}")

            # Get the file path without actually creating the file
            file_path = save_document_file(content.encode('utf-8'), document_id, skip_file_creation=True)

            # Get the content path without actually creating the file
            try:
                from .storage_manager import save_document_content
                content_path = save_document_content(document_id, content, skip_file_creation=True)
                logger.info(f"Generated content file path at {content_path} (file not created)")
            except Exception as e:
                logger.warning(f"Could not generate content file path: {str(e)}")

            # Use the provided document type
            doc_type = document_type

            # Create document object with content directly
            document = {
                "document_id": document_id,
                "title": title,
                "document_type": doc_type,
                "owner": owner,
                "project_id": project_id,
                "created_at": now,
                "updated_at": now,
                "version_count": 1,
                "file_path": file_path,
                "db_file_path": db_file_path,
                "original_filename": original_filename,
                "content": content,  # Store content directly
                "metadata": {"extraction_method": "direct"}
            }
        elif isinstance(file, bytes):
            # Handle bytes directly if that's what we received
            # First save to database with proper structure
            if not original_filename:
                original_filename = f"{title}.{document_type if document_type else 'bin'}"

            # Save to database with proper structure
            db_file_path, file_size = save_file_to_database(file, project_id, owner, original_filename)
            logger.info(f"Saved {file_size} bytes to database at {db_file_path}")

            # Get the file path without actually creating the file
            file_path = save_document_file(file, document_id, skip_file_creation=True)

            # Process the file based on its type
            logger.info(f"Processing file {original_filename} ({len(file)} bytes) using file processor")
            extracted_text, metadata = process_file(file, original_filename)

            # Store only an excerpt of the content in the document registry to save space
            try:
                from .document_registry import update_document_metadata
                # Create an excerpt (first 1000 characters) for the document registry
                content_excerpt = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                # Update the document metadata with the content excerpt
                update_document_metadata(document_id, {
                    "content_excerpt": content_excerpt,
                    "content_length": len(extracted_text),
                    "has_full_content": True
                })
                logger.info(f"Stored content excerpt in document registry for document {document_id} (full length: {len(extracted_text)})")
            except Exception as e:
                logger.warning(f"Could not store content excerpt in document registry: {str(e)}")

            # Get the content path without actually creating the file
            try:
                from .storage_manager import save_document_content
                content_path = save_document_content(document_id, extracted_text, skip_file_creation=True)
                logger.info(f"Generated content file path at {content_path} (file not created)")
            except Exception as e:
                logger.warning(f"Could not generate content file path: {str(e)}")

            # Get document type from metadata or fallback
            doc_type = document_type or metadata.get("file_type", _get_document_type(original_filename))

            # Create document object with extracted content
            document = {
                "document_id": document_id,
                "title": title,
                "document_type": doc_type,
                "owner": owner,
                "project_id": project_id,
                "created_at": now,
                "updated_at": now,
                "version_count": 1,
                "file_path": file_path,
                "db_file_path": db_file_path,
                "original_filename": original_filename,
                "content": extracted_text,  # Store extracted content
                "metadata": metadata
            }
        else:
            # For file-like objects
            # First read the content
            try:
                # If it's a SpooledTemporaryFile or similar with a file attribute
                if hasattr(file, 'file'):
                    file = file.file

                # Try to seek to the beginning if possible
                try:
                    file.seek(0)
                except (AttributeError, IOError):
                    # Some file objects might not support seek
                    pass

                # Read the content in binary mode
                file_data = file.read()

                # Get filename
                if not original_filename:
                    original_filename = getattr(file, 'name', title)
                    if not original_filename or original_filename == '':
                        original_filename = f"{title}.{document_type if document_type else 'bin'}"

                # Save to database with proper structure
                db_file_path, file_size = save_file_to_database(file_data, project_id, owner, original_filename)
                logger.info(f"Saved {file_size} bytes to database at {db_file_path}")

                # Get the file path without actually creating the file
                file_path = save_document_file(file_data, document_id, skip_file_creation=True)

                # Process the file based on its type
                logger.info(f"Processing file {original_filename} ({len(file_data)} bytes) using file processor")
                extracted_text, metadata = process_file(file_data, original_filename)

                # Store only an excerpt of the content in the document registry to save space
                try:
                    from .document_registry import update_document_metadata
                    # Create an excerpt (first 1000 characters) for the document registry
                    content_excerpt = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                    # Update the document metadata with the content excerpt
                    update_document_metadata(document_id, {
                        "content_excerpt": content_excerpt,
                        "content_length": len(extracted_text),
                        "has_full_content": True
                    })
                    logger.info(f"Stored content excerpt in document registry for document {document_id} (full length: {len(extracted_text)})")
                except Exception as e:
                    logger.warning(f"Could not store content excerpt in document registry: {str(e)}")

                # Get the content path without actually creating the file
                try:
                    from .storage_manager import save_document_content
                    content_path = save_document_content(document_id, extracted_text, skip_file_creation=True)
                    logger.info(f"Generated content file path at {content_path} (file not created)")
                except Exception as e:
                    logger.warning(f"Could not generate content file path: {str(e)}")

                # Get document type
                doc_type = document_type or _get_document_type(original_filename)

                # Create document object with extracted content
                document = {
                    "document_id": document_id,
                    "title": title,
                    "document_type": doc_type,
                    "owner": owner,
                    "project_id": project_id,
                    "created_at": now,
                    "updated_at": now,
                    "version_count": 1,
                    "file_path": file_path,
                    "db_file_path": db_file_path,
                    "original_filename": original_filename,
                    "content": extracted_text,  # Store extracted content
                    "metadata": metadata
                }
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                # Fallback to old method if reading fails, but skip file creation
                file_path = _save_file(file, document_id, skip_file_creation=True)

                # Get document type
                filename = getattr(file, 'name', '')
                doc_type = document_type or _get_document_type(filename)

                # Create document object
                document = {
                    "document_id": document_id,
                    "title": title,
                    "document_type": doc_type,
                    "owner": owner,
                    "project_id": project_id,
                    "created_at": now,
                    "updated_at": now,
                    "version_count": 1,
                    "file_path": file_path,
                    "content": "Content not loaded", # Content will be loaded on demand
                    "metadata": {"extraction_method": "deferred"}
                }

        # Save document
        documents = _load_documents()
        documents.append(document)
        _save_documents(documents)

        # Register document in registry if available
        try:
            from .document_registry import register_document, register_document_file, register_document_project

            # Register document in registry
            register_document(document_id, {
                "title": title,
                "document_type": document.get("document_type"),
                "owner": owner,
                "project_id": project_id,
                "created_at": now,
                "original_filename": document.get("original_filename"),
                "metadata": document.get("metadata", {})
            })

            # Register document file
            if "file_path" in document:
                register_document_file(document_id, document["file_path"], "storage")

            # Register project association
            if project_id:
                register_document_project(document_id, project_id)

            logger.info(f"Registered document {document_id} in registry")
        except ImportError:
            logger.warning("Document registry not available. Document not registered.")

        return document
    except Exception as e:
        # Clean up if there was an error
        try:
            # Delete file from storage if it exists
            delete_document_file(document_id)
        except Exception:
            pass

        # Re-raise the exception
        logger.error(f"Error creating document: {e}")
        raise

def get_documents(username: str = None, project_id: str = None) -> List[Dict[str, Any]]:
    """
    Get all documents or documents for a specific user or project.

    Args:
        username: Optional username to filter documents by
        project_id: Optional project ID to filter documents by

    Returns:
        List of document dictionaries
    """
    documents = _load_documents()

    if username:
        # Filter documents by username (owner)
        filtered_docs = []
        for d in documents:
            try:
                if d.get("owner") == username:
                    filtered_docs.append(d)
            except Exception as e:
                logger.warning(f"Error filtering document by owner: {e}")
        documents = filtered_docs

    if project_id:
        # Filter documents by project ID
        filtered_docs = []
        for d in documents:
            try:
                if d.get("project_id") == project_id:
                    filtered_docs.append(d)
            except Exception as e:
                logger.warning(f"Error filtering document by project_id: {e}")
        documents = filtered_docs

    return documents

def get_registry_content(document_id: str) -> Optional[str]:
    """
    Helper function to get content from the document registry.

    Args:
        document_id: Document ID

    Returns:
        Document content, excerpt, or None if not found
    """
    from .document_registry import get_document_registry
    doc_registry = get_document_registry(document_id)
    if doc_registry and "metadata" in doc_registry:
        # Check if full content is available
        if "content" in doc_registry["metadata"]:
            return doc_registry["metadata"]["content"]
        # If only excerpt is available, use that
        elif "content_excerpt" in doc_registry["metadata"]:
            excerpt = doc_registry["metadata"]["content_excerpt"]
            # Add a note that this is just an excerpt if it's truncated
            if doc_registry["metadata"].get("has_full_content", False) and "..." in excerpt:
                excerpt += "\n\n[Note: This is only an excerpt of the full content. The complete content is available in the content file.]"
            return excerpt
    return None

def get_document_content(document_id: str) -> Optional[str]:
    """
    Get the content of a document by ID.

    Args:
        document_id: Document ID

    Returns:
        Document content or None if not found
    """
    # Try multiple sources for content in order of preference

    # 1. Try to get from content file first (most reliable)
    try:
        from .storage_manager import read_document_content
        content_text = read_document_content(document_id)
        if content_text:
            logger.info(f"Retrieved content from content file for document {document_id}, length: {len(content_text)}")
            return content_text
    except Exception as e:
        logger.warning(f"Could not retrieve content from content file: {str(e)}")

    # 2. Try to get from document registry
    try:
        from .document_registry import get_document_registry
        doc_registry = get_document_registry(document_id)
        if doc_registry and "metadata" in doc_registry:
            # Check if full content is available
            if "content" in doc_registry["metadata"]:
                content = doc_registry["metadata"]["content"]
                logger.info(f"Retrieved full content from registry for document {document_id}, length: {len(content)}")
                return content
            # If only excerpt is available, note that in logs but still return it
            elif "content_excerpt" in doc_registry["metadata"]:
                excerpt = doc_registry["metadata"]["content_excerpt"]
                content_length = doc_registry["metadata"].get("content_length", len(excerpt))
                logger.info(f"Retrieved content excerpt from registry for document {document_id}, excerpt length: {len(excerpt)}, full length: {content_length}")
                # Add a note that this is just an excerpt if it's truncated
                if doc_registry["metadata"].get("has_full_content", False) and "..." in excerpt:
                    excerpt += "\n\n[Note: This is only an excerpt of the full content. The complete content is available in the content file.]"
                return excerpt
    except Exception as e:
        logger.warning(f"Could not retrieve content from registry: {str(e)}")

    # 3. Try to get from document object
    document = _get_document(document_id)
    if document and "content" in document and document["content"] != "Content not loaded":
        logger.info(f"Retrieved content from document object for document {document_id}, length: {len(document['content'])}")
        return document["content"]

    # 4. Try to extract content from file
    try:
        file_data = _read_file(document_id)
        if file_data:
            # Get original filename from document
            original_filename = None
            if document:
                original_filename = document.get("original_filename")

            # Process the file to extract content
            extracted_text, _ = process_file(file_data, original_filename)
            if extracted_text:
                logger.info(f"Extracted content from file for document {document_id}, length: {len(extracted_text)}")
                return extracted_text
    except Exception as e:
        logger.warning(f"Could not extract content from file: {str(e)}")

    logger.error(f"Could not retrieve content for document {document_id} from any source")
    return None

def get_document_metadata(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the metadata of a document by ID.

    Args:
        document_id: Document ID

    Returns:
        Document metadata or None if not found
    """
    # Initialize a combined metadata dictionary
    combined_metadata = {}

    # Try to get from document object first
    document = _get_document(document_id)
    if document:
        # Add basic document fields to metadata
        for key, value in document.items():
            if key != "content" and key != "metadata":  # Skip content and nested metadata
                combined_metadata[key] = value

        # Add document's metadata if available
        if "metadata" in document and isinstance(document["metadata"], dict):
            for key, value in document["metadata"].items():
                combined_metadata[key] = value

            logger.info(f"Retrieved metadata from document object for document {document_id}")

    # Try to get from document registry to supplement or override
    try:
        from .document_registry import get_document_registry
        doc_registry = get_document_registry(document_id)
        if doc_registry:
            # Add registry fields to metadata
            for key, value in doc_registry.items():
                if key != "metadata" and key != "files" and key != "projects" and key != "embeddings" and key != "indices":
                    # Add top-level registry fields
                    combined_metadata[key] = value

            # Add registry's metadata if available
            if "metadata" in doc_registry and isinstance(doc_registry["metadata"], dict):
                for key, value in doc_registry["metadata"].items():
                    # Registry metadata takes precedence
                    combined_metadata[key] = value

                logger.info(f"Retrieved and merged metadata from registry for document {document_id}")
    except Exception as e:
        logger.warning(f"Could not retrieve metadata from registry: {str(e)}")

    if combined_metadata:
        # Log the metadata keys we found
        logger.info(f"Combined metadata for document {document_id} has keys: {list(combined_metadata.keys())}")
        return combined_metadata

    logger.error(f"Could not retrieve metadata for document {document_id} from any source")
    return None

def get_document_by_id(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a document by ID.

    Args:
        document_id: Document ID

    Returns:
        Document dictionary or None if not found
    """
    document = _get_document(document_id)
    if document:
        # Check if content field exists and needs loading
        if "content" in document and document["content"] == "Content not loaded":
            # Try multiple sources for content in order of preference
            content_found = False

            # 1. Try to get from content file first (most reliable)
            try:
                from .storage_manager import read_document_content
                content_text = read_document_content(document_id)
                if content_text:
                    logger.info(f"Retrieved content from content file for document {document_id}, length: {len(content_text)}")
                    document["content"] = content_text
                    if "metadata" not in document:
                        document["metadata"] = {}
                    document["metadata"]["extraction_method"] = "content_file"
                    content_found = True
            except Exception as e:
                logger.warning(f"Could not retrieve content from content file: {str(e)}")

            # 2. Try to get from document file if not found in content file
            if not content_found:
                content = _read_file(document_id)
                if content:
                    # Get the filename from the document
                    title = document.get('title', 'untitled')
                    doc_type = document.get('document_type', 'txt')
                    filename = f"{title}.{doc_type}"

                    # Process the file based on its type
                    logger.info(f"Processing document {filename} ({len(content)} bytes) using file processor")
                    extracted_text, metadata = process_file(content, filename)

                    # Update document with extracted content and metadata
                    document["content"] = extracted_text
                    document["metadata"] = metadata

                    # Log the extraction method
                    logger.info(f"Document content extracted using {metadata.get('extraction_method', 'unknown')} method")

                    # Save to content file for future use
                    try:
                        from .storage_manager import save_document_content
                        save_document_content(document_id, extracted_text)
                        logger.info(f"Saved extracted content to content file for document {document_id}")
                    except Exception as e:
                        logger.warning(f"Could not save content to content file: {str(e)}")

                    content_found = True

            # 3. Try to get from document registry if not found yet
            if not content_found:
                try:
                    from .document_registry import get_document_registry
                    doc_registry = get_document_registry(document_id)
                    if doc_registry and "metadata" in doc_registry:
                        # Check if full content is available
                        if "content" in doc_registry["metadata"]:
                            content_text = doc_registry["metadata"]["content"]
                            logger.info(f"Retrieved full content from registry for document {document_id}, length: {len(content_text)}")
                            document["content"] = content_text
                            if "metadata" not in document:
                                document["metadata"] = {}
                            document["metadata"]["extraction_method"] = "registry"
                            content_found = True
                        # If only excerpt is available, use that
                        elif "content_excerpt" in doc_registry["metadata"]:
                            excerpt = doc_registry["metadata"]["content_excerpt"]
                            content_length = doc_registry["metadata"].get("content_length", len(excerpt))
                            logger.info(f"Retrieved content excerpt from registry for document {document_id}, excerpt length: {len(excerpt)}, full length: {content_length}")
                            document["content"] = excerpt
                            if "metadata" not in document:
                                document["metadata"] = {}
                            document["metadata"]["extraction_method"] = "registry_excerpt"
                            document["metadata"]["is_excerpt"] = True
                            document["metadata"]["full_content_length"] = content_length
                            # Add a note that this is just an excerpt if it's truncated
                            if doc_registry["metadata"].get("has_full_content", False) and "..." in excerpt:
                                document["content_note"] = "This is only an excerpt of the full content. The complete content is available in the content file."
                            content_found = True

                        # Save to content file for future use
                        try:
                            from .storage_manager import save_document_content
                            save_document_content(document_id, content_text)
                            logger.info(f"Saved registry content to content file for document {document_id}")
                        except Exception as e:
                            logger.warning(f"Could not save content to content file: {str(e)}")

                        content_found = True
                except Exception as e:
                    logger.warning(f"Could not retrieve content from registry: {str(e)}")

            # 4. Try to get from vector store if not found yet
            if not content_found:
                try:
                    from .metadata_optimizer import get_document_content
                    project_id = document.get("project_id")
                    content_text = get_document_content(document_id, project_id)
                    if content_text:
                        logger.info(f"Retrieved content from vector store for document {document_id}, length: {len(content_text)}")
                        document["content"] = content_text
                        if "metadata" not in document:
                            document["metadata"] = {}
                        document["metadata"]["extraction_method"] = "vector_store"

                        # Save to content file for future use
                        try:
                            from .storage_manager import save_document_content
                            save_document_content(document_id, content_text)
                            logger.info(f"Saved vector store content to content file for document {document_id}")
                        except Exception as e:
                            logger.warning(f"Could not save content to content file: {str(e)}")

                        content_found = True
                    else:
                        logger.warning(f"Could not retrieve content from vector store for document {document_id}")
                        document["content"] = "Content not available"
                except Exception as e:
                    logger.error(f"Error retrieving content from vector store: {str(e)}")
                    document["content"] = "Content not available"

        # If content field doesn't exist or is empty, try to get it from all sources
        elif "content" not in document or not document["content"] or document["content"] == "Content not available":
            logger.info(f"Content missing for document {document_id}, trying all sources")
            content_found = False

            # Try all sources in order of preference
            sources_to_try = [
                ("content_file", lambda: read_document_content(document_id)),
                ("registry", lambda: get_registry_content(document_id)),
                ("vector_store", lambda: get_document_content(document_id, document.get("project_id")))
            ]

            for source_name, get_content_func in sources_to_try:
                if content_found:
                    break

                try:
                    # Import necessary functions
                    if source_name == "content_file":
                        from .storage_manager import read_document_content
                    elif source_name == "registry":
                        from .document_registry import get_document_registry
                    elif source_name == "vector_store":
                        from .metadata_optimizer import get_document_content

                    # Try to get content
                    content_text = get_content_func()
                    if content_text:
                        logger.info(f"Retrieved content from {source_name} for document {document_id}, length: {len(content_text)}")
                        document["content"] = content_text
                        if "metadata" not in document:
                            document["metadata"] = {}
                        document["metadata"]["extraction_method"] = source_name

                        # Save to content file for future use if not already from content file
                        if source_name != "content_file":
                            try:
                                from .storage_manager import save_document_content
                                save_document_content(document_id, content_text)
                                logger.info(f"Saved {source_name} content to content file for document {document_id}")
                            except Exception as e:
                                logger.warning(f"Could not save content to content file: {str(e)}")

                        content_found = True
                except Exception as e:
                    logger.warning(f"Could not retrieve content from {source_name}: {str(e)}")

            # If content is still not available, add a content_excerpt field if not present
            if not content_found:
                logger.warning(f"Could not retrieve content from any source for document {document_id}")
                document["content"] = "Content not available"
                if "content_excerpt" not in document:
                    document["content_excerpt"] = "Content not available"

    return document

def update_document(document_id: str, title: str = None) -> Optional[Dict[str, Any]]:
    """
    Update a document.

    Args:
        document_id: Document ID
        title: New document title (optional)

    Returns:
        Updated document dictionary or None if not found
    """
    documents = _load_documents()

    for i, document in enumerate(documents):
        if document["document_id"] == document_id:
            if title:
                document["title"] = title

            document["updated_at"] = datetime.now().isoformat()
            _save_documents(documents)
            return document

    return None

def delete_document(document_id: str) -> bool:
    """
    Delete a document.

    Args:
        document_id: Document ID

    Returns:
        True if the document was deleted, False otherwise
    """
    # Get the document first to get its project_id and other metadata
    document = get_document_by_id(document_id)
    project_id = None
    if document:
        project_id = document.get("project_id")

    documents = _load_documents()
    original_count = len(documents)

    documents = [d for d in documents if d["document_id"] != document_id]

    if len(documents) < original_count:
        _save_documents(documents)

        # Delete file from storage
        delete_document_file(document_id)

        # First try to use the document registry if available
        try:
            from .document_registry import unregister_document

            # Unregister document from registry
            registry_data = unregister_document(document_id)
            logger.info(f"Unregistered document {document_id} from registry")
            logger.info(f"  - Files: {len(registry_data.get('files', []))}")
            logger.info(f"  - Projects: {registry_data.get('projects', [])}")
            logger.info(f"  - Embeddings: {len(registry_data.get('embeddings', []))}")
            logger.info(f"  - Indices: {len(registry_data.get('indices', []))}")
        except ImportError:
            logger.warning("Document registry not available. Using fallback cleanup.")

        # Perform thorough cleanup of all document files and references
        try:
            # Import here to avoid circular imports
            from .project_index_cleanup import thoroughly_clean_document_files
            cleanup_result = thoroughly_clean_document_files(document_id)
            logger.info(f"Thorough document cleanup for {document_id}: {cleanup_result['status']}")
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

        return True

    return False

def create_document_version(document_id: str, file, changes_summary: str = None, created_by: str = None, content: Optional[str] = None, original_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Create a new version of a document.

    Args:
        document_id: Document ID
        file: File object
        changes_summary: Optional summary of changes
        created_by: Optional username of the version creator
        content: Optional content string (for text documents)
        original_filename: Optional original filename

    Returns:
        Version dictionary or None if the document was not found
    """
    documents = _load_documents()
    version_id = str(uuid.uuid4())

    for document in documents:
        if document["document_id"] == document_id:
            try:
                # Create version
                now = datetime.now().isoformat()
                project_id = document.get("project_id")
                owner = created_by or document.get("owner")
                doc_type = document.get("document_type")

                # Check if we have direct content
                if content is not None:
                    # First save to database with proper structure
                    if not original_filename:
                        original_filename = f"version_{document_id}.{doc_type}"

                    # Convert content to bytes if it's a string
                    if isinstance(content, str):
                        file_data = content.encode('utf-8')
                    else:
                        file_data = content

                    # Save to database with proper structure
                    db_file_path, _ = save_file_to_database(file_data, project_id, owner, original_filename)
                    logger.info(f"Saved version content to database at {db_file_path}")

                    # Also save to versions directory for processing
                    _ensure_dirs()
                    version_path = os.path.join(VERSIONS_DIR, version_id)
                    with open(version_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    # Create metadata
                    metadata = {"extraction_method": "direct"}
                elif isinstance(file, bytes):
                    # If we received bytes directly
                    # First save to database with proper structure
                    if not original_filename:
                        original_filename = f"version_{document_id}.{doc_type}"

                    # Save to database with proper structure
                    db_file_path, file_size = save_file_to_database(file, project_id, owner, original_filename)
                    logger.info(f"Saved version {file_size} bytes to database at {db_file_path}")

                    # Also save to versions directory for processing
                    _ensure_dirs()
                    version_path = os.path.join(VERSIONS_DIR, version_id)

                    # Process the file based on its type
                    logger.info(f"Processing version file {original_filename} ({len(file)} bytes) using file processor")
                    extracted_text, metadata = process_file(file, original_filename)

                    # Save the original file to versions directory
                    with open(version_path, "wb") as f:
                        f.write(file)
                else:
                    # For file-like objects
                    # First read the content
                    try:
                        # If it's a SpooledTemporaryFile or similar with a file attribute
                        if hasattr(file, 'file'):
                            file = file.file

                        # Try to seek to the beginning if possible
                        try:
                            file.seek(0)
                        except (AttributeError, IOError):
                            # Some file objects might not support seek
                            pass

                        # Read the content in binary mode
                        file_data = file.read()

                        # Get filename
                        if not original_filename:
                            original_filename = getattr(file, 'name', f"version_{document_id}.{doc_type}")

                        # Save to database with proper structure
                        db_file_path, file_size = save_file_to_database(file_data, project_id, owner, original_filename)
                        logger.info(f"Saved version {file_size} bytes to database at {db_file_path}")

                        # Also save to versions directory for processing
                        version_path = os.path.join(VERSIONS_DIR, version_id)
                        with open(version_path, "wb") as f:
                            f.write(file_data)

                        # Process the file based on its type
                        logger.info(f"Processing version file {original_filename} ({len(file_data)} bytes) using file processor")
                        extracted_text, metadata = process_file(file_data, original_filename)
                    except Exception as e:
                        logger.error(f"Error reading file: {e}")
                        raise

                # Update document
                document["updated_at"] = now
                document["version_count"] = document.get("version_count", 1) + 1

                # Create version object with metadata
                if content is not None:
                    # For direct content
                    version = {
                        "version_id": version_id,
                        "document_id": document_id,
                        "created_at": now,
                        "changes_summary": changes_summary,
                        "created_by": created_by,
                        "file_path": version_path,
                        "db_file_path": db_file_path,
                        "original_filename": original_filename,
                        "content": content,
                        "metadata": {"extraction_method": "direct"}
                    }
                elif 'extracted_text' in locals():
                    # For processed files
                    version = {
                        "version_id": version_id,
                        "document_id": document_id,
                        "created_at": now,
                        "changes_summary": changes_summary,
                        "created_by": created_by,
                        "file_path": version_path,
                        "db_file_path": db_file_path,
                        "original_filename": original_filename,
                        "content": extracted_text,
                        "metadata": metadata
                    }
                else:
                    # Fallback
                    version = {
                        "version_id": version_id,
                        "document_id": document_id,
                        "created_at": now,
                        "changes_summary": changes_summary,
                        "created_by": created_by,
                        "file_path": version_path,
                        "metadata": {"extraction_method": "unknown"}
                    }

                # Save document
                _save_documents(documents)

                return version
            except Exception as e:
                # Clean up if there was an error
                try:
                    error_file_path = os.path.join(VERSIONS_DIR, version_id)
                    if os.path.exists(error_file_path):
                        os.remove(error_file_path)
                except Exception:
                    pass

                # Re-raise the exception
                logger.error(f"Error creating document version: {e}")
                raise

    return None

def get_document_versions(document_id: str) -> List[Dict[str, Any]]:
    """
    Get all versions of a document.

    Args:
        document_id: Document ID

    Returns:
        List of version dictionaries
    """
    # This is a simplified implementation
    # In a real application, you would store versions in a database
    logger.info(f"Getting versions for document {document_id}")
    return []

# Initialize
_ensure_dirs()
