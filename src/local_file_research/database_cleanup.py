"""
Database file cleanup utilities for Local File Deep Research.
This module provides functions to clean up database files after indexing.
It also serves as a central point for comprehensive cleanup operations.
"""

import os
import logging
import shutil
import sys
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Define legacy directories
BASE_DIR = os.environ.get("BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATABASE_DIR = os.environ.get("DATABASE_DIR", os.path.join(BASE_DIR, "database"))
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", os.path.join(BASE_DIR, "documents"))

# Import constants for other directories
from .storage_manager import STORAGE_DIR
from .embedding import EMBEDDING_CACHE_DIR as EMBEDDINGS_DIR

# Silence logging errors related to Unicode
def silence_logging_errors():
    """
    Silence logging errors related to Unicode encoding/decoding.
    This helps keep the command line output clean.
    """
    # Create a custom StreamHandler that handles Unicode errors
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream

                # Ensure msg is a string, not bytes
                if isinstance(msg, bytes):
                    try:
                        msg = msg.decode('utf-8', errors='replace')
                    except Exception:
                        msg = str(msg)

                # Ensure terminator is a string
                terminator = self.terminator
                if isinstance(terminator, bytes):
                    try:
                        terminator = terminator.decode('utf-8', errors='replace')
                    except Exception:
                        terminator = '\n'

                # Write with error handling
                try:
                    stream.write(msg + terminator)
                except UnicodeEncodeError:
                    # If Unicode error, try with replacement
                    stream.write(msg.encode('utf-8', 'replace').decode('utf-8') + terminator)
                except TypeError as e:
                    # Handle "write() argument must be str, not bytes" error
                    if "write() argument must be str, not bytes" in str(e):
                        if isinstance(msg, bytes):
                            stream.write(msg.decode('utf-8', errors='replace') + terminator)
                        else:
                            # If msg is already a string but we still get this error,
                            # it might be because terminator is bytes
                            stream.write(msg + '\n')
                    else:
                        # Re-raise other TypeError exceptions
                        raise

                self.flush()
            except Exception as e:
                # Log the error to stderr directly to avoid recursion
                import sys
                sys.stderr.write(f"Error in SafeStreamHandler.emit: {str(e)}\n")
                self.handleError(record)

    # Patch the logging.Formatter.formatMessage method to handle Unicode errors
    original_format_message = logging.Formatter.formatMessage

    def patched_format_message(self, record):
        # Ensure record.message is safe and is a string
        if isinstance(record.message, bytes):
            try:
                record.message = record.message.decode('utf-8', errors='replace')
            except Exception:
                record.message = str(record.message)

        try:
            return original_format_message(self, record)
        except Exception as e:
            # If formatting fails, provide a fallback
            return f"[Logging error: {e}] {str(record.message)}"

    # Apply the patch
    logging.Formatter.formatMessage = patched_format_message

    # Replace all StreamHandlers in the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, SafeStreamHandler):
            # Create new handler with same level and formatter
            new_handler = SafeStreamHandler(handler.stream)
            if handler.formatter:
                new_handler.setFormatter(handler.formatter)
            new_handler.setLevel(handler.level)

            # Remove old handler and add new one
            root_logger.removeHandler(handler)
            root_logger.addHandler(new_handler)

    # Also patch the original StreamHandler.emit method to catch errors in other handlers
    original_emit = logging.StreamHandler.emit

    def patched_emit(self, record):
        try:
            # Format the record
            msg = self.format(record)

            # Ensure msg is a string, not bytes
            if isinstance(msg, bytes):
                try:
                    msg = msg.decode('utf-8', errors='replace')
                except Exception:
                    msg = str(msg)

            # Ensure terminator is a string
            terminator = self.terminator
            if isinstance(terminator, bytes):
                try:
                    terminator = terminator.decode('utf-8', errors='replace')
                except Exception:
                    terminator = '\n'

            # Write with error handling
            try:
                self.stream.write(msg + terminator)
            except UnicodeEncodeError:
                # If Unicode error, try with replacement
                self.stream.write(msg.encode('utf-8', 'replace').decode('utf-8') + terminator)
            except TypeError as e:
                # Handle "write() argument must be str, not bytes" error
                if "write() argument must be str, not bytes" in str(e):
                    if isinstance(msg, bytes):
                        self.stream.write(msg.decode('utf-8', errors='replace') + terminator)
                    else:
                        # If msg is already a string but we still get this error,
                        # it might be because terminator is bytes
                        self.stream.write(msg + '\n')
                else:
                    # Re-raise other TypeError exceptions
                    raise

            self.flush()
        except Exception as e:
            # Log the error to stderr directly to avoid recursion
            import sys
            sys.stderr.write(f"Error in patched_emit: {str(e)}\n")
            self.handleError(record)

    # Apply the patch to all StreamHandler instances
    logging.StreamHandler.emit = patched_emit

    logger.info("Logging errors related to Unicode have been silenced")

def cleanup_database_files(document_ids: List[str] = None) -> Dict[str, Any]:
    """
    Clean up database files after indexing.
    This function also triggers comprehensive cleanup of all related folders:
    - Legacy database directory
    - Embeddings directory (cache files)
    - Storage/content directory
    - Storage/projects directory

    Args:
        document_ids: Optional list of document IDs to clean up. If None, clean up all.

    Returns:
        Dictionary with cleanup statistics
    """
    # Silence logging errors to keep command line output clean
    silence_logging_errors()

    # Initialize stats dictionary
    stats = {
        "status": "success",
        "total_files_removed": 0,
        "total_dirs_removed": 0,
        "total_bytes_freed": 0,
        "errors": []
    }

    # Check if legacy database directory exists
    if not os.path.exists(DATABASE_DIR):
        logger.warning(f"Legacy database directory {DATABASE_DIR} does not exist. Skipping legacy cleanup.")
        # Continue with other cleanups instead of returning early
    else:
        # Only clean up legacy database directory if it exists
        try:
            # Walk through the database directory
            for root, dirs, files in os.walk(DATABASE_DIR, topdown=False):
                # Remove files
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        stats["total_files_removed"] += 1
                        stats["total_bytes_freed"] += file_size
                        logger.info(f"Removed database file: {file_path} ({file_size} bytes)")
                    except Exception as e:
                        error_msg = f"Error removing file {os.path.join(root, file)}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

                # Remove empty directories
                for dir in dirs:
                    try:
                        dir_path = os.path.join(root, dir)
                        if not os.listdir(dir_path):  # Check if directory is empty
                            os.rmdir(dir_path)
                            stats["total_dirs_removed"] += 1
                            logger.info(f"Removed empty directory: {dir_path}")
                    except Exception as e:
                        error_msg = f"Error removing directory {os.path.join(root, dir)}: {str(e)}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

        except Exception as e:
            error_msg = f"Error during database cleanup: {str(e)}"
            logger.error(error_msg)
            stats["status"] = "error"
            stats["errors"].append(error_msg)

    # Format the bytes freed in a human-readable format for legacy database cleanup
    bytes_freed = stats["total_bytes_freed"]
    if bytes_freed > 1024 * 1024 * 1024:
        human_readable = f"{bytes_freed / (1024 * 1024 * 1024):.2f} GB"
    elif bytes_freed > 1024 * 1024:
        human_readable = f"{bytes_freed / (1024 * 1024):.2f} MB"
    elif bytes_freed > 1024:
        human_readable = f"{bytes_freed / 1024:.2f} KB"
    else:
        human_readable = f"{bytes_freed} bytes"

    if stats["total_files_removed"] > 0 or stats["total_dirs_removed"] > 0:
        logger.info(f"Legacy database cleanup completed: {stats['total_files_removed']} files and {stats['total_dirs_removed']} directories removed, freeing {human_readable}")

    # We'll update the human-readable size at the end of the function after all cleanups

    # Now perform comprehensive cleanup of all related folders
    logger.info("Starting comprehensive cleanup of all related folders...")

    # Initialize additional stats
    stats["embedding_files_removed"] = 0
    stats["storage_files_removed"] = 0
    stats["projects_files_removed"] = 0

    # 1. Clean up embeddings directory
    logger.info("Cleaning up embeddings directory...")
    try:
        # First try using document_registry.cleanup_embeddings
        try:
            from .document_registry import cleanup_embeddings
            embedding_stats = cleanup_embeddings(document_ids, force_cleanup=True)
            stats["embedding_files_removed"] = embedding_stats["total_files_removed"]
            stats["total_bytes_freed"] += embedding_stats["total_bytes_freed"]
            stats["errors"].extend(embedding_stats["errors"])
            logger.info(f"Embeddings cleanup: removed {embedding_stats['total_files_removed']} files, freed {embedding_stats['human_readable_bytes_freed']}")
        except ImportError:
            logger.warning("Document registry cleanup_embeddings not available. Falling back to basic cleanup.")
            # Fall back to basic cleanup
            if os.path.exists(EMBEDDINGS_DIR):
                embedding_stats = cleanup_embeddings_directory()
                stats["embedding_files_removed"] = embedding_stats["total_files_removed"]
                stats["total_bytes_freed"] += embedding_stats["total_bytes_freed"]
                stats["errors"].extend(embedding_stats["errors"])
                logger.info(f"Basic embeddings cleanup: removed {embedding_stats['total_files_removed']} files, freed {embedding_stats['human_readable_bytes_freed']}")
    except Exception as e:
        error_msg = f"Error during embeddings cleanup: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # 2. Clean up storage/content and storage/projects directories
    logger.info("Cleaning up storage directories...")
    try:
        from .document_cleanup import cleanup_storage_files, cleanup_projects_folder

        # Clean up storage/content
        storage_stats = cleanup_storage_files()
        stats["storage_files_removed"] = storage_stats["total_files_removed"]
        stats["total_bytes_freed"] += storage_stats["total_bytes_freed"]
        stats["errors"].extend(storage_stats["errors"])
        logger.info(f"Storage cleanup: removed {storage_stats['total_files_removed']} files, freed {storage_stats['human_readable_bytes_freed']}")

        # Clean up storage/projects
        projects_stats = cleanup_projects_folder()
        stats["projects_files_removed"] = projects_stats["total_files_removed"]
        stats["total_bytes_freed"] += projects_stats["total_bytes_freed"]
        stats["errors"].extend(projects_stats["errors"])
        logger.info(f"Projects folder cleanup: removed {projects_stats['total_files_removed']} files, freed {projects_stats['human_readable_bytes_freed']}")
    except ImportError as ie:
        logger.warning(f"Document cleanup functions not available: {str(ie)}. Skipping storage cleanup.")
    except Exception as e:
        error_msg = f"Error during storage cleanup: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # 3. Try to use organization_manager for any additional cleanup
    try:
        from .organization_manager import cleanup_after_indexing
        logger.info("Running additional cleanup through organization_manager...")
        comprehensive_stats = cleanup_after_indexing(document_ids)

        # Add any additional stats not already counted
        if "session_files_removed" in comprehensive_stats:
            stats["session_files_removed"] = comprehensive_stats["session_files_removed"]
            logger.info(f"Session cleanup: removed {comprehensive_stats['session_files_removed']} files")

        # Add any bytes freed that weren't already counted
        additional_bytes = comprehensive_stats.get("total_bytes_freed", 0)
        if additional_bytes > 0:
            stats["total_bytes_freed"] += additional_bytes
            from .document_cleanup import _format_bytes
            logger.info(f"Additional cleanup freed: {_format_bytes(additional_bytes)}")

        # Add any errors
        stats["errors"].extend(comprehensive_stats.get("errors", []))
    except ImportError:
        logger.info("Organization manager not available. Skipping additional cleanup.")
    except Exception as e:
        error_msg = f"Error during additional cleanup: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

    # Update human-readable size
    bytes_freed = stats["total_bytes_freed"]
    if bytes_freed > 1024 * 1024 * 1024:
        human_readable = f"{bytes_freed / (1024 * 1024 * 1024):.2f} GB"
    elif bytes_freed > 1024 * 1024:
        human_readable = f"{bytes_freed / (1024 * 1024):.2f} MB"
    elif bytes_freed > 1024:
        human_readable = f"{bytes_freed / 1024:.2f} KB"
    else:
        human_readable = f"{bytes_freed} bytes"

    stats["human_readable_bytes_freed"] = human_readable

    logger.info(f"Comprehensive cleanup completed: freed {human_readable} in total")

    return stats

def cleanup_embeddings_directory() -> Dict[str, Any]:
    """
    Clean up embedding files after indexing.
    After indexing, embedding files are no longer needed as they're stored in the vector store.

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "status": "success",
        "total_files_removed": 0,
        "total_bytes_freed": 0,
        "errors": []
    }

    if not os.path.exists(EMBEDDINGS_DIR):
        logger.warning(f"Embeddings directory {EMBEDDINGS_DIR} does not exist. Creating it.")
        try:
            os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
            logger.info(f"Created embeddings directory: {EMBEDDINGS_DIR}")
        except Exception as e:
            error_msg = f"Error creating embeddings directory: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats

    try:
        # Walk through the embeddings directory
        for root, dirs, files in os.walk(EMBEDDINGS_DIR, topdown=False):
            # Remove files
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    stats["total_files_removed"] += 1
                    stats["total_bytes_freed"] += file_size
                    logger.info(f"Removed embedding file: {file_path} ({file_size} bytes)")
                except Exception as e:
                    error_msg = f"Error removing file {os.path.join(root, file)}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

            # Remove empty directories (except the root embeddings directory)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Check if directory is empty
                        os.rmdir(dir_path)
                        logger.info(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    error_msg = f"Error removing directory {dir_path}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

        # If the embeddings directory is still empty after cleanup, try to recreate it
        if not os.listdir(EMBEDDINGS_DIR):
            logger.info(f"Embeddings directory is empty after cleanup: {EMBEDDINGS_DIR}")
            try:
                # Remove and recreate the directory to ensure it's clean
                shutil.rmtree(EMBEDDINGS_DIR)
                os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
                logger.info(f"Recreated empty embeddings directory: {EMBEDDINGS_DIR}")
            except Exception as e:
                error_msg = f"Error recreating embeddings directory: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

    except Exception as e:
        error_msg = f"Error during embeddings cleanup: {str(e)}"
        logger.error(error_msg)
        stats["status"] = "error"
        stats["errors"].append(error_msg)

    # Format the bytes freed in a human-readable format
    from .document_cleanup import _format_bytes
    stats["human_readable_bytes_freed"] = _format_bytes(stats["total_bytes_freed"])

    logger.info(f"Embeddings cleanup completed: {stats['total_files_removed']} files removed, freeing {stats['human_readable_bytes_freed']}")

    return stats
