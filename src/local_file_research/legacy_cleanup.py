"""
Legacy cleanup utilities for Local File Deep Research.
This module provides functions to clean up legacy folders after migration.
"""

import os
import shutil
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
BASE_DIR = os.environ.get("BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", os.path.join(BASE_DIR, "documents"))
DATABASE_DIR = os.environ.get("DATABASE_DIR", os.path.join(BASE_DIR, "database"))

def remove_legacy_folders() -> Dict[str, Any]:
    """
    Remove legacy folders (documents and database) after migration.
    
    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        "status": "success",
        "folders_removed": [],
        "errors": []
    }
    
    # Remove documents folder
    if os.path.exists(DOCUMENTS_DIR):
        try:
            shutil.rmtree(DOCUMENTS_DIR)
            stats["folders_removed"].append(DOCUMENTS_DIR)
            logger.info(f"Removed legacy folder: {DOCUMENTS_DIR}")
        except Exception as e:
            error_msg = f"Error removing legacy folder {DOCUMENTS_DIR}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
    
    # Remove database folder
    if os.path.exists(DATABASE_DIR):
        try:
            shutil.rmtree(DATABASE_DIR)
            stats["folders_removed"].append(DATABASE_DIR)
            logger.info(f"Removed legacy folder: {DATABASE_DIR}")
        except Exception as e:
            error_msg = f"Error removing legacy folder {DATABASE_DIR}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
    
    if not stats["folders_removed"]:
        stats["status"] = "skipped"
        stats["reason"] = "no_legacy_folders_found"
    
    return stats
