"""
API endpoints for admin functions.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import os
import json
import uuid
import time
from datetime import datetime

# Import local modules
from src.local_file_research.auth import get_current_user

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/admin/status", response_model=Dict[str, Any])
async def get_status(
    current_user: str = Depends(get_current_user)
):
    """
    Get system status.
    """
    try:
        # Get disk usage
        import shutil

        # Get total, used, and free space
        total, used, free = shutil.disk_usage("/")

        # Convert to human-readable format
        def human_readable_size(size, decimal_places=2):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
                if size < 1024.0 or unit == 'PB':
                    break
                size /= 1024.0
            return f"{size:.{decimal_places}f} {unit}"

        # Get directory sizes
        def get_dir_size(path):
            total_size = 0
            if os.path.exists(path):
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
            return total_size

        # Get sizes of important directories
        storage_size = get_dir_size("storage")
        project_indices_size = get_dir_size("project_indices")
        sessions_size = get_dir_size("sessions")

        # Get counts
        project_count = len(os.listdir("project_indices")) if os.path.exists("project_indices") else 0
        session_count = len(os.listdir("sessions")) if os.path.exists("sessions") else 0
        document_count = len(os.listdir("storage")) if os.path.exists("storage") else 0

        return {
            "disk_usage": {
                "total": human_readable_size(total),
                "used": human_readable_size(used),
                "free": human_readable_size(free),
                "percent_used": f"{(used / total) * 100:.2f}%"
            },
            "directory_sizes": {
                "storage": human_readable_size(storage_size),
                "project_indices": human_readable_size(project_indices_size),
                "sessions": human_readable_size(sessions_size),
                "total": human_readable_size(storage_size + project_indices_size + sessions_size)
            },
            "counts": {
                "projects": project_count,
                "sessions": session_count,
                "documents": document_count
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in get_status endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/admin/cleanup", response_model=Dict[str, Any])
async def cleanup_system(
    current_user: str = Depends(get_current_user)
):
    """
    Clean up the system.
    """
    try:
        # Clean up database files
        from .database_cleanup import cleanup_database_files
        cleanup_result = cleanup_database_files()

        # Clean up expired sessions
        from .auth import cleanup_expired_sessions
        expired_sessions = cleanup_expired_sessions()

        return {
            "status": "success",
            "cleanup_result": cleanup_result,
            "expired_sessions_removed": expired_sessions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in cleanup_system endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clean up system: {str(e)}")

def setup_admin_routes(app):
    """
    Set up admin routes for the FastAPI app.
    """
    app.include_router(router, prefix="/admin", tags=["Admin"])
    logger.info("Set up admin routes")
