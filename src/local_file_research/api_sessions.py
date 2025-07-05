"""
API endpoints for session management.
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

@router.get("/sessions", response_model=List[Dict[str, Any]])
async def list_sessions(
    current_user: str = Depends(get_current_user)
):
    """
    List all sessions.
    """
    try:
        # Get all session files
        sessions_dir = "sessions"
        if not os.path.exists(sessions_dir):
            return []

        # Find all session vector store files
        vector_store_files = [f for f in os.listdir(sessions_dir) if f.endswith("_vector_store")]

        # Extract session IDs
        sessions = []
        for vector_store_file in vector_store_files:
            session_id = vector_store_file.replace("_vector_store", "")
            sessions.append({
                "session_id": session_id,
                "created": datetime.fromtimestamp(os.path.getctime(os.path.join(sessions_dir, vector_store_file))).isoformat()
            })

        return sessions
    except Exception as e:
        logger.error(f"Error in list_sessions endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@router.delete("/sessions/{session_id}", response_model=Dict[str, Any])
async def delete_session(
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Delete a session.
    """
    try:
        # Check if session exists
        sessions_dir = "sessions"
        vector_store_dir = os.path.join(sessions_dir, f"{session_id}_vector_store")

        if not os.path.exists(vector_store_dir):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Delete session files
        import shutil
        shutil.rmtree(vector_store_dir)

        return {
            "session_id": session_id,
            "status": "success",
            "message": f"Session {session_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error in delete_session endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

def setup_session_routes(app):
    """
    Set up session routes for the FastAPI app.
    """
    app.include_router(router, prefix="/sessions", tags=["Sessions"])
    logger.info("Set up session routes")
