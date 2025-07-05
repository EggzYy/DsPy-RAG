"""
Research management for Local File Deep Research.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
RESEARCH_DIR = os.environ.get("RESEARCH_DIR", "research")

# --- Helper Functions ---
def _ensure_research_dir():
    """Ensure the research directory exists."""
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    
    # Create research file if it doesn't exist
    research_file = os.path.join(RESEARCH_DIR, "research.json")
    if not os.path.exists(research_file):
        with open(research_file, "w") as f:
            json.dump([], f)

def _load_research() -> List[Dict[str, Any]]:
    """Load research from file."""
    _ensure_research_dir()
    try:
        with open(os.path.join(RESEARCH_DIR, "research.json"), "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def _save_research(research_list: List[Dict[str, Any]]):
    """Save research to file."""
    _ensure_research_dir()
    with open(os.path.join(RESEARCH_DIR, "research.json"), "w") as f:
        json.dump(research_list, f, indent=2)

# --- Research Functions ---
def create_research(query: str, username: str, project_id: Optional[str] = None, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create a new research.
    
    Args:
        query: Research query
        username: Username of the researcher
        project_id: Optional project ID
        document_ids: Optional list of document IDs
        
    Returns:
        Research dictionary
    """
    if not query or not username:
        raise ValueError("Research query and username are required")
    
    # Create research
    research_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    research = {
        "research_id": research_id,
        "query": query,
        "username": username,
        "project_id": project_id,
        "document_ids": document_ids or [],
        "created_at": now,
        "status": "pending",
        "result": None
    }
    
    # Save research
    research_list = _load_research()
    research_list.append(research)
    _save_research(research_list)
    
    return research

def get_research(research_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a research by ID.
    
    Args:
        research_id: Research ID
        
    Returns:
        Research dictionary or None if not found
    """
    research_list = _load_research()
    
    for research in research_list:
        if research["research_id"] == research_id:
            return research
    
    return None

def get_research_list(username: Optional[str] = None, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get all research or research for a specific user or project.
    
    Args:
        username: Optional username to filter research by
        project_id: Optional project ID to filter research by
        
    Returns:
        List of research dictionaries
    """
    research_list = _load_research()
    
    if username:
        # Filter research by username
        research_list = [r for r in research_list if r["username"] == username]
    
    if project_id:
        # Filter research by project ID
        research_list = [r for r in research_list if r.get("project_id") == project_id]
    
    return research_list

def update_research_status(research_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Update the status of a research.
    
    Args:
        research_id: Research ID
        status: New status (pending, running, completed, failed)
        result: Optional research result
        
    Returns:
        Updated research dictionary or None if not found
    """
    if status not in ["pending", "running", "completed", "failed"]:
        raise ValueError("Status must be 'pending', 'running', 'completed', or 'failed'")
    
    research_list = _load_research()
    
    for i, research in enumerate(research_list):
        if research["research_id"] == research_id:
            research["status"] = status
            if result is not None:
                research["result"] = result
            
            _save_research(research_list)
            return research
    
    return None

def delete_research(research_id: str, username: str) -> bool:
    """
    Delete a research.
    
    Args:
        research_id: Research ID
        username: Username of the user trying to delete the research
        
    Returns:
        True if the research was deleted, False otherwise
    """
    research_list = _load_research()
    original_count = len(research_list)
    
    # Find the research
    research = next((r for r in research_list if r["research_id"] == research_id), None)
    if not research:
        return False
    
    # Check if the user is the researcher
    if research["username"] != username:
        return False
    
    # Remove the research
    research_list = [r for r in research_list if r["research_id"] != research_id]
    
    if len(research_list) < original_count:
        _save_research(research_list)
        return True
    
    return False

# Initialize
_ensure_research_dir()
