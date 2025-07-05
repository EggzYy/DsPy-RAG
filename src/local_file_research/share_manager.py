"""
Share management for Local File Deep Research.
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
SHARES_DIR = os.environ.get("SHARES_DIR", "shares")

# --- Helper Functions ---
def _ensure_shares_dir():
    """Ensure the shares directory exists."""
    os.makedirs(SHARES_DIR, exist_ok=True)
    
    # Create shares file if it doesn't exist
    shares_file = os.path.join(SHARES_DIR, "shares.json")
    if not os.path.exists(shares_file):
        with open(shares_file, "w") as f:
            json.dump([], f)

def _load_shares() -> List[Dict[str, Any]]:
    """Load shares from file."""
    _ensure_shares_dir()
    try:
        with open(os.path.join(SHARES_DIR, "shares.json"), "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def _save_shares(shares: List[Dict[str, Any]]):
    """Save shares to file."""
    _ensure_shares_dir()
    with open(os.path.join(SHARES_DIR, "shares.json"), "w") as f:
        json.dump(shares, f, indent=2)

# --- Share Functions ---
def create_share(project_id: str, sharer: str, recipient: str, permission: str = "read") -> Dict[str, Any]:
    """
    Create a new share.
    
    Args:
        project_id: Project ID
        sharer: Username of the user sharing the project
        recipient: Username of the user receiving the share
        permission: Permission level (read, write)
        
    Returns:
        Share dictionary
    """
    if not project_id or not sharer or not recipient:
        raise ValueError("Project ID, sharer, and recipient are required")
    
    if permission not in ["read", "write"]:
        raise ValueError("Permission must be 'read' or 'write'")
    
    # Create share
    share_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    share = {
        "share_id": share_id,
        "project_id": project_id,
        "sharer": sharer,
        "recipient": recipient,
        "permission": permission,
        "created_at": now
    }
    
    # Save share
    shares = _load_shares()
    shares.append(share)
    _save_shares(shares)
    
    return share

def get_shares(project_id: Optional[str] = None, username: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get shares for a project or user.
    
    Args:
        project_id: Optional project ID
        username: Optional username
        
    Returns:
        List of share dictionaries
    """
    shares = _load_shares()
    
    if project_id:
        # Filter shares by project ID
        shares = [s for s in shares if s["project_id"] == project_id]
    
    if username:
        # Filter shares by username (either sharer or recipient)
        shares = [s for s in shares if s["sharer"] == username or s["recipient"] == username]
    
    return shares

def delete_share(share_id: str, username: str) -> bool:
    """
    Delete a share.
    
    Args:
        share_id: Share ID
        username: Username of the user trying to delete the share
        
    Returns:
        True if the share was deleted, False otherwise
    """
    shares = _load_shares()
    original_count = len(shares)
    
    # Find the share
    share = next((s for s in shares if s["share_id"] == share_id), None)
    if not share:
        return False
    
    # Check if the user is the sharer
    if share["sharer"] != username:
        return False
    
    # Remove the share
    shares = [s for s in shares if s["share_id"] != share_id]
    
    if len(shares) < original_count:
        _save_shares(shares)
        return True
    
    return False

# Initialize
_ensure_shares_dir()
