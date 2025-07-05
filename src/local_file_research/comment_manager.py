"""
Comment management for Local File Deep Research.
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
COMMENTS_DIR = os.environ.get("COMMENTS_DIR", "comments")

# --- Helper Functions ---
def _ensure_comments_dir():
    """Ensure the comments directory exists."""
    os.makedirs(COMMENTS_DIR, exist_ok=True)
    
    # Create comments file if it doesn't exist
    comments_file = os.path.join(COMMENTS_DIR, "comments.json")
    if not os.path.exists(comments_file):
        with open(comments_file, "w") as f:
            json.dump([], f)

def _load_comments() -> List[Dict[str, Any]]:
    """Load comments from file."""
    _ensure_comments_dir()
    try:
        with open(os.path.join(COMMENTS_DIR, "comments.json"), "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def _save_comments(comments: List[Dict[str, Any]]):
    """Save comments to file."""
    _ensure_comments_dir()
    with open(os.path.join(COMMENTS_DIR, "comments.json"), "w") as f:
        json.dump(comments, f, indent=2)

# --- Comment Functions ---
def create_comment(content: str, username: str, project_id: Optional[str] = None, document_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new comment.
    
    Args:
        content: Comment content
        username: Username of the comment creator
        project_id: Optional project ID
        document_id: Optional document ID
        
    Returns:
        Comment dictionary
    """
    if not content or not username:
        raise ValueError("Comment content and username are required")
    
    if not project_id and not document_id:
        raise ValueError("Either project_id or document_id must be provided")
    
    # Create comment
    comment_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    comment = {
        "comment_id": comment_id,
        "content": content,
        "username": username,
        "project_id": project_id,
        "document_id": document_id,
        "created_at": now
    }
    
    # Save comment
    comments = _load_comments()
    comments.append(comment)
    _save_comments(comments)
    
    return comment

def get_comments(project_id: Optional[str] = None, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get comments for a project or document.
    
    Args:
        project_id: Optional project ID
        document_id: Optional document ID
        
    Returns:
        List of comment dictionaries
    """
    comments = _load_comments()
    
    if project_id:
        # Filter comments by project ID
        return [c for c in comments if c.get("project_id") == project_id]
    
    if document_id:
        # Filter comments by document ID
        return [c for c in comments if c.get("document_id") == document_id]
    
    return []

def delete_comment(comment_id: str, username: str) -> bool:
    """
    Delete a comment.
    
    Args:
        comment_id: Comment ID
        username: Username of the user trying to delete the comment
        
    Returns:
        True if the comment was deleted, False otherwise
    """
    comments = _load_comments()
    original_count = len(comments)
    
    # Find the comment
    comment = next((c for c in comments if c["comment_id"] == comment_id), None)
    if not comment:
        return False
    
    # Check if the user is the comment creator
    if comment["username"] != username:
        return False
    
    # Remove the comment
    comments = [c for c in comments if c["comment_id"] != comment_id]
    
    if len(comments) < original_count:
        _save_comments(comments)
        return True
    
    return False

# Initialize
_ensure_comments_dir()
