"""
Collaboration module for Local File Deep Research.
"""

import os
import json
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Constants
COLLAB_DIR = os.environ.get("COLLAB_DIR", "collaboration")
PROJECTS_FILE = os.path.join(COLLAB_DIR, "projects.json")
COMMENTS_DIR = os.path.join(COLLAB_DIR, "comments")
SHARES_DIR = os.path.join(COLLAB_DIR, "shares")

class CollaborationError(Exception):
    """Base exception for collaboration errors."""
    pass

def _ensure_collab_dirs():
    """Ensure the collaboration directories exist."""
    os.makedirs(COLLAB_DIR, exist_ok=True)
    os.makedirs(COMMENTS_DIR, exist_ok=True)
    os.makedirs(SHARES_DIR, exist_ok=True)
    
    # Create projects file if it doesn't exist
    if not os.path.exists(PROJECTS_FILE):
        with open(PROJECTS_FILE, 'w') as f:
            json.dump({}, f)

def _load_projects() -> Dict[str, Dict[str, Any]]:
    """Load the projects data."""
    _ensure_collab_dirs()
    try:
        with open(PROJECTS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def _save_projects(projects: Dict[str, Dict[str, Any]]):
    """Save the projects data."""
    _ensure_collab_dirs()
    with open(PROJECTS_FILE, 'w') as f:
        json.dump(projects, f, indent=2)

def _get_comments_file(project_id: str) -> str:
    """Get the path to a project's comments file."""
    return os.path.join(COMMENTS_DIR, f"{project_id}_comments.json")

def _load_comments(project_id: str) -> List[Dict[str, Any]]:
    """Load comments for a project."""
    comments_file = _get_comments_file(project_id)
    try:
        with open(comments_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def _save_comments(project_id: str, comments: List[Dict[str, Any]]):
    """Save comments for a project."""
    comments_file = _get_comments_file(project_id)
    with open(comments_file, 'w') as f:
        json.dump(comments, f, indent=2)

def _get_shares_file(project_id: str) -> str:
    """Get the path to a project's shares file."""
    return os.path.join(SHARES_DIR, f"{project_id}_shares.json")

def _load_shares(project_id: str) -> Dict[str, Dict[str, Any]]:
    """Load shares for a project."""
    shares_file = _get_shares_file(project_id)
    try:
        with open(shares_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def _save_shares(project_id: str, shares: Dict[str, Dict[str, Any]]):
    """Save shares for a project."""
    shares_file = _get_shares_file(project_id)
    with open(shares_file, 'w') as f:
        json.dump(shares, f, indent=2)

def create_project(name: str, description: str, owner: str, documents: List[str] = None) -> str:
    """
    Create a new collaborative project.
    
    Args:
        name: Project name
        description: Project description
        owner: Username of the project owner
        documents: Optional list of document IDs to include
        
    Returns:
        Project ID
    """
    _ensure_collab_dirs()
    
    # Load projects
    projects = _load_projects()
    
    # Generate project ID
    project_id = f"proj_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Create project data
    project_data = {
        "project_id": project_id,
        "name": name,
        "description": description,
        "owner": owner,
        "documents": documents or [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "members": [owner]
    }
    
    # Save project
    projects[project_id] = project_data
    _save_projects(projects)
    
    # Initialize comments and shares
    _save_comments(project_id, [])
    _save_shares(project_id, {})
    
    return project_id

def get_project(project_id: str) -> Dict[str, Any]:
    """
    Get a project.
    
    Args:
        project_id: Project ID
        
    Returns:
        Project data
        
    Raises:
        CollaborationError: If the project is not found
    """
    # Load projects
    projects = _load_projects()
    
    # Check if project exists
    if project_id not in projects:
        raise CollaborationError(f"Project '{project_id}' not found")
    
    return projects[project_id]

def update_project(project_id: str, name: str = None, description: str = None, documents: List[str] = None) -> Dict[str, Any]:
    """
    Update a project.
    
    Args:
        project_id: Project ID
        name: Optional new project name
        description: Optional new project description
        documents: Optional new list of document IDs
        
    Returns:
        Updated project data
        
    Raises:
        CollaborationError: If the project is not found
    """
    # Load projects
    projects = _load_projects()
    
    # Check if project exists
    if project_id not in projects:
        raise CollaborationError(f"Project '{project_id}' not found")
    
    # Update project data
    if name is not None:
        projects[project_id]["name"] = name
    if description is not None:
        projects[project_id]["description"] = description
    if documents is not None:
        projects[project_id]["documents"] = documents
    
    projects[project_id]["updated_at"] = datetime.now().isoformat()
    
    # Save projects
    _save_projects(projects)
    
    return projects[project_id]

def delete_project(project_id: str) -> bool:
    """
    Delete a project.
    
    Args:
        project_id: Project ID
        
    Returns:
        True if the project was deleted, False otherwise
    """
    # Load projects
    projects = _load_projects()
    
    # Check if project exists
    if project_id not in projects:
        return False
    
    # Delete project
    del projects[project_id]
    _save_projects(projects)
    
    # Delete comments and shares
    try:
        os.remove(_get_comments_file(project_id))
    except FileNotFoundError:
        pass
    
    try:
        os.remove(_get_shares_file(project_id))
    except FileNotFoundError:
        pass
    
    return True

def list_projects(owner: str = None) -> List[Dict[str, Any]]:
    """
    List projects.
    
    Args:
        owner: Optional owner username to filter by
        
    Returns:
        List of project data
    """
    # Load projects
    projects = _load_projects()
    
    # Filter by owner if specified
    if owner:
        return [p for p in projects.values() if p["owner"] == owner]
    
    return list(projects.values())

def add_project_member(project_id: str, username: str) -> Dict[str, Any]:
    """
    Add a member to a project.
    
    Args:
        project_id: Project ID
        username: Username to add
        
    Returns:
        Updated project data
        
    Raises:
        CollaborationError: If the project is not found
    """
    # Load projects
    projects = _load_projects()
    
    # Check if project exists
    if project_id not in projects:
        raise CollaborationError(f"Project '{project_id}' not found")
    
    # Add member if not already a member
    if username not in projects[project_id]["members"]:
        projects[project_id]["members"].append(username)
        projects[project_id]["updated_at"] = datetime.now().isoformat()
        _save_projects(projects)
    
    return projects[project_id]

def remove_project_member(project_id: str, username: str) -> Dict[str, Any]:
    """
    Remove a member from a project.
    
    Args:
        project_id: Project ID
        username: Username to remove
        
    Returns:
        Updated project data
        
    Raises:
        CollaborationError: If the project is not found or the user is the owner
    """
    # Load projects
    projects = _load_projects()
    
    # Check if project exists
    if project_id not in projects:
        raise CollaborationError(f"Project '{project_id}' not found")
    
    # Check if user is the owner
    if username == projects[project_id]["owner"]:
        raise CollaborationError(f"Cannot remove the owner from the project")
    
    # Remove member if they are a member
    if username in projects[project_id]["members"]:
        projects[project_id]["members"].remove(username)
        projects[project_id]["updated_at"] = datetime.now().isoformat()
        _save_projects(projects)
    
    return projects[project_id]

def add_comment(project_id: str, username: str, content: str, document_id: str = None) -> Dict[str, Any]:
    """
    Add a comment to a project.
    
    Args:
        project_id: Project ID
        username: Username of the commenter
        content: Comment content
        document_id: Optional document ID to associate with the comment
        
    Returns:
        Comment data
        
    Raises:
        CollaborationError: If the project is not found or the user is not a member
    """
    # Check if project exists and user is a member
    project = get_project(project_id)
    if username not in project["members"]:
        raise CollaborationError(f"User '{username}' is not a member of project '{project_id}'")
    
    # Load comments
    comments = _load_comments(project_id)
    
    # Create comment data
    comment_id = f"comment_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    comment_data = {
        "comment_id": comment_id,
        "project_id": project_id,
        "username": username,
        "content": content,
        "document_id": document_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # Add comment
    comments.append(comment_data)
    _save_comments(project_id, comments)
    
    return comment_data

def get_comments(project_id: str, document_id: str = None) -> List[Dict[str, Any]]:
    """
    Get comments for a project.
    
    Args:
        project_id: Project ID
        document_id: Optional document ID to filter by
        
    Returns:
        List of comment data
        
    Raises:
        CollaborationError: If the project is not found
    """
    # Check if project exists
    get_project(project_id)
    
    # Load comments
    comments = _load_comments(project_id)
    
    # Filter by document ID if specified
    if document_id:
        return [c for c in comments if c.get("document_id") == document_id]
    
    return comments

def update_comment(project_id: str, comment_id: str, username: str, content: str) -> Dict[str, Any]:
    """
    Update a comment.
    
    Args:
        project_id: Project ID
        comment_id: Comment ID
        username: Username of the commenter
        content: New comment content
        
    Returns:
        Updated comment data
        
    Raises:
        CollaborationError: If the project or comment is not found, or the user is not the commenter
    """
    # Check if project exists
    get_project(project_id)
    
    # Load comments
    comments = _load_comments(project_id)
    
    # Find the comment
    for i, comment in enumerate(comments):
        if comment["comment_id"] == comment_id:
            # Check if user is the commenter
            if comment["username"] != username:
                raise CollaborationError(f"User '{username}' is not the author of comment '{comment_id}'")
            
            # Update comment
            comments[i]["content"] = content
            comments[i]["updated_at"] = datetime.now().isoformat()
            _save_comments(project_id, comments)
            
            return comments[i]
    
    raise CollaborationError(f"Comment '{comment_id}' not found in project '{project_id}'")

def delete_comment(project_id: str, comment_id: str, username: str) -> bool:
    """
    Delete a comment.
    
    Args:
        project_id: Project ID
        comment_id: Comment ID
        username: Username of the commenter or project owner
        
    Returns:
        True if the comment was deleted, False otherwise
        
    Raises:
        CollaborationError: If the project is not found or the user is not authorized
    """
    # Check if project exists and get owner
    project = get_project(project_id)
    is_owner = username == project["owner"]
    
    # Load comments
    comments = _load_comments(project_id)
    
    # Find the comment
    for i, comment in enumerate(comments):
        if comment["comment_id"] == comment_id:
            # Check if user is the commenter or project owner
            if comment["username"] != username and not is_owner:
                raise CollaborationError(f"User '{username}' is not authorized to delete comment '{comment_id}'")
            
            # Delete comment
            del comments[i]
            _save_comments(project_id, comments)
            
            return True
    
    return False

def share_project(project_id: str, username: str, recipient: str, permission: str = "read") -> Dict[str, Any]:
    """
    Share a project with a user.
    
    Args:
        project_id: Project ID
        username: Username of the sharer (must be a member)
        recipient: Username of the recipient
        permission: Permission level ("read" or "write")
        
    Returns:
        Share data
        
    Raises:
        CollaborationError: If the project is not found or the user is not a member
    """
    # Check if project exists and user is a member
    project = get_project(project_id)
    if username not in project["members"]:
        raise CollaborationError(f"User '{username}' is not a member of project '{project_id}'")
    
    # Validate permission
    if permission not in ["read", "write"]:
        raise CollaborationError(f"Invalid permission '{permission}'. Must be 'read' or 'write'.")
    
    # Load shares
    shares = _load_shares(project_id)
    
    # Create share data
    share_id = f"share_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    share_data = {
        "share_id": share_id,
        "project_id": project_id,
        "sharer": username,
        "recipient": recipient,
        "permission": permission,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # Add share
    shares[share_id] = share_data
    _save_shares(project_id, shares)
    
    return share_data

def get_shares(project_id: str, username: str = None) -> List[Dict[str, Any]]:
    """
    Get shares for a project.
    
    Args:
        project_id: Project ID
        username: Optional username to filter by (as recipient)
        
    Returns:
        List of share data
        
    Raises:
        CollaborationError: If the project is not found
    """
    # Check if project exists
    get_project(project_id)
    
    # Load shares
    shares = _load_shares(project_id)
    
    # Filter by username if specified
    if username:
        return [s for s in shares.values() if s["recipient"] == username]
    
    return list(shares.values())

def revoke_share(project_id: str, share_id: str, username: str) -> bool:
    """
    Revoke a share.
    
    Args:
        project_id: Project ID
        share_id: Share ID
        username: Username of the sharer or project owner
        
    Returns:
        True if the share was revoked, False otherwise
        
    Raises:
        CollaborationError: If the project is not found or the user is not authorized
    """
    # Check if project exists and get owner
    project = get_project(project_id)
    is_owner = username == project["owner"]
    
    # Load shares
    shares = _load_shares(project_id)
    
    # Check if share exists
    if share_id not in shares:
        return False
    
    # Check if user is the sharer or project owner
    if shares[share_id]["sharer"] != username and not is_owner:
        raise CollaborationError(f"User '{username}' is not authorized to revoke share '{share_id}'")
    
    # Revoke share
    del shares[share_id]
    _save_shares(project_id, shares)
    
    return True

def get_user_projects(username: str) -> List[Dict[str, Any]]:
    """
    Get projects that a user is a member of or has access to via shares.
    
    Args:
        username: Username
        
    Returns:
        List of project data with access level
    """
    # Load projects
    projects = _load_projects()
    
    # Get projects where user is a member
    user_projects = []
    for project_id, project in projects.items():
        if username in project["members"]:
            # User is a member
            access_level = "owner" if project["owner"] == username else "member"
            user_projects.append({
                **project,
                "access_level": access_level
            })
        else:
            # Check if user has access via shares
            shares = get_shares(project_id, username)
            if shares:
                # User has access via share
                access_level = "shared_" + shares[0]["permission"]
                user_projects.append({
                    **project,
                    "access_level": access_level,
                    "shared_by": shares[0]["sharer"]
                })
    
    return user_projects
