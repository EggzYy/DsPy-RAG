"""
Real-time collaboration module for Local File Deep Research.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from fastapi.responses import JSONResponse

# Configure logging
logger = logging.getLogger(__name__)

# --- WebSocket Connection Manager ---
class ConnectionManager:
    """Manage WebSocket connections for real-time collaboration."""
    
    def __init__(self):
        # Active connections: {connection_id: {"websocket": websocket, "user": username, "projects": [project_ids], "documents": [document_ids]}}
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        # Project subscriptions: {project_id: set(connection_ids)}
        self.project_subscribers: Dict[str, Set[str]] = {}
        # Document subscriptions: {document_id: set(connection_ids)}
        self.document_subscribers: Dict[str, Set[str]] = {}
        # User connections: {username: set(connection_ids)}
        self.user_connections: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, user: str) -> str:
        """
        Connect a new WebSocket client.
        
        Args:
            websocket: WebSocket connection
            user: Username
            
        Returns:
            Connection ID
        """
        await websocket.accept()
        connection_id = f"conn_{int(time.time())}_{id(websocket)}"
        
        # Store connection
        self.active_connections[connection_id] = {
            "websocket": websocket,
            "user": user,
            "projects": set(),
            "documents": set(),
            "connected_at": datetime.now().isoformat()
        }
        
        # Add to user connections
        if user not in self.user_connections:
            self.user_connections[user] = set()
        self.user_connections[user].add(connection_id)
        
        logger.info(f"WebSocket connection established for user '{user}' with ID '{connection_id}'")
        
        # Send welcome message
        await self.send_personal_message(
            {"type": "connection", "status": "connected", "connection_id": connection_id},
            connection_id
        )
        
        # Broadcast user online status
        await self.broadcast_user_status(user, "online")
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """
        Disconnect a WebSocket client.
        
        Args:
            connection_id: Connection ID
        """
        if connection_id not in self.active_connections:
            return
        
        # Get user before removing connection
        user = self.active_connections[connection_id]["user"]
        
        # Remove from project subscriptions
        for project_id in self.active_connections[connection_id]["projects"]:
            if project_id in self.project_subscribers:
                self.project_subscribers[project_id].discard(connection_id)
                if not self.project_subscribers[project_id]:
                    del self.project_subscribers[project_id]
        
        # Remove from document subscriptions
        for document_id in self.active_connections[connection_id]["documents"]:
            if document_id in self.document_subscribers:
                self.document_subscribers[document_id].discard(connection_id)
                if not self.document_subscribers[document_id]:
                    del self.document_subscribers[document_id]
        
        # Remove from user connections
        if user in self.user_connections:
            self.user_connections[user].discard(connection_id)
            if not self.user_connections[user]:
                del self.user_connections[user]
                # Broadcast user offline status if no more connections
                asyncio.create_task(self.broadcast_user_status(user, "offline"))
        
        # Remove connection
        del self.active_connections[connection_id]
        
        logger.info(f"WebSocket connection closed for user '{user}' with ID '{connection_id}'")
    
    async def subscribe_to_project(self, connection_id: str, project_id: str):
        """
        Subscribe a connection to a project.
        
        Args:
            connection_id: Connection ID
            project_id: Project ID
        """
        if connection_id not in self.active_connections:
            return
        
        # Add to project subscribers
        if project_id not in self.project_subscribers:
            self.project_subscribers[project_id] = set()
        self.project_subscribers[project_id].add(connection_id)
        
        # Add to connection's projects
        self.active_connections[connection_id]["projects"].add(project_id)
        
        # Send confirmation
        await self.send_personal_message(
            {"type": "subscription", "status": "subscribed", "project_id": project_id},
            connection_id
        )
        
        # Broadcast user joined project
        user = self.active_connections[connection_id]["user"]
        await self.broadcast_to_project(
            project_id,
            {"type": "project_event", "event": "user_joined", "user": user, "project_id": project_id},
            exclude=connection_id
        )
        
        logger.info(f"User '{user}' subscribed to project '{project_id}'")
    
    async def unsubscribe_from_project(self, connection_id: str, project_id: str):
        """
        Unsubscribe a connection from a project.
        
        Args:
            connection_id: Connection ID
            project_id: Project ID
        """
        if connection_id not in self.active_connections:
            return
        
        # Remove from project subscribers
        if project_id in self.project_subscribers:
            self.project_subscribers[project_id].discard(connection_id)
            if not self.project_subscribers[project_id]:
                del self.project_subscribers[project_id]
        
        # Remove from connection's projects
        self.active_connections[connection_id]["projects"].discard(project_id)
        
        # Send confirmation
        await self.send_personal_message(
            {"type": "subscription", "status": "unsubscribed", "project_id": project_id},
            connection_id
        )
        
        # Broadcast user left project
        user = self.active_connections[connection_id]["user"]
        await self.broadcast_to_project(
            project_id,
            {"type": "project_event", "event": "user_left", "user": user, "project_id": project_id},
            exclude=connection_id
        )
        
        logger.info(f"User '{user}' unsubscribed from project '{project_id}'")
    
    async def subscribe_to_document(self, connection_id: str, document_id: str):
        """
        Subscribe a connection to a document.
        
        Args:
            connection_id: Connection ID
            document_id: Document ID
        """
        if connection_id not in self.active_connections:
            return
        
        # Add to document subscribers
        if document_id not in self.document_subscribers:
            self.document_subscribers[document_id] = set()
        self.document_subscribers[document_id].add(connection_id)
        
        # Add to connection's documents
        self.active_connections[connection_id]["documents"].add(document_id)
        
        # Send confirmation
        await self.send_personal_message(
            {"type": "subscription", "status": "subscribed", "document_id": document_id},
            connection_id
        )
        
        # Broadcast user viewing document
        user = self.active_connections[connection_id]["user"]
        await self.broadcast_to_document(
            document_id,
            {"type": "document_event", "event": "user_viewing", "user": user, "document_id": document_id},
            exclude=connection_id
        )
        
        logger.info(f"User '{user}' subscribed to document '{document_id}'")
    
    async def unsubscribe_from_document(self, connection_id: str, document_id: str):
        """
        Unsubscribe a connection from a document.
        
        Args:
            connection_id: Connection ID
            document_id: Document ID
        """
        if connection_id not in self.active_connections:
            return
        
        # Remove from document subscribers
        if document_id in self.document_subscribers:
            self.document_subscribers[document_id].discard(connection_id)
            if not self.document_subscribers[document_id]:
                del self.document_subscribers[document_id]
        
        # Remove from connection's documents
        self.active_connections[connection_id]["documents"].discard(document_id)
        
        # Send confirmation
        await self.send_personal_message(
            {"type": "subscription", "status": "unsubscribed", "document_id": document_id},
            connection_id
        )
        
        # Broadcast user stopped viewing document
        user = self.active_connections[connection_id]["user"]
        await self.broadcast_to_document(
            document_id,
            {"type": "document_event", "event": "user_left", "user": user, "document_id": document_id},
            exclude=connection_id
        )
        
        logger.info(f"User '{user}' unsubscribed from document '{document_id}'")
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """
        Send a message to a specific connection.
        
        Args:
            message: Message to send
            connection_id: Connection ID
        """
        if connection_id not in self.active_connections:
            return
        
        websocket = self.active_connections[connection_id]["websocket"]
        await websocket.send_json(message)
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """
        Broadcast a message to all connections.
        
        Args:
            message: Message to broadcast
            exclude: Optional connection ID to exclude
        """
        disconnected = []
        
        for connection_id, connection in self.active_connections.items():
            if connection_id != exclude:
                try:
                    await connection["websocket"].send_json(message)
                except RuntimeError:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def broadcast_to_project(self, project_id: str, message: Dict[str, Any], exclude: Optional[str] = None):
        """
        Broadcast a message to all connections subscribed to a project.
        
        Args:
            project_id: Project ID
            message: Message to broadcast
            exclude: Optional connection ID to exclude
        """
        if project_id not in self.project_subscribers:
            return
        
        disconnected = []
        
        for connection_id in self.project_subscribers[project_id]:
            if connection_id != exclude and connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id]["websocket"].send_json(message)
                except RuntimeError:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def broadcast_to_document(self, document_id: str, message: Dict[str, Any], exclude: Optional[str] = None):
        """
        Broadcast a message to all connections subscribed to a document.
        
        Args:
            document_id: Document ID
            message: Message to broadcast
            exclude: Optional connection ID to exclude
        """
        if document_id not in self.document_subscribers:
            return
        
        disconnected = []
        
        for connection_id in self.document_subscribers[document_id]:
            if connection_id != exclude and connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id]["websocket"].send_json(message)
                except RuntimeError:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def broadcast_to_user(self, username: str, message: Dict[str, Any]):
        """
        Broadcast a message to all connections of a specific user.
        
        Args:
            username: Username
            message: Message to broadcast
        """
        if username not in self.user_connections:
            return
        
        disconnected = []
        
        for connection_id in self.user_connections[username]:
            if connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id]["websocket"].send_json(message)
                except RuntimeError:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def broadcast_user_status(self, username: str, status: str):
        """
        Broadcast a user's status to all connections.
        
        Args:
            username: Username
            status: Status ("online" or "offline")
        """
        await self.broadcast({
            "type": "user_status",
            "user": username,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_active_users(self) -> List[str]:
        """
        Get a list of active users.
        
        Returns:
            List of usernames
        """
        return list(self.user_connections.keys())
    
    def get_project_users(self, project_id: str) -> List[str]:
        """
        Get a list of users subscribed to a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of usernames
        """
        if project_id not in self.project_subscribers:
            return []
        
        users = set()
        for connection_id in self.project_subscribers[project_id]:
            if connection_id in self.active_connections:
                users.add(self.active_connections[connection_id]["user"])
        
        return list(users)
    
    def get_document_users(self, document_id: str) -> List[str]:
        """
        Get a list of users subscribed to a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of usernames
        """
        if document_id not in self.document_subscribers:
            return []
        
        users = set()
        for connection_id in self.document_subscribers[document_id]:
            if connection_id in self.active_connections:
                users.add(self.active_connections[connection_id]["user"])
        
        return list(users)

# Create a global connection manager
manager = ConnectionManager()

# --- WebSocket Event Handlers ---
async def handle_client_message(connection_id: str, message: Dict[str, Any]):
    """
    Handle a message from a client.
    
    Args:
        connection_id: Connection ID
        message: Message from client
    """
    if connection_id not in manager.active_connections:
        return
    
    message_type = message.get("type")
    user = manager.active_connections[connection_id]["user"]
    
    if message_type == "subscribe":
        # Subscribe to project or document
        if "project_id" in message:
            await manager.subscribe_to_project(connection_id, message["project_id"])
        elif "document_id" in message:
            await manager.subscribe_to_document(connection_id, message["document_id"])
    
    elif message_type == "unsubscribe":
        # Unsubscribe from project or document
        if "project_id" in message:
            await manager.unsubscribe_from_project(connection_id, message["project_id"])
        elif "document_id" in message:
            await manager.unsubscribe_from_document(connection_id, message["document_id"])
    
    elif message_type == "project_message":
        # Send message to project
        if "project_id" in message and "content" in message:
            project_id = message["project_id"]
            
            # Check if user is subscribed to project
            if project_id not in manager.active_connections[connection_id]["projects"]:
                await manager.send_personal_message(
                    {"type": "error", "message": "Not subscribed to project"},
                    connection_id
                )
                return
            
            # Broadcast message to project
            await manager.broadcast_to_project(
                project_id,
                {
                    "type": "project_message",
                    "project_id": project_id,
                    "user": user,
                    "content": message["content"],
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    elif message_type == "document_message":
        # Send message to document
        if "document_id" in message and "content" in message:
            document_id = message["document_id"]
            
            # Check if user is subscribed to document
            if document_id not in manager.active_connections[connection_id]["documents"]:
                await manager.send_personal_message(
                    {"type": "error", "message": "Not subscribed to document"},
                    connection_id
                )
                return
            
            # Broadcast message to document
            await manager.broadcast_to_document(
                document_id,
                {
                    "type": "document_message",
                    "document_id": document_id,
                    "user": user,
                    "content": message["content"],
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    elif message_type == "document_edit":
        # Document edit
        if "document_id" in message and "changes" in message:
            document_id = message["document_id"]
            
            # Check if user is subscribed to document
            if document_id not in manager.active_connections[connection_id]["documents"]:
                await manager.send_personal_message(
                    {"type": "error", "message": "Not subscribed to document"},
                    connection_id
                )
                return
            
            # Broadcast edit to document
            await manager.broadcast_to_document(
                document_id,
                {
                    "type": "document_edit",
                    "document_id": document_id,
                    "user": user,
                    "changes": message["changes"],
                    "timestamp": datetime.now().isoformat()
                },
                exclude=connection_id
            )
    
    elif message_type == "cursor_position":
        # Cursor position update
        if "document_id" in message and "position" in message:
            document_id = message["document_id"]
            
            # Check if user is subscribed to document
            if document_id not in manager.active_connections[connection_id]["documents"]:
                await manager.send_personal_message(
                    {"type": "error", "message": "Not subscribed to document"},
                    connection_id
                )
                return
            
            # Broadcast cursor position to document
            await manager.broadcast_to_document(
                document_id,
                {
                    "type": "cursor_position",
                    "document_id": document_id,
                    "user": user,
                    "position": message["position"],
                    "timestamp": datetime.now().isoformat()
                },
                exclude=connection_id
            )
    
    elif message_type == "ping":
        # Ping to keep connection alive
        await manager.send_personal_message(
            {"type": "pong", "timestamp": datetime.now().isoformat()},
            connection_id
        )

# --- WebSocket Route Handler ---
async def websocket_endpoint(websocket: WebSocket, user: str):
    """
    Handle WebSocket connections.
    
    Args:
        websocket: WebSocket connection
        user: Username from authentication
    """
    connection_id = await manager.connect(websocket, user)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_client_message(connection_id, message)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON"},
                    connection_id
                )
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(connection_id)

# --- API Functions ---
def get_active_users() -> List[str]:
    """
    Get a list of active users.
    
    Returns:
        List of usernames
    """
    return manager.get_active_users()

def get_project_users(project_id: str) -> List[str]:
    """
    Get a list of users subscribed to a project.
    
    Args:
        project_id: Project ID
        
    Returns:
        List of usernames
    """
    return manager.get_project_users(project_id)

def get_document_users(document_id: str) -> List[str]:
    """
    Get a list of users subscribed to a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        List of usernames
    """
    return manager.get_document_users(document_id)

async def notify_project_update(project_id: str, update_type: str, data: Dict[str, Any]):
    """
    Notify all users subscribed to a project about an update.
    
    Args:
        project_id: Project ID
        update_type: Type of update
        data: Update data
    """
    await manager.broadcast_to_project(
        project_id,
        {
            "type": "project_update",
            "project_id": project_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    )

async def notify_document_update(document_id: str, update_type: str, data: Dict[str, Any]):
    """
    Notify all users subscribed to a document about an update.
    
    Args:
        document_id: Document ID
        update_type: Type of update
        data: Update data
    """
    await manager.broadcast_to_document(
        document_id,
        {
            "type": "document_update",
            "document_id": document_id,
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    )

async def notify_user(username: str, notification_type: str, data: Dict[str, Any]):
    """
    Send a notification to a specific user.
    
    Args:
        username: Username
        notification_type: Type of notification
        data: Notification data
    """
    await manager.broadcast_to_user(
        username,
        {
            "type": "notification",
            "notification_type": notification_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    )
