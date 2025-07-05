"""
Analytics module for Local File Deep Research.
"""

import os
import json
import time
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import threading

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
ANALYTICS_DIR = os.environ.get("ANALYTICS_DIR", "analytics")
ANALYTICS_DB = os.path.join(ANALYTICS_DIR, "analytics.db")
EVENTS_FILE = os.path.join(ANALYTICS_DIR, "events.jsonl")
METRICS_FILE = os.path.join(ANALYTICS_DIR, "metrics.json")

# --- Analytics Errors ---
class AnalyticsError(Exception):
    """Base exception for analytics errors."""
    pass

# --- Database Setup ---
def _ensure_analytics_dir():
    """Ensure the analytics directory exists."""
    os.makedirs(ANALYTICS_DIR, exist_ok=True)

def _init_database():
    """Initialize the analytics database."""
    _ensure_analytics_dir()
    
    conn = sqlite3.connect(ANALYTICS_DB)
    cursor = conn.cursor()
    
    # Create events table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        event_data TEXT,
        user_id TEXT,
        username TEXT,
        timestamp TEXT NOT NULL,
        session_id TEXT,
        ip_address TEXT,
        user_agent TEXT
    )
    ''')
    
    # Create metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        dimensions TEXT,
        timestamp TEXT NOT NULL
    )
    ''')
    
    # Create sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT,
        username TEXT,
        start_time TEXT NOT NULL,
        end_time TEXT,
        duration INTEGER,
        ip_address TEXT,
        user_agent TEXT,
        events_count INTEGER DEFAULT 0
    )
    ''')
    
    # Create performance table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operation TEXT NOT NULL,
        duration_ms REAL NOT NULL,
        details TEXT,
        timestamp TEXT NOT NULL
    )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_user ON events (user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_time ON events (timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (metric_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_time ON metrics (timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_op ON performance (operation)')
    
    conn.commit()
    conn.close()

# Initialize database on module load
_init_database()

# --- Event Tracking ---
def track_event(event_type: str, event_data: Dict[str, Any] = None, user_id: str = None, username: str = None, session_id: str = None):
    """
    Track an event.
    
    Args:
        event_type: Type of event
        event_data: Optional event data
        user_id: Optional user ID
        username: Optional username
        session_id: Optional session ID
    """
    try:
        # Ensure analytics directory exists
        _ensure_analytics_dir()
        
        # Create event
        event = {
            "event_type": event_type,
            "event_data": json.dumps(event_data) if event_data else None,
            "user_id": user_id,
            "username": username,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "ip_address": os.environ.get("REMOTE_ADDR", "unknown"),
            "user_agent": os.environ.get("HTTP_USER_AGENT", "unknown")
        }
        
        # Insert into database
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO events (event_type, event_data, user_id, username, timestamp, session_id, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event["event_type"],
            event["event_data"],
            event["user_id"],
            event["username"],
            event["timestamp"],
            event["session_id"],
            event["ip_address"],
            event["user_agent"]
        ))
        
        # Update session events count if session_id is provided
        if session_id:
            cursor.execute('''
            UPDATE sessions SET events_count = events_count + 1 WHERE session_id = ?
            ''', (session_id,))
        
        conn.commit()
        conn.close()
        
        # Also log to events file for backup
        with open(EVENTS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"Failed to track event: {e}")

def get_events(event_type: Optional[str] = None, user_id: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get events.
    
    Args:
        event_type: Optional event type to filter by
        user_id: Optional user ID to filter by
        start_time: Optional start time to filter by
        end_time: Optional end time to filter by
        limit: Maximum number of events to return
        
    Returns:
        List of events
    """
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        # Execute query
        cursor.execute(query, params)
        
        # Convert rows to dictionaries
        events = []
        for row in cursor.fetchall():
            event = dict(row)
            
            # Parse event data
            if event["event_data"]:
                try:
                    event["event_data"] = json.loads(event["event_data"])
                except:
                    pass
            
            events.append(event)
        
        conn.close()
        
        return events
    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        return []

def get_event_counts(group_by: str = "event_type", start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, int]:
    """
    Get event counts grouped by a field.
    
    Args:
        group_by: Field to group by (event_type, user_id, etc.)
        start_time: Optional start time to filter by
        end_time: Optional end time to filter by
        
    Returns:
        Dictionary of counts
    """
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        # Validate group_by field
        valid_fields = ["event_type", "user_id", "username", "session_id", "ip_address"]
        if group_by not in valid_fields:
            group_by = "event_type"
        
        # Build query
        query = f"SELECT {group_by}, COUNT(*) as count FROM events WHERE {group_by} IS NOT NULL"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += f" GROUP BY {group_by} ORDER BY count DESC"
        
        # Execute query
        cursor.execute(query, params)
        
        # Convert to dictionary
        counts = {}
        for row in cursor.fetchall():
            counts[row[0]] = row[1]
        
        conn.close()
        
        return counts
    except Exception as e:
        logger.error(f"Failed to get event counts: {e}")
        return {}

# --- Metrics Tracking ---
def track_metric(metric_name: str, metric_value: float, dimensions: Dict[str, Any] = None):
    """
    Track a metric.
    
    Args:
        metric_name: Name of the metric
        metric_value: Value of the metric
        dimensions: Optional dimensions for the metric
    """
    try:
        # Ensure analytics directory exists
        _ensure_analytics_dir()
        
        # Create metric
        metric = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "dimensions": json.dumps(dimensions) if dimensions else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Insert into database
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO metrics (metric_name, metric_value, dimensions, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (
            metric["metric_name"],
            metric["metric_value"],
            metric["dimensions"],
            metric["timestamp"]
        ))
        
        conn.commit()
        conn.close()
        
        # Also update metrics file
        try:
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            metrics = {}
        
        if metric_name not in metrics:
            metrics[metric_name] = {
                "values": [],
                "count": 0,
                "sum": 0,
                "min": None,
                "max": None,
                "last_value": None,
                "last_update": None
            }
        
        # Update metrics
        metrics[metric_name]["values"].append({
            "value": metric_value,
            "timestamp": metric["timestamp"],
            "dimensions": dimensions
        })
        
        # Keep only the last 100 values
        if len(metrics[metric_name]["values"]) > 100:
            metrics[metric_name]["values"] = metrics[metric_name]["values"][-100:]
        
        metrics[metric_name]["count"] += 1
        metrics[metric_name]["sum"] += metric_value
        metrics[metric_name]["min"] = min(metrics[metric_name]["min"] or metric_value, metric_value)
        metrics[metric_name]["max"] = max(metrics[metric_name]["max"] or metric_value, metric_value)
        metrics[metric_name]["last_value"] = metric_value
        metrics[metric_name]["last_update"] = metric["timestamp"]
        
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to track metric: {e}")

def get_metrics(metric_name: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Get metrics.
    
    Args:
        metric_name: Optional metric name to filter by
        start_time: Optional start time to filter by
        end_time: Optional end time to filter by
        
    Returns:
        Dictionary of metrics
    """
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        # Build query
        if metric_name:
            query = "SELECT metric_name, AVG(metric_value) as avg, MIN(metric_value) as min, MAX(metric_value) as max, COUNT(*) as count, SUM(metric_value) as sum FROM metrics WHERE metric_name = ?"
            params = [metric_name]
        else:
            query = "SELECT metric_name, AVG(metric_value) as avg, MIN(metric_value) as min, MAX(metric_value) as max, COUNT(*) as count, SUM(metric_value) as sum FROM metrics GROUP BY metric_name"
            params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        # Execute query
        cursor.execute(query, params)
        
        # Convert to dictionary
        metrics = {}
        for row in cursor.fetchall():
            metrics[row[0]] = {
                "avg": row[1],
                "min": row[2],
                "max": row[3],
                "count": row[4],
                "sum": row[5]
            }
            
            # Get recent values
            if metric_name:
                sub_query = "SELECT metric_value, dimensions, timestamp FROM metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT 100"
                sub_params = [metric_name]
                
                if start_time:
                    sub_query += " AND timestamp >= ?"
                    sub_params.append(start_time.isoformat())
                
                if end_time:
                    sub_query += " AND timestamp <= ?"
                    sub_params.append(end_time.isoformat())
                
                cursor.execute(sub_query, sub_params)
                
                values = []
                for sub_row in cursor.fetchall():
                    dimensions = None
                    if sub_row[1]:
                        try:
                            dimensions = json.loads(sub_row[1])
                        except:
                            pass
                    
                    values.append({
                        "value": sub_row[0],
                        "dimensions": dimensions,
                        "timestamp": sub_row[2]
                    })
                
                metrics[row[0]]["values"] = values
        
        conn.close()
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {}

# --- Session Tracking ---
def start_session(session_id: str, user_id: Optional[str] = None, username: Optional[str] = None):
    """
    Start a new session.
    
    Args:
        session_id: Session ID
        user_id: Optional user ID
        username: Optional username
    """
    try:
        # Ensure analytics directory exists
        _ensure_analytics_dir()
        
        # Create session
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "username": username,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration": None,
            "ip_address": os.environ.get("REMOTE_ADDR", "unknown"),
            "user_agent": os.environ.get("HTTP_USER_AGENT", "unknown"),
            "events_count": 0
        }
        
        # Insert into database
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO sessions (session_id, user_id, username, start_time, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session["session_id"],
            session["user_id"],
            session["username"],
            session["start_time"],
            session["ip_address"],
            session["user_agent"]
        ))
        
        conn.commit()
        conn.close()
        
        # Track session start event
        track_event("session_start", {"session_id": session_id}, user_id, username, session_id)
    except Exception as e:
        logger.error(f"Failed to start session: {e}")

def end_session(session_id: str):
    """
    End a session.
    
    Args:
        session_id: Session ID
    """
    try:
        # Get session start time
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        cursor.execute("SELECT start_time, user_id, username FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            start_time = datetime.fromisoformat(row[0])
            user_id = row[1]
            username = row[2]
            end_time = datetime.now()
            duration = int((end_time - start_time).total_seconds())
            
            # Update session
            cursor.execute('''
            UPDATE sessions SET end_time = ?, duration = ? WHERE session_id = ?
            ''', (end_time.isoformat(), duration, session_id))
            
            conn.commit()
            
            # Track session end event
            track_event("session_end", {"session_id": session_id, "duration": duration}, user_id, username, session_id)
            
            # Track session duration metric
            track_metric("session_duration", duration, {"session_id": session_id, "user_id": user_id})
        
        conn.close()
    except Exception as e:
        logger.error(f"Failed to end session: {e}")

def get_sessions(user_id: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get sessions.
    
    Args:
        user_id: Optional user ID to filter by
        start_time: Optional start time to filter by
        end_time: Optional end time to filter by
        limit: Maximum number of sessions to return
        
    Returns:
        List of sessions
    """
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM sessions WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        
        # Execute query
        cursor.execute(query, params)
        
        # Convert rows to dictionaries
        sessions = []
        for row in cursor.fetchall():
            sessions.append(dict(row))
        
        conn.close()
        
        return sessions
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        return []

# --- Performance Tracking ---
def track_performance(operation: str, duration_ms: float, details: Dict[str, Any] = None):
    """
    Track a performance metric.
    
    Args:
        operation: Operation name
        duration_ms: Duration in milliseconds
        details: Optional details
    """
    try:
        # Ensure analytics directory exists
        _ensure_analytics_dir()
        
        # Create performance record
        performance = {
            "operation": operation,
            "duration_ms": duration_ms,
            "details": json.dumps(details) if details else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Insert into database
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO performance (operation, duration_ms, details, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (
            performance["operation"],
            performance["duration_ms"],
            performance["details"],
            performance["timestamp"]
        ))
        
        conn.commit()
        conn.close()
        
        # Track as metric
        track_metric(f"performance.{operation}", duration_ms, details)
    except Exception as e:
        logger.error(f"Failed to track performance: {e}")

def performance_decorator(operation: str):
    """
    Decorator to track function performance.
    
    Args:
        operation: Operation name
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # Track performance
            track_performance(operation, duration_ms)
            
            return result
        return wrapper
    return decorator

def get_performance_stats(operation: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Get performance statistics.
    
    Args:
        operation: Optional operation to filter by
        start_time: Optional start time to filter by
        end_time: Optional end time to filter by
        
    Returns:
        Dictionary of performance statistics
    """
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        # Build query
        if operation:
            query = "SELECT operation, AVG(duration_ms) as avg, MIN(duration_ms) as min, MAX(duration_ms) as max, COUNT(*) as count FROM performance WHERE operation = ?"
            params = [operation]
        else:
            query = "SELECT operation, AVG(duration_ms) as avg, MIN(duration_ms) as min, MAX(duration_ms) as max, COUNT(*) as count FROM performance GROUP BY operation"
            params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        # Execute query
        cursor.execute(query, params)
        
        # Convert to dictionary
        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {
                "avg_ms": row[1],
                "min_ms": row[2],
                "max_ms": row[3],
                "count": row[4]
            }
            
            # Get percentiles
            if operation:
                sub_query = "SELECT duration_ms FROM performance WHERE operation = ? ORDER BY duration_ms"
                sub_params = [operation]
                
                if start_time:
                    sub_query += " AND timestamp >= ?"
                    sub_params.append(start_time.isoformat())
                
                if end_time:
                    sub_query += " AND timestamp <= ?"
                    sub_params.append(end_time.isoformat())
                
                cursor.execute(sub_query, sub_params)
                
                durations = [row[0] for row in cursor.fetchall()]
                if durations:
                    durations.sort()
                    stats[row[0]]["p50"] = durations[int(len(durations) * 0.5)]
                    stats[row[0]]["p90"] = durations[int(len(durations) * 0.9)]
                    stats[row[0]]["p95"] = durations[int(len(durations) * 0.95)]
                    stats[row[0]]["p99"] = durations[int(len(durations) * 0.99)]
        
        conn.close()
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return {}

# --- Usage Analytics ---
def get_usage_stats(start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Get usage statistics.
    
    Args:
        start_time: Optional start time to filter by
        end_time: Optional end time to filter by
        
    Returns:
        Dictionary of usage statistics
    """
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        cursor = conn.cursor()
        
        # Set default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=30)
        
        # Format times for SQL
        start_time_str = start_time.isoformat()
        end_time_str = end_time.isoformat()
        
        # Get active users
        cursor.execute('''
        SELECT COUNT(DISTINCT user_id) FROM events
        WHERE user_id IS NOT NULL AND timestamp BETWEEN ? AND ?
        ''', (start_time_str, end_time_str))
        active_users = cursor.fetchone()[0]
        
        # Get total sessions
        cursor.execute('''
        SELECT COUNT(*) FROM sessions
        WHERE start_time BETWEEN ? AND ?
        ''', (start_time_str, end_time_str))
        total_sessions = cursor.fetchone()[0]
        
        # Get average session duration
        cursor.execute('''
        SELECT AVG(duration) FROM sessions
        WHERE duration IS NOT NULL AND start_time BETWEEN ? AND ?
        ''', (start_time_str, end_time_str))
        avg_session_duration = cursor.fetchone()[0] or 0
        
        # Get total events
        cursor.execute('''
        SELECT COUNT(*) FROM events
        WHERE timestamp BETWEEN ? AND ?
        ''', (start_time_str, end_time_str))
        total_events = cursor.fetchone()[0]
        
        # Get events by type
        cursor.execute('''
        SELECT event_type, COUNT(*) FROM events
        WHERE timestamp BETWEEN ? AND ?
        GROUP BY event_type
        ORDER BY COUNT(*) DESC
        ''', (start_time_str, end_time_str))
        events_by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get daily active users
        cursor.execute('''
        SELECT substr(timestamp, 1, 10) as date, COUNT(DISTINCT user_id)
        FROM events
        WHERE user_id IS NOT NULL AND timestamp BETWEEN ? AND ?
        GROUP BY date
        ORDER BY date
        ''', (start_time_str, end_time_str))
        daily_active_users = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        # Compile stats
        stats = {
            "active_users": active_users,
            "total_sessions": total_sessions,
            "avg_session_duration": avg_session_duration,
            "total_events": total_events,
            "events_by_type": events_by_type,
            "daily_active_users": daily_active_users,
            "start_time": start_time_str,
            "end_time": end_time_str
        }
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        return {}

# --- Analytics Dashboard Data ---
def get_dashboard_data() -> Dict[str, Any]:
    """
    Get data for the analytics dashboard.
    
    Returns:
        Dictionary of dashboard data
    """
    try:
        # Get time ranges
        now = datetime.now()
        last_day = now - timedelta(days=1)
        last_week = now - timedelta(days=7)
        last_month = now - timedelta(days=30)
        
        # Get usage stats
        daily_stats = get_usage_stats(last_day, now)
        weekly_stats = get_usage_stats(last_week, now)
        monthly_stats = get_usage_stats(last_month, now)
        
        # Get performance stats
        performance_stats = get_performance_stats(start_time=last_week)
        
        # Get top events
        top_events = get_event_counts(start_time=last_week)
        
        # Get metrics
        metrics = get_metrics(start_time=last_week)
        
        # Compile dashboard data
        dashboard = {
            "usage": {
                "daily": daily_stats,
                "weekly": weekly_stats,
                "monthly": monthly_stats
            },
            "performance": performance_stats,
            "top_events": dict(list(top_events.items())[:10]),
            "metrics": metrics,
            "generated_at": now.isoformat()
        }
        
        return dashboard
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return {}
