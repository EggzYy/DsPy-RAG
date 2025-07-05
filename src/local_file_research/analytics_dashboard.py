"""
Advanced analytics dashboard for Local File Deep Research.
"""

import os
import json
import time
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Charts will not be displayed.")
    PLOTLY_AVAILABLE = False

try:
    from .analytics import get_dashboard_data, get_usage_stats, get_performance_stats
    from .analytics import get_events, get_event_counts, get_metrics
except ImportError:
    logger.warning("Analytics module not found. Using simple implementation.")
    from .analytics_simple import get_dashboard_data, get_usage_stats, get_performance_stats
    from .analytics_simple import get_events, get_event_counts, get_metrics

# --- Dashboard Configuration ---
REFRESH_INTERVAL = 60  # seconds
TIME_RANGES = {
    "Last Hour": timedelta(hours=1),
    "Last Day": timedelta(days=1),
    "Last Week": timedelta(days=7),
    "Last Month": timedelta(days=30),
    "Last Year": timedelta(days=365)
}

# --- Helper Functions ---
def format_number(num):
    """Format a number with thousands separator."""
    return f"{num:,}"

def format_time(seconds):
    """Format seconds into a readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_time_range(range_name):
    """Get start and end time for a time range."""
    end_time = datetime.now()
    start_time = end_time - TIME_RANGES.get(range_name, timedelta(days=7))
    return start_time, end_time

# --- Dashboard Components ---
def render_header():
    """Render the dashboard header."""
    st.title("📊 Analytics Dashboard")
    st.markdown("*Real-time analytics for Local File Deep Research*")

    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Time range selector
    with col1:
        time_range = st.selectbox("Time Range", list(TIME_RANGES.keys()), index=2)

    return time_range

def render_overview_metrics(time_range):
    """Render overview metrics."""
    st.header("Overview")

    # Get data
    start_time, end_time = get_time_range(time_range)
    stats = get_usage_stats(start_time, end_time)

    # Create metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Users", format_number(stats.get("active_users", 0)))

    with col2:
        st.metric("Total Sessions", format_number(stats.get("total_sessions", 0)))

    with col3:
        avg_duration = stats.get("avg_session_duration", 0)
        st.metric("Avg. Session Duration", format_time(avg_duration))

    with col4:
        st.metric("Total Events", format_number(stats.get("total_events", 0)))

    return stats

def render_user_activity(stats):
    """Render user activity charts."""
    st.header("User Activity")

    # Daily active users chart
    daily_users = stats.get("daily_active_users", {})
    if daily_users:
        df = pd.DataFrame({
            "Date": list(daily_users.keys()),
            "Active Users": list(daily_users.values())
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        fig = px.line(df, x="Date", y="Active Users",
                     title="Daily Active Users",
                     labels={"Date": "", "Active Users": "Users"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No user activity data available for the selected time range.")

    # Events by type
    events_by_type = stats.get("events_by_type", {})
    if events_by_type:
        # Sort by count
        events_sorted = sorted(events_by_type.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame({
            "Event Type": [e[0] for e in events_sorted],
            "Count": [e[1] for e in events_sorted]
        })

        fig = px.bar(df, x="Event Type", y="Count",
                    title="Events by Type",
                    labels={"Event Type": "", "Count": "Number of Events"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No event data available for the selected time range.")

def render_performance_metrics(time_range):
    """Render performance metrics."""
    st.header("Performance Metrics")

    # Get data
    start_time, end_time = get_time_range(time_range)
    performance_stats = get_performance_stats(start_time=start_time)

    if not performance_stats:
        st.info("No performance data available for the selected time range.")
        return

    # Create dataframe
    data = []
    for op, stats in performance_stats.items():
        data.append({
            "Operation": op,
            "Average (ms)": stats.get("avg_ms", 0),
            "Min (ms)": stats.get("min_ms", 0),
            "Max (ms)": stats.get("max_ms", 0),
            "Count": stats.get("count", 0),
            "p95 (ms)": stats.get("p95", 0)
        })

    df = pd.DataFrame(data)

    # Sort by average time
    df = df.sort_values("Average (ms)", ascending=False)

    # Display table
    st.dataframe(df, use_container_width=True)

    # Display chart for top operations
    top_ops = df.head(5)

    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=("Average Duration (ms)", "Operation Count"),
                       specs=[[{"type": "bar"}, {"type": "bar"}]])

    fig.add_trace(
        go.Bar(x=top_ops["Operation"], y=top_ops["Average (ms)"], name="Avg Duration"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=top_ops["Operation"], y=top_ops["Count"], name="Count"),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_document_metrics(time_range):
    """Render document metrics."""
    st.header("Document Metrics")

    # Get data
    start_time, end_time = get_time_range(time_range)

    # Get document events
    document_events = get_events("document_processed", start_time=start_time, end_time=end_time)

    if not document_events:
        st.info("No document processing data available for the selected time range.")
        return

    # Process document events
    doc_types = {}
    doc_sizes = []
    processing_times = []

    for event in document_events:
        event_data = event.get("event_data", {})
        if isinstance(event_data, str):
            try:
                event_data = json.loads(event_data)
            except:
                event_data = {}

        doc_type = event_data.get("document_type", "unknown")
        if doc_type in doc_types:
            doc_types[doc_type] += 1
        else:
            doc_types[doc_type] = 1

        size = event_data.get("size_bytes", 0)
        if size > 0:
            doc_sizes.append(size)

        time_ms = event_data.get("processing_time_ms", 0)
        if time_ms > 0:
            processing_times.append(time_ms)

    # Create document type chart
    col1, col2 = st.columns(2)

    with col1:
        if doc_types:
            # Sort by count
            types_sorted = sorted(doc_types.items(), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame({
                "Document Type": [t[0] for t in types_sorted],
                "Count": [t[1] for t in types_sorted]
            })

            fig = px.pie(df, values="Count", names="Document Type",
                        title="Documents by Type")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if doc_sizes:
            # Create size histogram
            fig = px.histogram(
                x=doc_sizes,
                nbins=20,
                labels={"x": "Size (bytes)"},
                title="Document Size Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Processing time vs size scatter plot
    if doc_sizes and processing_times and len(doc_sizes) == len(processing_times):
        df = pd.DataFrame({
            "Size (KB)": [s / 1024 for s in doc_sizes],
            "Processing Time (ms)": processing_times
        })

        fig = px.scatter(df, x="Size (KB)", y="Processing Time (ms)",
                        title="Processing Time vs Document Size",
                        trendline="ols")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_search_metrics(time_range):
    """Render search metrics."""
    st.header("Search Metrics")

    # Get data
    start_time, end_time = get_time_range(time_range)

    # Get search events
    search_events = get_events("search_performed", start_time=start_time, end_time=end_time)

    if not search_events:
        st.info("No search data available for the selected time range.")
        return

    # Process search events
    search_modes = {}
    report_modes = {}
    search_times = []
    result_counts = []

    for event in search_events:
        event_data = event.get("event_data", {})
        if isinstance(event_data, str):
            try:
                event_data = json.loads(event_data)
            except:
                event_data = {}

        search_mode = event_data.get("search_mode", "unknown")
        if search_mode in search_modes:
            search_modes[search_mode] += 1
        else:
            search_modes[search_mode] = 1

        report_mode = event_data.get("report_mode", "unknown")
        if report_mode in report_modes:
            report_modes[report_mode] += 1
        else:
            report_modes[report_mode] = 1

        time_ms = event_data.get("search_time_ms", 0)
        if time_ms > 0:
            search_times.append(time_ms)

        results = event_data.get("result_count", 0)
        if results > 0:
            result_counts.append(results)

    # Create charts
    col1, col2 = st.columns(2)

    with col1:
        if search_modes:
            # Sort by count
            modes_sorted = sorted(search_modes.items(), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame({
                "Search Mode": [m[0] for m in modes_sorted],
                "Count": [m[1] for m in modes_sorted]
            })

            fig = px.bar(df, x="Search Mode", y="Count",
                        title="Searches by Mode",
                        labels={"Search Mode": "", "Count": "Number of Searches"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if report_modes:
            # Sort by count
            modes_sorted = sorted(report_modes.items(), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame({
                "Report Mode": [m[0] for m in modes_sorted],
                "Count": [m[1] for m in modes_sorted]
            })

            fig = px.bar(df, x="Report Mode", y="Count",
                        title="Reports by Mode",
                        labels={"Report Mode": "", "Count": "Number of Reports"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Search time histogram
    if search_times:
        fig = px.histogram(
            x=search_times,
            nbins=20,
            labels={"x": "Search Time (ms)"},
            title="Search Time Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Result count histogram
    if result_counts:
        fig = px.histogram(
            x=result_counts,
            nbins=20,
            labels={"x": "Result Count"},
            title="Search Result Count Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_system_metrics(time_range):
    """Render system metrics."""
    st.header("System Metrics")

    # Get data
    start_time, end_time = get_time_range(time_range)
    metrics = get_metrics(start_time=start_time)

    if not metrics:
        st.info("No system metrics available for the selected time range.")
        return

    # Filter for system metrics
    system_metrics = {}
    for name, data in metrics.items():
        if name.startswith("system."):
            system_metrics[name] = data

    if not system_metrics:
        st.info("No system metrics available for the selected time range.")
        return

    # Create metrics table
    data = []
    for name, metric in system_metrics.items():
        display_name = name.replace("system.", "")
        data.append({
            "Metric": display_name,
            "Average": metric.get("avg", 0),
            "Min": metric.get("min", 0),
            "Max": metric.get("max", 0),
            "Count": metric.get("count", 0)
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Create time series charts for metrics with values
    for name, metric in system_metrics.items():
        if "values" in metric and metric["values"]:
            display_name = name.replace("system.", "")

            # Extract values and timestamps
            values = []
            timestamps = []

            for entry in metric["values"]:
                values.append(entry.get("value", 0))
                timestamps.append(entry.get("timestamp", ""))

            if values and timestamps:
                df = pd.DataFrame({
                    "Timestamp": timestamps,
                    "Value": values
                })

                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df = df.sort_values("Timestamp")

                fig = px.line(df, x="Timestamp", y="Value",
                             title=f"{display_name} Over Time",
                             labels={"Timestamp": "", "Value": display_name})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def render_export_metrics(time_range):
    """Render export metrics."""
    st.header("Export Metrics")

    # Get data
    start_time, end_time = get_time_range(time_range)

    # Get export events
    export_events = get_events("report_exported", start_time=start_time, end_time=end_time)

    if not export_events:
        st.info("No export data available for the selected time range.")
        return

    # Process export events
    export_formats = {}
    export_sizes = []

    for event in export_events:
        event_data = event.get("event_data", {})
        if isinstance(event_data, str):
            try:
                event_data = json.loads(event_data)
            except:
                event_data = {}

        export_format = event_data.get("format", "unknown")
        if export_format in export_formats:
            export_formats[export_format] += 1
        else:
            export_formats[export_format] = 1

        size = event_data.get("size_bytes", 0)
        if size > 0:
            export_sizes.append((export_format, size))

    # Create export format chart
    if export_formats:
        # Sort by count
        formats_sorted = sorted(export_formats.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame({
            "Export Format": [f[0] for f in formats_sorted],
            "Count": [f[1] for f in formats_sorted]
        })

        fig = px.pie(df, values="Count", names="Export Format",
                    title="Exports by Format")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Create export size by format chart
    if export_sizes:
        df = pd.DataFrame(export_sizes, columns=["Format", "Size (bytes)"])
        df["Size (KB)"] = df["Size (bytes)"] / 1024

        fig = px.box(df, x="Format", y="Size (KB)",
                    title="Export Size by Format",
                    labels={"Format": "Export Format", "Size (KB)": "Size (KB)"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_collaboration_metrics(time_range):
    """Render collaboration metrics."""
    st.header("Collaboration Metrics")

    # Get data
    start_time, end_time = get_time_range(time_range)

    # Get collaboration events
    collab_events = []
    for event_type in ["project_created", "project_member_added", "comment_added", "project_shared"]:
        events = get_events(event_type, start_time=start_time, end_time=end_time)
        collab_events.extend(events)

    if not collab_events:
        st.info("No collaboration data available for the selected time range.")
        return

    # Process collaboration events
    event_counts = {}
    user_activity = {}
    project_activity = {}

    for event in collab_events:
        event_type = event.get("event_type", "unknown")
        if event_type in event_counts:
            event_counts[event_type] += 1
        else:
            event_counts[event_type] = 1

        username = event.get("username", "unknown")
        if username in user_activity:
            user_activity[username] += 1
        else:
            user_activity[username] = 1

        event_data = event.get("event_data", {})
        if isinstance(event_data, str):
            try:
                event_data = json.loads(event_data)
            except:
                event_data = {}

        project_id = event_data.get("project_id", "unknown")
        if project_id in project_activity:
            project_activity[project_id] += 1
        else:
            project_activity[project_id] = 1

    # Create event counts chart
    if event_counts:
        # Sort by count
        events_sorted = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame({
            "Event Type": [e[0] for e in events_sorted],
            "Count": [e[1] for e in events_sorted]
        })

        fig = px.bar(df, x="Event Type", y="Count",
                    title="Collaboration Events",
                    labels={"Event Type": "", "Count": "Number of Events"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Create user activity chart
    if user_activity:
        # Sort by count and take top 10
        users_sorted = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        df = pd.DataFrame({
            "Username": [u[0] for u in users_sorted],
            "Activity": [u[1] for u in users_sorted]
        })

        fig = px.bar(df, x="Username", y="Activity",
                    title="Top 10 Users by Activity",
                    labels={"Username": "", "Activity": "Number of Actions"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Create project activity chart
    if project_activity:
        # Sort by count and take top 10
        projects_sorted = sorted(project_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        df = pd.DataFrame({
            "Project ID": [p[0] for p in projects_sorted],
            "Activity": [p[1] for p in projects_sorted]
        })

        fig = px.bar(df, x="Project ID", y="Activity",
                    title="Top 10 Projects by Activity",
                    labels={"Project ID": "", "Activity": "Number of Actions"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# --- Main Dashboard ---
def main(set_config=False):
    """Main dashboard function.

    Args:
        set_config: Whether to set the page config. Set to False if called from another module
                   that has already set the page config.
    """
    # Page config is now set in the page_config.py module
    # This parameter is kept for backward compatibility
    if set_config:
        try:
            st.set_page_config(
                page_title="Analytics Dashboard",
                page_icon="📊",
                layout="wide"
            )
        except Exception as e:
            # Ignore errors if page config is already set
            pass

    if not PLOTLY_AVAILABLE:
        st.title("📊 Analytics Dashboard")
        st.markdown("*Real-time analytics for Local File Deep Research*")
        st.error("Plotly is not available. Please install it with `pip install plotly` to view charts.")
        return

    # Render header and get time range
    time_range = render_header()

    # Render overview metrics
    stats = render_overview_metrics(time_range)

    # Render user activity
    render_user_activity(stats)

    # Create tabs for different metric categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Performance", "Documents", "Search", "Exports", "Collaboration"
    ])

    with tab1:
        render_performance_metrics(time_range)

    with tab2:
        render_document_metrics(time_range)

    with tab3:
        render_search_metrics(time_range)

    with tab4:
        render_export_metrics(time_range)

    with tab5:
        render_collaboration_metrics(time_range)

    # System metrics at the bottom
    render_system_metrics(time_range)

    # Auto-refresh
    st.markdown(f"""
    <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {REFRESH_INTERVAL * 1000});
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
