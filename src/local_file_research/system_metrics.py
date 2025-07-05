"""
System metrics collection for Local File Deep Research.
"""

import os
import time
import logging
import threading
import platform
import psutil
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from analytics import track_metric
except ImportError:
    # Create a simple placeholder if the module is not available
    def track_metric(metric_name, metric_value, dimensions=None):
        logger.debug(f"Tracking metric {metric_name}: {metric_value}")

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_COLLECTION_INTERVAL = 60  # seconds
SYSTEM_METRICS = [
    "cpu_percent",
    "memory_percent",
    "disk_usage_percent",
    "disk_io_read_bytes",
    "disk_io_write_bytes",
    "network_bytes_sent",
    "network_bytes_recv",
    "process_count",
    "thread_count",
    "open_files"
]

# --- Metrics Collection ---
class SystemMetricsCollector:
    """Collect system metrics periodically."""

    def __init__(self, interval: int = DEFAULT_COLLECTION_INTERVAL):
        """
        Initialize the system metrics collector.

        Args:
            interval: Collection interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.last_disk_io = None
        self.last_network_io = None
        self.last_collection_time = None

    def start(self):
        """Start collecting metrics."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()

        logger.info(f"System metrics collection started with interval {self.interval}s")

    def stop(self):
        """Stop collecting metrics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

        logger.info("System metrics collection stopped")

    def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            # Sleep until next collection
            time.sleep(self.interval)

    def _collect_metrics(self):
        """Collect and track system metrics."""
        current_time = time.time()

        # Initialize IO counters on first run
        if self.last_disk_io is None:
            self.last_disk_io = psutil.disk_io_counters()
            self.last_network_io = psutil.net_io_counters()
            self.last_collection_time = current_time
            return

        # Calculate time delta
        time_delta = current_time - self.last_collection_time
        if time_delta <= 0:
            return

        # Collect metrics
        metrics = {}

        # CPU usage
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        metrics["memory_percent"] = memory.percent
        metrics["memory_used_gb"] = memory.used / (1024 ** 3)
        metrics["memory_total_gb"] = memory.total / (1024 ** 3)

        # Disk usage
        disk = psutil.disk_usage("/")
        metrics["disk_usage_percent"] = disk.percent
        metrics["disk_used_gb"] = disk.used / (1024 ** 3)
        metrics["disk_total_gb"] = disk.total / (1024 ** 3)

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_bytes = disk_io.read_bytes - self.last_disk_io.read_bytes
        disk_write_bytes = disk_io.write_bytes - self.last_disk_io.write_bytes
        metrics["disk_io_read_bytes"] = disk_read_bytes / time_delta
        metrics["disk_io_write_bytes"] = disk_write_bytes / time_delta
        self.last_disk_io = disk_io

        # Network I/O
        network_io = psutil.net_io_counters()
        net_bytes_sent = network_io.bytes_sent - self.last_network_io.bytes_sent
        net_bytes_recv = network_io.bytes_recv - self.last_network_io.bytes_recv
        metrics["network_bytes_sent"] = net_bytes_sent / time_delta
        metrics["network_bytes_recv"] = net_bytes_recv / time_delta
        self.last_network_io = network_io

        # Process info
        metrics["process_count"] = len(psutil.pids())

        # Current process info
        process = psutil.Process()
        metrics["process_cpu_percent"] = process.cpu_percent(interval=0.1)
        metrics["process_memory_percent"] = process.memory_percent()
        metrics["thread_count"] = process.num_threads()
        metrics["open_files"] = len(process.open_files())

        # Track metrics
        for name, value in metrics.items():
            track_metric(f"system.{name}", value)

        # Update last collection time
        self.last_collection_time = current_time

        logger.debug(f"Collected system metrics: {metrics}")

# --- Singleton Instance ---
_collector_instance = None

def start_metrics_collection(interval: int = DEFAULT_COLLECTION_INTERVAL):
    """
    Start collecting system metrics.

    Args:
        interval: Collection interval in seconds
    """
    global _collector_instance

    if _collector_instance is None:
        _collector_instance = SystemMetricsCollector(interval)

    _collector_instance.start()

def stop_metrics_collection():
    """Stop collecting system metrics."""
    global _collector_instance

    if _collector_instance is not None:
        _collector_instance.stop()
        _collector_instance = None

def get_system_info() -> Dict[str, Any]:
    """
    Get system information.

    Returns:
        System information dictionary
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        "disk_total_gb": psutil.disk_usage("/").total / (1024 ** 3),
        "hostname": platform.node(),
        "collected_at": datetime.now().isoformat()
    }

    return info
