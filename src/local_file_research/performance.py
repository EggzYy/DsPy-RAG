"""
Performance optimization utilities for the Local File Deep Research system.
"""

import os
import time
import logging
import functools
import threading
import multiprocessing
import heapq
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast, Generic, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')

# Global cache for memoization
_MEMOIZATION_CACHE = {}
_CACHE_LOCK = threading.RLock()

# --- Caching Vector Store ---
class CachingVectorStore:
    """Vector store with caching."""

    def __init__(self, vector_store: Any, cache_size: int = 100):
        """
        Initialize the caching vector store.

        Args:
            vector_store: Base vector store
            cache_size: Maximum cache size
        """
        self.vector_store = vector_store
        self.cache_size = cache_size
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_order = []  # For LRU eviction

    def search(self, query_embedding: List[float], k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search the vector store with caching.

        Args:
            query_embedding: Query embedding
            k: Number of results
            **kwargs: Additional arguments

        Returns:
            Search results
        """
        # Create cache key
        key = f"{str(query_embedding)[:100]}:{k}:{str(kwargs)}"

        # Check cache
        if key in self.cache:
            # Move to front of LRU list
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self.cache[key]

        # Perform search
        results = self.vector_store.search(query_embedding, k=k, **kwargs)

        # Update cache
        self.cache[key] = results
        self.cache_order.append(key)

        # Evict if necessary
        if len(self.cache) > self.cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]

        return results

    def add(self, *args, **kwargs):
        """
        Add to the vector store and clear cache.

        Args:
            *args: Arguments
            **kwargs: Keyword arguments
        """
        self.vector_store.add(*args, **kwargs)
        self.clear_cache()

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.cache_order.clear()

    def __getattr__(self, name):
        """
        Forward attribute access to the base vector store.

        Args:
            name: Attribute name

        Returns:
            Attribute value
        """
        return getattr(self.vector_store, name)

# --- Priority Queue for Document Processing ---
class PriorityDocumentProcessor:
    """Process documents with priority."""

    def __init__(self, processor_func: Callable[[str], Any], max_workers: int = 4):
        """
        Initialize the priority document processor.

        Args:
            processor_func: Function to process a document
            max_workers: Maximum number of worker threads
        """
        self.processor_func = processor_func
        self.max_workers = max_workers
        self.queue = []  # Priority queue
        self.lock = threading.RLock()
        self.processing = False
        self.results = {}

    def add_document(self, file_path: str, priority: int = 0):
        """
        Add a document to the processing queue.

        Args:
            file_path: Path to the document
            priority: Processing priority (lower is higher priority)
        """
        with self.lock:
            heapq.heappush(self.queue, (priority, file_path))

    def process_all(self) -> Dict[str, Any]:
        """
        Process all documents in the queue.

        Returns:
            Processing results
        """
        with self.lock:
            if self.processing:
                return self.results

            self.processing = True

        try:
            # Extract documents from queue
            documents = []
            with self.lock:
                while self.queue:
                    _, file_path = heapq.heappop(self.queue)
                    documents.append(file_path)

            # Process documents in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {executor.submit(self.processor_func, path): path for path in documents}

                import concurrent.futures
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        self.results[path] = {"status": "success", "result": result}
                    except Exception as e:
                        self.results[path] = {"status": "error", "error": str(e)}

            return self.results
        finally:
            with self.lock:
                self.processing = False

    def clear(self):
        """Clear the queue and results."""
        with self.lock:
            self.queue = []
            self.results = {}

def timeit(func):
    """
    Decorator to measure and log the execution time of a function.

    Args:
        func: Function to time

    Returns:
        Decorated function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

def memoize(max_size: int = 128, ttl: Optional[float] = None):
    """
    Decorator for memoizing function results.

    Args:
        max_size: Maximum number of results to cache
        ttl: Time-to-live in seconds (None for no expiration)

    Returns:
        Decorator function
    """
    def decorator(func):
        cache_key = f"memo_{func.__module__}_{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from the arguments
            key = str(args) + str(sorted(kwargs.items()))

            with _CACHE_LOCK:
                # Initialize cache for this function if it doesn't exist
                if cache_key not in _MEMOIZATION_CACHE:
                    _MEMOIZATION_CACHE[cache_key] = {}

                cache = _MEMOIZATION_CACHE[cache_key]

                # Check if result is in cache and not expired
                if key in cache:
                    result, timestamp = cache[key]
                    if ttl is None or time.time() - timestamp < ttl:
                        return result

                # Compute result if not in cache or expired
                result = func(*args, **kwargs)

                # Store result in cache with timestamp
                cache[key] = (result, time.time())

                # Trim cache if it exceeds max_size
                if len(cache) > max_size:
                    # Remove oldest entries
                    oldest_keys = sorted(cache.keys(), key=lambda k: cache[k][1])[:len(cache) - max_size]
                    for old_key in oldest_keys:
                        del cache[old_key]

                return result

        # Add method to clear the cache
        def clear_cache():
            with _CACHE_LOCK:
                if cache_key in _MEMOIZATION_CACHE:
                    _MEMOIZATION_CACHE[cache_key].clear()

        wrapper.clear_cache = clear_cache
        return wrapper

    return decorator

def parallelize(func, items, max_workers=None, use_processes=False, chunk_size=1):
    """
    Execute a function on multiple items in parallel.

    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of workers (default: CPU count)
        use_processes: Whether to use processes instead of threads
        chunk_size: Number of items to process per worker

    Returns:
        List of results
    """
    if not items:
        return []

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with executor_class(max_workers=max_workers) as executor:
        results = list(executor.map(func, items, chunksize=chunk_size))

    return results

def batch_process(items, batch_size, process_func):
    """
    Process items in batches to avoid memory issues with large datasets.

    Args:
        items: List of items to process
        batch_size: Number of items per batch
        process_func: Function to process each batch

    Returns:
        List of combined results from all batches
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    return results

class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self, total_items, description="Processing"):
        """
        Initialize the progress tracker.

        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total = total_items
        self.description = description
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.RLock()

        logger.info(f"Starting {description}: 0/{total_items} (0%)")

    def update(self, increment=1):
        """
        Update progress.

        Args:
            increment: Number of items completed
        """
        with self.lock:
            self.completed += increment
            percentage = (self.completed / self.total) * 100 if self.total > 0 else 0
            elapsed = time.time() - self.start_time

            # Calculate estimated time remaining
            if self.completed > 0:
                items_per_second = self.completed / elapsed
                remaining_items = self.total - self.completed
                eta = remaining_items / items_per_second if items_per_second > 0 else 0
                eta_str = f", ETA: {eta:.1f}s" if eta > 0 else ""
            else:
                eta_str = ""

            # Log progress at reasonable intervals
            if self.completed == self.total or self.completed % max(1, self.total // 20) == 0:
                logger.info(f"{self.description}: {self.completed}/{self.total} ({percentage:.1f}%), elapsed: {elapsed:.1f}s{eta_str}")

    def finish(self):
        """Mark the operation as complete and log final statistics."""
        elapsed = time.time() - self.start_time
        items_per_second = self.completed / elapsed if elapsed > 0 else 0
        logger.info(f"Completed {self.description}: {self.completed}/{self.total} items in {elapsed:.2f}s ({items_per_second:.2f} items/s)")
        return {
            "total": self.total,
            "completed": self.completed,
            "elapsed_seconds": elapsed,
            "items_per_second": items_per_second
        }

def optimize_embedding_batch_size(embedding_func, sample_texts, min_batch=1, max_batch=64, step=None):
    """
    Find the optimal batch size for embedding by benchmarking different sizes.

    Args:
        embedding_func: Function that takes a list of texts and returns embeddings
        sample_texts: Sample texts to use for benchmarking
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try

    Returns:
        Optimal batch size
    """
    if len(sample_texts) < max_batch:
        # Duplicate texts to have enough samples
        sample_texts = sample_texts * (max_batch // len(sample_texts) + 1)
        sample_texts = sample_texts[:max_batch]

    results = {}

    # Try different batch sizes
    batch_sizes = []
    if step:
        batch_sizes = list(range(min_batch, max_batch + 1, step))
    else:
        batch_sizes = [min_batch, max_batch // 4, max_batch // 2, max_batch]

    for batch_size in batch_sizes:
        if batch_size < min_batch:
            continue

        # Measure time for this batch size
        start_time = time.time()
        for i in range(0, len(sample_texts), batch_size):
            batch = sample_texts[i:i + batch_size]
            embedding_func(batch)
        elapsed = time.time() - start_time

        # Calculate throughput (texts per second)
        throughput = len(sample_texts) / elapsed
        results[batch_size] = throughput

        logger.info(f"Batch size {batch_size}: {throughput:.2f} texts/second")

    # Find batch size with highest throughput
    optimal_batch = max(results.items(), key=lambda x: x[1])[0]
    logger.info(f"Optimal embedding batch size: {optimal_batch}")

    return optimal_batch

def optimize_vector_search(vector_store, query_embedding, min_k: int = 5, max_k: int = 100, step: int = 5):
    """
    Benchmark vector search performance with different top_k values.

    Args:
        vector_store: Vector store to search
        query_embedding: Query embedding to use
        min_k: Minimum k value
        max_k: Maximum k value
        step: Step size

    Returns:
        Dictionary with benchmark results and optimal k value
    """
    results = {}

    for top_k in range(min_k, max_k + 1, step):
        start_time = time.time()
        vector_store.search(query_embedding, top_k=top_k)
        elapsed = time.time() - start_time

        results[top_k] = elapsed
        logger.info(f"Vector search with top_k={top_k}: {elapsed:.4f} seconds")

    # Find optimal k (best time/k ratio)
    optimal_k = min(results.items(), key=lambda x: x[1] / x[0])[0]

    return {
        "results": results,
        "optimal_k": optimal_k,
        "times": list(results.values()),
        "k_values": list(results.keys())
    }
