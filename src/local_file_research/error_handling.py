"""
Error handling and recovery mechanisms for the Local File Deep Research system.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')

class ResearchError(Exception):
    """Base exception class for research-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class IndexingError(ResearchError):
    """Exception raised for errors during file indexing."""
    pass

class VectorStoreError(ResearchError):
    """Exception raised for errors related to vector store operations."""
    pass

class DSPyConfigError(ResearchError):
    """Exception raised for errors in DSPy configuration."""
    pass

class DocumentAnalysisError(ResearchError):
    """Exception raised for errors during document analysis."""
    pass

class APIError(ResearchError):
    """Exception raised for API-related errors."""
    pass

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,), logger_func: Optional[Callable] = None):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (e.g., 2.0 means delay doubles after each retry)
        exceptions: Tuple of exceptions to catch and retry
        logger_func: Optional function to use for logging
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger_func or logger.warning
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        log(f"Final attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}")
                        raise
                    
                    log(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. Retrying in {current_delay:.2f}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
        
        return wrapper
    return decorator

def safe_execute(default_return: Any = None, log_exception: bool = True):
    """
    Decorator to safely execute a function and return a default value on exception.
    
    Args:
        default_return: Value to return if an exception occurs
        log_exception: Whether to log the exception
    
    Returns:
        Decorated function that catches exceptions
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_exception:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Exception details: {traceback.format_exc()}")
                return default_return
        
        return wrapper
    return decorator

def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Format an exception into a standardized error response.
    
    Args:
        error: The exception to format
    
    Returns:
        Dictionary with error details
    """
    if isinstance(error, ResearchError):
        response = {
            "error": error.__class__.__name__,
            "message": error.message,
            "details": error.details
        }
    else:
        response = {
            "error": error.__class__.__name__,
            "message": str(error),
            "details": {"traceback": traceback.format_exc()}
        }
    
    return response

def handle_dspy_errors(func):
    """
    Decorator to handle DSPy-specific errors.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function with DSPy error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            if "dspy" in str(e).lower():
                logger.error(f"DSPy import error: {e}")
                raise DSPyConfigError(f"DSPy is not properly installed: {e}")
            raise
        except Exception as e:
            if "dspy" in str(e).lower() or "llm" in str(e).lower() or "model" in str(e).lower():
                logger.error(f"DSPy error: {e}")
                raise DSPyConfigError(f"Error in DSPy configuration or execution: {e}")
            raise
    
    return wrapper

def handle_vector_store_errors(func):
    """
    Decorator to handle vector store errors.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function with vector store error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "faiss" in str(e).lower():
                logger.error(f"FAISS error: {e}")
                raise VectorStoreError(f"FAISS vector store error: {e}")
            elif "vector" in str(e).lower() or "embedding" in str(e).lower() or "index" in str(e).lower():
                logger.error(f"Vector store error: {e}")
                raise VectorStoreError(f"Vector store operation failed: {e}")
            raise
    
    return wrapper

def recover_from_indexing_error(error: Exception, partial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attempt to recover from indexing errors by returning partial results.
    
    Args:
        error: The exception that occurred
        partial_results: Partial indexing results collected before the error
    
    Returns:
        Partial results that can be used despite the error
    """
    logger.warning(f"Recovering from indexing error: {error}")
    logger.info(f"Returning {len(partial_results)} partial results")
    
    # Add error information to each result
    for result in partial_results:
        result["_error_during_indexing"] = str(error)
    
    return partial_results

def recover_from_dspy_error(error: Exception, fallback_mode: str = "basic") -> Dict[str, Any]:
    """
    Attempt to recover from DSPy errors by using fallback mechanisms.
    
    Args:
        error: The exception that occurred
        fallback_mode: Fallback mode to use ("basic", "none", or "mock")
    
    Returns:
        Dictionary with fallback configuration
    """
    logger.warning(f"Recovering from DSPy error: {error}")
    
    if fallback_mode == "basic":
        # Use basic text processing without DSPy
        return {
            "mode": "basic",
            "message": f"Using basic text processing due to DSPy error: {error}"
        }
    elif fallback_mode == "mock":
        # Use mock responses for testing
        return {
            "mode": "mock",
            "message": f"Using mock responses due to DSPy error: {error}"
        }
    else:
        # No fallback, just return error information
        return {
            "mode": "none",
            "error": str(error),
            "message": "No fallback available for this operation"
        }
