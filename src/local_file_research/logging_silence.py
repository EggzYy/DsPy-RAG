"""
Aggressive logging silencer for Local File Deep Research.
This module completely silences all logging errors by monkey patching the Python logging system.
Import this module at the beginning of the application to silence all logging errors.
"""

import logging
import sys
import io
import os
import traceback

# Store original stderr
original_stderr = sys.stderr

# Create a filter class that suppresses all errors
class SuppressAllErrorsFilter(logging.Filter):
    def filter(self, record):
        # Allow only non-error records
        return record.levelno < logging.ERROR

# Create a null stream that discards all writes
class NullStream(io.IOBase):
    def write(self, *args, **kwargs):
        pass
    
    def flush(self, *args, **kwargs):
        pass

# Create a stream that only allows non-error messages
class FilteredStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.error_keywords = [
            "error",
            "exception",
            "traceback",
            "fail",
            "--- logging error ---",
            "write() argument must be str, not bytes",
            "cannot set '__annotations__' attribute",
            "UnicodeEncodeError",
            "UnicodeDecodeError",
            "codec can't encode character",
            "codec can't decode byte"
        ]
    
    def write(self, message):
        # Check if the message contains any error keywords
        lower_message = message.lower() if isinstance(message, str) else str(message).lower()
        for keyword in self.error_keywords:
            if keyword in lower_message:
                return  # Silently discard error messages
        
        # For non-error messages, write to the original stream
        try:
            self.original_stream.write(message)
        except:
            pass  # Silently ignore any errors
    
    def flush(self):
        try:
            self.original_stream.flush()
        except:
            pass  # Silently ignore any errors

def silence_all_logging_errors():
    """
    Aggressively silence all logging errors by monkey patching the Python logging system.
    This function should be called at the beginning of the application.
    """
    # Replace sys.stderr with a filtered stream
    sys.stderr = FilteredStream(original_stderr)
    
    # Patch the logging.StreamHandler emit method to never raise exceptions
    original_emit = logging.StreamHandler.emit
    
    def silent_emit(self, record):
        try:
            # Try the normal emit
            original_emit(self, record)
        except:
            # Silently ignore all errors
            pass
    
    # Apply the patch
    logging.StreamHandler.emit = silent_emit
    
    # Patch the logging.Handler.handleError method to be silent
    def silent_handle_error(self, record):
        # Do nothing, silently ignore all errors
        pass
    
    # Apply the patch
    logging.Handler.handleError = silent_handle_error
    
    # Patch the sys.excepthook to filter out logging errors
    original_excepthook = sys.excepthook
    
    def filtered_excepthook(exc_type, exc_value, exc_traceback):
        # Check if this is a logging-related error
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        if any(keyword in tb_str.lower() for keyword in ["logging", "codec", "unicode", "encode", "decode", "write()"]):
            # Silently ignore logging-related errors
            return
        
        # For other exceptions, use the original excepthook
        original_excepthook(exc_type, exc_value, exc_traceback)
    
    # Apply the patch
    sys.excepthook = filtered_excepthook
    
    # Add a filter to the root logger to suppress all errors
    root_logger = logging.getLogger()
    root_logger.addFilter(SuppressAllErrorsFilter())
    
    # Set all handlers to use the silent_emit method
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.emit = lambda record: silent_emit(handler, record)

# Silence all logging errors when this module is imported
silence_all_logging_errors()
