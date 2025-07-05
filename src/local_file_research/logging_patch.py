"""
Patch for logging to handle Unicode characters properly.
This module provides a function to patch the logging system to handle Unicode characters.
"""

import logging
import sys
import codecs
import io

def patch_logging():
    """
    Patch the logging system to handle Unicode characters properly.
    This function should be called at application startup.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Create a UTF-8 stream wrapper for stdout
    utf8_stream = io.TextIOWrapper(sys.stdout.buffer, 
                                  encoding='utf-8', 
                                  errors='backslashreplace')
    
    # Create handler with the UTF-8 stream
    console_handler = logging.StreamHandler(utf8_stream)
    
    # Create formatter - include timestamp, logger name, level, message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handler to the root logger
    root_logger.addHandler(console_handler)
    
    # Log configuration complete
    root_logger.info("Logging patched with Unicode support")
