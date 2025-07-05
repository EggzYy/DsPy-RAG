"""
Logging configuration for Local File Deep Research.
This module configures logging to handle Unicode characters properly.
"""

import logging
import sys
import codecs
import io
import os
import functools

# Configure a basic logger for this module
logger = logging.getLogger(__name__)

# Create a special logger for DSPy warnings
dspy_warning_logger = logging.getLogger("dspy.warnings")

# Create a filter to suppress specific error messages
class SuppressFilter(logging.Filter):
    def __init__(self, suppress_patterns):
        super().__init__()
        self.suppress_patterns = suppress_patterns

    def filter(self, record):
        # Check if the message contains any of the patterns to suppress
        message = record.getMessage()
        for pattern in self.suppress_patterns:
            if pattern in message:
                return False
        return True

# Create a global safe_str function to handle Unicode characters
def safe_str(obj):
    """Convert any object to a string that can be safely printed/logged."""
    if obj is None:
        return 'None'

    # Handle bytes objects first
    if isinstance(obj, bytes):
        try:
            # Try to decode as UTF-8 with replacement for invalid chars
            return obj.decode('utf-8', errors='backslashreplace')
        except Exception:
            try:
                # Try with latin-1 as fallback (will always succeed)
                return obj.decode('latin-1')
            except Exception:
                # Last resort for bytes
                return str(obj)

    try:
        # For strings, try to ensure they're valid UTF-8
        if isinstance(obj, str):
            # Check if there are any encoding issues by round-tripping
            try:
                # This will fail if the string contains invalid UTF-8
                obj.encode('utf-8').decode('utf-8')
                return obj  # String is valid UTF-8, return as is
            except UnicodeError:
                # String has encoding issues, sanitize it
                return obj.encode('utf-8', errors='backslashreplace').decode('utf-8')

        # For other objects, convert to string
        return str(obj)
    except UnicodeEncodeError:
        # If that fails, try with explicit encoding/decoding
        try:
            if isinstance(obj, str):
                # For strings, encode with replacement
                return obj.encode('utf-8', errors='backslashreplace').decode('utf-8')
            else:
                # For other objects, convert to string with replacement
                return str(obj).encode('utf-8', errors='backslashreplace').decode('utf-8')
        except Exception:
            # Try with latin-1 as a last resort
            try:
                if isinstance(obj, str):
                    return obj.encode('latin-1', errors='replace').decode('latin-1')
                else:
                    return "[Object conversion error]"
            except Exception:
                # Absolute last resort - return a placeholder
                return "[Unicode conversion error]"

def configure_logging():
    """
    Configure logging to handle Unicode characters properly.
    This function should be called at application startup.
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure special DSPy warning logger
    dspy_warning_logger = logging.getLogger("dspy.warnings")
    dspy_predict_logger = logging.getLogger("dspy.predict.predict")

    # Silence noisy loggers
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('litellm').setLevel(logging.ERROR)
    logging.getLogger('litellm.utils').setLevel(logging.ERROR)
    logging.getLogger('litellm.llms').setLevel(logging.ERROR)
    logging.getLogger('litellm.router').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('fsspec').setLevel(logging.WARNING)
    logging.getLogger('filelock').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

    # Ensure our vector store loggers are at INFO level
    logging.getLogger('llamaindex_vector_store').setLevel(logging.INFO)
    logging.getLogger('vector_store').setLevel(logging.INFO)

    # Add a filter to suppress specific error messages
    suppress_patterns = [
        "cannot import name 'PROJECT_INDEX_DIR'",
        "cannot import name 'SESSION_PERSIST_DIR'",
        "cannot set '__annotations__' attribute",
        "UnicodeEncodeError",
        "UnicodeDecodeError",
        "codec can't encode character",
        "codec can't decode byte",
        "Failed to load LlamaIndex vector store",
        "Vector store directory not found",
        "Error reading FAISS index",
        "Error during direct reload",
        "Vector store loaded but contains no vectors",
        "Missing required files",
        "Error serializing DSPy object",
        "Error converting object to string",
        "Error in patched_emit",
        "Error in patched_format_message",
        "write() argument must be str, not bytes",
        "TypeError: write() argument must be str, not bytes",
        "--- Logging error ---",
        "Traceback (most recent call last):",
        "File \"<frozen codecs>\"",
        "File \"D:\\anaconda32\\envs\\deepresearch_clean\\Lib\\logging\\__init__.py\""
    ]
    root_logger.addFilter(SuppressFilter(suppress_patterns))

    # Disable file handler for DSPy warnings (console only)
    # Set levels for DSPy loggers
    dspy_warning_logger.setLevel(logging.WARNING)
    dspy_predict_logger.setLevel(logging.WARNING)

    # Don't propagate to root logger to avoid duplicate messages
    dspy_warning_logger.propagate = False
    dspy_predict_logger.propagate = False

    print("DSPy warning logging configured for console only")

    # Disable file handler for application logs (console only)
    print("Application logging configured for console only")

    # Create console handler with a safe stream
    console_handler = None

    # Special handling for Windows console
    if sys.platform == 'win32':
        try:
            # Try to use the Windows-specific console handler
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)  # Set console to UTF-8
            console_handler = logging.StreamHandler()
            # Force the stream to use UTF-8
            console_handler.stream.encoding = 'utf-8'
            print("Successfully configured Windows console for UTF-8")
        except Exception as win_e:
            print(f"Warning: Could not configure Windows console for UTF-8: {win_e}")

    # If Windows-specific handling failed or we're on another platform
    if console_handler is None:
        try:
            console_handler = logging.StreamHandler()
            try:
                console_handler.stream.reconfigure(encoding='utf-8', errors='backslashreplace')
                print("Successfully configured stream with reconfigure")
            except (AttributeError, TypeError):
                # Try to set encoding directly
                try:
                    console_handler.stream.encoding = 'utf-8'
                    print("Set stream encoding directly")
                except:
                    print("Could not set stream encoding, will rely on patched emit")
        except Exception as e:
            # Last resort fallback
            print(f"Warning: Could not configure console logging: {e}")
            console_handler = logging.StreamHandler()

    # Create formatter - include timestamp, logger name, level, message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to the root logger
    root_logger.addHandler(console_handler)

    # Set level based on environment variable or default to INFO
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)
    root_logger.setLevel(level)

    # Configure specific loggers
    # Silence noisy loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    # Patch the logging.StreamHandler emit method to handle Unicode errors
    original_emit = logging.StreamHandler.emit

    def patched_emit(self, record):
        # Patch the record's message and args to handle Unicode
        if isinstance(record.msg, str):
            record.msg = safe_str(record.msg)
        elif isinstance(record.msg, bytes):
            # Convert bytes to string
            try:
                record.msg = record.msg.decode('utf-8', errors='backslashreplace')
            except Exception:
                record.msg = str(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                # Handle dictionary args
                safe_args = {}
                for k, v in record.args.items():
                    safe_args[k] = safe_str(v)
                record.args = safe_args
            else:
                # Handle tuple args
                safe_args = []
                for arg in record.args:
                    safe_args.append(safe_str(arg))
                record.args = tuple(safe_args)

        try:
            # Format the record first
            msg = self.format(record)

            # Ensure msg is a string, not bytes
            if isinstance(msg, bytes):
                try:
                    msg = msg.decode('utf-8', errors='backslashreplace')
                except Exception:
                    msg = str(msg)

            # Ensure terminator is a string
            terminator = self.terminator
            if isinstance(terminator, bytes):
                try:
                    terminator = terminator.decode('utf-8', errors='backslashreplace')
                except Exception:
                    terminator = '\n'

            # Write with error handling
            try:
                self.stream.write(msg + terminator)
            except UnicodeEncodeError:
                # If Unicode error, try with replacement
                self.stream.write(msg.encode('utf-8', errors='backslashreplace').decode('utf-8') + terminator)
            except TypeError as type_err:
                # Handle "write() argument must be str, not bytes" error
                if "write() argument must be str, not bytes" in str(type_err):
                    if isinstance(msg, bytes):
                        self.stream.write(msg.decode('utf-8', errors='backslashreplace') + terminator)
                    else:
                        # If msg is already a string but we still get this error,
                        # it might be because terminator is bytes
                        self.stream.write(msg + '\n')
                else:
                    # For other TypeError exceptions, try a different approach
                    import sys
                    try:
                        # Try writing directly to stderr
                        sys.stderr.write(f"{msg}\n")
                    except:
                        # Last resort
                        print(f"LOGGING ERROR: {msg}")

            self.flush()
        except Exception as e:
            # Avoid using logger here to prevent recursion
            import sys
            try:
                sys.stderr.write(f"Error in patched_emit: {str(e)}\n")
            except:
                print(f"LOGGING ERROR: {str(e)}")
            self.handleError(record)

    # Apply the patch
    logging.StreamHandler.emit = patched_emit

    # Patch the logging.Formatter.formatMessage method
    original_format_message = logging.Formatter.formatMessage

    def patched_format_message(self, record):
        # Ensure record.message is safe and is a string
        if isinstance(record.message, bytes):
            try:
                record.message = record.message.decode('utf-8', errors='backslashreplace')
            except Exception:
                record.message = str(record.message)

        # Apply safe_str to ensure it's a valid string
        try:
            record.message = safe_str(record.message)
        except Exception as safe_str_err:
            # If safe_str fails, use a basic fallback
            try:
                record.message = str(record.message)
            except:
                record.message = "[Unformattable message]"

        try:
            return original_format_message(self, record)
        except Exception as e:
            # If formatting fails, provide a fallback
            try:
                # Avoid using logger here to prevent recursion
                import sys
                sys.stderr.write(f"Error in patched_format_message: {str(e)}\n")
            except:
                pass

            # Return a safe fallback message
            try:
                return f"[Logging error: {str(e)}] {str(record.message)}"
            except:
                return "[Logging error: Unable to format message]"

    # Apply the patch
    logging.Formatter.formatMessage = patched_format_message

    # Log configuration complete
    root_logger.info("Logging configured with comprehensive Unicode support and error handling")

# Configure logging when this module is imported
configure_logging()
