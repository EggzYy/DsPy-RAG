"""
Serialization utilities for handling DSPy objects in API responses.
"""

import json
import logging
from typing import Any, Dict, List, Union

# Configure logging
logger = logging.getLogger(__name__)

def is_dspy_object(obj: Any) -> bool:
    """Check if an object is a DSPy-related object."""
    # Check for None
    if obj is None:
        return False

    # Check class name
    class_name = obj.__class__.__name__
    module_name = getattr(obj.__class__, "__module__", "")

    # Check if it's from dspy module
    if "dspy" in module_name:
        logger.debug(f"Found DSPy object: {class_name} from {module_name}")
        return True

    # Check for SignatureMeta specifically
    if class_name == "SignatureMeta" or "SignatureMeta" in str(type(obj)):
        logger.debug(f"Found SignatureMeta object: {obj}")
        return True

    # Check for type objects that might be DSPy-related
    if isinstance(obj, type) and hasattr(obj, "__module__") and "dspy" in getattr(obj, "__module__", ""):
        logger.debug(f"Found DSPy type object: {obj}")
        return True

    return False

def serialize_dspy_object(obj: Any) -> Dict[str, Any]:
    """Serialize a DSPy object to a dictionary."""
    try:
        # Handle None case
        if obj is None:
            return None

        # For SignatureMeta objects, return a simplified representation
        if "SignatureMeta" in str(type(obj)) or (isinstance(obj, type) and hasattr(obj, "__module__") and "dspy" in getattr(obj, "__module__", "")):
            return {
                "type": "DSPySignature",
                "name": getattr(obj, "__name__", str(obj)),
                "description": getattr(obj, "__doc__", "No description available")
            }

        # Try to convert to dict if it has __dict__
        if hasattr(obj, "__dict__"):
            # Filter out private attributes and recursively serialize values
            serialized_dict = {}
            for k, v in obj.__dict__.items():
                if not k.startswith("_"):
                    # Recursively serialize any nested DSPy objects
                    if is_dspy_object(v):
                        serialized_dict[k] = serialize_dspy_object(v)
                    elif isinstance(v, dict):
                        # Handle dictionaries with potential DSPy objects
                        serialized_dict[k] = {
                            dk: serialize_dspy_object(dv) if is_dspy_object(dv) else dv
                            for dk, dv in v.items()
                        }
                    elif isinstance(v, (list, tuple)):
                        # Handle lists with potential DSPy objects
                        serialized_dict[k] = [
                            serialize_dspy_object(item) if is_dspy_object(item) else item
                            for item in v
                        ]
                    else:
                        # Use the value as is for non-DSPy objects
                        serialized_dict[k] = v
            return serialized_dict

        # For other DSPy objects, convert to string with more metadata
        return {
            "type": obj.__class__.__name__,
            "module": getattr(obj.__class__, "__module__", "unknown"),
            "value": str(obj)
        }
    except Exception as e:
        logger.warning(f"Error serializing DSPy object {type(obj)}: {e}")
        return {"type": "UnserializableDSPyObject", "value": str(obj)}

def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to be JSON serializable.
    Handles DSPy objects, nested dictionaries, lists, and other types.
    """
    if obj is None:
        return None

    # Handle basic types that are already JSON serializable
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle DSPy objects
    if is_dspy_object(obj):
        return serialize_dspy_object(obj)

    # Handle type objects (classes)
    if isinstance(obj, type):
        return {
            "type": "Class",
            "name": obj.__name__,
            "module": getattr(obj, "__module__", "unknown"),
            "doc": getattr(obj, "__doc__", "No documentation available")
        }

    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]

    # Handle sets by converting to list
    if isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]

    # Handle bytes by decoding to string
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8', errors='replace')
        except Exception:
            return str(obj)

    # Handle objects with __dict__
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        try:
            return {k: make_json_serializable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception as e:
            logger.warning(f"Error serializing object {type(obj)}: {e}")
            return str(obj)

    # Convert anything else to string
    try:
        return str(obj)
    except Exception as e:
        logger.warning(f"Error converting object to string: {e}")
        return f"<Unserializable object of type {type(obj).__name__}>"

class DSPyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles DSPy objects."""
    def default(self, obj):
        return make_json_serializable(obj)
