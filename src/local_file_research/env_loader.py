"""
Environment variable loader for Local File Deep Research.
This module loads environment variables from .env files.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

def load_dotenv(env_file: str = ".env") -> bool:
    """
    Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file

    Returns:
        True if the .env file was loaded successfully, False otherwise
    """
    try:
        # Try to import python-dotenv
        try:
            from dotenv import load_dotenv as dotenv_load

            # Get the absolute path to the .env file
            env_path = Path(env_file).resolve()

            # Check if the .env file exists
            if not env_path.exists():
                logger.warning(f".env file not found at {env_path}")
                return False

            # Load the .env file
            result = dotenv_load(dotenv_path=env_path)

            if result:
                logger.info(f"Loaded environment variables from {env_path}")
            else:
                logger.warning(f"No environment variables loaded from {env_path}")

            return result

        except ImportError:
            # If python-dotenv is not installed, use a simple implementation
            logger.warning("python-dotenv not installed. Using simple .env loader.")
            return _simple_load_dotenv(env_file)

    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        return False

def _simple_load_dotenv(env_file: str) -> bool:
    """
    Simple implementation of dotenv loading without dependencies.

    Args:
        env_file: Path to the .env file

    Returns:
        True if the .env file was loaded successfully, False otherwise
    """
    try:
        # Get the absolute path to the .env file
        env_path = Path(env_file).resolve()

        # Check if the .env file exists
        if not env_path.exists():
            logger.warning(f".env file not found at {env_path}")
            return False

        # Read the .env file
        with open(env_path, "r") as f:
            lines = f.readlines()

        # Parse each line
        loaded_vars = 0
        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse key-value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value and value[0] == value[-1] and value[0] in ["'", "\""]:
                    value = value[1:-1]

                # Set environment variable if not already set
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded_vars += 1

        logger.info(f"Loaded {loaded_vars} environment variables from {env_path}")
        return loaded_vars > 0

    except Exception as e:
        logger.error(f"Error in simple .env loader: {e}")
        return False

def get_env_vars() -> Dict[str, Any]:
    """
    Get all environment variables used by the application.

    Returns:
        Dictionary of environment variables and their values
    """
    env_vars = {}

    # API Configuration
    env_vars["API_HOST"] = os.environ.get("API_HOST", "0.0.0.0")
    env_vars["API_PORT"] = int(os.environ.get("API_PORT", "8006"))

    # Session Configuration
    env_vars["SESSION_PERSIST_DIR"] = os.environ.get("SESSION_PERSIST_DIR", "sessions")

    # Vector Store Configuration
    env_vars["USE_FAISS"] = os.environ.get("USE_FAISS", "true").lower() in ("true", "1", "yes")
    env_vars["FAISS_INDEX_TYPE"] = os.environ.get("FAISS_INDEX_TYPE", "hnsw")
    env_vars["CHUNK_SIZE"] = int(os.environ.get("CHUNK_SIZE", "1024"))
    env_vars["CHUNK_OVERLAP"] = int(os.environ.get("CHUNK_OVERLAP", "0"))

    # DSPy Configuration
    env_vars["DSPY_LLM_PROVIDER"] = os.environ.get("DSPY_LLM_PROVIDER", "openai")
    env_vars["DSPY_LLM_MODEL"] = os.environ.get("DSPY_LLM_MODEL", "gpt-4.1-2025-04-14")
    env_vars["DSPY_TEMPERATURE"] = float(os.environ.get("DSPY_TEMPERATURE", "0.4"))
    env_vars["DSPY_MAX_TOKENS"] = int(os.environ.get("DSPY_MAX_TOKENS", "8192"))
    env_vars["DSPY_API_BASE"] = os.environ.get("DSPY_API_BASE", None)

    # Embedding Configuration
    env_vars["EMBEDDING_MODEL_NAME"] = os.environ.get("EMBEDDING_MODEL_NAME", "mxbai-embed-large:latest")
    env_vars["EMBEDDING_DIMENSION"] = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))
    env_vars["OLLAMA_API_BASE"] = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    env_vars["EMBEDDING_CACHE_DIR"] = os.environ.get("EMBEDDING_CACHE_DIR", "embeddings")

    # Logging Configuration
    env_vars["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO").upper()
    env_vars["LOG_FILE"] = os.environ.get("LOG_FILE", "local_file_research.log")

    # Document Analysis Configuration
    env_vars["DEFAULT_DOCUMENT_ANALYSIS_MODE"] = os.environ.get("DEFAULT_DOCUMENT_ANALYSIS_MODE", "summarize")

    # Research System Configuration
    env_vars["DEFAULT_RESEARCH_MODE"] = os.environ.get("DEFAULT_RESEARCH_MODE", "rag")
    env_vars["DEFAULT_REPORT_MODE"] = os.environ.get("DEFAULT_REPORT_MODE", "normal")

    # Security Configuration
    env_vars["API_KEY"] = os.environ.get("API_KEY", None)
    env_vars["ENABLE_CORS"] = os.environ.get("ENABLE_CORS", "true").lower() in ("true", "1", "yes")
    env_vars["CORS_ORIGINS"] = os.environ.get("CORS_ORIGINS", "*").split(",")

    # Authentication Configuration
    env_vars["AUTH_DIR"] = os.environ.get("AUTH_DIR", "auth_data")
    env_vars["JWT_SECRET"] = os.environ.get("JWT_SECRET", None)
    env_vars["JWT_ALGORITHM"] = os.environ.get("JWT_ALGORITHM", "HS256")
    env_vars["JWT_EXPIRY_DAYS"] = int(os.environ.get("JWT_EXPIRY_DAYS", "7"))
    env_vars["PASSWORD_SALT_SIZE"] = int(os.environ.get("PASSWORD_SALT_SIZE", "16"))
    env_vars["MIN_PASSWORD_LENGTH"] = int(os.environ.get("MIN_PASSWORD_LENGTH", "8"))

    # Directory Configuration
    env_vars["BASE_DIR"] = os.environ.get("BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env_vars["DATABASE_DIR"] = os.environ.get("DATABASE_DIR", os.path.join(env_vars["BASE_DIR"], "database"))
    env_vars["DOCUMENTS_DIR"] = os.environ.get("DOCUMENTS_DIR", os.path.join(env_vars["BASE_DIR"], "documents"))
    env_vars["VERSIONS_DIR"] = os.environ.get("VERSIONS_DIR", os.path.join(env_vars["BASE_DIR"], "versions"))
    env_vars["COLLAB_DIR"] = os.environ.get("COLLAB_DIR", os.path.join(env_vars["BASE_DIR"], "collaboration"))
    env_vars["EXPORTS_DIR"] = os.environ.get("EXPORTS_DIR", os.path.join(env_vars["BASE_DIR"], "exports"))
    env_vars["SECURITY_DIR"] = os.environ.get("SECURITY_DIR", os.path.join(env_vars["BASE_DIR"], "security"))
    env_vars["ANALYTICS_DIR"] = os.environ.get("ANALYTICS_DIR", os.path.join(env_vars["BASE_DIR"], "analytics"))

    # Analytics Configuration
    env_vars["TRACK_ANALYTICS"] = os.environ.get("TRACK_ANALYTICS", "true").lower() in ("true", "1", "yes")

    # Testing Configuration
    env_vars["TEST_TIMEOUT"] = int(os.environ.get("TEST_TIMEOUT", "10"))
    env_vars["MAX_WORKERS"] = int(os.environ.get("MAX_WORKERS", "4"))
    env_vars["API_URL"] = os.environ.get("API_URL", "http://localhost:8000")

    return env_vars

def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    Get an environment variable.

    Args:
        key: Name of the environment variable
        default: Default value if the environment variable is not set

    Returns:
        Value of the environment variable or the default value
    """
    return os.environ.get(key, default)

# Load environment variables when the module is imported
load_dotenv()
