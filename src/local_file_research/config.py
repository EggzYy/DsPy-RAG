"""
Configuration settings for the Local File Deep Research application.
Reads from environment variables with sensible defaults.
"""
import os
import logging

# Load environment variables from .env file
try:
    from .env_loader import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    # If env_loader is not available, try to import python-dotenv directly
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # If python-dotenv is not installed, continue without loading .env
        pass

# API Configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8006"))

# Session and Project Configuration
SESSION_PERSIST_DIR = os.environ.get("SESSION_PERSIST_DIR", "sessions")
PROJECT_INDEX_DIR = os.environ.get("PROJECT_INDEX_DIR", "project_indices")

# Vector Store Configuration
USE_FAISS = os.environ.get("USE_FAISS", "true").lower() in ("true", "1", "yes")
FAISS_INDEX_TYPE = os.environ.get("FAISS_INDEX_TYPE", "hnsw")  # Options: "flat", "hnsw", "ivf"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "0"))  # Not used with semantic chunking

# DSPy Default Configuration (can be overridden via env vars in dspy_config.py or API request)
DEFAULT_DSPY_LLM_PROVIDER = os.environ.get("DSPY_LLM_PROVIDER", "openai")
DEFAULT_DSPY_LLM_MODEL = os.environ.get("DSPY_LLM_MODEL", "gpt-4.1-2025-04-14")
DEFAULT_DSPY_TEMPERATURE = float(os.environ.get("DSPY_TEMPERATURE", 0.4))
DEFAULT_DSPY_MAX_TOKENS = int(os.environ.get("DSPY_MAX_TOKENS", 8192))
DEFAULT_DSPY_API_BASE = os.environ.get("DSPY_API_BASE", None)

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "mxbai-embed-large:latest")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Configure logging (console only, no file logging)
try:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
except Exception as e:
    # Avoid duplicate logging configuration errors
    pass

# Document Analysis Configuration
DOCUMENT_ANALYSIS_MODES = ["summarize", "answer", "extract", "chain_of_thought", "analyze"]
DEFAULT_DOCUMENT_ANALYSIS_MODE = os.environ.get("DEFAULT_DOCUMENT_ANALYSIS_MODE", "summarize")

# Research System Configuration
RESEARCH_MODES = ["rag", "multi_iteration"]
DEFAULT_RESEARCH_MODE = os.environ.get("DEFAULT_RESEARCH_MODE", "rag")

REPORT_MODES = ["normal", "chain_of_thought", "enhanced"]
DEFAULT_REPORT_MODE = os.environ.get("DEFAULT_REPORT_MODE", "normal")

# Security Configuration
API_KEY = os.environ.get("API_KEY", None)  # Simple API key for basic protection
ENABLE_CORS = os.environ.get("ENABLE_CORS", "true").lower() in ("true", "1", "yes")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Authentication Configuration
AUTH_DIR = os.environ.get("AUTH_DIR", "auth_data")
JWT_SECRET = os.environ.get("JWT_SECRET", None)  # Will be auto-generated if None
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_DAYS = int(os.environ.get("JWT_EXPIRY_DAYS", "7"))
PASSWORD_SALT_SIZE = int(os.environ.get("PASSWORD_SALT_SIZE", "16"))
MIN_PASSWORD_LENGTH = int(os.environ.get("MIN_PASSWORD_LENGTH", "8"))

# Versioning Configuration
VERSIONS_DIR = os.environ.get("VERSIONS_DIR", "versions")

# Collaboration Configuration
COLLAB_DIR = os.environ.get("COLLAB_DIR", "collaboration")

# Export Configuration
EXPORTS_DIR = os.environ.get("EXPORTS_DIR", "exports")

# Security Configuration
SECURITY_DIR = os.environ.get("SECURITY_DIR", "security")

# Analytics Configuration
ANALYTICS_DIR = os.environ.get("ANALYTICS_DIR", "analytics")
TRACK_ANALYTICS = os.environ.get("TRACK_ANALYTICS", "true").lower() in ("true", "1", "yes")

# --- Security Instructions ---
# For HTTPS: Use a reverse proxy like Nginx or Caddy to handle TLS termination.
# For Authentication: Implement middleware (e.g., using FastAPI's security utilities)
#   - Basic Auth, API Keys, or OAuth2/JWT depending on requirements.
#   - Example: Add dependency `python-jose[cryptography]` and `passlib[bcrypt]`
#   - See FastAPI security docs: https://fastapi.tiangolo.com/tutorial/security/

# --- Backup Instructions ---
# Regularly back up the directory specified by SESSION_PERSIST_DIR.
# Use tools like rsync, cron jobs, or cloud storage backup solutions.

# --- Scaling Instructions ---
# API Scaling: Run multiple instances of the FastAPI app using a process manager
#              (e.g., Gunicorn, Uvicorn workers) behind a load balancer.
# Vector Store Scaling: FAISS is now integrated for improved performance.
#                       For even larger scale, consider Milvus, Weaviate, or Qdrant.
# Session State: If scaling API instances, ensure session persistence (vector store files)
#                is accessible by all instances (e.g., shared network storage, S3).
