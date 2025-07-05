# filename: src/local_file_research/api_llamaindex_main.py
"""
Main API file for the local file research system using LlamaIndex.
"""

# Import the logging silencer first to suppress all logging errors
try:
    from .logging_silence import silence_all_logging_errors
except ImportError:
    try:
        from src.local_file_research.logging_silence import silence_all_logging_errors
    except ImportError:
        print("Warning: Could not import logging_silence module")

import os
import logging
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import time
import json
import sys # +++ ADD THIS +++
from pathlib import Path

# Apply LiteLLM patch to fix __annotations__ error
try:
    from .litellm_patch import apply_patch, patch_dspy
    apply_patch()
    patch_dspy()
    print("Applied LiteLLM patch to fix __annotations__ error")
except ImportError:
    try:
        from src.local_file_research.litellm_patch import apply_patch, patch_dspy
        apply_patch()
        patch_dspy()
        print("Applied LiteLLM patch to fix __annotations__ error")
    except ImportError:
        print("Warning: Could not import litellm_patch module")

try:
    from .logging_config import configure_logging
    configure_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# Set up logging (console only, no file logging)
# Logging config should ideally be centralized, perhaps in logging_config.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- MODIFY DSPY INITIALIZATION SECTION ---
# ** CRITICAL: Initialize DSPy ONCE before creating the FastAPI app **
try:
    from src.local_file_research.dspy_config import initialize_dspy, DSPY_CONFIGURED
    logger.info("Attempting DSPy Initialization...")
    initialize_dspy() # Call initialization early
    if DSPY_CONFIGURED:
        logger.info("DSPy Initialization successful.")
    else:
        # +++ ADD EXIT ON FAILURE +++
        logger.error("CRITICAL: DSPy Initialization FAILED. Check dspy_config logs. Agents will not function. Exiting application.")
        #sys.exit("DSPy initialization failed, cannot start application.")
except Exception as init_err:
    logger.error(f"CRITICAL ERROR during DSPy initialization: {init_err}", exc_info=True)
    # +++ ADD EXIT ON FAILURE +++
    #sys.exit("DSPy initialization failed, cannot start application.")
# --- END OF MODIFIED DSPY INITIALIZATION ---

# Create FastAPI app
app = FastAPI(title="Local File Research API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and set up routes
from src.local_file_research.api_llamaindex import setup_llamaindex_routes
from src.local_file_research.api_documents import setup_document_routes
from src.local_file_research.api_projects import setup_project_routes
from src.local_file_research.api_sessions import setup_session_routes
from src.local_file_research.api_admin import setup_admin_routes
from src.local_file_research.api_auth import setup_auth_routes

# Set up routes
setup_llamaindex_routes(app)
setup_document_routes(app)
setup_project_routes(app)
setup_session_routes(app)
setup_admin_routes(app)
setup_auth_routes(app)

# Initialize document registry
from src.local_file_research.document_registry import initialize_document_registry
initialize_document_registry()

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    # Use the serialization utility to handle potential DSPy objects in the exception details
    from .serialization_utils import make_json_serializable # Import utility here
    serializable_detail = make_json_serializable({"detail": f"Internal server error: {str(exc)}"})
    return JSONResponse(
        status_code=500,
        content=serializable_detail
    )


# --- MODIFY HEALTH CHECK ---
@app.get("/health", tags=["Health"])
async def health_check():
    """ Health check endpoint. """
    from src.local_file_research.dspy_config import DSPY_CONFIGURED # Check current status
    return {"status": "ok", "version": "1.0.0", "dspy_configured": DSPY_CONFIGURED}
# --- END OF MODIFIED HEALTH CHECK ---

# Add migration endpoint
@app.post("/migrate-to-llamaindex", tags=["Admin"])
async def migrate_to_llamaindex():
    """
    Migrate existing vector stores to LlamaIndex format.
    """
    try:
        from .migrate_to_llamaindex import main as migrate_main
        migrate_main()
        return {"status": "success", "message": "Migration complete"}
    except Exception as e:
        logger.error(f"Error in migrate_to_llamaindex endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")

def run_api():
    """ Run the API server. """
    import os
    port = int(os.environ.get("API_PORT", "8006"))
    # +++ ENSURE ALL DIRS ARE CREATED +++
    os.makedirs("project_indices", exist_ok=True)
    os.makedirs("sessions", exist_ok=True)
    os.makedirs("storage", exist_ok=True)
    os.makedirs("storage/projects", exist_ok=True)
    os.makedirs("storage/registry", exist_ok=True) # Ensure registry dir exists
    os.makedirs("auth_data", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    logger.info(f"Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_api()