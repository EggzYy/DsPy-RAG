# filename: src/local_file_research/api_llamaindex.py

from fastapi import APIRouter, Depends, HTTPException, Query, Body, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import os
import json
import uuid
import time
from datetime import datetime

# Import local modules
from src.local_file_research.pipeline_llamaindex import build_index, search_index, deep_research
from src.local_file_research.llamaindex_vector_store import LlamaIndexVectorStore, get_llamaindex_vector_store
from src.local_file_research.document_manager import save_file_to_database
from src.local_file_research.storage_manager import read_document_content
from src.local_file_research.auth import get_current_user
from src.local_file_research.models import ResearchRequest, ResearchResponse, IndexingRequest, IndexingResponse
from src.local_file_research.research_system import ResearchSystem # Ensure ResearchSystem is imported
from src.local_file_research.serialization_utils import make_json_serializable # Import utility

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# In-memory cache for vector stores
vector_store_cache = {}

@router.post("/research", response_model=ResearchResponse)
async def research_endpoint(
    request: ResearchRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Research endpoint that performs vector search and optionally runs LLM analysis.
    Returns a comprehensive report string and detailed findings.
    """
    try:
        logger.info(f"Received research request from user '{current_user}' for session '{request.session_id}' with query '{request.query}', research_mode='{request.research_mode}', report_mode='{request.report_mode}', chain='{request.chain}', project_filter='{request.project_filter}'.")

        # Get vector store (keep this logic)
        vector_store = None
        if request.session_id in vector_store_cache:
            logger.info(f"Using cached vector store for session '{request.session_id}'.")
            vector_store = vector_store_cache[request.session_id]
        else:
            logger.info(f"Cache miss. Loading vector store from disk for session '{request.session_id}'.")
            if request.project_filter:
                from .document_registry import get_project_by_name
                project = get_project_by_name(request.project_filter)
                if project:
                    project_id = project.get("project_id")
                    if project_id:
                        vector_store = get_llamaindex_vector_store(project_id=project_id)
                        if vector_store: logger.info(f"Successfully loaded vector store for project '{project_id}'"); vector_store_cache[request.session_id] = vector_store
            if not vector_store:
                vector_store = get_llamaindex_vector_store(session_id=request.session_id)
                if vector_store: logger.info(f"Successfully loaded store from disk for session '{request.session_id}'. Caching it."); vector_store_cache[request.session_id] = vector_store
            # --- If still no vector store, raise error ---
            if not vector_store:
                raise HTTPException(status_code=404, detail=f"Vector store not found for session '{request.session_id}' or project '{request.project_filter}'. Please index first.")


        # --- Execute Research using ResearchSystem ---
        research_system = ResearchSystem(vector_store)
        # research_response_data contains 'report', 'sources', 'findings', etc.
        research_response_data = research_system.run(
            query=request.query,
            research_mode=request.research_mode,
            report_mode=request.report_mode,
            agent_chain_name=request.chain,
            top_k=request.top_k,
            context_filter=request.context_filter,
            session_id=request.session_id,
            max_iterations=request.max_iterations if hasattr(request, 'max_iterations') else 3
            # Note: analysis_focus parameter has been removed from ResearchSystem.run()
        )

        # --- Construct Final API Response (Simplified, focusing on the report) ---
        final_report = research_response_data.get("report", "")
        if not final_report or len(final_report) < 150: # Check for valid report content
            logger.warning(f"Report seems short or missing. Length: {len(final_report)}. Content: {final_report[:100]}...")
            if "error" in research_response_data: # If system returned an error message
                final_report = f"# Report Error\n\n{research_response_data['error']}"
            else:
                final_report = f"# Report Error\n\nNo valid report was generated for query: '{request.query}'"

        # Get the final findings used by the report (they should have ref numbers now)
        report_findings = research_response_data.get("findings", []) # Use findings returned by ResearchSystem.run
        logger.info(f"Final response will include {len(report_findings)} detailed findings.")

        # Add 'type': 'finding' to each finding dict for potential UI filtering
        for finding in report_findings:
            finding['type'] = 'finding'

        # Structure the response more like the legacy system
        response_content = {
            "query": request.query,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat(),
            "user": current_user,
            "research_mode": research_response_data.get("research_mode", request.research_mode),
            "report_mode": research_response_data.get("report_mode", request.report_mode),
            "elapsed_time": research_response_data.get("elapsed_time", "N/A"),
            "sources": research_response_data.get("sources", []), # List of source dicts from ReportGenerator
            "questions_by_iteration": research_response_data.get("questions_by_iteration"),
            # --- The main report string ---
            "report": final_report,
            # --- The detailed findings used in the report ---
            # The UI will primarily display the 'report'. This list is for drilling down.
            "results": report_findings # List of finding dictionaries
        }

        logger.info(f"Final response structure: report string + {len(report_findings)} findings in results list.")
        return JSONResponse(content=response_content) # Use JSONResponse to ensure correct serialization

    except HTTPException as http_exc:
        logger.error(f"HTTP exception in research endpoint: {http_exc.detail}", exc_info=True)
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in research endpoint: {e}", exc_info=True)
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}\nTraceback:\n{tb_str}")

@router.post("/documents/index", response_model=IndexingResponse)
async def index_documents(
    request: IndexingRequest = Body(...),
    current_user: str = Depends(get_current_user)
):
    """
    Index documents for a user or project.
    """
    try:
        logger.info(f"Received document indexing request from user '{current_user}' for project '{request.project_id}'")

        # Determine root directory
        root_dir = request.root_dir if request.root_dir else "."
        if request.project_id:
            from .document_registry import get_project_directory
            root_dir = get_project_directory(request.project_id)

        # Generate session ID if not provided
        session_id = request.session_id or f"session_{int(time.time())}"

        # Build index
        vector_store = build_index(
            root_dir=root_dir,
            context_filter=request.context_filter,
            external_sources=request.external_sources,
            session_id=session_id,
            project_id=request.project_id,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        # Cache vector store
        vector_store_cache[session_id] = vector_store

        # Perform comprehensive cleanup after indexing
        from .database_cleanup import cleanup_database_files
        cleanup_result = cleanup_database_files()

        # Log detailed cleanup information
        logger.info(f"Comprehensive cleanup after indexing: {cleanup_result['status']}")
        logger.info(f"- Legacy database: removed {cleanup_result.get('total_files_removed', 0)} files")
        logger.info(f"- Embedding files: removed {cleanup_result.get('embedding_files_removed', 0)} files")
        logger.info(f"- Storage files: removed {cleanup_result.get('storage_files_removed', 0)} files")
        logger.info(f"- Projects files: removed {cleanup_result.get('projects_files_removed', 0)} files")
        logger.info(f"- Total space freed: {cleanup_result.get('human_readable_bytes_freed', '0B')}")

        # Create response
        response = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user": current_user,
            "status": "success",
            "message": f"Indexed documents successfully. Session ID: {session_id}"
        }

        return response

    except Exception as e:
        logger.error(f"Error in index_documents endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@router.post("/projects/{project_id}/index", response_model=IndexingResponse)
async def index_project(
    project_id: str,
    request: IndexingRequest = Body(...),
    current_user: str = Depends(get_current_user)
):
    """
    Index documents for a specific project.
    """
    try:
        logger.info(f"Received project indexing request from user '{current_user}' for project '{project_id}'")

        # Update request with project_id
        request.project_id = project_id

        # Call index_documents endpoint
        return await index_documents(request, current_user)

    except Exception as e:
        logger.error(f"Error in index_project endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Project indexing failed: {str(e)}")

@router.get("/vector-stores", response_model=List[Dict[str, Any]])
async def list_vector_stores(
    current_user: str = Depends(get_current_user),
    include_sessions: bool = Query(False, description="Whether to include session vector stores")
):
    """
    List available vector stores.

    By default, only project vector stores are returned.
    Set include_sessions=true to include session vector stores.
    """
    try:
        # Import config for proper directory paths
        from .config import PROJECT_INDEX_DIR, SESSION_PERSIST_DIR

        # List project vector stores
        project_vector_stores = []

        if os.path.exists(PROJECT_INDEX_DIR):
            logger.info(f"Checking for vector stores in {PROJECT_INDEX_DIR}")

            # Find all project vector store directories
            vector_store_dirs = [
                d for d in os.listdir(PROJECT_INDEX_DIR)
                if os.path.isdir(os.path.join(PROJECT_INDEX_DIR, d)) and d.endswith("_vector_store")
            ]

            # Also check for index info files for backward compatibility
            index_files = [f for f in os.listdir(PROJECT_INDEX_DIR) if f.endswith("_index_info.json")]

            # Process vector store directories
            for vector_store_dir in vector_store_dirs:
                try:
                    # Extract project ID from directory name
                    project_id = vector_store_dir.replace("_vector_store", "")

                    # Check if the FAISS index file exists
                    faiss_file_path = os.path.join(PROJECT_INDEX_DIR, vector_store_dir, "vector_store", "index.faiss")
                    docstore_path = os.path.join(PROJECT_INDEX_DIR, vector_store_dir, "docstore.json")

                    if os.path.exists(faiss_file_path) and os.path.exists(docstore_path):
                        # Add to list with status
                        project_vector_stores.append({
                            "project_id": project_id,
                            "type": "project",
                            "status": "valid",
                            "path": os.path.join(PROJECT_INDEX_DIR, vector_store_dir)
                        })
                    else:
                        # Add to list with invalid status
                        missing = []
                        if not os.path.exists(faiss_file_path): missing.append("FAISS index")
                        if not os.path.exists(docstore_path): missing.append("docstore")

                        project_vector_stores.append({
                            "project_id": project_id,
                            "type": "project",
                            "status": "invalid",
                            "missing": missing,
                            "path": os.path.join(PROJECT_INDEX_DIR, vector_store_dir)
                        })
                except Exception as e:
                    logger.error(f"Error processing vector store directory {vector_store_dir}: {e}")

            # Process index info files for backward compatibility
            for index_file in index_files:
                try:
                    # Extract project ID from filename
                    project_id = index_file.replace("_index_info.json", "")

                    # Skip if we already have this project ID from vector store directories
                    if any(vs["project_id"] == project_id for vs in project_vector_stores):
                        continue

                    # Load index info
                    with open(os.path.join(PROJECT_INDEX_DIR, index_file), "r") as f:
                        index_info = json.load(f)

                    # Add to list
                    project_vector_stores.append({
                        "project_id": project_id,
                        "session_id": index_info.get("session_id"),
                        "document_count": index_info.get("document_count", 0),
                        "chunk_count": index_info.get("chunk_count", 0),
                        "last_indexed": index_info.get("last_indexed"),
                        "type": "project",
                        "status": "legacy",
                        "path": os.path.join(PROJECT_INDEX_DIR, f"{project_id}_vector_store")
                    })
                except Exception as e:
                    logger.error(f"Error loading index info for {index_file}: {e}")

        # If include_sessions is False, return only project vector stores
        if not include_sessions:
            logger.info("Returning only project vector stores (include_sessions=False)")
            return project_vector_stores

        # List session vector stores if include_sessions is True
        logger.info("Including session vector stores (include_sessions=True)")
        session_vector_stores = []

        if os.path.exists(SESSION_PERSIST_DIR):
            logger.info(f"Checking for vector stores in {SESSION_PERSIST_DIR}")

            # Find all session vector store directories
            session_dirs = [
                d for d in os.listdir(SESSION_PERSIST_DIR)
                if os.path.isdir(os.path.join(SESSION_PERSIST_DIR, d)) and d.endswith("_vector_store")
            ]

            for session_dir in session_dirs:
                try:
                    # Extract session ID from directory name
                    session_id = session_dir.replace("_vector_store", "")

                    # Check if the FAISS index file exists
                    faiss_file_path = os.path.join(SESSION_PERSIST_DIR, session_dir, "vector_store", "index.faiss")
                    docstore_path = os.path.join(SESSION_PERSIST_DIR, session_dir, "docstore.json")

                    if os.path.exists(faiss_file_path) and os.path.exists(docstore_path):
                        # Add to list with status
                        session_vector_stores.append({
                            "session_id": session_id,
                            "type": "session",
                            "status": "valid",
                            "path": os.path.join(SESSION_PERSIST_DIR, session_dir)
                        })
                    else:
                        # Add to list with invalid status
                        missing = []
                        if not os.path.exists(faiss_file_path): missing.append("FAISS index")
                        if not os.path.exists(docstore_path): missing.append("docstore")

                        session_vector_stores.append({
                            "session_id": session_id,
                            "type": "session",
                            "status": "invalid",
                            "missing": missing,
                            "path": os.path.join(SESSION_PERSIST_DIR, session_dir)
                        })
                except Exception as e:
                    logger.error(f"Error processing session vector store directory {session_dir}: {e}")

            # Also check for legacy session vector store files
            vector_store_files = [f for f in os.listdir(SESSION_PERSIST_DIR) if f.endswith("_vector_store.faiss")]

            for vector_store_file in vector_store_files:
                try:
                    # Extract session ID from filename
                    session_id = vector_store_file.replace("_vector_store.faiss", "")

                    # Skip if we already have this session ID from vector store directories
                    if any(vs["session_id"] == session_id for vs in session_vector_stores):
                        continue

                    # Add to list
                    session_vector_stores.append({
                        "session_id": session_id,
                        "type": "session",
                        "status": "legacy",
                        "path": os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
                    })
                except Exception as e:
                    logger.error(f"Error loading session vector store for {vector_store_file}: {e}")

        # Combine lists
        vector_stores = project_vector_stores + session_vector_stores

        return vector_stores

    except Exception as e:
        logger.error(f"Error in list_vector_stores endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list vector stores: {str(e)}")

@router.delete("/vector-stores/{store_id}", response_model=Dict[str, Any])
async def delete_vector_store(
    store_id: str,
    store_type: str = Query(..., description="Type of store: 'project' or 'session'"),
    current_user: str = Depends(get_current_user)
):
    """
    Delete a vector store.
    """
    try:
        # Import config for proper directory paths
        from .config import PROJECT_INDEX_DIR, SESSION_PERSIST_DIR
        import shutil

        logger.info(f"Received delete vector store request from user '{current_user}' for {store_type} '{store_id}'")

        if store_type == "project":
            # Delete project vector store
            # Check if vector store directory exists
            vector_store_dir = os.path.join(PROJECT_INDEX_DIR, f"{store_id}_vector_store")

            if os.path.isdir(vector_store_dir):
                # Delete the entire directory
                shutil.rmtree(vector_store_dir)
                logger.info(f"Deleted project vector store directory: {vector_store_dir}")

                # Also check for legacy files
                index_file = os.path.join(PROJECT_INDEX_DIR, f"{store_id}_index_info.json")
                if os.path.exists(index_file):
                    os.remove(index_file)
                    logger.info(f"Deleted legacy index info file: {index_file}")

                # Check for legacy FAISS files
                legacy_faiss = os.path.join(PROJECT_INDEX_DIR, f"{store_id}_vector_store.faiss")
                if os.path.exists(legacy_faiss):
                    os.remove(legacy_faiss)
                    logger.info(f"Deleted legacy FAISS file: {legacy_faiss}")

                # Check for legacy JSON files
                legacy_json = os.path.join(PROJECT_INDEX_DIR, f"{store_id}_vector_store.json")
                if os.path.exists(legacy_json):
                    os.remove(legacy_json)
                    logger.info(f"Deleted legacy JSON file: {legacy_json}")

                return {"status": "success", "message": f"Deleted project vector store {store_id}"}

            # Check for legacy files if directory doesn't exist
            index_file = os.path.join(PROJECT_INDEX_DIR, f"{store_id}_index_info.json")
            if os.path.exists(index_file):
                # Load index info to get session_id
                with open(index_file, "r") as f:
                    index_info = json.load(f)

                session_id = index_info.get("session_id")

                # Delete vector store files
                if session_id:
                    vector_store_path = os.path.join(PROJECT_INDEX_DIR, f"{session_id}_vector_store")
                    if os.path.exists(f"{vector_store_path}.faiss"):
                        os.remove(f"{vector_store_path}.faiss")
                        logger.info(f"Deleted legacy FAISS file: {vector_store_path}.faiss")
                    if os.path.exists(f"{vector_store_path}.json"):
                        os.remove(f"{vector_store_path}.json")
                        logger.info(f"Deleted legacy JSON file: {vector_store_path}.json")

                # Delete index info file
                os.remove(index_file)
                logger.info(f"Deleted legacy index info file: {index_file}")

                return {"status": "success", "message": f"Deleted legacy project vector store {store_id}"}
            else:
                raise HTTPException(status_code=404, detail=f"Project vector store {store_id} not found")

        elif store_type == "session":
            # Delete session vector store
            # Check if vector store directory exists
            vector_store_dir = os.path.join(SESSION_PERSIST_DIR, f"{store_id}_vector_store")

            if os.path.isdir(vector_store_dir):
                # Delete the entire directory
                shutil.rmtree(vector_store_dir)
                logger.info(f"Deleted session vector store directory: {vector_store_dir}")

                # Also check for legacy files
                legacy_faiss = os.path.join(SESSION_PERSIST_DIR, f"{store_id}_vector_store.faiss")
                if os.path.exists(legacy_faiss):
                    os.remove(legacy_faiss)
                    logger.info(f"Deleted legacy FAISS file: {legacy_faiss}")

                # Check for legacy JSON files
                legacy_json = os.path.join(SESSION_PERSIST_DIR, f"{store_id}_vector_store.json")
                if os.path.exists(legacy_json):
                    os.remove(legacy_json)
                    logger.info(f"Deleted legacy JSON file: {legacy_json}")

                return {"status": "success", "message": f"Deleted session vector store {store_id}"}

            # Check for legacy files if directory doesn't exist
            legacy_faiss = os.path.join(SESSION_PERSIST_DIR, f"{store_id}_vector_store.faiss")
            legacy_json = os.path.join(SESSION_PERSIST_DIR, f"{store_id}_vector_store.json")

            if os.path.exists(legacy_faiss) or os.path.exists(legacy_json):
                # Delete vector store files
                if os.path.exists(legacy_faiss):
                    os.remove(legacy_faiss)
                    logger.info(f"Deleted legacy FAISS file: {legacy_faiss}")
                if os.path.exists(legacy_json):
                    os.remove(legacy_json)
                    logger.info(f"Deleted legacy JSON file: {legacy_json}")

                return {"status": "success", "message": f"Deleted legacy session vector store {store_id}"}
            else:
                raise HTTPException(status_code=404, detail=f"Session vector store {store_id} not found")

        else:
            raise HTTPException(status_code=400, detail=f"Invalid store type: {store_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_vector_store endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete vector store: {str(e)}")

def setup_llamaindex_routes(app):
    """
    Set up LlamaIndex routes for the FastAPI app.
    """
    app.include_router(router, tags=["LlamaIndex"])
    logger.info("Set up LlamaIndex routes")