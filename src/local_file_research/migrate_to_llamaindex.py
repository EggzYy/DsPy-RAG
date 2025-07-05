"""
Script to migrate existing vector stores to LlamaIndex format.
"""

import os
import logging
import json
import glob
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_project_indices():
    """
    Migrate all project indices to LlamaIndex format.
    """
    # Get all project index directories
    project_indices_dir = "project_indices"
    if not os.path.exists(project_indices_dir):
        logger.warning(f"Project indices directory {project_indices_dir} does not exist")
        return
    
    # Find all project index files
    index_files = glob.glob(os.path.join(project_indices_dir, "*_index_info.json"))
    logger.info(f"Found {len(index_files)} project index files")
    
    for index_file in index_files:
        try:
            # Extract project ID from filename
            filename = os.path.basename(index_file)
            project_id = filename.replace("_index_info.json", "")
            
            # Load index info
            with open(index_file, "r") as f:
                index_info = json.load(f)
            
            # Check if session ID is available
            session_id = index_info.get("session_id")
            if not session_id:
                logger.warning(f"No session ID found in index info for project {project_id}")
                continue
            
            # Check if vector store files exist
            vector_store_path = os.path.join(project_indices_dir, f"{session_id}_vector_store")
            if not os.path.exists(f"{vector_store_path}.index") or not os.path.exists(f"{vector_store_path}.data"):
                logger.warning(f"Vector store files not found for project {project_id} at {vector_store_path}")
                continue
            
            # Migrate vector store
            migrate_vector_store(project_id=project_id, session_id=session_id)
            
        except Exception as e:
            logger.error(f"Error migrating project index {index_file}: {e}", exc_info=True)

def migrate_session_indices():
    """
    Migrate all session indices to LlamaIndex format.
    """
    # Get all session directories
    sessions_dir = "sessions"
    if not os.path.exists(sessions_dir):
        logger.warning(f"Sessions directory {sessions_dir} does not exist")
        return
    
    # Find all session vector store files
    vector_store_files = glob.glob(os.path.join(sessions_dir, "*_vector_store.index"))
    logger.info(f"Found {len(vector_store_files)} session vector store files")
    
    for vector_store_file in vector_store_files:
        try:
            # Extract session ID from filename
            filename = os.path.basename(vector_store_file)
            session_id = filename.replace("_vector_store.index", "")
            
            # Migrate vector store
            migrate_vector_store(session_id=session_id)
            
        except Exception as e:
            logger.error(f"Error migrating session vector store {vector_store_file}: {e}", exc_info=True)

def migrate_vector_store(project_id: Optional[str] = None, session_id: Optional[str] = None):
    """
    Migrate a vector store to LlamaIndex format.
    
    Args:
        project_id: Project ID
        session_id: Session ID
    """
    try:
        # Import required modules
        from .vector_store import FAISSVectorStore
        from .llamaindex_vector_store import LlamaIndexVectorStore
        from .config import EMBEDDING_DIMENSION, FAISS_INDEX_TYPE
        
        # Load old vector store
        if project_id and session_id:
            old_vector_store_path = os.path.join("project_indices", f"{session_id}_vector_store")
            new_vector_store_path = os.path.join("project_indices", f"{project_id}_vector_store")
            logger.info(f"Migrating project vector store {project_id} from {old_vector_store_path} to {new_vector_store_path}")
            
            # Load old vector store
            old_vector_store = FAISSVectorStore.load(old_vector_store_path)
            
            # Create new vector store
            new_vector_store = LlamaIndexVectorStore(
                dimension=EMBEDDING_DIMENSION,
                index_type=FAISS_INDEX_TYPE,
                project_id=project_id
            )
            
            # Add chunks to new vector store
            new_vector_store.add_chunks(old_vector_store.chunks)
            
            # Save new vector store
            new_vector_store.save()
            
            logger.info(f"Successfully migrated project vector store {project_id} with {len(old_vector_store.chunks)} chunks")
            
        elif session_id:
            old_vector_store_path = os.path.join("sessions", f"{session_id}_vector_store")
            new_vector_store_path = os.path.join("sessions", f"{session_id}_vector_store")
            logger.info(f"Migrating session vector store {session_id} from {old_vector_store_path} to {new_vector_store_path}")
            
            # Load old vector store
            old_vector_store = FAISSVectorStore.load(old_vector_store_path)
            
            # Create new vector store
            new_vector_store = LlamaIndexVectorStore(
                dimension=EMBEDDING_DIMENSION,
                index_type=FAISS_INDEX_TYPE,
                session_id=session_id
            )
            
            # Add chunks to new vector store
            new_vector_store.add_chunks(old_vector_store.chunks)
            
            # Save new vector store
            new_vector_store.save()
            
            logger.info(f"Successfully migrated session vector store {session_id} with {len(old_vector_store.chunks)} chunks")
        
        else:
            logger.error("Either project_id or session_id must be provided")
            
    except Exception as e:
        logger.error(f"Error migrating vector store: {e}", exc_info=True)

def main():
    """
    Main function to migrate all vector stores.
    """
    logger.info("Starting migration of vector stores to LlamaIndex format")
    
    # Migrate project indices
    migrate_project_indices()
    
    # Migrate session indices
    migrate_session_indices()
    
    logger.info("Migration complete")

if __name__ == "__main__":
    main()
