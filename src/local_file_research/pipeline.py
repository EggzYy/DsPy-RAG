from .file_indexer import scan_directory, scan_data_source
from .vector_store import get_vector_store, chunk_file_content, VectorStore
from .document_analysis import DocumentAnalyzer
from .database_cleanup import cleanup_database_files
from typing import List, Dict, Any, Optional, Union
import os
import pickle
import logging
import time
from pathlib import Path

# Get logger instance (basic config happens in api.py)
logger = logging.getLogger(__name__)

# --- Embedding Functionality ---

def embed_text(text: str, model_name: str = None):
    """
    Embed text using the configured embedding model.

    This function now delegates to the embedding module to handle different embedding models,
    including Ollama models like mxbai-embed-large.

    Args:
        text: Text to embed
        model_name: Name of the embedding model to use

    Returns:
        List of floats representing the embedding
    """
    if not text:
        logger.warning("Attempted to embed empty text")
        from .config import EMBEDDING_DIMENSION
        return [0.0] * EMBEDDING_DIMENSION  # Return zero vector of configured dimension

    # Use the embedding module to get embeddings
    from .embedding import get_embeddings
    embeddings = get_embeddings([text])

    if embeddings and len(embeddings) > 0:
        return embeddings[0]
    else:
        logger.error("Failed to get embedding, returning zero vector")
        from .config import EMBEDDING_DIMENSION
        return [0.0] * EMBEDDING_DIMENSION

# --- Indexing Pipeline ---

def build_index(
    root_dir: str,
    context_filter: str = None,
    external_sources: List[Dict[str, Any]] = None,
    session_id: str = None,
    use_faiss: bool = True,
    chunk_size: int = 1024,
    chunk_overlap: int = 0  # Not used with semantic chunking
) -> VectorStore:
    """
    Index all supported files in the given directory and external sources.

    Args:
        root_dir: Directory to scan for files
        context_filter: Optional filter for file paths
        external_sources: List of external data sources to include
        session_id: Optional session ID for persistence
        use_faiss: Whether to use FAISS for vector search
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Vector store with indexed content
    """
    start_time = time.time()
    logger.info(f"Starting indexing of {root_dir}")

    # Check if we can load from session first
    if session_id:
        from .config import SESSION_PERSIST_DIR
        session_path = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
        if os.path.exists(f"{session_path}.data") or os.path.exists(session_path):
            logger.info(f"Found existing session data for '{session_id}', attempting to load")
            vector_store = load_session_vector_store(session_id, use_faiss=use_faiss)
            if vector_store:
                logger.info(f"Successfully loaded vector store for session '{session_id}'")
                return vector_store
            logger.warning(f"Failed to load session data, will create new index")

    # Scan files and external sources
    file_records = scan_directory(root_dir)
    logger.info(f"Found {len(file_records)} files in {root_dir}")

    if context_filter:
        file_records = [fr for fr in file_records if context_filter.lower() in fr["path"].lower()]
        logger.info(f"After filtering with '{context_filter}', {len(file_records)} files remain")

    # Add records from external sources
    if external_sources:
        for src in external_sources:
            source_type = src.get("source_type")
            params = src.get("params", {})
            logger.info(f"Scanning external source: {source_type}")
            ext_records = scan_data_source(source_type, **params)
            logger.info(f"Found {len(ext_records)} records from {source_type}")
            file_records.extend(ext_records)

    # Create vector store with appropriate configuration
    from .config import FAISS_INDEX_TYPE, EMBEDDING_DIMENSION
    vector_store = get_vector_store(
        use_faiss=use_faiss,
        dimension=EMBEDDING_DIMENSION,
        index_type=FAISS_INDEX_TYPE
    )
    logger.info(f"Created vector store with FAISS index type: {FAISS_INDEX_TYPE}, dimension: {EMBEDDING_DIMENSION}")
    total_chunks = 0

    # Process each file
    for i, file_record in enumerate(file_records):
        try:
            # Log progress periodically
            if i % 10 == 0 or i == len(file_records) - 1:
                logger.info(f"Processing file {i+1}/{len(file_records)}: {file_record.get('name', file_record.get('path'))}")

            # Chunk the file content
            chunks = chunk_file_content(file_record, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Embed each chunk
            for chunk in chunks:
                chunk["embedding"] = embed_text(chunk["content"])

            # Add chunks to vector store
            vector_store.add_chunks(chunks)
            total_chunks += len(chunks)
        except Exception as e:
            logger.error(f"Error processing file {file_record.get('path')}: {e}", exc_info=True)

    # Persist vector store if session_id provided
    if session_id:
        persist_vector_store(vector_store, session_id)

        # Clean up database files after indexing
        cleanup_result = cleanup_database_files()
        logger.info(f"Database cleanup after indexing: {cleanup_result['status']}, removed {cleanup_result['total_files_removed']} files, freed {cleanup_result['human_readable_bytes_freed']}")

    elapsed_time = time.time() - start_time
    logger.info(f"Indexing complete. Processed {len(file_records)} files into {total_chunks} chunks in {elapsed_time:.2f} seconds")

    return vector_store

def persist_vector_store(vector_store: VectorStore, session_id: str) -> bool:
    """
    Persist a vector store for session-based memory.

    Args:
        vector_store: The vector store to persist
        session_id: Session ID for the persistence

    Returns:
        True if successful, False otherwise
    """
    from .config import SESSION_PERSIST_DIR
    os.makedirs(SESSION_PERSIST_DIR, exist_ok=True)
    path = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")

    try:
        logger.info(f"Persisting vector store for session '{session_id}' to {path}")
        success = vector_store.save(path)
        if success:
            logger.info(f"Successfully persisted vector store for session '{session_id}'")
        else:
            logger.error(f"Failed to persist vector store for session '{session_id}'")
        return success
    except Exception as e:
        logger.error(f"Error persisting vector store for session '{session_id}': {e}", exc_info=True)
        return False

def load_session_vector_store(session_id: str, use_faiss: bool = True) -> Optional[VectorStore]:
    """
    Load a persisted vector store for session-based memory.

    Args:
        session_id: Session ID to load
        use_faiss: Whether to try loading a FAISS vector store first

    Returns:
        Loaded vector store or None if not found or error
    """
    from .config import SESSION_PERSIST_DIR

    # Try FAISS first if requested
    if use_faiss:
        from .vector_store import FAISSVectorStore, FAISS_AVAILABLE
        if FAISS_AVAILABLE:
            faiss_path = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
            logger.info(f"Attempting to load FAISS vector store for session '{session_id}' from {faiss_path}")

            if os.path.exists(f"{faiss_path}.data") and os.path.exists(f"{faiss_path}.index"):
                try:
                    # The load method will automatically detect the index type from the saved data
                    vector_store = FAISSVectorStore.load(faiss_path)
                    if vector_store:
                        index_type = getattr(vector_store, 'index_type', 'unknown')
                        logger.info(f"Successfully loaded FAISS {index_type} vector store for session '{session_id}'. Chunks: {len(vector_store.chunks)}")
                        return vector_store
                except Exception as e:
                    logger.error(f"Failed to load FAISS vector store for session '{session_id}': {e}", exc_info=True)

    # Try InMemoryVectorStore as fallback
    from .vector_store import InMemoryVectorStore
    path = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
    logger.info(f"Attempting to load in-memory vector store for session '{session_id}' from {path}")

    if os.path.exists(path):
        try:
            vector_store = InMemoryVectorStore.load(path)
            if vector_store:
                logger.info(f"Successfully loaded in-memory vector store for session '{session_id}'. Chunks: {len(vector_store.chunks)}")
                return vector_store
            else:
                logger.error(f"Failed to load in-memory vector store for session '{session_id}'")
        except Exception as e:
            logger.error(f"Failed to load in-memory vector store for session '{session_id}': {e}", exc_info=True)
    else:
        logger.warning(f"Vector store file not found for session '{session_id}' at {path}")

    return None

# --- Search Pipeline ---

def search_index(
    vector_store: VectorStore,
    query: str,
    top_k: int = 5,
    context_filter: str = None,
    session_id: str = None,
    use_faiss: bool = True
) -> List[Dict]:
    """
    Search the index for the query, optionally narrowing by context.

    Args:
        vector_store: Vector store to search
        query: Search query
        top_k: Number of results to return
        context_filter: Optional filter for results
        session_id: Optional session ID for loading vector store
        use_faiss: Whether to use FAISS for vector search

    Returns:
        List of search results
    """
    # Load vector store if not provided and session_id is available
    if session_id and not vector_store:
        logger.info(f"No vector store provided, attempting to load from session '{session_id}'")
        vs = load_session_vector_store(session_id, use_faiss=use_faiss)
        if vs is not None:
            vector_store = vs
            logger.info(f"Successfully loaded vector store for session '{session_id}'")
        else:
            logger.warning(f"Failed to load vector store for session '{session_id}'")
            return []  # Return empty results if no vector store available

    # Ensure we have a vector store
    if not vector_store:
        logger.error("No vector store available for search")
        return []

    # Embed the query and search
    logger.info(f"Searching for: '{query}' (top_k={top_k})")
    query_embedding = embed_text(query)
    results = vector_store.search(query_embedding, top_k=top_k)
    logger.info(f"Found {len(results)} results")

    # Apply context filter if provided
    if context_filter:
        filtered_results = [r for r in results if context_filter.lower() in r.get("file_path", "").lower()]
        logger.info(f"After filtering with '{context_filter}', {len(filtered_results)} results remain")
        results = filtered_results

    # Attach citation metadata to each result
    for r in results:
        # Get similarity score if available
        similarity = r.get("similarity", 0.0)

        # Add score to the result itself
        r["score"] = similarity

        # Ensure content is included in the result
        if "content" in r:
            # Content is already present, make sure it's not empty
            if not r["content"]:
                logger.warning(f"Empty content found in result for {r.get('file_path')}")

                # Try to get content from the vector store if document_id is available
                if r.get("document_id"):
                    try:
                        # First try to get from content file (most reliable)
                        try:
                            from .storage_manager import read_document_content
                            content = read_document_content(r["document_id"])
                            if content:
                                logger.info(f"Retrieved content for document {r['document_id']} from content file")
                                r["content"] = content
                                continue  # Skip to next iteration if content found
                        except Exception as content_e:
                            logger.warning(f"Error retrieving content from content file: {str(content_e)}")

                        # Try to get from document registry
                        try:
                            from .document_registry import get_document_registry
                            doc_registry = get_document_registry(r["document_id"])
                            if doc_registry and "metadata" in doc_registry and "content" in doc_registry["metadata"]:
                                content = doc_registry["metadata"]["content"]
                                logger.info(f"Retrieved content for document {r['document_id']} from registry")
                                r["content"] = content
                                continue  # Skip to next iteration if content found
                        except Exception as registry_e:
                            logger.warning(f"Error retrieving content from registry: {str(registry_e)}")

                        # Try to get from vector store
                        try:
                            # Get project_id from document registry
                            from .document_registry import get_document_registry
                            doc_registry = get_document_registry(r["document_id"])
                            if doc_registry and "projects" in doc_registry and doc_registry["projects"]:
                                project_id = doc_registry["projects"][0]

                                from .metadata_optimizer import get_document_content
                                content = get_document_content(r["document_id"], project_id)
                                if content:
                                    logger.info(f"Retrieved content for document {r['document_id']} from vector store")
                                    r["content"] = content
                            else:
                                logger.warning(f"No project_id found for document {r['document_id']}")
                        except Exception as e:
                            logger.error(f"Error retrieving content for document {r.get('document_id')}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error retrieving content for document {r.get('document_id')}: {str(e)}")
        else:
            logger.warning(f"No content found in result for {r.get('file_path')}")

            # Try to get content from the vector store if document_id is available
            if r.get("document_id") and session_id:
                try:
                    from .metadata_optimizer import get_document_content
                    content = get_document_content(r["document_id"], session_id)
                    if content:
                        logger.info(f"Retrieved content for document {r['document_id']} from vector store")
                        r["content"] = content
                except Exception as e:
                    logger.error(f"Error retrieving content for document {r.get('document_id')}: {str(e)}")

        r["citation"] = {
            "source_path": r.get("source_path", r.get("file_path")),
            "source_name": r.get("source_name"),
            "source_type": r.get("source_type"),
            "start": r.get("start"),
            "end": r.get("end"),
            "score": similarity  # Add similarity score to citation
        }

    return results

# --- Deep Research Pipeline (DSPy Agent Integration) ---

from .dspy_config import DSPyAgentRegistry #, initialize_dspy

def deep_research(
    vector_store: VectorStore,
    query: str,
    top_k: int = 5,
    mode: str = "summarize",
    context_filter: str = None,
    session_id: str = None,
    use_faiss: bool = True
) -> List[Dict]:
    """
    After retrieving top-k chunks, run DSPy agent for summarization, Q&A, or info extraction.

    Args:
        vector_store: Vector store to search
        query: Search query
        top_k: Number of results to return
        mode: Analysis mode ("summarize", "answer", "extract", "chain_of_thought", "analyze")
        context_filter: Optional filter for results
        session_id: Optional session ID for loading vector store
        use_faiss: Whether to use FAISS for vector search

    Returns:
        List of search results with agent-generated content
    """
    # Ensure DSPy is configured and agents are registered
    #initialize_dspy()  # Make sure DSPy is ready

    # Get search results
    results = search_index(
        vector_store,
        query,
        top_k=top_k,
        context_filter=context_filter,
        session_id=session_id,
        use_faiss=use_faiss
    )

    if not results:  # Handle case where search returns nothing
        logger.warning(f"No search results found for query: '{query}'")
        return []

    # Create document analyzer for advanced analysis
    document_analyzer = DocumentAnalyzer()

    try:
        # Get the appropriate agent based on mode
        agent = None
        if mode == "summarize":
            agent = DSPyAgentRegistry.get_agent("summarizer")
        elif mode == "answer":
            agent = DSPyAgentRegistry.get_agent("answerer")
        elif mode == "extract":
            agent = DSPyAgentRegistry.get_agent("extractor")
        elif mode == "chain_of_thought":
            agent = DSPyAgentRegistry.get_agent("chain_of_thought")
        elif mode == "analyze":
            agent = DSPyAgentRegistry.get_agent("document_analyzer")

        if agent is None:
            logger.warning(f"No DSPy agent registered for mode '{mode}', falling back to document analyzer")
            # Fall back to document analyzer
            for r in results:
                analysis = document_analyzer.analyze_document(r["content"], r.get("file_path"), query)
                r["analysis"] = analysis
                r["summary"] = analysis.get("summary", "")
        else:
            # Use the selected agent
            logger.info(f"Using DSPy agent for mode: {mode}")
            for r in results:
                if mode == "summarize":
                    # Provide all possible input fields with defaults to avoid warnings
                    r["summary"] = agent(
                        content=r["content"],
                        document=r["content"],  # Duplicate content in document field
                        context="",  # Empty context
                        query="Summarize this document"  # Default query
                    ).get("summary", "")
                elif mode == "answer":
                    # Provide all possible input fields with defaults to avoid warnings
                    r["answer"] = agent(
                        context=r["content"],
                        query=query,
                        document=r["content"],  # Duplicate content in document field
                        content=r["content"]    # Duplicate content in content field
                    ).get("answer", "")
                elif mode == "extract":
                    # Provide all possible input fields with defaults to avoid warnings
                    r["info"] = agent(
                        content=r["content"],
                        document=r["content"],  # Duplicate content in document field
                        context="",  # Empty context
                        query="Extract key information"  # Default query
                    ).get("info", "")
                elif mode == "chain_of_thought":
                    # Provide all possible input fields with defaults to avoid warnings
                    r["cot"] = agent(
                        content=r["content"],
                        document=r["content"],  # Duplicate content in document field
                        context="",  # Empty context
                        query=query if query else "Analyze this document"  # Use query or default
                    ).get("cot", "")
                elif mode == "analyze":
                    # Provide all possible input fields with defaults to avoid warnings
                    analysis = agent(
                        document=r["content"],
                        context=query if query else "",  # Use query as context or empty string
                        query=query if query else "Analyze this document"  # Use query or default
                    )
                    r["analysis"] = {
                        "summary": analysis.get("summary", ""),
                        "key_points": analysis.get("key_points", []),
                        "entities": analysis.get("entities", []),
                        "sentiment": analysis.get("sentiment", "")
                    }
    except ImportError as ie:
        # If DSPy not available, use document analyzer as fallback
        logger.error(f"DSPy not available: {ie}. Using document analyzer fallback.")
        for r in results:
            try:
                analysis = document_analyzer._basic_analysis(r["content"], query)
                r["analysis"] = analysis
                r["summary"] = analysis.get("summary", "")
                if mode == "summarize":
                    r["summary"] = analysis.get("summary", "")
                elif mode == "answer":
                    r["answer"] = f"Unable to answer: {ie}"
                elif mode == "extract":
                    r["info"] = analysis.get("key_points", [])
                elif mode == "chain_of_thought":
                    r["cot"] = f"Unable to provide chain of thought: {ie}"
            except Exception as inner_e:
                logger.error(f"Error in document analyzer fallback: {inner_e}", exc_info=True)
                r["error"] = str(inner_e)
    except Exception as e:
        # Log and handle DSPy errors gracefully
        logger.error(f"Error in deep_research: {e}", exc_info=True)
        for r in results:
            r["dspy_error"] = str(e)

    # Always attach citation metadata
    for r in results:
        # Get similarity score if available
        similarity = r.get("similarity", 0.0)

        # Add score to the result itself
        r["score"] = similarity

        # Ensure content is included in the result
        if "content" in r:
            # Content is already present, make sure it's not empty
            if not r["content"]:
                logger.warning(f"Empty content found in result for {r.get('file_path')}")

                # Try to get content from the vector store if document_id is available
                if r.get("document_id"):
                    try:
                        # First try to get from content file (most reliable)
                        try:
                            from .storage_manager import read_document_content
                            content = read_document_content(r["document_id"])
                            if content:
                                logger.info(f"Retrieved content for document {r['document_id']} from content file")
                                r["content"] = content
                                continue  # Skip to next iteration if content found
                        except Exception as content_e:
                            logger.warning(f"Error retrieving content from content file: {str(content_e)}")

                        # Try to get from document registry
                        try:
                            from .document_registry import get_document_registry
                            doc_registry = get_document_registry(r["document_id"])
                            if doc_registry and "metadata" in doc_registry and "content" in doc_registry["metadata"]:
                                content = doc_registry["metadata"]["content"]
                                logger.info(f"Retrieved content for document {r['document_id']} from registry")
                                r["content"] = content
                                continue  # Skip to next iteration if content found
                        except Exception as registry_e:
                            logger.warning(f"Error retrieving content from registry: {str(registry_e)}")

                        # Try to get from vector store
                        try:
                            # Get project_id from document registry
                            from .document_registry import get_document_registry
                            doc_registry = get_document_registry(r["document_id"])
                            if doc_registry and "projects" in doc_registry and doc_registry["projects"]:
                                project_id = doc_registry["projects"][0]

                                from .metadata_optimizer import get_document_content
                                content = get_document_content(r["document_id"], project_id)
                                if content:
                                    logger.info(f"Retrieved content for document {r['document_id']} from vector store")
                                    r["content"] = content
                            else:
                                logger.warning(f"No project_id found for document {r['document_id']}")
                        except Exception as e:
                            logger.error(f"Error retrieving content for document {r.get('document_id')}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error retrieving content for document {r.get('document_id')}: {str(e)}")
        else:
            logger.warning(f"No content found in result for {r.get('file_path')}")

            # Try to get content from the vector store if document_id is available
            if r.get("document_id"):
                try:
                    # First try to get from content file (most reliable)
                    try:
                        from .storage_manager import read_document_content
                        content = read_document_content(r["document_id"])
                        if content:
                            logger.info(f"Retrieved content for document {r['document_id']} from content file")
                            r["content"] = content
                            continue  # Skip to next iteration if content found
                    except Exception as content_e:
                        logger.warning(f"Error retrieving content from content file: {str(content_e)}")

                    # Try to get from document registry
                    try:
                        from .document_registry import get_document_registry
                        doc_registry = get_document_registry(r["document_id"])
                        if doc_registry and "metadata" in doc_registry and "content" in doc_registry["metadata"]:
                            content = doc_registry["metadata"]["content"]
                            logger.info(f"Retrieved content for document {r['document_id']} from registry")
                            r["content"] = content
                            continue  # Skip to next iteration if content found
                    except Exception as registry_e:
                        logger.warning(f"Error retrieving content from registry: {str(registry_e)}")

                    # Try to get from vector store
                    try:
                        # Get project_id from document registry
                        from .document_registry import get_document_registry
                        doc_registry = get_document_registry(r["document_id"])
                        if doc_registry and "projects" in doc_registry and doc_registry["projects"]:
                            project_id = doc_registry["projects"][0]

                            from .metadata_optimizer import get_document_content
                            content = get_document_content(r["document_id"], project_id)
                            if content:
                                logger.info(f"Retrieved content for document {r['document_id']} from vector store")
                                r["content"] = content
                        else:
                            logger.warning(f"No project_id found for document {r['document_id']}")
                    except Exception as e:
                        logger.error(f"Error retrieving content for document {r.get('document_id')}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error retrieving content for document {r.get('document_id')}: {str(e)}")

        r["citation"] = {
            "source_path": r.get("source_path", r.get("file_path")),
            "source_name": r.get("source_name"),
            "source_type": r.get("source_type"),
            "start": r.get("start"),
            "end": r.get("end"),
            "score": similarity  # Add similarity score to citation
        }

    return results

# --- Extensibility Instructions ---
# To add a new data source (e.g., MySQL, Notion, SharePoint), implement a loader function
# that returns a list of file_records (dicts with keys: path, name, size, modified, content).
# Register the loader in file_indexer.py or call it in scan_directory/scan_data_source.
# To add a new LLM or embedding model, update embed_text() to select the model based on config.
# For session memory, use build_index(..., session_id=...) and load_session_vector_store.
# For security, deploy behind HTTPS, add authentication middleware, and follow best practices (see config.py).
# For backup, periodically copy the directory defined by SESSION_PERSIST_DIR in config.py.