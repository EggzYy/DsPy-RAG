"""
Pipeline for document processing, indexing, and search using LlamaIndex.
"""

from .file_indexer import scan_directory, scan_data_source
from .llamaindex_vector_store import get_llamaindex_vector_store, LlamaIndexVectorStore
from .document_analysis import DocumentAnalyzer
from .database_cleanup import cleanup_database_files
from .storage_manager import read_document_content
from .document_registry import get_document_registry
from typing import List, Dict, Any, Optional, Union
import os
import logging
import time
from pathlib import Path

# Get logger instance (basic config happens in api.py)
logger = logging.getLogger(__name__)

# --- Embedding Functionality ---

def embed_text(text: str, model_name: str = None):
    """
    Embed text using the configured embedding model.

    This function delegates to the embedding module to handle different embedding models,
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
    project_id: str = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 0  # Not used with semantic chunking
) -> LlamaIndexVectorStore:
    """
    Index all supported files in the given directory and external sources.

    Args:
        root_dir: Directory to scan for files
        context_filter: Optional filter for file paths
        external_sources: List of external data sources to include
        session_id: Optional session ID for persistence
        project_id: Optional project ID for persistence
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Vector store with indexed content
    """
    start_time = time.time()
    logger.info(f"Starting indexing of {root_dir}")

    # Check if we can load from session or project first
    vector_store = None
    if session_id:
        vector_store = load_session_vector_store(session_id)
        if vector_store:
            logger.info(f"Successfully loaded vector store for session '{session_id}'")
            return vector_store
        logger.warning(f"Failed to load session data, will create new index")
    elif project_id:
        vector_store = load_project_vector_store(project_id)
        if vector_store:
            logger.info(f"Successfully loaded vector store for project '{project_id}'")
            return vector_store
        logger.warning(f"Failed to load project data, will create new index")

    # Scan files and external sources
    file_records = scan_directory(root_dir)
    logger.info(f"Found {len(file_records)} files in {root_dir}")

    # Debug the file records to see what's being found
    if file_records:
        logger.info(f"First file found: {file_records[0].get('path', 'unknown')}")
    else:
        logger.warning(f"No files found in {root_dir}. Check if the directory exists and contains supported files.")
        # Check if directory exists
        if not os.path.exists(root_dir):
            logger.error(f"Directory {root_dir} does not exist!")
        else:
            # List files in directory to debug
            files_in_dir = os.listdir(root_dir)
            logger.info(f"Files in directory: {files_in_dir}")

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
    from .config import FAISS_INDEX_TYPE, EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
    vector_store = get_llamaindex_vector_store(
        project_id=project_id,
        session_id=session_id,
        dimension=EMBEDDING_DIMENSION,
        index_type=FAISS_INDEX_TYPE,
        embedding_model=EMBEDDING_MODEL_NAME
    )
    logger.info(f"Created LlamaIndex vector store with FAISS index type: {FAISS_INDEX_TYPE}, dimension: {EMBEDDING_DIMENSION}, embedding model: {EMBEDDING_MODEL_NAME}")
    total_chunks = 0

    # Process each file
    for i, file_record in enumerate(file_records):
        try:
            # Log progress periodically
            if i % 10 == 0 or i == len(file_records) - 1:
                logger.info(f"Processing file {i+1}/{len(file_records)}: {file_record.get('name', file_record.get('path'))}")

            # Chunk the file content
            from .vector_store import chunk_file_content
            chunks = chunk_file_content(file_record, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Embed each chunk
            for chunk in chunks:
                chunk["embedding"] = embed_text(chunk["content"])

            # Add chunks to vector store
            vector_store.add_chunks(chunks)
            total_chunks += len(chunks)
        except Exception as e:
            logger.error(f"Error processing file {file_record.get('path')}: {e}", exc_info=True)

    # Persist vector store
    vector_store.save()

    # Perform comprehensive cleanup after indexing
    cleanup_result = cleanup_database_files()

    # Log detailed cleanup information
    logger.info(f"Comprehensive cleanup after indexing: {cleanup_result['status']}")
    logger.info(f"- Legacy database: removed {cleanup_result.get('total_files_removed', 0)} files")
    logger.info(f"- Embedding files: removed {cleanup_result.get('embedding_files_removed', 0)} files")
    logger.info(f"- Storage files: removed {cleanup_result.get('storage_files_removed', 0)} files")
    logger.info(f"- Projects files: removed {cleanup_result.get('projects_files_removed', 0)} files")
    logger.info(f"- Total space freed: {cleanup_result.get('human_readable_bytes_freed', '0B')}")

    elapsed_time = time.time() - start_time
    logger.info(f"Indexing complete. Processed {len(file_records)} files into {total_chunks} chunks in {elapsed_time:.2f} seconds")

    return vector_store

def load_session_vector_store(session_id: str) -> Optional[LlamaIndexVectorStore]:
    """
    Load a persisted vector store for session-based memory.

    Args:
        session_id: Session ID to load

    Returns:
        Loaded vector store or None if not found or error
    """
    from .config import SESSION_PERSIST_DIR
    persist_dir = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
    logger.info(f"Attempting to load LlamaIndex vector store for session '{session_id}' from {persist_dir}")

    # Check if the directory exists
    if os.path.exists(persist_dir):
        # Check if the FAISS index file exists
        faiss_file_path = os.path.join(persist_dir, "vector_store", "index.faiss")
        docstore_path = os.path.join(persist_dir, "docstore.json")

        if os.path.exists(faiss_file_path) and os.path.exists(docstore_path):
            logger.info(f"Found FAISS index file at {faiss_file_path} and docstore at {docstore_path}")

            try:
                # Try to read the FAISS index to verify it's valid
                import faiss
                try:
                    loaded_faiss_index = faiss.read_index(faiss_file_path)
                    logger.info(f"Successfully read FAISS index with {loaded_faiss_index.ntotal} vectors")
                except Exception as faiss_err:
                    logger.error(f"Error reading FAISS index: {faiss_err}", exc_info=True)
                    return None

                # Load the vector store
                from .config import EMBEDDING_DIMENSION, FAISS_INDEX_TYPE, EMBEDDING_MODEL_NAME
                vector_store = get_llamaindex_vector_store(
                    session_id=session_id,
                    dimension=EMBEDDING_DIMENSION,
                    index_type=FAISS_INDEX_TYPE,
                    embedding_model=EMBEDDING_MODEL_NAME
                )

                # Verify the vector store was loaded correctly
                if vector_store and vector_store.faiss_store and vector_store.faiss_store._faiss_index.ntotal > 0:
                    logger.info(f"Successfully loaded LlamaIndex vector store for session '{session_id}' with {vector_store.faiss_store._faiss_index.ntotal} vectors")
                    return vector_store
                else:
                    logger.warning(f"Vector store loaded but contains no vectors. Attempting direct reload...")

                    # Try direct reload
                    try:
                        # Create a fresh vector store
                        vector_store = get_llamaindex_vector_store(
                            session_id=session_id,
                            dimension=EMBEDDING_DIMENSION,
                            index_type=FAISS_INDEX_TYPE,
                            embedding_model=EMBEDDING_MODEL_NAME
                        )

                        # Replace the FAISS index
                        vector_store.faiss_store._faiss_index = loaded_faiss_index

                        # Reload the storage context
                        from llama_index.core import StorageContext, load_index_from_storage
                        vector_store.storage_context = StorageContext.from_defaults(
                            persist_dir=persist_dir,
                            vector_store=vector_store.faiss_store
                        )

                        # Reload the index
                        vector_store.index = load_index_from_storage(
                            storage_context=vector_store.storage_context
                        )

                        logger.info(f"Successfully reloaded vector store with {loaded_faiss_index.ntotal} vectors")
                        return vector_store
                    except Exception as reload_err:
                        logger.error(f"Error during direct reload: {reload_err}", exc_info=True)
                        return None
            except Exception as e:
                logger.error(f"Failed to load LlamaIndex vector store for session '{session_id}': {e}", exc_info=True)
        else:
            missing = []
            if not os.path.exists(faiss_file_path): missing.append("FAISS index")
            if not os.path.exists(docstore_path): missing.append("docstore")
            logger.warning(f"Vector store directory exists but missing required files: {', '.join(missing)}")
    else:
        logger.warning(f"Vector store directory not found for session '{session_id}' at {persist_dir}")

    return None

def load_project_vector_store(project_id: str) -> Optional[LlamaIndexVectorStore]:
    """
    Load a persisted vector store for project-based memory.

    Args:
        project_id: Project ID to load

    Returns:
        Loaded vector store or None if not found or error
    """
    from .config import PROJECT_INDEX_DIR
    persist_dir = os.path.join(PROJECT_INDEX_DIR, f"{project_id}_vector_store")
    logger.info(f"Attempting to load LlamaIndex vector store for project '{project_id}' from {persist_dir}")

    # Check if the directory exists
    if os.path.exists(persist_dir):
        # Check if the FAISS index file exists
        faiss_file_path = os.path.join(persist_dir, "vector_store", "index.faiss")
        docstore_path = os.path.join(persist_dir, "docstore.json")

        if os.path.exists(faiss_file_path) and os.path.exists(docstore_path):
            logger.info(f"Found FAISS index file at {faiss_file_path} and docstore at {docstore_path}")

            try:
                # Try to read the FAISS index to verify it's valid
                import faiss
                try:
                    loaded_faiss_index = faiss.read_index(faiss_file_path)
                    logger.info(f"Successfully read FAISS index with {loaded_faiss_index.ntotal} vectors")
                except Exception as faiss_err:
                    logger.error(f"Error reading FAISS index: {faiss_err}", exc_info=True)
                    return None

                # Load the vector store
                from .config import EMBEDDING_DIMENSION, FAISS_INDEX_TYPE, EMBEDDING_MODEL_NAME
                vector_store = get_llamaindex_vector_store(
                    project_id=project_id,
                    dimension=EMBEDDING_DIMENSION,
                    index_type=FAISS_INDEX_TYPE,
                    embedding_model=EMBEDDING_MODEL_NAME
                )

                # Verify the vector store was loaded correctly
                if vector_store and vector_store.faiss_store and vector_store.faiss_store._faiss_index.ntotal > 0:
                    logger.info(f"Successfully loaded LlamaIndex vector store for project '{project_id}' with {vector_store.faiss_store._faiss_index.ntotal} vectors")
                    return vector_store
                else:
                    logger.warning(f"Vector store loaded but contains no vectors. Attempting direct reload...")

                    # Try direct reload
                    try:
                        # Create a fresh vector store
                        vector_store = get_llamaindex_vector_store(
                            project_id=project_id,
                            dimension=EMBEDDING_DIMENSION,
                            index_type=FAISS_INDEX_TYPE,
                            embedding_model=EMBEDDING_MODEL_NAME
                        )

                        # Replace the FAISS index
                        vector_store.faiss_store._faiss_index = loaded_faiss_index

                        # Reload the storage context
                        from llama_index.core import StorageContext, load_index_from_storage
                        vector_store.storage_context = StorageContext.from_defaults(
                            persist_dir=persist_dir,
                            vector_store=vector_store.faiss_store
                        )

                        # Reload the index
                        vector_store.index = load_index_from_storage(
                            storage_context=vector_store.storage_context
                        )

                        logger.info(f"Successfully reloaded vector store with {loaded_faiss_index.ntotal} vectors")
                        return vector_store
                    except Exception as reload_err:
                        logger.error(f"Error during direct reload: {reload_err}", exc_info=True)
                        return None
            except Exception as e:
                logger.error(f"Failed to load LlamaIndex vector store for project '{project_id}': {e}", exc_info=True)
        else:
            missing = []
            if not os.path.exists(faiss_file_path): missing.append("FAISS index")
            if not os.path.exists(docstore_path): missing.append("docstore")
            logger.warning(f"Vector store directory exists but missing required files: {', '.join(missing)}")
    else:
        logger.warning(f"Vector store directory not found for project '{project_id}' at {persist_dir}")

    return None

# --- Search Pipeline ---

def search_index(
    vector_store: Optional[LlamaIndexVectorStore] = None,
    query: str = "",
    top_k: int = 5,
    context_filter: str = None,
    session_id: str = None,
    project_id: str = None
) -> List[Dict]:
    """
    Search the index for the query, optionally narrowing by context.

    Args:
        vector_store: Vector store to search
        query: Search query
        top_k: Number of results to return
        context_filter: Optional filter for results
        session_id: Optional session ID for loading vector store
        project_id: Optional project ID for loading vector store

    Returns:
        List of search results
    """
    # Load vector store if not provided
    if not vector_store:
        if project_id:
            logger.info(f"No vector store provided, attempting to load from project '{project_id}'")
            vector_store = load_project_vector_store(project_id)
            if vector_store:
                logger.info(f"Successfully loaded vector store for project '{project_id}'")
            else:
                logger.warning(f"Failed to load vector store for project '{project_id}'")
                return []  # Return empty results if no vector store available
        elif session_id:
            logger.info(f"No vector store provided, attempting to load from session '{session_id}'")
            vector_store = load_session_vector_store(session_id)
            if vector_store:
                logger.info(f"Successfully loaded vector store for session '{session_id}'")
            else:
                logger.warning(f"Failed to load vector store for session '{session_id}'")
                return []  # Return empty results if no vector store available
        else:
            logger.error("No vector store, project_id, or session_id provided for search")
            return []

    # Ensure we have a vector store
    if not vector_store:
        logger.error("No vector store available for search")
        return []

    # Use the query engine approach for better results
    logger.info(f"Searching for: '{query}' (top_k={top_k})")
    try:
        # Check if we have a valid index
        if vector_store.index:
            # Create a query engine with the specified top_k
            query_engine = vector_store.index.as_query_engine(similarity_top_k=top_k)

            # Execute the query
            response = query_engine.query(query)

            # Extract source nodes
            results = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                logger.info(f"Found {len(response.source_nodes)} source nodes using query engine")
                for i, node in enumerate(response.source_nodes):
                    # Convert node to dictionary format
                    result = {
                        "node_id": node.node_id,
                        "content": node.text,
                        "score": node.score if hasattr(node, 'score') else 0.0
                    }

                    # Add metadata if available
                    if hasattr(node, 'metadata') and node.metadata:
                        result.update(node.metadata)

                    results.append(result)
            else:
                logger.warning("Query engine did not return source nodes, falling back to traditional search")
                query_embedding = embed_text(query)
                results = vector_store.search(query_embedding, top_k=top_k)
        else:
            # Fallback to traditional search if index is not available
            logger.warning("Vector store does not have a valid index, falling back to traditional search")
            query_embedding = embed_text(query)
            results = vector_store.search(query_embedding, top_k=top_k)
    except Exception as e:
        logger.error(f"Error using query engine: {e}. Falling back to traditional search.")
        # Fallback to traditional search
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
        similarity = r.get("score", 0.0)

        # Add score to the result itself
        r["score"] = similarity
        r["similarity"] = similarity

        # Ensure content is included in the result
        if "content" in r:
            # Content is already present, make sure it's not empty
            if not r["content"]:
                logger.warning(f"Empty content found in result for {r.get('file_path')}")
                _retrieve_content_for_result(r)
        else:
            logger.warning(f"No content found in result for {r.get('file_path')}")
            _retrieve_content_for_result(r)

        r["citation"] = {
            "source_path": r.get("source_path", r.get("file_path")),
            "source_name": r.get("source_name"),
            "source_type": r.get("source_type"),
            "start": r.get("start"),
            "end": r.get("end"),
            "score": similarity  # Add similarity score to citation
        }

    return results

def _retrieve_content_for_result(result: Dict) -> None:
    """
    Retrieve content for a search result using multiple fallback methods.

    Args:
        result: Search result to retrieve content for
    """
    document_id = result.get("document_id")
    if not document_id:
        logger.warning("Cannot retrieve content: no document_id in result")
        return

    # Method 1: Try to get from content file (most reliable)
    try:
        content = read_document_content(document_id)
        if content:
            logger.info(f"Retrieved content for document {document_id} from content file")
            result["content"] = content
            return
    except Exception as content_e:
        logger.warning(f"Error retrieving content from content file: {str(content_e)}")

    # Method 2: Try to get from document registry
    try:
        doc_registry = get_document_registry(document_id)
        if doc_registry and "metadata" in doc_registry and "content" in doc_registry["metadata"]:
            content = doc_registry["metadata"]["content"]
            logger.info(f"Retrieved content for document {document_id} from registry")
            result["content"] = content
            return
    except Exception as registry_e:
        logger.warning(f"Error retrieving content from registry: {str(registry_e)}")

    # Method 3: Try to get from vector store
    try:
        # Get project_id from document registry
        doc_registry = get_document_registry(document_id)
        if doc_registry and "projects" in doc_registry and doc_registry["projects"]:
            project_id = doc_registry["projects"][0]

            # Load vector store for project
            vector_store = load_project_vector_store(project_id)
            if vector_store:
                # Get chunks for document
                chunks = vector_store.get_chunks_by_document_id(document_id)
                if chunks:
                    # Combine content from all chunks
                    content = "\n".join([chunk.get("content", "") for chunk in chunks])
                    if content:
                        logger.info(f"Retrieved content for document {document_id} from vector store")
                        result["content"] = content
                        return
        else:
            logger.warning(f"No project_id found for document {document_id}")
    except Exception as e:
        logger.error(f"Error retrieving content for document {document_id}: {str(e)}")

    # If all methods fail, set empty content
    result["content"] = "Content not available"
    logger.error(f"Failed to retrieve content for document {document_id} using all methods")

# --- Deep Research Pipeline (DSPy Agent Integration) ---

from .dspy_config import DSPyAgentRegistry #, initialize_dspy

def deep_research(
    vector_store: Optional[LlamaIndexVectorStore] = None,
    query: str = "",
    top_k: int = 5,
    mode: str = "summarize",
    context_filter: str = None,
    session_id: str = None,
    project_id: str = None
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
        project_id: Optional project ID for loading vector store

    Returns:
        List of search results with agent-generated content
    """
    # Ensure DSPy is configured and agents are registered
    #initialize_dspy()  # Make sure DSPy is ready

    # Get search results
    results = search_index(
        vector_store=vector_store,
        query=query,
        top_k=top_k,
        context_filter=context_filter,
        session_id=session_id,
        project_id=project_id
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
                    # Provide all required fields for summarizer
                    r["summary"] = agent(
                        content=r["content"],
                        document=r["content"],  # Duplicate content in document field
                        context=r["content"],   # Use content as context
                        query="Summarize this document"  # Default query
                    ).get("summary", "")
                elif mode == "answer":
                    # Provide all required fields for answerer
                    r["answer"] = agent(
                        context=r["content"],
                        query=query if query else "Answer based on this content",
                        document=r["content"],  # Duplicate content in document field
                        content=r["content"]    # Duplicate content in content field
                    ).get("answer", "")
                elif mode == "extract":
                    # Provide all required fields for extractor
                    r["info"] = agent(
                        content=r["content"],
                        document=r["content"],  # Duplicate content in document field
                        context=r["content"],   # Use content as context
                        query="Extract key information"  # Default query
                    ).get("info", "")
                elif mode == "chain_of_thought":
                    # Provide all required fields for chain_of_thought
                    r["cot"] = agent(
                        content=r["content"],
                        document=r["content"],  # Duplicate content in document field
                        context=r["content"],   # Use content as context
                        query=query if query else "Analyze this document"  # Use query or default
                    ).get("cot", "")
                elif mode == "analyze":
                    # Provide all required fields for document_analyzer
                    analysis = agent(
                        document=r["content"],
                        context=r["content"],  # Use content as context
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

    return results
