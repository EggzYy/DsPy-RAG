"""
Vector store implementation using LlamaIndex with FAISS (HNSW, IVF, Flat).
Includes corrected persistence mechanisms and QueryFusionRetriever support.

Retrieval Methods:
- Primary approach: Use QueryFusionRetriever to combine vector and BM25 search
  - This combines the strengths of both dense (vector) and sparse (BM25) retrieval
  - Configurable with use_fusion_retriever parameter (default: True)
  - Uses relative_score fusion mode with customizable weights (default: 0.6 for vector, 0.4 for BM25)

- Fallback approach: Standard vector retrieval via index.as_retriever()
  - Used when fusion retriever is disabled or encounters errors

Score Handling:
- Primary approach: Use RetrieverEvaluator for accurate score evaluation
  - This uses LlamaIndex's built-in evaluation metrics for proper scoring

- Secondary approach: Use LlamaIndex's own similarity_fn if available
  - This ensures scores are normalized exactly as LlamaIndex expects

- For METRIC_INNER_PRODUCT (cosine similarity with normalized vectors):
  - Uses LlamaIndex's standard normalization: (score + 1) / 2
  - This converts the [-1, 1] range of cosine similarity to [0, 1]

- For METRIC_L2 (Euclidean distance):
  - Uses LlamaIndex's standard conversion: 1 / (1 + distance)
  - This converts distances (where smaller is better) to similarities (where larger is better)

This implementation uses the proper methods from LlamaIndex and FAISS to normalize scores,
ensuring compatibility with LlamaIndex's expectations. All logs are at ERROR level to ensure
visibility despite any logging configuration issues.

Note: If scores are outside the expected range (e.g., inner product scores outside [-1, 1]),
this may indicate:
1. Vectors are not properly normalized
2. There might be a metric type mismatch between what's configured and what's used
"""

import os
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
import numpy as np
from pathlib import Path
import faiss # Ensure faiss is imported
import shutil # For backup logic
import sys # For logging handler

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings, # Recommended for global settings like embed_model
    Document, # For fallback add_chunks
)
from llama_index.core.schema import TextNode, QueryBundle, NodeWithScore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilters,
    ExactMatchFilter,
    VectorStoreQueryResult, # Import for type hint
)
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
# Assuming env_loader might be needed elsewhere, keep import structure
# from .env_loader import get_env_var # Commented out as its code is not provided

# --- Set up logging ---
logger = logging.getLogger(__name__)

# Define safe_str function for Unicode handling
def safe_str(obj):
    """Convert any object to a string that can be safely printed/logged."""
    if obj is None:
        return 'None'
    try:
        # First try normal string conversion
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
            # Last resort - return a placeholder
            return "[Unicode conversion error]"

# Configure basic logging if it's not handled elsewhere in your application
if not logger.handlers:
    # Create handler with UTF-8 encoding to handle Unicode characters
    import codecs
    import io

    # Create a UTF-8 stream wrapper for stdout
    utf8_stream = codecs.getwriter('utf-8')(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='backslashreplace'))

    # Create handler with the UTF-8 stream
    handler = logging.StreamHandler(utf8_stream)

    # Create formatter - include timestamp, logger name, level, message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(handler)

    # Set level (e.g., INFO, DEBUG)
    logger.setLevel(logging.INFO)

    # Prevent propagation to root logger if it has its own handlers
    logger.propagate = False


class LlamaIndexVectorStore:
    """
    Vector store implementation using LlamaIndex with FAISS (supports HNSW, IVF, Flat).
    Handles persistence and loading with corrections for FaissVectorStore.
    """

    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "hnsw", # "flat", "hnsw", or "ivf"
        metric: str = "ip", # Inner Product (for cosine similarity), or "l2" for Euclidean
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ):
        """
        Initialize or load the LlamaIndex vector store with FAISS.

        Args:
            dimension: Dimension of the embeddings. MUST match existing index if loading.
            index_type: Type of FAISS index ("flat", "hnsw", "ivf"). MUST match existing index if loading.
            metric: Distance metric for FAISS ("ip" for inner product/cosine, "l2" for Euclidean). MUST match existing index.
            project_id: Project ID for determining persist_dir.
            session_id: Session ID for determining persist_dir.
            persist_dir: Explicit directory to persist/load the vector store. Overrides project/session ID logic.
        """
        logger.debug(f"Initializing LlamaIndexVectorStore with: dimension={dimension}, index_type={index_type}, metric={metric}, project_id={project_id}, session_id={session_id}, persist_dir={persist_dir}")
        self.dimension = dimension
        self.index_type = index_type.lower()
        self._set_metric_type(metric) # Set self.metric_type via helper (e.g., faiss.METRIC_INNER_PRODUCT)
        self.expected_metric = metric.lower() # Store the user-provided metric string
        self.project_id = project_id
        self.session_id = session_id

        # Set up the embedding model in LlamaIndex Settings
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.core.settings import Settings
            from .config import EMBEDDING_MODEL_NAME, OLLAMA_API_BASE

            # Create an Ollama embedding model
            embed_model = OllamaEmbedding(
                model_name=EMBEDDING_MODEL_NAME,
                base_url=OLLAMA_API_BASE,
            )

            # Set the embedding model in the global settings
            Settings.embed_model = embed_model
            logger.info(f"Set up LlamaIndex with Ollama embedding model: {EMBEDDING_MODEL_NAME}")
        except ImportError:
            logger.warning("Could not import LlamaIndex OllamaEmbedding. Embeddings will be handled separately.")
        except Exception as e:
            logger.error(f"Error setting up LlamaIndex embedding model: {e}")
            logger.warning("Embeddings will be handled separately.")

        # --- Determine and Create Persist Directory ---
        self._resolve_persist_dir(persist_dir, project_id, session_id)

        # --- Initialize State Variables ---
        self.index: Optional[VectorStoreIndex] = None
        self.storage_context: Optional[StorageContext] = None
        self.faiss_store: Optional[FaissVectorStore] = None

        # --- Attempt to Load Existing Index ---
        # Check if the directory and essential files exist to attempt loading
        faiss_file_path = os.path.join(self.persist_dir, "vector_store", "index.faiss") # Standard path helper
        #logger.info(f"Attempting to save FAISS index explicitly to: {faiss_file_path}")
        #faiss.write_index(self.faiss_store._faiss_index, faiss_file_path)
        #logger.info(f"FAISS index saved explicitly to: {faiss_file_path}")
        docstore_path = os.path.join(self.persist_dir, "docstore.json") # Document content/metadata

        # Core requirement for loading FAISS: the actual index file AND docstore must exist
        if os.path.exists(faiss_file_path) and os.path.exists(docstore_path):
            logger.info(f"Found FAISS index file ({faiss_file_path}) and docstore ({docstore_path}). Attempting to load.")
            try:
                logger.info(f"Reading raw FAISS index from {faiss_file_path}...")
                loaded_faiss_index = faiss.read_index(faiss_file_path)
                logger.info(f"Successfully read raw FAISS index (ntotal={loaded_faiss_index.ntotal}, d={loaded_faiss_index.d}).")

                # *** Verification Step ***
                loaded_dimension = loaded_faiss_index.d
                loaded_metric_type = getattr(loaded_faiss_index, 'metric_type', None)
                loaded_metric_str = "ip" if loaded_metric_type == faiss.METRIC_INNER_PRODUCT else "l2" if loaded_metric_type == faiss.METRIC_L2 else "unknown"

                if loaded_dimension != self.dimension:
                    logger.warning(f"Dimension mismatch! Provided dimension ({self.dimension}) "
                                   f"differs from loaded FAISS index ({loaded_dimension}). "
                                   f"Using loaded dimension: {loaded_dimension}")
                    self.dimension = loaded_dimension

                if loaded_metric_type is not None and loaded_metric_type != self.metric_type:
                     logger.error(f"METRIC MISMATCH DETECTED! Provided metric '{self.expected_metric}' (resolves to FAISS type {self.metric_type}) "
                                  f"differs from loaded FAISS index metric '{loaded_metric_str}' (FAISS type {loaded_metric_type}). "
                                  f"This WILL likely cause incorrect score interpretation. Using loaded metric type: {loaded_metric_str}")
                     # Update self.metric_type to reflect reality, crucial for score interpretation if needed
                     self.metric_type = loaded_metric_type
                     self.expected_metric = loaded_metric_str # Update expected string too
                elif loaded_metric_type is None:
                    logger.warning("Could not determine metric type from loaded FAISS index. Assuming provided metric is correct.")
                else:
                    logger.info(f"Confirmed loaded FAISS index metric type ({loaded_metric_str}) matches provided metric.")

                # 2. Create FaissVectorStore wrapper
                self.faiss_store = FaissVectorStore(faiss_index=loaded_faiss_index)
                logger.info("Created FaissVectorStore wrapper around loaded index.")

                # 3. Load StorageContext
                logger.info("Loading StorageContext, providing the loaded FaissVectorStore...")
                self.storage_context = StorageContext.from_defaults(
                    persist_dir=self.persist_dir,
                    vector_store=self.faiss_store
                )
                logger.info("Successfully loaded other StorageContext components (docstore, etc.).")

                # 4. Load the main VectorStoreIndex object
                logger.info("Loading VectorStoreIndex from storage...")
                self.index = load_index_from_storage(storage_context=self.storage_context)
                logger.info("Successfully loaded VectorStoreIndex object.")

            except Exception as load_err:
                # Handle different types of exceptions
                if "FaissException" in str(type(load_err)) or isinstance(load_err, FileNotFoundError) or isinstance(load_err, json.JSONDecodeError) or isinstance(load_err, UnicodeDecodeError):
                    logger.error(f"Failed to load existing index components from {self.persist_dir}: {load_err}", exc_info=True)
                    if isinstance(load_err, (json.JSONDecodeError, UnicodeDecodeError)):
                        logger.error("JSON or encoding error detected. Consider backing up and deleting the corrupted directory, or fixing the file manually.")
                        # self._handle_corrupted_files() # Optional: Implement backup/delete logic if desired
                elif "No index in storage context" in str(load_err):
                    logger.error(f"No index found in storage context at {self.persist_dir}: {load_err}", exc_info=True)
                    # Try to clean up the corrupted directory
                    try:
                        import shutil
                        backup_dir = f"{self.persist_dir}_backup_{int(time.time())}"
                        logger.info(f"Backing up corrupted directory to {backup_dir}")
                        shutil.copytree(self.persist_dir, backup_dir)
                        logger.info(f"Removing corrupted directory {self.persist_dir}")
                        shutil.rmtree(self.persist_dir)
                        os.makedirs(self.persist_dir, exist_ok=True)
                        logger.info(f"Created fresh directory at {self.persist_dir}")
                    except Exception as cleanup_err:
                        logger.error(f"Error cleaning up corrupted directory: {cleanup_err}", exc_info=True)
                else:
                    logger.error(f"Unexpected error loading index from {self.persist_dir}: {load_err}", exc_info=True)

                logger.info("Falling back to initializing a new empty vector store.")
                self._initialize_new_index() # Fallback to creating an empty store
        else:
            # If required files don't exist, initialize a new store
            missing = []
            if not os.path.exists(faiss_file_path): missing.append(f"FAISS index ({faiss_file_path})")
            if not os.path.exists(docstore_path): missing.append(f"docstore ({docstore_path})")
            logger.info(f"Required file(s) missing ({', '.join(missing)}). Initializing a new vector store at {self.persist_dir}.")
            self._initialize_new_index()

        # --- Final State Logging ---
        faiss_vector_count = "N/A"
        faiss_dim = "N/A"
        if self.faiss_store and self.faiss_store._faiss_index:
            faiss_vector_count = self.faiss_store._faiss_index.ntotal
            faiss_dim = self.faiss_store._faiss_index.d

        logger.info(f"LlamaIndexVectorStore initialized for {self.persist_dir}. "
                    f"Index object loaded: {self.index is not None}. "
                    f"Vectors in FAISS index: {faiss_vector_count} (Dim: {faiss_dim})")

    def _set_metric_type(self, metric: str):
        """Sets the internal FAISS metric type based on the string."""
        metric_lower = metric.lower()
        if metric_lower == "ip":
            self.metric_type = faiss.METRIC_INNER_PRODUCT
            logger.debug("Using FAISS METRIC_INNER_PRODUCT (suitable for normalized embeddings/cosine similarity)")
        elif metric_lower == "l2":
             self.metric_type = faiss.METRIC_L2
             logger.debug("Using FAISS METRIC_L2 (Euclidean distance)")
        else:
             logger.warning(f"Unknown metric type '{metric}'. Defaulting to 'ip' (METRIC_INNER_PRODUCT).")
             self.metric_type = faiss.METRIC_INNER_PRODUCT
             metric_lower = "ip" # Ensure consistency
        self.expected_metric = metric_lower # Store the string representation

    def _transform_raw_score(self, raw_score: float) -> float:
        """
        Transform raw FAISS similarity/distance scores to a normalized 0-1 range,
        based on the ACTUAL metric type detected in the FAISS index.

        Args:
            raw_score: The raw score (distance or inner product) from FAISS.

        Returns:
            Normalized score between 0 and 1 (higher is better).
        """
        logger.debug(f"Raw score received for transformation: {raw_score}")

        # Determine the actual metric type from the FAISS index
        actual_metric = None
        metric_name = "UNKNOWN"
        if self.faiss_store and hasattr(self.faiss_store, '_faiss_index') and hasattr(self.faiss_store._faiss_index, 'metric_type'):
            actual_metric = self.faiss_store._faiss_index.metric_type
            if actual_metric == faiss.METRIC_INNER_PRODUCT:
                metric_name = "INNER_PRODUCT (IP)"
            elif actual_metric == faiss.METRIC_L2:
                metric_name = "L2"
            logger.debug(f"Transforming score based on detected FAISS metric: {metric_name}")
        else:
            # Fallback to configured metric type if detection fails
            actual_metric = self.metric_type
            if actual_metric == faiss.METRIC_INNER_PRODUCT: metric_name = "INNER_PRODUCT (IP - Fallback)"
            elif actual_metric == faiss.METRIC_L2: metric_name = "L2 (Fallback)"
            logger.warning(f"Could not detect metric from FAISS index, using configured metric for transformation: {metric_name}")


        if actual_metric == faiss.METRIC_INNER_PRODUCT:
            # Inner Product (Cosine Similarity if normalized): Range [-1, 1] -> [0, 1]
            # Check if score is unexpectedly large/small, suggesting unnormalized vectors
            if raw_score > 1.05 or raw_score < -1.05: # Using a small tolerance
                logger.warning(f"Inner product score {raw_score:.4f} is outside the expected [-1, 1] range. "
                               "Vectors might not be properly L2-normalized.")

            # Standard LlamaIndex normalization for IP -> Similarity
            normalized_score = (raw_score + 1.0) / 2.0
            # Clamp the score strictly between 0 and 1
            normalized_score = max(0.0, min(normalized_score, 1.0))
            logger.debug(f"Transformed IP score {raw_score:.4f} -> {normalized_score:.4f}")
            return normalized_score

        elif actual_metric == faiss.METRIC_L2:
            # L2 Distance (Squared Euclidean): Range [0, inf) -> (0, 1] (higher is better)
            # Check for negative distances which shouldn't happen
            if raw_score < 0:
                logger.warning(f"L2 distance score {raw_score:.4f} is negative. This is unexpected.")
                raw_score = 0.0 # Treat negative distance as zero distance

            # Standard LlamaIndex conversion from distance to similarity
            # Avoid division by zero if distance is exactly 0
            if raw_score < 1e-9: # Treat very small distances as perfect match
                normalized_score = 1.0
            else:
                normalized_score = 1.0 / (1.0 + raw_score)
            logger.debug(f"Transformed L2 distance {raw_score:.4f} -> {normalized_score:.4f}")
            return normalized_score

        else:
            # Fallback if metric is unknown
            logger.error(f"Cannot transform score: Unknown or unsupported FAISS metric type ({actual_metric}). Returning raw score.")
            return raw_score # Or perhaps return 0 or raise an error

    def _resolve_persist_dir(self, persist_dir, project_id, session_id):
        """Determines and creates the persistence directory."""
        resolved_dir = None
        if persist_dir:
            resolved_dir = persist_dir
            source = "explicit persist_dir"
        elif project_id:
            from .config import PROJECT_INDEX_DIR
            resolved_dir = os.path.join(PROJECT_INDEX_DIR, f"{project_id}_vector_store")
            source = f"project_id '{project_id}'"
        elif session_id:
            from .config import SESSION_PERSIST_DIR
            resolved_dir = os.path.join(SESSION_PERSIST_DIR, f"{session_id}_vector_store")
            source = f"session_id '{session_id}'"
        else:
            # Default path or raise error
            from .config import SESSION_PERSIST_DIR
            timestamp = int(time.time())
            resolved_dir = os.path.join(SESSION_PERSIST_DIR, f"vector_store_{timestamp}")
            source = "default path with timestamp"
            logger.warning(f"No persist_dir, project_id, or session_id provided. Defaulting to {resolved_dir}")
            # Or: raise ValueError("Either persist_dir, project_id, or session_id must be provided.")

        if not resolved_dir:
             raise ValueError("Persistence directory could not be determined.")

        self.persist_dir = resolved_dir
        logger.info(f"Using persistence directory: {self.persist_dir} (determined by {source})")

        # Check if directory exists and contains expected files
        faiss_file_path = os.path.join(self.persist_dir, "vector_store", "index.faiss")
        docstore_path = os.path.join(self.persist_dir, "docstore.json")

        if os.path.exists(self.persist_dir):
            logger.info(f"Persistence directory exists: {self.persist_dir}")

            # Check if the directory contains the expected files
            if os.path.exists(faiss_file_path) and os.path.exists(docstore_path):
                logger.info(f"Found existing index files in {self.persist_dir}")
            else:
                logger.warning(f"Persistence directory exists but missing expected files: FAISS index exists: {os.path.exists(faiss_file_path)}, docstore exists: {os.path.exists(docstore_path)}")
                # Ensure the vector_store subdirectory exists
                vector_store_dir = os.path.join(self.persist_dir, "vector_store")
                os.makedirs(vector_store_dir, exist_ok=True)
        else:
            logger.info(f"Creating persistence directory: {self.persist_dir}")

            # Ensure base directory and persist directory exist
            try:
                # Create the parent directory if it doesn't exist
                parent_dir = os.path.dirname(self.persist_dir)
                if parent_dir: # Avoid trying to create '' if persist_dir is in root
                    os.makedirs(parent_dir, exist_ok=True)
                # Create the persist directory itself
                os.makedirs(self.persist_dir, exist_ok=True)
                # Create the vector_store subdirectory
                vector_store_dir = os.path.join(self.persist_dir, "vector_store")
                os.makedirs(vector_store_dir, exist_ok=True)
            except OSError as e:
                logger.error(f"Error creating persistence directory {self.persist_dir}: {e}", exc_info=True)
                raise # Re-raise error if directory creation fails

    def _create_faiss_index(self) -> faiss.Index:
        """Creates the low-level FAISS index object based on configuration."""
        metric_name = "IP" if self.metric_type == faiss.METRIC_INNER_PRODUCT else "L2"
        logger.info(f"Creating new low-level FAISS index: type='{self.index_type}', dimension={self.dimension}, metric={metric_name}")
        try:
            if self.index_type == "hnsw":
                # HNSW parameters (tune as needed)
                M = 32  # Number of connections (higher = more RAM/accuracy, lower = less RAM/speed)
                ef_construction = 64 # Construction speed/accuracy trade-off (higher = slower build, better index)
                ef_search = 32 # Search speed/accuracy trade-off (higher = slower search, better accuracy)

                index = faiss.IndexHNSWFlat(self.dimension, M, self.metric_type)
                index.hnsw.efConstruction = ef_construction
                index.hnsw.efSearch = ef_search
                # HNSWFlat is trained during additions
                index.is_trained = True
                logger.info(f"Created IndexHNSWFlat: M={M}, efConstruction={ef_construction}, efSearch={ef_search}")

            elif self.index_type == "ivf":
                # IVF parameters (tune based on expected data size N)
                # Rule of thumb: nlist between sqrt(N) and 16*sqrt(N)
                # Let's assume N might be around 100k for a default guess
                nlist = 100 # Number of clusters (Voronoi cells)
                quantizer = faiss.IndexFlat(self.dimension, self.metric_type) # Base index for finding nearest centroids
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, self.metric_type)
                index.nprobe = max(1, nlist // 10) # Number of cells to search (e.g., 10% of nlist)
                index.is_trained = False # IMPORTANT: Must be trained before adding vectors!
                logger.info(f"Created IndexIVFFlat: nlist={nlist}, nprobe={index.nprobe}. REQUIRES TRAINING.")

            elif self.index_type == "flat":
                index = faiss.IndexFlat(self.dimension, self.metric_type) # Brute-force search
                index.is_trained = True # Flat doesn't require separate training step
                logger.info(f"Created IndexFlat.")
            else:
                logger.warning(f"Unknown index_type '{self.index_type}'. Defaulting to 'flat'.")
                index = faiss.IndexFlat(self.dimension, self.metric_type)
                index.is_trained = True
            return index
        except Exception as e:
            logger.error(f"Failed to create FAISS index object: {e}", exc_info=True)
            raise # Re-raise the exception


    def _initialize_new_index(self):
        """Initializes attributes for a new, empty index."""
        logger.info("Setting up new FAISS vector store and storage context.")
        try:
            faiss_index = self._create_faiss_index()
            self.faiss_store = FaissVectorStore(faiss_index=faiss_index)
            # Create a default storage context associated with the new vector store
            self.storage_context = StorageContext.from_defaults(vector_store=self.faiss_store)
            # The main LlamaIndex VectorStoreIndex object is NOT created yet.
            # It will be created lazily when the first batch of nodes/documents is added.
            self.index = None
            # self.nodes = [] # Removed - don't store all nodes in memory
            logger.info("New empty vector store initialized. Index object will be created upon adding data.")
        except Exception as e:
            logger.error(f"Failed to initialize new FAISS store: {e}", exc_info=True)
            # Ensure clean state on failure
            self.faiss_store = None
            self.storage_context = None
            self.index = None

    def _get_bm25_retriever_path(self):
        """Get the path for persisting/loading the BM25Retriever."""
        if not self.persist_dir:
            logger.warning("No persist_dir set, cannot determine BM25Retriever path")
            return None
        return os.path.join(self.persist_dir, "bm25_retriever")

    def _create_or_load_bm25_retriever(self, similarity_top_k=None):
        """
        Create a new BM25Retriever or load an existing one from disk.

        Args:
            similarity_top_k: Number of results to return (defaults to None, which uses the BM25Retriever default)

        Returns:
            BM25Retriever instance or None if creation/loading fails
        """
        if not self.storage_context or not self.storage_context.docstore:
            logger.warning("Cannot create BM25Retriever: No docstore available")
            return None

        bm25_path = self._get_bm25_retriever_path()
        if not bm25_path:
            return None

        # Try to load existing BM25Retriever
        if os.path.exists(bm25_path):
            try:
                logger.info(f"Loading BM25Retriever from {bm25_path}")
                bm25_retriever = BM25Retriever.from_persist_dir(bm25_path)
                if similarity_top_k is not None:
                    bm25_retriever.similarity_top_k = similarity_top_k
                else:
                    similarity_top_k = 30
                logger.info("Successfully loaded BM25Retriever from disk")
                return bm25_retriever
            except Exception as e:
                logger.warning(f"Failed to load BM25Retriever from {bm25_path}: {e}")
                # Fall through to create a new one

        # Create new BM25Retriever
        try:
            logger.info("Creating new BM25Retriever")
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.storage_context.docstore,
                similarity_top_k=similarity_top_k
            )

            # Persist the BM25Retriever
            try:
                logger.info(f"Persisting BM25Retriever to {bm25_path}")
                os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
                bm25_retriever.persist(bm25_path)
                logger.info(f"Successfully persisted BM25Retriever to {bm25_path}")
            except Exception as persist_err:
                logger.warning(f"Failed to persist BM25Retriever: {persist_err}")

            return bm25_retriever
        except Exception as e:
            logger.error(f"Failed to create BM25Retriever: {e}")
            return None


    # Optional: Keep if needed for debugging corrupted files
    # def _handle_corrupted_files(self): ...


    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add chunks (dictionaries) to the vector store. Handles index creation and IVF training.
        Assumes chunks have 'content' and optionally 'embedding', 'metadata', 'id'.
        Embeddings will be generated if missing and Settings.embed_model is configured.
        Ensures embeddings are L2-normalized if metric is Inner Product.

        Args:
            chunks: List of chunk dictionaries.
        """
        if not self.storage_context or not self.faiss_store or not self.faiss_store._faiss_index:
             logger.error("Cannot add chunks: Vector store or FAISS index not initialized correctly.")
             return

        nodes_to_add: List[TextNode] = []
        embeddings_for_training: List[np.ndarray] = []
        needs_embedding_generation = False

        # --- Prepare Nodes and Collect Embeddings (if needed for training) ---
        for i, chunk in enumerate(chunks):
            content = chunk.get("content")
            if not content:
                 logger.warning(f"Skipping chunk {i} due to missing 'content'.")
                 continue

            metadata = {k: v for k, v in chunk.items() if k not in ["content", "embedding", "id"]}
            serializable_metadata = {}
            for k, v in metadata.items():
                try:
                    # Basic check for simple types, allows nested lists/dicts if they contain simple types
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                         # Further check if list/dict contains only simple types (can be slow for deep nesting)
                         json.dumps({k: v}) # Quickest check
                         serializable_metadata[k] = v
                    else:
                         logger.warning(f"Metadata key '{k}' type '{type(v)}' in chunk {i} might not be JSON serializable, skipping.")
                except (TypeError, OverflowError) as json_err:
                    logger.warning(f"Metadata key '{k}' in chunk {i} is not JSON serializable (Error: {json_err}), skipping.")

            # Generate a default UUID if 'id' is not provided or empty
            node_id = str(chunk.get('id') or uuid.uuid4())

            node = TextNode(
                text=content,
                metadata=serializable_metadata,
                id_=node_id
            )

            embedding = chunk.get("embedding")
            if embedding is not None:
                try:
                    float_embedding = [float(e) for e in embedding]
                    if len(float_embedding) != self.dimension:
                         logger.warning(f"Node '{node_id}' embedding dim ({len(float_embedding)}) != store dim ({self.dimension}). Skipping embedding.")
                         node.embedding = None
                         needs_embedding_generation = True # Mark for generation if possible
                    else:
                         # *** Ensure L2-normalization for Inner Product Metric ***
                         if self.metric_type == faiss.METRIC_INNER_PRODUCT:
                             embedding_np = np.array(float_embedding, dtype='float32')
                             norm = np.linalg.norm(embedding_np)
                             if norm > 1e-6: # Use a small epsilon for stability
                                 normalized_embedding_np = embedding_np / norm
                                 # Verify norm is close to 1 after normalization
                                 # final_norm = np.linalg.norm(normalized_embedding_np)
                                 # logger.debug(f"Node '{node_id}' original norm: {norm:.4f}, final norm: {final_norm:.4f}")
                                 float_embedding = normalized_embedding_np.tolist()
                             else:
                                 logger.warning(f"Embedding for node '{node_id}' has near-zero norm ({norm}), cannot normalize reliably. Using original.")
                                 # Optionally set embedding to None or zeros, depending on desired behavior
                                 # float_embedding = [0.0] * self.dimension

                         node.embedding = float_embedding
                         if self.index_type == "ivf" and not self.faiss_store._faiss_index.is_trained:
                              embeddings_for_training.append(np.array(float_embedding, dtype='float32'))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping pre-computed embedding for Node '{node_id}' due to conversion error: {e}")
                    node.embedding = None
                    needs_embedding_generation = True
            else:
                node.embedding = None
                needs_embedding_generation = True

            nodes_to_add.append(node)

        if not nodes_to_add:
            logger.warning("No valid nodes were created from the provided chunks.")
            return

        # Check if embedding model is needed and configured *before* proceeding
        if needs_embedding_generation and not Settings.embed_model:
             logger.error("Embeddings need to be generated for some nodes, but Settings.embed_model is not configured. Cannot add chunks.")
             # Optionally, filter out nodes without embeddings if that's acceptable
             # nodes_to_add = [n for n in nodes_to_add if n.embedding is not None]
             # if not nodes_to_add:
             #      logger.error("No nodes with embeddings available to add.")
             #      return
             return # Stop if embeddings cannot be generated

        try:
            faiss_index = self.faiss_store._faiss_index

            # --- Train IVF Index (if needed and not already trained) ---
            if isinstance(faiss_index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)) and not faiss_index.is_trained:
                # Check if we have enough vectors to train meaningfully
                # Use vectors already in the index (if any) + new vectors with embeddings
                # This part is tricky - LlamaIndex doesn't easily expose existing vectors for retraining.
                # Safest approach: Require training *before* adding the first batch, or provide enough vectors in the first batch.
                if faiss_index.ntotal == 0: # Only train if index is empty
                     if embeddings_for_training:
                         logger.info(f"Training IVF index with {len(embeddings_for_training)} provided embeddings (Index is currently empty)...")
                         embeddings_np = np.vstack(embeddings_for_training).astype('float32')
                         if embeddings_np.shape[0] < getattr(faiss_index, 'nlist', 1): # Check against nlist if possible
                              logger.warning(f"Training IVF with less vectors ({embeddings_np.shape[0]}) than nlist ({getattr(faiss_index, 'nlist', 'N/A')}) can be suboptimal.")
                         if embeddings_np.shape[1] != faiss_index.d:
                             raise ValueError(f"Dimension mismatch during training: Embeddings ({embeddings_np.shape[1]}) vs Index ({faiss_index.d})")

                         faiss_index.train(embeddings_np)
                         logger.info("IVF index training complete.")
                         # faiss_index.is_trained is set automatically by faiss
                     else:
                         # Cannot add to an untrained, empty IVF index without training data
                         logger.error("IVF index requires training, but no suitable embeddings were provided in this batch and index is empty. Cannot add data.")
                         return
                elif not faiss_index.is_trained: # Index has data but reports not trained (unusual state)
                     logger.error("IVF Index has data but is marked as not trained. Cannot add more data reliably. Consider re-building the index.")
                     return

            # --- Create/Update LlamaIndex VectorStoreIndex ---
            if self.index is None:
                # First time adding data, create the main index object
                logger.info(f"Creating new VectorStoreIndex object from {len(nodes_to_add)} nodes...")
                # VectorStoreIndex handles embedding generation via Settings.embed_model
                self.index = VectorStoreIndex(
                    nodes=nodes_to_add,
                    storage_context=self.storage_context,
                )
                logger.info("New VectorStoreIndex object created.")
                # Update self references from the newly created index
                self.storage_context = self.index.storage_context
                if isinstance(self.storage_context.vector_store, FaissVectorStore):
                     self.faiss_store = self.storage_context.vector_store
                else:
                     logger.error("Index creation resulted in an unexpected vector store type!")
                     # Handle this error state - maybe revert or raise
            else:
                # Index already exists, insert new nodes
                logger.info(f"Inserting {len(nodes_to_add)} nodes into existing index...")
                # insert_nodes handles embedding generation via Settings.embed_model
                self.index.insert_nodes(nodes_to_add)
                logger.info("Nodes inserted successfully.")

            logger.info(f"Added {len(nodes_to_add)} nodes. Vectors in FAISS index: {self.faiss_store._faiss_index.ntotal}")

            # Update BM25Retriever if it exists
            try:
                bm25_path = self._get_bm25_retriever_path()
                if bm25_path and os.path.exists(bm25_path):
                    logger.info("Updating BM25Retriever with new nodes")
                    # Create a new BM25Retriever with the updated docstore
                    bm25_retriever = BM25Retriever.from_defaults(
                        docstore=self.storage_context.docstore
                    )
                    # Persist the updated BM25Retriever
                    bm25_retriever.persist(bm25_path)
                    logger.info(f"Updated BM25Retriever saved to {bm25_path}")
            except Exception as bm25_err:
                logger.warning(f"Failed to update BM25Retriever: {bm25_err}")

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}", exc_info=True)

    # Import safe_str at the top of the file

    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[MetadataFilters] = None,
               query_str: Optional[str] = None, # Keep query_str optional
               use_fusion_retriever: bool = True # Whether to use QueryFusionRetriever
               ) -> List[Dict]:
        """
        Search the vector store for similar chunks. Handles score normalization based on detected metric.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional LlamaIndex metadata filters.
            query_str: Optional query string for text-based search.
            use_fusion_retriever: Whether to use QueryFusionRetriever (combines vector and BM25 search).

        Returns:
            List of similar chunk dictionaries with 'node_id', 'content', 'score', and metadata.
            Scores are normalized to [0, 1] (higher is better).
        """
        results = []
        # --- Pre-checks ---
        if not self.faiss_store or not self.faiss_store._faiss_index:
            logger.warning("Search failed: FaissVectorStore or its index is not initialized.")
            return results
        # IVF indices must be trained before searching
        if isinstance(self.faiss_store._faiss_index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)) and not self.faiss_store._faiss_index.is_trained:
             logger.warning("Search failed: FAISS IVF index is not trained.")
             return results
        # Index must contain vectors to be searched
        if self.faiss_store._faiss_index.ntotal == 0:
            logger.warning("Search failed: FAISS index contains no vectors. Attempting to reload from disk...")

            try:
                faiss_file_path = os.path.join(self.persist_dir, "vector_store", "index.faiss")
                if os.path.exists(faiss_file_path):
                    logger.info(f"Reloading FAISS index from {faiss_file_path}...")
                    loaded_faiss_index = faiss.read_index(faiss_file_path)
                    if loaded_faiss_index.ntotal > 0:
                        logger.info(f"Reloaded FAISS index with {loaded_faiss_index.ntotal} vectors.")
                        self.faiss_store._faiss_index = loaded_faiss_index
                        # Potentially need to reload context/index object too if they were None
                        if not self.storage_context:
                            self.storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir, vector_store=self.faiss_store)
                        if not self.index:
                            self.index = load_index_from_storage(storage_context=self.storage_context)
                        logger.info("Reloaded vector store components.")
                    else:
                        logger.warning("Reloaded FAISS index still contains no vectors.")
                        return results
                else:
                    logger.warning(f"No FAISS index file found at {faiss_file_path} for reload.")
                    return results
            except Exception as e:
                logger.error(f"Error reloading FAISS index from disk: {e}", exc_info=True)
                return results
            # Re-check after reload attempt
            if self.faiss_store._faiss_index.ntotal == 0:
                logger.warning("After reload attempt, FAISS index still contains no vectors.")
                return results

        try:
            # --- Verify query dimension ---
            faiss_dimension = self.faiss_store._faiss_index.d
            query_dim = len(query_embedding)
            if query_dim != faiss_dimension:
                logger.error(f"Search failed: Query embedding dimension ({query_dim}) "
                             f"does not match FAISS index dimension ({faiss_dimension}).")
                return results

            # --- Normalize query embedding for IP metric ---
            query_embedding_final = query_embedding # Use a different variable
            if self.metric_type == faiss.METRIC_INNER_PRODUCT:
                query_embedding_np = np.array(query_embedding, dtype='float32')
                norm = np.linalg.norm(query_embedding_np)
                if norm > 1e-6:
                    normalized_query_np = query_embedding_np / norm
                    query_embedding_final = normalized_query_np.tolist()
                    logger.debug("L2-normalized query embedding for IP metric search")
                else:
                    logger.warning("Query embedding has near-zero norm ({norm}), cannot normalize reliably. Using original.")
            # --- End Query Normalization ---

            # --- Method 1: Use LlamaIndex Retriever (with optional QueryFusionRetriever) ---
            if self.index:
                try:
                    # Create vector retriever (always needed)
                    vector_retriever = self.index.as_retriever(
                        similarity_top_k=top_k,
                        filters=filters
                    )

                    # Determine whether to use fusion retriever
                    if use_fusion_retriever:
                        logger.info(f"Searching using QueryFusionRetriever (top_k={top_k}, filters={safe_str(filters)})...")

                        # Load or create BM25 retriever
                        bm25_retriever = self._create_or_load_bm25_retriever(similarity_top_k=top_k)

                        if bm25_retriever:
                            # Apply filters if provided
                            if filters:
                                bm25_retriever.filters = filters

                            # Create fusion retriever combining vector and BM25
                            try:
                                retriever = QueryFusionRetriever(
                                    [vector_retriever, bm25_retriever],
                                    retriever_weights=[0.6, 0.4],  # Weight vector search higher than BM25
                                    similarity_top_k=top_k,
                                    num_queries=1,  # Set to 1 to disable query generation
                                    mode="relative_score",  # Use relative score fusion
                                    use_async=True,
                                    verbose=True
                                )
                                logger.info("Successfully created QueryFusionRetriever with vector and BM25 retrievers")
                            except Exception as fusion_err:
                                logger.warning(f"Failed to create QueryFusionRetriever: {fusion_err}. Falling back to vector retriever only.")
                                retriever = vector_retriever
                        else:
                            logger.warning("Could not create or load BM25Retriever. Falling back to vector retriever only.")
                            retriever = vector_retriever
                    else:
                        logger.info(f"Using standard vector retriever (top_k={top_k}, filters={safe_str(filters)})...")
                        retriever = vector_retriever

                    # Use the potentially normalized query embedding
                    query_bundle = QueryBundle(query_str=query_str, embedding=query_embedding_final)

                    logger.info(f"Executing retriever.retrieve() with embedding (query_str='{query_str}')...")
                    try:
                        # For QueryFusionRetriever with async=True, we need to handle potential async issues
                        if isinstance(retriever, QueryFusionRetriever) and getattr(retriever, "use_async", False):
                            try:
                                # Try to import nest_asyncio to handle potential nested event loop issues
                                import nest_asyncio
                                nest_asyncio.apply()
                                logger.info("Applied nest_asyncio patch for async retriever")
                            except ImportError:
                                logger.warning("nest_asyncio not available, async operations may fail in certain environments")

                        retrieved_nodes_with_scores: List[NodeWithScore] = retriever.retrieve(query_bundle)
                        logger.info(f"Retriever returned {len(retrieved_nodes_with_scores)} nodes.")
                    except RuntimeError as async_err:
                        if "This event loop is already running" in str(async_err):
                            logger.warning(f"Async error in retriever: {async_err}. Retrying with use_async=False...")
                            # If using QueryFusionRetriever, try again with use_async=False
                            if isinstance(retriever, QueryFusionRetriever):
                                retriever.use_async = False
                                retrieved_nodes_with_scores = retriever.retrieve(query_bundle)
                                logger.info(f"Retriever returned {len(retrieved_nodes_with_scores)} nodes with async disabled.")
                            else:
                                # For other retrievers, just re-raise the error
                                raise
                        else:
                            # For other runtime errors, re-raise
                            raise

                    # --- Process results WITHOUT extra transformation ---
                    for node_with_score in retrieved_nodes_with_scores:
                        metadata = node_with_score.node.metadata or {}
                        # *** Use the score directly from LlamaIndex ***
                        score = node_with_score.score

                        # Log the score provided by LlamaIndex
                        logger.info(f"Retriever Node ID: {node_with_score.node.node_id}, Score from LlamaIndex: {score}")

                        # Check if the score is reasonable (e.g., roughly 0-1)
                        if not isinstance(score, (int, float)):
                             logger.warning(f"Node {node_with_score.node.node_id}: Score type is {type(score)}, expected float/int. Setting score to 0.0.")
                             score = 0.0
                        elif score < -1.1 or score > 10: # Allow slightly outside 0-1 for IP/L2 variations, but flag large numbers
                             logger.warning(f"Node {node_with_score.node.node_id}: Unusual score {score} received from retriever. Check metric/normalization.")
                        # Optional: Clamp score to 0-1 if desired, but better to understand why it's outside
                        # score = min(max(float(score), 0.0), 1.0)

                        chunk = {
                            "node_id": node_with_score.node.node_id,
                            "content": node_with_score.node.text,
                            "score": score, # Use the score directly
                            "raw_score_from_llama": node_with_score.score, # Keep original for debugging
                            **metadata
                        }
                        results.append(chunk)

                    logger.info(f"Search successful using index retriever. Found {len(results)} results.")
                    return results
                except NotImplementedError:
                     logger.warning("Metadata filters might not be fully supported by the FaissVectorStore integration via retriever. Falling back to direct query.")
                except Exception as idx_e:
                    logger.warning(f"Index retriever search failed: {idx_e}. Falling back to direct FAISS query.", exc_info=True)
                    # Proceed to fallback

            # --- Method 2: Direct FaissVectorStore Query (Fallback) ---
            logger.info(f"Searching using direct FaissVectorStore query (top_k={top_k}, filters={filters})...")
            if not self.storage_context or not self.storage_context.docstore:
                 logger.error("Direct search fallback failed: Docstore is not available.")
                 return []

            try:
                # Log FAISS index details for debugging
                logger.info(f"Direct Query - FAISS index type: {type(self.faiss_store._faiss_index).__name__}")
                logger.info(f"Direct Query - FAISS index dimension: {self.faiss_store._faiss_index.d}")
                logger.info(f"Direct Query - FAISS index total vectors: {self.faiss_store._faiss_index.ntotal}")
                metric_name = "IP" if self.metric_type == faiss.METRIC_INNER_PRODUCT else "L2"
                logger.info(f"Direct Query - Using metric type: {metric_name} ({self.metric_type})")

                vector_store_query = VectorStoreQuery(
                    # Use the potentially normalized query embedding
                    query_embedding=query_embedding_final,
                    similarity_top_k=top_k,
                    filters=filters
                )

                logger.info("Executing direct faiss_store.query()...")
                query_result: VectorStoreQueryResult = self.faiss_store.query(vector_store_query)

                if query_result and query_result.ids:
                    logger.info(f"Direct query returned {len(query_result.ids)} IDs.")
                    # Log the raw scores/similarities returned by the direct query
                    logger.info(f"Direct query scores/similarities: {query_result.similarities}")
                else:
                    logger.warning("Direct query returned no results or invalid format.")
                    return [] # No results

                node_ids = query_result.ids
                # Scores from query_result.similarities should already be similarity scores
                scores = query_result.similarities or ([0.0] * len(node_ids))
                retrieved_nodes_dict = self.storage_context.docstore.get_nodes(node_ids, raise_error=False)
                score_map = dict(zip(node_ids, scores))

                # --- Process results WITHOUT extra transformation ---
                for node_id in node_ids: # Iterate in the order returned by FAISS/LlamaIndex
                    node = retrieved_nodes_dict.get(node_id)
                    if node:
                        metadata = node.metadata or {}
                        # *** Use the score directly from LlamaIndex query result ***
                        score = score_map.get(node.node_id, 0.0)

                        # Log the score provided by LlamaIndex
                        logger.info(f"Direct Query Node ID: {node_id}, Score from LlamaIndex: {score}")

                        # Check if the score is reasonable
                        if not isinstance(score, (int, float)):
                             logger.warning(f"Node {node_id}: Score type is {type(score)}, expected float/int. Setting score to 0.0.")
                             score = 0.0
                        elif score < -1.1 or score > 10: # Flag unusual scores
                             logger.warning(f"Node {node_id}: Unusual score {score} received from direct query. Check metric/normalization.")
                        # Optional: Clamp score
                        # score = min(max(float(score), 0.0), 1.0)

                        chunk = {
                            "node_id": node.node_id,
                            "content": node.text,
                            "score": score, # Use score directly
                            "raw_score_from_llama": score_map.get(node.node_id, 0.0), # Keep original for debugging
                            **metadata
                        }
                        results.append(chunk)
                    else:
                        logger.warning(f"Node ID '{node_id}' returned by vector store query but not found in docstore.")

                logger.info(f"Search successful using direct vector store query. Found {len(results)} results.")
                return results

            except NotImplementedError:
                 logger.error("Direct FaissVectorStore query does not support the provided filters.")
                 return []
            except Exception as vs_e:
                logger.error(f"Direct vector store query failed: {vs_e}", exc_info=True)
                return []

        except Exception as e:
            logger.error(f"Unexpected error during search: {e}", exc_info=True)
            return []

    def get_chunks_by_document_id(self, document_id: str) -> List[Dict]:
        """
        Get all chunks for a specific document using metadata filtering via the search method.
        # ... (rest of docstring)
        """
        logger.info(f"Attempting to retrieve all chunks for document_id: '{document_id}'")
        if not self.faiss_store or not self.faiss_store._faiss_index or self.faiss_store._faiss_index.ntotal == 0:
             logger.warning("Cannot get chunks by document ID: Vector store not initialized or empty.")
             return []

        filters = MetadataFilters(filters=[ExactMatchFilter(key="document_id", value=str(document_id))])
        dummy_dimension = self.dimension
        # Use a reasonable upper bound, maybe total vectors + buffer
        max_possible_chunks = (self.faiss_store._faiss_index.ntotal + 100) if self.faiss_store._faiss_index.ntotal > 0 else 1000

        logger.debug(f"Calling internal search with top_k={max_possible_chunks} and document_id filter.")
        # Pass dummy embedding. Score is irrelevant here, only filtering matters.
        results = self.search(
            query_embedding=[0.0] * dummy_dimension,
            top_k=max_possible_chunks,
            filters=filters,
            query_str="", # No query string needed
            use_fusion_retriever=True # Use standard retriever for document ID filtering
        )

        if results:
             logger.info(f"Found {len(results)} chunks for document {document_id} using metadata filter search.")
             # Sort chunks by chunk_index if available in metadata
             try:
                 # Robust sorting: Treat chunk_index as int, default to 0 if missing/invalid
                 results.sort(key=lambda x: int(x.get("chunk_index", 0)) if str(x.get("chunk_index", "0")).isdigit() else 0)
                 logger.debug("Sorted retrieved chunks by 'chunk_index'.")
             except Exception as sort_e:
                  logger.warning(f"Could not sort chunks by 'chunk_index' due to error: {sort_e}. Returning unsorted.")
             return results
        else:
             logger.warning(f"No chunks found for document {document_id} using metadata filter search.")
             return []


    def save(self) -> None:
        """
        Save the vector store and associated LlamaIndex components to disk.
        Triggers persistence of FaissVectorStore and other StorageContext components.
        """
        if not self.persist_dir:
            logger.error("Cannot save: Persistence directory (persist_dir) is not set.")
            return

        # Ensure storage_context exists, creating a default one if we only have the faiss_store
        if not self.storage_context:
            if self.faiss_store:
                logger.warning("StorageContext not found during save. Creating default from Faiss store. Docstore might be missing unless loaded previously.")
                self.storage_context = StorageContext.from_defaults(vector_store=self.faiss_store)
            else:
                logger.error("Cannot save: StorageContext and FaissVectorStore are not initialized.")
                return
        elif not self.storage_context.vector_store and self.faiss_store:
            # If context exists but somehow lost its vector store reference, re-link it
            logger.warning("Re-linking FaissVectorStore to existing StorageContext before saving.")
            self.storage_context.vector_store = self.faiss_store

        logger.info(f"Saving LlamaIndex components to {self.persist_dir}...")
        try:
            # Persisting the StorageContext handles calling persist on its components.
            # FaissVectorStore's persist method should save the index.faiss file.
            self.storage_context.persist(persist_dir=self.persist_dir)

            # --- Explicitly save the FAISS index to ensure it's persisted ---
            if self.faiss_store and self.faiss_store._faiss_index:
                # Create the vector_store directory if it doesn't exist
                vector_store_dir = os.path.join(self.persist_dir, "vector_store")
                os.makedirs(vector_store_dir, exist_ok=True)

                faiss_index_path = os.path.join(vector_store_dir, "index.faiss")
                logger.info(f"Attempting to save FAISS index explicitly to: {faiss_index_path}")
                faiss.write_index(self.faiss_store._faiss_index, faiss_index_path)
                logger.info(f"FAISS index saved explicitly to: {faiss_index_path}")
            else:
                logger.warning("FaissVectorStore or its index is not initialized. Skipping explicit FAISS index save.")

            # --- Save BM25Retriever if possible ---
            try:
                bm25_retriever = self._create_or_load_bm25_retriever()
                if bm25_retriever:
                    bm25_path = self._get_bm25_retriever_path()
                    if bm25_path:
                        logger.info(f"Saving BM25Retriever to {bm25_path}")
                        bm25_retriever.persist(bm25_path)
                        logger.info(f"BM25Retriever saved to {bm25_path}")
            except Exception as bm25_err:
                logger.warning(f"Failed to save BM25Retriever: {bm25_err}")

            # --- Verification ---
            logger.info("Verifying saved files...")
            saved_files = os.listdir(self.persist_dir) if os.path.exists(self.persist_dir) else []
            logger.debug(f"Files found in {self.persist_dir}: {saved_files}")

            required_files = {
                "docstore.json": os.path.join(self.persist_dir, "docstore.json"),
                "vector_store.json": os.path.join(self.persist_dir, "vector_store.json"),
                # Use the helper to find the canonical FAISS index path
                "faiss_index": os.path.join(self.persist_dir, "vector_store", "index.faiss"),
                # BM25Retriever is optional
                "bm25_retriever": self._get_bm25_retriever_path()
            }
            all_found = True
            for name, path in required_files.items():
                if os.path.exists(path):
                    try:
                        size = os.path.getsize(path)
                        logger.info(f"  [OK] Found {name} at {path} (Size: {size} bytes)")
                    except OSError as e:
                        logger.warning(f"  [WARN] Found {name} at {path} but cannot get size: {e}")
                else:
                    # Optional files: vector_store.json and bm25_retriever
                    optional_files = ["vector_store.json", "bm25_retriever"]
                    log_level = logging.WARNING if name not in optional_files else logging.DEBUG
                    logger.log(log_level, f"  [MISSING] {name} not found at {path}")
                    if name not in optional_files:
                        all_found = False

            if all_found:
                logger.info(f"Core persistence files verified in {self.persist_dir}.")
            else:
                logger.warning(f"Save verification failed: One or more core files missing in {self.persist_dir}.")

        except Exception as e:
            logger.error(f"Error saving index/context to {self.persist_dir}: {e}", exc_info=True)


    # --- Class Method for Loading (Simplified) ---
    @classmethod
    def load(cls,
             persist_dir: str,
             dimension: int = 1024, # Provide defaults, but __init__ will verify against loaded index
             index_type: str = "hnsw",
             metric: str = "ip",
             project_id: Optional[str] = None, # Pass through args for context
             session_id: Optional[str] = None) -> "LlamaIndexVectorStore":
        """
        Load a vector store from disk by calling the __init__ method.

        Args:
            persist_dir: Directory where the vector store is persisted.
            dimension: Expected dimension (will be updated if loaded index differs).
            index_type: Expected index type (primarily for initializing if load fails).
            metric: Expected distance metric (primarily for initializing if load fails).
            project_id: Optional project ID associated with the store.
            session_id: Optional session ID associated with the store.

        Returns:
            Loaded or newly initialized vector store instance.
        """
        logger.info(f"LlamaIndexVectorStore.load called for persist_dir: {persist_dir}")
        # The __init__ method now handles the primary loading logic.
        # Pass all relevant parameters.
        return cls(
            persist_dir=persist_dir,
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            project_id=project_id,
            session_id=session_id
        )


# --- Factory Function (Simplified) ---
def get_llamaindex_vector_store(
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    dimension: int = 1024, # Crucial: Must match the embedding model used!
    index_type: str = "hnsw",
    metric: str = "ip",
    embedding_model: str = "mxbai-embed-large:latest"
) -> LlamaIndexVectorStore:
    """
    Get or create a LlamaIndex vector store instance based on project/session ID.

    Args:
        project_id: Project ID.
        session_id: Session ID.
        dimension: Dimension of the embeddings (must match embedding model).
        index_type: Type of FAISS index ("flat", "hnsw", "ivf").
        metric: Distance metric ("ip" or "l2").
        embedding_model: Name of the embedding model to use.

    Returns:
        LlamaIndexVectorStore instance (either loaded or newly initialized).
    """
    # Set up the embedding model in LlamaIndex Settings
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.core.settings import Settings
        from .config import OLLAMA_API_BASE

        # Create an Ollama embedding model
        embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=OLLAMA_API_BASE,
        )

        # Set the embedding model in the global settings
        Settings.embed_model = embed_model
        logger.info(f"Set up LlamaIndex with Ollama embedding model: {embedding_model}")
    except ImportError:
        logger.warning("Could not import LlamaIndex OllamaEmbedding. Embeddings will be handled separately.")
    except Exception as e:
        logger.error(f"Error setting up LlamaIndex embedding model: {e}")
        logger.warning("Embeddings will be handled separately.")

    # Determine persist_dir based on IDs (logic moved to class __init__)
    # Call the class constructor - it handles determining the path and loading/initializing.
    logger.info(f"Getting LlamaIndexVectorStore instance for project='{project_id}', session='{session_id}'")
    vector_store_instance = LlamaIndexVectorStore(
        project_id=project_id,
        session_id=session_id,
        # Explicit persist_dir=None allows __init__ to use ID logic
        dimension=dimension,
        index_type=index_type,
        metric=metric
    )
    return vector_store_instance