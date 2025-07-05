from typing import List, Dict, Any, Optional, Union
import os
import math
import pickle
import logging
import numpy as np
from pathlib import Path

# Import semantic chunker
from .semantic_chunker import chunk_document_semantically

# Configure logging
logger = logging.getLogger(__name__)

# Try to import FAISS, fall back to in-memory if not available
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS is available for vector search")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using simple in-memory vector search instead. Install FAISS for better performance.")

class VectorStore:
    """Base class for vector stores"""

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add chunks to the vector store"""
        raise NotImplementedError()

    def remove_chunks(self, document_ids: List[str] = None, project_id: str = None) -> int:
        """
        Remove chunks from the vector store based on document_ids or project_id.

        Args:
            document_ids: Optional list of document IDs to remove
            project_id: Optional project ID to remove all documents from

        Returns:
            Number of chunks removed
        """
        raise NotImplementedError()

    def get_chunks_by_document_id(self, document_id: str) -> List[Dict]:
        """
        Get all chunks for a specific document.

        Args:
            document_id: Document ID

        Returns:
            List of chunks
        """
        raise NotImplementedError()

    def search(self, query_embedding, top_k=5) -> List[Dict]:
        """Search for similar chunks"""
        raise NotImplementedError()

    def save(self, path: str):
        """Save the vector store to disk"""
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str):
        """Load a vector store from disk"""
        raise NotImplementedError()

class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using cosine similarity"""

    def __init__(self):
        self.chunks = []  # List of dicts: {embedding, content, metadata}

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        self.chunks.extend(chunks)

    def remove_chunks(self, document_ids: List[str] = None, project_id: str = None) -> int:
        """
        Remove chunks from the vector store based on document_ids or project_id.

        Args:
            document_ids: Optional list of document IDs to remove
            project_id: Optional project ID to remove all documents from

        Returns:
            Number of chunks removed
        """
        if not document_ids and not project_id:
            return 0

        original_count = len(self.chunks)

        if document_ids:
            # Remove chunks for specific document IDs
            self.chunks = [chunk for chunk in self.chunks
                          if chunk.get("document_id") not in document_ids]

        if project_id:
            # Remove chunks for a specific project
            self.chunks = [chunk for chunk in self.chunks
                          if chunk.get("project_id") != project_id]

        removed_count = original_count - len(self.chunks)
        logger.info(f"Removed {removed_count} chunks from in-memory vector store")
        return removed_count

    def get_chunks_by_document_id(self, document_id: str) -> List[Dict]:
        """
        Get all chunks for a specific document.

        Args:
            document_id: Document ID

        Returns:
            List of chunks
        """
        chunks = [chunk for chunk in self.chunks if chunk.get("document_id") == document_id]

        # Log warning if no chunks found
        if not chunks:
            logger.warning(f"No chunks found for document {document_id}")

        # Log warning if any chunks don't have content
        for i, chunk in enumerate(chunks):
            if "content" not in chunk or not chunk["content"]:
                logger.warning(f"Chunk {i} for document {document_id} has no content")

        # Sort chunks by chunk_index if available
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        return chunks

    @staticmethod
    def cosine_similarity(vec1, vec2):
        # Simple cosine similarity for lists of floats
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query_embedding, top_k=5) -> List[Dict]:
        # Return top_k chunks by cosine similarity
        if not self.chunks or not query_embedding:
            return []
        scored = []
        for chunk in self.chunks:
            emb = chunk.get("embedding")
            if emb is None:
                continue
            sim = self.cosine_similarity(query_embedding, emb)
            scored.append((sim, chunk))
        scored.sort(reverse=True, key=lambda x: x[0])

        results = []
        for sim, chunk in scored[:top_k]:
            # Create a copy of the chunk to avoid modifying the original
            chunk_copy = dict(chunk)
            # Add similarity score
            chunk_copy["similarity"] = sim
            chunk_copy["score"] = sim  # Add score for consistency with FAISS

            # Ensure content is included in the result
            if "content" not in chunk_copy or not chunk_copy["content"]:
                logger.warning(f"Missing or empty content in chunk for {chunk_copy.get('file_path')}")

            results.append(chunk_copy)

        return results

    def save(self, path: str):
        """Save the vector store to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self, f)
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False

    @classmethod
    def load(cls, path: str):
        """Load a vector store from disk"""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None

class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search"""

    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        Initialize a FAISS vector store

        Args:
            dimension: Dimension of the embeddings
            index_type: Type of FAISS index to use ("flat", "hnsw", or "ivf")
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")

        self.dimension = dimension
        self.index_type = index_type

        # Create the appropriate index based on the index_type
        if index_type == "hnsw":
            # HNSW index parameters
            # M: Number of connections per layer (default: 32)
            # efConstruction: Build-time accuracy/speed trade-off (default: 40)
            M = 32
            efConstruction = 200  # Higher value = better accuracy but slower construction

            # Create HNSW index with inner product metric (for cosine similarity with normalized vectors)
            self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = efConstruction
            self.index.hnsw.efSearch = 128  # Search-time accuracy/speed trade-off

            logger.info(f"Created FAISS HNSW index with dimension={dimension}, M={M}, efConstruction={efConstruction}")
        elif index_type == "ivf":
            # IVF (Inverted File Index) parameters
            nlist = 100  # Number of clusters/cells (adjust based on dataset size)

            # Create a quantizer (the index used to assign vectors to cells)
            quantizer = faiss.IndexFlatIP(dimension)

            # Create IVF index with inner product metric
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            # IVF index needs to be trained before adding vectors
            self.needs_training = True

            logger.info(f"Created FAISS IVF index with dimension={dimension}, nlist={nlist}")
        else:
            # Default to flat index (exact search, no approximation)
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity on normalized vectors)
            logger.info(f"Created FAISS Flat index with dimension={dimension}")

        self.chunks = []  # Store the original chunks
        self.embeddings = []
        self.needs_training = getattr(self, 'needs_training', False)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add chunks to the FAISS index"""
        if not chunks:
            return

        # Extract embeddings
        new_embeddings = []
        new_chunks = []  # Store chunks that have valid embeddings

        for chunk in chunks:
            emb = chunk.get("embedding")
            if emb is None or len(emb) != self.dimension:
                logger.warning(f"Skipping chunk with invalid embedding dimension: {len(emb) if emb else None} (expected {self.dimension})")
                continue
            new_embeddings.append(emb)
            new_chunks.append(chunk)

        if not new_embeddings:
            return

        # Convert to numpy array and normalize
        embeddings_array = np.array(new_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity

        # Handle IVF index training if needed
        if self.index_type == "ivf" and self.needs_training:
            if len(embeddings_array) < 100:
                logger.warning("Not enough vectors to train IVF index. Need at least 100 vectors.")
                # Generate random vectors for training if we don't have enough
                if len(embeddings_array) < 100:
                    logger.info("Generating random vectors for IVF training")
                    np.random.seed(42)  # For reproducibility
                    random_vectors = np.random.randn(max(100, self.dimension), self.dimension).astype('float32')
                    faiss.normalize_L2(random_vectors)
                    self.index.train(random_vectors)
            else:
                logger.info(f"Training IVF index with {len(embeddings_array)} vectors")
                self.index.train(embeddings_array)

            # Set nprobe for search (higher = more accurate but slower)
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(20, self.index.nlist)

            self.needs_training = False

        # For HNSW, we might need to adjust efSearch based on dataset size
        if self.index_type == "hnsw" and hasattr(self.index, 'hnsw'):
            # Adjust efSearch based on total vectors (higher for larger datasets)
            total_vectors = len(self.chunks) + len(new_chunks)
            if total_vectors > 10000:
                self.index.hnsw.efSearch = 256
            elif total_vectors > 1000:
                self.index.hnsw.efSearch = 128
            else:
                self.index.hnsw.efSearch = 64

            logger.info(f"Set HNSW efSearch to {self.index.hnsw.efSearch} for {total_vectors} total vectors")

        # Add to index
        try:
            self.index.add(embeddings_array)
            self.chunks.extend(new_chunks)
            self.embeddings.extend(new_embeddings)
            logger.info(f"Added {len(new_embeddings)} vectors to {self.index_type} index. Total vectors: {len(self.chunks)}")
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS index: {e}", exc_info=True)
            # Try to recover by adding vectors one by one
            if len(new_embeddings) > 1:
                logger.info("Trying to add vectors one by one")
                successful_adds = 0
                for i, (emb, chunk) in enumerate(zip(new_embeddings, new_chunks)):
                    try:
                        emb_array = np.array([emb]).astype('float32')
                        faiss.normalize_L2(emb_array)
                        self.index.add(emb_array)
                        self.chunks.append(chunk)
                        self.embeddings.append(emb)
                        successful_adds += 1
                    except Exception as inner_e:
                        logger.error(f"Error adding vector {i}: {inner_e}")

                logger.info(f"Successfully added {successful_adds}/{len(new_embeddings)} vectors individually")

    def remove_chunks(self, document_ids: List[str] = None, project_id: str = None) -> int:
        """
        Remove chunks from the FAISS index based on document_ids or project_id.

        Args:
            document_ids: Optional list of document IDs to remove
            project_id: Optional project ID to remove all documents from

        Returns:
            Number of chunks removed
        """
        if not document_ids and not project_id:
            return 0

        if not self.chunks:
            return 0

        # FAISS doesn't support direct removal of vectors, so we need to:
        # 1. Identify which vectors to keep
        # 2. Create a new index with only those vectors

        # Identify indices to keep
        keep_indices = []
        for i, chunk in enumerate(self.chunks):
            should_keep = True

            if document_ids and chunk.get("document_id") in document_ids:
                should_keep = False

            if project_id and chunk.get("project_id") == project_id:
                should_keep = False

            if should_keep:
                keep_indices.append(i)

        # If nothing to remove, return early
        if len(keep_indices) == len(self.chunks):
            return 0

        removed_count = len(self.chunks) - len(keep_indices)
        logger.info(f"Removing {removed_count} vectors from FAISS {self.index_type} index")

        if not keep_indices:
            # If removing all vectors, just reset the index
            logger.info("Removing all vectors from index")

            # Create a new empty index
            if self.index_type == "hnsw":
                M = 32
                efConstruction = 200
                self.index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = efConstruction
                self.index.hnsw.efSearch = 128
            elif self.index_type == "ivf":
                nlist = 100
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                self.needs_training = True
            else:
                self.index = faiss.IndexFlatIP(self.dimension)

            # Clear chunks and embeddings
            self.chunks = []
            self.embeddings = []

            return removed_count

        # Create new lists for kept chunks and embeddings
        new_chunks = [self.chunks[i] for i in keep_indices]
        new_embeddings = [self.embeddings[i] for i in keep_indices]

        # Create a new index
        if self.index_type == "hnsw":
            M = 32
            efConstruction = 200
            new_index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
            new_index.hnsw.efConstruction = efConstruction
            new_index.hnsw.efSearch = self.index.hnsw.efSearch
        elif self.index_type == "ivf":
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            # Copy nprobe setting if available
            if hasattr(self.index, 'nprobe'):
                new_index.nprobe = self.index.nprobe
        else:
            new_index = faiss.IndexFlatIP(self.dimension)

        # Add vectors to the new index
        embeddings_array = np.array(new_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)

        # Train IVF index if needed
        if self.index_type == "ivf":
            if len(embeddings_array) < 100:
                logger.warning("Not enough vectors to train IVF index after removal. Using random vectors.")
                np.random.seed(42)
                random_vectors = np.random.randn(max(100, self.dimension), self.dimension).astype('float32')
                faiss.normalize_L2(random_vectors)
                new_index.train(random_vectors)
            else:
                logger.info(f"Training IVF index with {len(embeddings_array)} vectors after removal")
                new_index.train(embeddings_array)

        # Add vectors to the new index
        new_index.add(embeddings_array)

        # Replace the old index, chunks, and embeddings
        self.index = new_index
        self.chunks = new_chunks
        self.embeddings = new_embeddings

        logger.info(f"Rebuilt FAISS {self.index_type} index with {len(self.chunks)} vectors after removing {removed_count} vectors")
        return removed_count

    def get_chunks_by_document_id(self, document_id: str) -> List[Dict]:
        """
        Get all chunks for a specific document.

        Args:
            document_id: Document ID

        Returns:
            List of chunks
        """
        # Log detailed information about all chunks
        logger.info(f"Looking for document_id {document_id} in {len(self.chunks)} total chunks")

        # Check if document_id exists in any chunks
        doc_ids = set(chunk.get("document_id") for chunk in self.chunks if "document_id" in chunk)
        logger.info(f"Available document_ids in chunks: {doc_ids}")

        # Get chunks for the document
        chunks = [chunk for chunk in self.chunks if chunk.get("document_id") == document_id]

        # Log warning if no chunks found
        if not chunks:
            logger.warning(f"No chunks found for document {document_id} in FAISS vector store")

            # Try to find chunks with similar document_id (in case of string/UUID formatting issues)
            similar_chunks = []
            for chunk in self.chunks:
                chunk_doc_id = chunk.get("document_id", "")
                if isinstance(chunk_doc_id, str) and isinstance(document_id, str):
                    if document_id in chunk_doc_id or chunk_doc_id in document_id:
                        similar_chunks.append(chunk)

            if similar_chunks:
                logger.info(f"Found {len(similar_chunks)} chunks with similar document_id")
                chunks = similar_chunks
            else:
                # If still no chunks, return empty list
                return []

        # Log detailed information about found chunks
        logger.info(f"Found {len(chunks)} chunks for document {document_id}")

        # Check content in chunks
        content_status = []
        for i, chunk in enumerate(chunks):
            has_content = "content" in chunk and bool(chunk["content"])
            content_len = len(chunk["content"]) if has_content else 0
            content_status.append(f"Chunk {i}: has_content={has_content}, length={content_len}")

            # Log warning and try to fix if chunk doesn't have content
            if not has_content:
                logger.warning(f"Chunk {i} for document {document_id} has no content in FAISS vector store")

                # Try to find content from other chunks with the same document_id
                for other_chunk in self.chunks:
                    if (other_chunk.get("document_id") == document_id and
                        "content" in other_chunk and
                        other_chunk["content"] and
                        other_chunk != chunk):

                        # Copy content from other chunk
                        chunk["content"] = other_chunk["content"]
                        logger.info(f"Copied content from another chunk for document {document_id}")
                        break

        logger.info(f"Content status for chunks: {content_status}")

        # Sort chunks by chunk_index if available
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        # Log the first few characters of each chunk's content
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            preview = content[:50] + "..." if content else "NO CONTENT"
            logger.info(f"Chunk {i} content preview: {preview}")

        return chunks

    def search(self, query_embedding, top_k=5) -> List[Dict]:
        """Search for similar chunks using FAISS"""
        if not self.chunks or not query_embedding or len(query_embedding) != self.dimension:
            return []

        # Convert query to numpy array and normalize
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)

        # Search
        distances, indices = self.index.search(query_array, min(top_k, len(self.chunks)))

        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue  # Skip invalid indices

            # Create a copy of the chunk to avoid modifying the original
            chunk = dict(self.chunks[idx])
            # Add similarity score (convert distance to similarity)
            # FAISS returns cosine similarity which is between -1 and 1, normalize to 0-1 range
            similarity = float(distances[0][i])  # Convert numpy float to Python float
            # Normalize to 0-1 range (cosine similarity is between -1 and 1)
            normalized_score = (similarity + 1) / 2
            chunk["similarity"] = similarity
            chunk["score"] = normalized_score  # Add normalized score for confidence

            # Ensure content is included in the result
            if "content" not in chunk or not chunk["content"]:
                logger.warning(f"Missing or empty content in chunk {idx} for {chunk.get('file_path')}")

            results.append(chunk)

        return results

    def save(self, path: str):
        """Save the FAISS index and chunks to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save the index
            index_path = f"{path}.index"
            faiss.write_index(self.index, index_path)

            # Save the chunks and metadata
            data = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "chunks": self.chunks,
                "embeddings": self.embeddings,
                "needs_training": self.needs_training
            }

            # For HNSW, save additional parameters
            if self.index_type == "hnsw" and hasattr(self.index, 'hnsw'):
                data["hnsw_params"] = {
                    "efConstruction": self.index.hnsw.efConstruction,
                    "efSearch": self.index.hnsw.efSearch
                }

            # For IVF, save additional parameters
            if self.index_type == "ivf" and hasattr(self.index, 'nprobe'):
                data["ivf_params"] = {
                    "nprobe": self.index.nprobe,
                    "nlist": self.index.nlist
                }

            with open(f"{path}.data", "wb") as f:
                pickle.dump(data, f)

            logger.info(f"Saved FAISS {self.index_type} index with {len(self.chunks)} vectors to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS vector store: {e}", exc_info=True)
            return False

    @classmethod
    def load(cls, path: str):
        """Load a FAISS index and chunks from disk"""
        try:
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS is not available. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")

            # Load the data
            with open(f"{path}.data", "rb") as f:
                data = pickle.load(f)

            # Get index type from data or default to "flat"
            index_type = data.get("index_type", "flat")

            # Create a new instance with the correct index type
            instance = cls(dimension=data["dimension"], index_type=index_type)

            # Load the index
            instance.index = faiss.read_index(f"{path}.index")

            # Restore HNSW parameters if available
            if index_type == "hnsw" and hasattr(instance.index, 'hnsw') and "hnsw_params" in data:
                hnsw_params = data["hnsw_params"]
                instance.index.hnsw.efConstruction = hnsw_params.get("efConstruction", 40)
                instance.index.hnsw.efSearch = hnsw_params.get("efSearch", 64)
                logger.info(f"Restored HNSW parameters: efConstruction={instance.index.hnsw.efConstruction}, efSearch={instance.index.hnsw.efSearch}")

            # Restore IVF parameters if available
            if index_type == "ivf" and hasattr(instance.index, 'nprobe') and "ivf_params" in data:
                ivf_params = data["ivf_params"]
                instance.index.nprobe = ivf_params.get("nprobe", 1)
                logger.info(f"Restored IVF parameters: nprobe={instance.index.nprobe}")

            # Set the chunks and embeddings
            instance.chunks = data["chunks"]
            instance.embeddings = data["embeddings"]
            instance.needs_training = data.get("needs_training", False)

            logger.info(f"Loaded FAISS {index_type} index with {len(instance.chunks)} vectors from {path}")
            return instance
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}", exc_info=True)
            return None

def get_vector_store(use_faiss: bool = True, dimension: int = 1024, index_type: str = "hnsw") -> VectorStore:
    """
    Get an appropriate vector store based on availability and configuration.

    Args:
        use_faiss: Whether to use FAISS for vector search
        dimension: Dimension of the embeddings
        index_type: Type of FAISS index to use ("flat", "hnsw", or "ivf")

    Returns:
        Vector store instance
    """
    if use_faiss and FAISS_AVAILABLE:
        # Validate index type
        if index_type not in ["flat", "hnsw", "ivf"]:
            logger.warning(f"Invalid index type: {index_type}. Using 'hnsw' instead.")
            index_type = "hnsw"

        logger.info(f"Creating FAISS vector store with {index_type} index and dimension {dimension}")
        return FAISSVectorStore(dimension=dimension, index_type=index_type)
    else:
        if use_faiss and not FAISS_AVAILABLE:
            logger.warning("FAISS is not available. Using in-memory vector store instead.")
        return InMemoryVectorStore()

def debug_vector_store_content(project_id: str = None, document_id: str = None, session_id: str = None) -> Dict[str, Any]:
    """
    Debug function to check content in vector store.

    Args:
        project_id: Optional project ID to check
        document_id: Optional document ID to check
        session_id: Optional session ID to check

    Returns:
        Dictionary with debug information
    """
    result = {
        "status": "success",
        "vector_stores_checked": [],
        "chunks_found": 0,
        "chunks_with_content": 0,
        "total_content_length": 0,
        "document_ids_found": set(),
        "errors": []
    }

    try:
        # Try to get vector store from project
        if project_id:
            try:
                from .project_indexer import get_project_vector_store
                vector_store = get_project_vector_store(project_id)
                if vector_store:
                    result["vector_stores_checked"].append(f"project_{project_id}")

                    # Check all chunks
                    chunks = vector_store.chunks
                    result["chunks_found"] += len(chunks)

                    # Count chunks with content
                    chunks_with_content = sum(1 for chunk in chunks if "content" in chunk and chunk["content"])
                    result["chunks_with_content"] += chunks_with_content

                    # Calculate total content length
                    total_content_length = sum(len(chunk["content"]) for chunk in chunks if "content" in chunk and chunk["content"])
                    result["total_content_length"] += total_content_length

                    # Get document IDs
                    doc_ids = set(chunk["document_id"] for chunk in chunks if "document_id" in chunk)
                    result["document_ids_found"].update(doc_ids)

                    # Check specific document if requested
                    if document_id:
                        doc_chunks = vector_store.get_chunks_by_document_id(document_id)
                        result[f"document_{document_id}_chunks"] = len(doc_chunks)
                        result[f"document_{document_id}_chunks_with_content"] = sum(1 for chunk in doc_chunks if "content" in chunk and chunk["content"])

                        # Get content preview
                        if doc_chunks:
                            content_previews = []
                            for i, chunk in enumerate(doc_chunks):
                                content = chunk.get("content", "")
                                preview = content[:50] + "..." if content else "NO CONTENT"
                                content_previews.append(f"Chunk {i}: {preview}")
                            result[f"document_{document_id}_content_previews"] = content_previews
            except Exception as e:
                error_msg = f"Error checking project vector store: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

        # Try to get vector store from session
        if session_id:
            try:
                from .pipeline import load_session_vector_store
                vector_store = load_session_vector_store(session_id, use_faiss=True)
                if vector_store:
                    result["vector_stores_checked"].append(f"session_{session_id}")

                    # Check all chunks
                    chunks = vector_store.chunks
                    result["chunks_found"] += len(chunks)

                    # Count chunks with content
                    chunks_with_content = sum(1 for chunk in chunks if "content" in chunk and chunk["content"])
                    result["chunks_with_content"] += chunks_with_content

                    # Calculate total content length
                    total_content_length = sum(len(chunk["content"]) for chunk in chunks if "content" in chunk and chunk["content"])
                    result["total_content_length"] += total_content_length

                    # Get document IDs
                    doc_ids = set(chunk["document_id"] for chunk in chunks if "document_id" in chunk)
                    result["document_ids_found"].update(doc_ids)

                    # Check specific document if requested
                    if document_id:
                        doc_chunks = vector_store.get_chunks_by_document_id(document_id)
                        result[f"document_{document_id}_chunks_in_session"] = len(doc_chunks)
                        result[f"document_{document_id}_chunks_with_content_in_session"] = sum(1 for chunk in doc_chunks if "content" in chunk and chunk["content"])
            except Exception as e:
                error_msg = f"Error checking session vector store: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

        # Convert set to list for JSON serialization
        result["document_ids_found"] = list(result["document_ids_found"])

        return result
    except Exception as e:
        logger.error(f"Error in debug_vector_store_content: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def chunk_file_content(file_record: Dict, chunk_size: int = 1024, chunk_overlap: int = 0) -> List[Dict]:
    """
    Split file content into chunks for embedding.

    This function now uses semantic chunking by default with a chunk size of 1024.
    The chunk_overlap parameter is kept for backward compatibility but is not used
    with semantic chunking.

    Args:
        file_record: Dictionary with file metadata and content
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks (used only for legacy chunking)

    Returns:
        List of chunk dictionaries
    """
    # Use semantic chunking by default
    return chunk_document_semantically(file_record, chunk_size)

def legacy_chunk_file_content(file_record: Dict, chunk_size: int = 512, chunk_overlap: int = 50) -> List[Dict]:
    """
    Legacy method to split file content into chunks with character-based overlap.
    Kept for backward compatibility.

    Args:
        file_record: Dictionary with file metadata and content
        chunk_size: Size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of chunk dictionaries
    """
    content = file_record["content"]
    path = file_record["path"]
    name = file_record.get("name", os.path.basename(path))
    source_type = file_record.get("source_type", os.path.splitext(name)[1].lstrip("."))

    chunks = []
    i = 0
    while i < len(content):
        # Get chunk text with proper bounds checking
        end_idx = min(i + chunk_size, len(content))
        chunk_text = content[i:end_idx]

        # Create chunk with metadata
        chunks.append({
            "content": chunk_text,
            "file_path": path,
            "source_name": name,
            "source_type": source_type,
            "start": i,
            "end": end_idx,
            # "embedding": ... (to be added in pipeline.py)
        })

        # Move to next chunk position with overlap
        i += (chunk_size - chunk_overlap)
        # Ensure we make progress even with large overlap
        if i <= 0:
            i = 1  # Ensure we make at least some progress

    return chunks