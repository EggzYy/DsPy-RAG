"""
Embedding utilities for Local File Deep Research.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import json
import pickle
import requests

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
EMBEDDING_CACHE_DIR = os.environ.get("EMBEDDING_CACHE_DIR", "embeddings")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "384"))
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

# --- Helper Functions ---
def _ensure_cache_dir():
    """Ensure the embedding cache directory exists."""
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

def _get_cache_path(text: str) -> str:
    """Get the cache path for a text."""
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return os.path.join(EMBEDDING_CACHE_DIR, f"{text_hash}.pkl")

def _cache_embedding(text: str, embedding: List[float]):
    """Cache an embedding."""
    _ensure_cache_dir()
    cache_path = _get_cache_path(text)
    with open(cache_path, "wb") as f:
        pickle.dump(embedding, f)

def _get_cached_embedding(text: str) -> Optional[List[float]]:
    """Get a cached embedding."""
    cache_path = _get_cache_path(text)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached embedding: {e}")
    return None

# --- Embedding Functions ---
def get_ollama_embeddings(texts: List[str], model_name: str = "mxbai-embed-large:latest") -> List[List[float]]:
    """
    Get embeddings for a list of texts using Ollama API.

    Args:
        texts: List of texts to embed
        model_name: Name of the Ollama model to use

    Returns:
        List of embeddings
    """
    try:
        # Try to use the LlamaIndex OllamaEmbedding class if available
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding

            # Create an Ollama embedding model
            embed_model = OllamaEmbedding(
                model_name=model_name,
                base_url=OLLAMA_API_BASE,
            )

            # Get embeddings using the LlamaIndex OllamaEmbedding class
            embeddings = embed_model.get_text_embeddings(texts)
            logger.info(f"Generated {len(embeddings)} embeddings using LlamaIndex OllamaEmbedding with model {model_name}")
            return embeddings

        except ImportError:
            logger.warning("LlamaIndex OllamaEmbedding not available, falling back to direct API calls")
            # Fall back to direct API calls if LlamaIndex OllamaEmbedding is not available
    except Exception as e:
        logger.error(f"Error using LlamaIndex OllamaEmbedding: {e}")
        logger.warning("Falling back to direct API calls")

    # Fallback: Direct API calls to Ollama
    embeddings = []
    for text in texts:
        try:
            # Call Ollama API
            response = requests.post(
                f"{OLLAMA_API_BASE}/api/embeddings",
                json={"model": model_name, "prompt": text}
            )

            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                embeddings.append(embedding)
                logger.info(f"Generated embedding using Ollama API with model {model_name}")
            else:
                logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * EMBEDDING_DIMENSION)

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            # Return a zero vector as fallback
            embeddings.append([0.0] * EMBEDDING_DIMENSION)

    return embeddings

def get_sentence_transformer_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Get embeddings for a list of texts using sentence-transformers.

    Args:
        texts: List of texts to embed
        model_name: Name of the sentence-transformer model to use

    Returns:
        List of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        logger.info(f"Generated {len(embeddings)} embeddings using sentence-transformers model {model_name}")
        return embeddings
    except ImportError:
        logger.warning("sentence-transformers not available. Using fallback embeddings.")
        return get_fallback_embeddings(texts)
    except Exception as e:
        logger.error(f"Error generating embeddings with sentence-transformers: {e}")
        return get_fallback_embeddings(texts)

def get_fallback_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate fallback embeddings when other methods fail.

    Args:
        texts: List of texts to embed

    Returns:
        List of embeddings
    """
    embeddings = []
    for text in texts:
        # Use text hash as seed for reproducibility
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
        np.random.seed(text_hash)
        # Generate random embedding and normalize
        embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())

    logger.info(f"Generated {len(embeddings)} fallback embeddings")
    return embeddings

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of texts to embed

    Returns:
        List of embeddings
    """
    if not texts:
        return []

    # Check cache first
    embeddings = []
    texts_to_embed = []
    indices_to_embed = []

    for i, text in enumerate(texts):
        cached_embedding = _get_cached_embedding(text)
        if cached_embedding is not None:
            embeddings.append(cached_embedding)
        else:
            embeddings.append(None)  # Placeholder
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    # If we have texts to embed, use the embedding model
    if texts_to_embed:
        try:
            # Determine which embedding method to use based on EMBEDDING_MODEL_NAME
            if EMBEDDING_MODEL_NAME.startswith("mxbai-embed") or "ollama" in EMBEDDING_MODEL_NAME.lower():
                # Use Ollama for embedding
                new_embeddings = get_ollama_embeddings(texts_to_embed, EMBEDDING_MODEL_NAME)
            elif EMBEDDING_MODEL_NAME.startswith("all-MiniLM") or "sentence-transformers" in EMBEDDING_MODEL_NAME.lower():
                # Use sentence-transformers
                new_embeddings = get_sentence_transformer_embeddings(texts_to_embed, EMBEDDING_MODEL_NAME)
            else:
                # Try sentence-transformers first, then fall back
                try:
                    new_embeddings = get_sentence_transformer_embeddings(texts_to_embed, EMBEDDING_MODEL_NAME)
                except Exception:
                    new_embeddings = get_fallback_embeddings(texts_to_embed)

            # Update embeddings and cache them
            for i, (idx, embedding) in enumerate(zip(indices_to_embed, new_embeddings)):
                embeddings[idx] = embedding
                _cache_embedding(texts_to_embed[i], embedding)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fill in missing embeddings with zeros
            for idx in indices_to_embed:
                embeddings[idx] = [0.0] * EMBEDDING_DIMENSION

    return embeddings

# Initialize
_ensure_cache_dir()
