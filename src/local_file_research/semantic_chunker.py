"""
Semantic chunking implementation for document processing.
This module provides functions to split text into semantically meaningful chunks.
"""
import re
import logging
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

def split_by_semantic_boundaries(text: str, max_chunk_size: int = 1024) -> List[str]:
    """
    Split text into chunks based on semantic boundaries like paragraphs, sections, etc.

    Args:
        text: The text to split
        max_chunk_size: Maximum size of each chunk in characters

    Returns:
        List of text chunks
    """
    # If text is shorter than max_chunk_size, return it as a single chunk
    if len(text) <= max_chunk_size:
        return [text]

    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If paragraph is too long, split it further by sentences
        if len(paragraph) > max_chunk_size:
            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                # If a single sentence is too long, split it by size
                if len(sentence) > max_chunk_size:
                    sentence_chunks = [sentence[i:i+max_chunk_size] for i in range(0, len(sentence), max_chunk_size)]
                    for sc in sentence_chunks:
                        chunks.append(sc)
                # Otherwise, try to add to current chunk or create a new one
                elif len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += sentence
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence
        # If adding paragraph would exceed max size, start a new chunk
        elif len(current_chunk) + len(paragraph) <= max_chunk_size:
            if current_chunk and not current_chunk.endswith("\n"):
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: The text to split

    Returns:
        List of sentences
    """
    # Simple sentence splitting - can be improved with NLP libraries
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    return sentences

def chunk_document_semantically(file_record: Dict, chunk_size: int = 1024) -> List[Dict]:
    """
    Split document content into semantic chunks.

    Args:
        file_record: Dictionary with file metadata and content
        chunk_size: Maximum size of each chunk in characters

    Returns:
        List of chunk dictionaries with metadata
    """
    content = file_record.get("content", "")
    path = file_record.get("path", "")
    name = file_record.get("name", "")
    source_type = file_record.get("source_type", "")
    document_id = file_record.get("document_id", "")
    project_id = file_record.get("project_id", "")

    if not content:
        logger.warning(f"Empty content for file: {path}")
        return []

    # Get semantic chunks
    semantic_chunks = split_by_semantic_boundaries(content, chunk_size)

    # Create chunk objects with metadata
    chunks = []
    start_pos = 0

    for i, chunk_text in enumerate(semantic_chunks):
        end_pos = start_pos + len(chunk_text)

        # Create chunk with all necessary metadata
        chunk = {
            "content": chunk_text,  # Always include content
            "file_path": path,
            "source_name": name,
            "source_type": source_type,
            "start": start_pos,
            "end": end_pos,
            "chunk_index": i,  # Add chunk index for proper ordering
        }

        # Add document_id and project_id if available
        if document_id:
            chunk["document_id"] = document_id

        if project_id:
            chunk["project_id"] = project_id

        chunks.append(chunk)

        start_pos = end_pos

    logger.info(f"Split document into {len(chunks)} semantic chunks")
    return chunks
