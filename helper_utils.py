"""
Helper utilities for PDF processing, text chunking, and ChromaDB integration.
This module provides functions to read PDFs, split text into chunks, and load them into ChromaDB.
"""

import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
from pypdf import PdfReader
from tqdm import tqdm


def _read_pdf(filename):
    """
    Read and extract text from a PDF file.
    
    Args:
        filename: Path to the PDF file to read
        
    Returns:
        List of strings containing the text from each non-empty page
    """
    # Initialize PDF reader with the given filename
    reader = PdfReader(filename)
    
    # Extract text from each page and remove leading/trailing whitespace
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter out any empty strings (pages with no extractable text)
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts


def _chunk_texts(texts):
    """
    Split texts into smaller chunks suitable for embedding and retrieval.
    Uses a two-stage splitting process: character-based then token-based.
    
    Args:
        texts: List of text strings to be chunked
        
    Returns:
        List of text chunks optimized for embedding (max 256 tokens each)
    """
    # Stage 1: Split text by characters using recursive splitting
    # This respects natural text boundaries (paragraphs, sentences, etc.)
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],  # Try splitting by paragraphs, then lines, then sentences
        chunk_size=1000,  # Maximum characters per chunk
        chunk_overlap=0   # No overlap between chunks
    )
    # Join all texts and split them based on the character rules
    character_split_texts = character_splitter.split_text('\n\n'.join(texts))

    # Stage 2: Split by tokens to ensure chunks fit within embedding model limits
    # This ensures each chunk is no more than 256 tokens (suitable for most embedding models)
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    # Apply token-based splitting to each character chunk
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


def load_chroma(filename, collection_name, embedding_function):
    """
    Load a PDF into ChromaDB by extracting text, chunking it, and storing with embeddings.
    
    Args:
        filename: Path to the PDF file
        collection_name: Name for the ChromaDB collection to create
        embedding_function: Function to generate embeddings for text chunks
        
    Returns:
        ChromaDB collection object containing the embedded document chunks
    """
    # Step 1: Extract text from the PDF file
    texts = _read_pdf(filename)
    
    # Step 2: Split the text into appropriately-sized chunks
    chunks = _chunk_texts(texts)

    # Step 3: Initialize ChromaDB client (in-memory database)
    chroma_client = chromadb.Client()
    
    # Step 4: Create a new collection with the specified embedding function
    chroma_collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)

    # Step 5: Generate unique IDs for each chunk (sequential numbers as strings)
    ids = [str(i) for i in range(len(chunks))]

    # Step 6: Add all chunks to the collection (embeddings are generated automatically)
    chroma_collection.add(ids=ids, documents=chunks)

    return chroma_collection

def word_wrap(string, n_chars=72):
    """
    Wrap a string at word boundaries to fit within a specified character width.
    Uses recursion to handle strings longer than n_chars.
    
    Args:
        string: The text string to wrap
        n_chars: Maximum number of characters per line (default: 72)
        
    Returns:
        String with newline characters inserted at appropriate word boundaries
    """
    # Base case: if string fits within the limit, return it as-is
    if len(string) < n_chars:
        return string
    else:
        # Find the last space before the character limit
        # Split at that space to avoid breaking words
        wrapped_line = string[:n_chars].rsplit(' ', 1)[0]
        # Recursively wrap the remaining text and append with newline
        return wrapped_line + '\n' + word_wrap(string[len(wrapped_line)+1:], n_chars)

   
def project_embeddings(embeddings, umap_transform):
    """
    Project high-dimensional embeddings to 2D space using UMAP transformation.
    Useful for visualization of embedding spaces.
    
    Args:
        embeddings: List or array of high-dimensional embedding vectors
        umap_transform: Pre-fitted UMAP transformer object
        
    Returns:
        NumPy array of shape (n_embeddings, 2) containing 2D projections
    """
    # Initialize empty array to store 2D projections
    umap_embeddings = np.empty((len(embeddings), 2))
    
    # Transform each embedding individually with progress bar
    for i, embedding in enumerate(tqdm(embeddings)): 
        # Apply UMAP transformation to project embedding to 2D
        umap_embeddings[i] = umap_transform.transform([embedding])
    
    return umap_embeddings