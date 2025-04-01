from typing import List, Optional, Any
import logging

import numpy as np
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.vectorstores.utils import maximal_marginal_relevance

# pylint: disable=too-many-positional-arguments

def max_marginal_relevance_search_with_threshold_by_vector(
    vectorstore: Chroma,
    embedding: List[float],
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold: Optional[float] = None,
    **kwargs: Any,
) -> List[Document]:
    """
    Return documents selected using MMR with a score threshold, based on an embedding.

    Args:
        vectorstore (Chroma): Instance for vector search.
        embedding (List[float]): Embedding vector of the query.
        k (int): Number of Documents to return. Defaults to 4.
        fetch_k (int): Number of Documents to fetch for MMR. Default is 20.
        lambda_mult (float): Controls diversity vs. relevance in MMR. Defaults to 0.5.
        score_threshold (Optional[float]): Maximum distance score to filter documents.
        **kwargs (Any): Additional arguments to pass to similarity search.

    Returns:
        List[Document]: Documents selected by MMR, filtered by score.
    """
    # Perform similarity search with scores
    try:
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
            embedding, k=fetch_k, **kwargs
        )
    except Exception as e:
        logging.error(f"Similarity search failed: {e}")
        return []

    # Filter based on score_threshold
    if score_threshold is not None:
        docs_and_scores = [
            (doc, score) for doc, score in docs_and_scores if score <= score_threshold
        ]

    if not docs_and_scores:
        logging.info("No documents found after applying score threshold.")
        return []

    # Extract document IDs and ensure they exist
    doc_ids = []
    for doc, _ in docs_and_scores:
        doc_id = doc.metadata.get("doc_id")
        if doc_id is None:
            logging.error("Document missing 'doc_id' in metadata.")
            raise ValueError("All documents must have a 'doc_id' in their metadata.")
        doc_ids.append(doc_id)

    # Retrieve embeddings for the documents
    try:
        # Accessing protected member _collection
        embeddings = vectorstore._collection.get( # pylint: disable=protected-access
            ids=doc_ids, include=["embeddings"]
        ).get("embeddings", [])
        if not embeddings:
            logging.error(f"No embeddings found for document IDs: {doc_ids}")
            raise ValueError("Embeddings retrieval failed.")
    except Exception as e:
        logging.error(f"Failed to retrieve embeddings: {e}")
        return []

    # Convert embeddings to numpy arrays
    embeddings = np.array(embeddings)
    query_embedding = np.array(embedding)

    # Apply Maximal Marginal Relevance
    mmr_selected = maximal_marginal_relevance(
        query_embedding, embeddings, k=k, lambda_mult=lambda_mult
    )

    selected_docs = [docs_and_scores[i][0] for i in mmr_selected]

    return selected_docs

def max_marginal_relevance_search_with_threshold(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold: Optional[float] = None,
    **kwargs: Any,
) -> List[Document]:
    """
    Return documents selected using MMR with a score threshold, based on a query.

    Args:
        vectorstore (Chroma): Instance for vector search.
        query (str): Text to look up similar documents.
        k (int): Number of Documents to return. Defaults to 4.
        fetch_k (int): Number of Documents to fetch for MMR. Default is 20.
        lambda_mult (float): Controls diversity vs. relevance in MMR. Defaults to 0.5.
        score_threshold (Optional[float]): Maximum distance score to filter documents.
        **kwargs (Any): Additional arguments to pass to similarity search.

    Returns:
        List[Document]: Documents selected by MMR, filtered by score.
    """
    try:
        # Access embedding function
        embedding_function = vectorstore.embedding_function
        if embedding_function is None:
            logging.error("Chroma instance lacks an embedding function.")
            raise ValueError("The vector store must have an embedding function.")
        query_embedding = embedding_function.embed_query(query)
    except Exception as e:
        logging.error(f"Embedding of query failed: {e}")
        return []

    return max_marginal_relevance_search_with_threshold_by_vector(
        vectorstore=vectorstore,
        embedding=query_embedding,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        score_threshold=score_threshold,
        **kwargs,
    )
