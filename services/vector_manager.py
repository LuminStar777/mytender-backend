"""
This module contains functions for interacting with vector stores and retrieving documents.
"""

import asyncio
import logging
import re
import time
from functools import wraps
from typing import List, Dict, Any, Tuple

import chromadb
from langchain_chroma import Chroma

from config import (
    embedder,
    doc_entry_collection,
    CHROMA_FOLDER,
    RETRIEVE_SUBTOPIC_CHUNKS,
)
from utils import get_parent_user

log = logging.getLogger(__name__)


# Helper function for timing async functions
def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        log.info(f"Starting {func_name}")
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        log.info(f"Completed {func_name} in {duration:.2f}s")
        return result

    return wrapper


@async_timer
async def retrieve_docs(vectorstore, query: str, k: int, parent_user: str) -> List[Dict[str, str]]:
    """
    Retrieve documents from a vectorstore with timing metrics.
    """
    try:
        collection_name = vectorstore._collection.name  # pylint: disable=protected-access
        retriever_start = time.time()
        log.info(
            "Creating retriever for collection '%s' with query: '%s...'",
            collection_name,
            query[:50],
        )
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
        retriever_time = time.time() - retriever_start
        log.info("Retriever created for collection '%s' in %.2fs", collection_name, retriever_time)

        docs_start = time.time()

        async def _retrieve_with_logging():
            try:
                # Do the actual search using ainvoke
                log.info("Performing vector search in collection '%s'...", collection_name)
                search_start = time.time()
                docs = await retriever.ainvoke(query)

                # Log retrieved documents preview
                for i, doc in enumerate(docs):
                    preview = doc.page_content[:50].replace('\n', ' ').strip()
                    mongo_id = doc.metadata.get('mongo_unique_id', 'Unknown')
                    log.debug(
                        "[%s] Doc[%d] Preview: '%s...' [Source: %s]",
                        collection_name,
                        i,
                        preview,
                        mongo_id,
                    )

                # Get unique IDs from the retrieved documents
                unique_ids = [
                    doc.metadata.get('mongo_unique_id')
                    for doc in docs
                    if 'mongo_unique_id' in doc.metadata
                ]

                # Fetch corresponding document metadata from MongoDB
                documents = await doc_entry_collection.find(
                    {"user": parent_user, "unique_id": {"$in": unique_ids}},
                    {"_id": 0, "filename": 1, "unique_id": 1},
                ).to_list(length=None)

                # Create a mapping of unique_id to filename
                id_to_filename = {doc['unique_id']: doc['filename'] for doc in documents}

                # Format documents with correct source information
                formatted_docs = []
                for doc in docs:
                    source = id_to_filename.get(doc.metadata.get('mongo_unique_id'), 'Unknown')
                    formatted_docs.append(
                        {
                            "content": doc.page_content,
                            "source": source,
                        }
                    )
                    preview = doc.page_content[:50].replace('\n', ' ')
                    log.info(
                        "[%s] Formatted doc from '%s', content preview: '%s...'",
                        collection_name,
                        source,
                        preview,
                    )

                log.info(
                    "Search completed in collection '%s' in %.2fs",
                    collection_name,
                    time.time() - search_start,
                )
                return formatted_docs
            except Exception as e:
                log.error(
                    "Error in retrieve function for collection '%s': %s",
                    collection_name,
                    str(e),
                    exc_info=True,
                )
                raise

        try:
            docs = await asyncio.wait_for(_retrieve_with_logging(), timeout=45.0)
            docs_time = time.time() - docs_start
            log.info(
                "Retrieved %d documents from collection '%s' in %.2fs",
                len(docs),
                collection_name,
                docs_time,
            )
            return docs
        except asyncio.TimeoutError:
            log.error(
                "Retriever invocation timed out after 45 seconds for collection '%s'",
                collection_name,
            )
            return []
        except Exception as e:
            log.error(
                "Error during retriever invocation for collection '%s': %s",
                collection_name,
                str(e),
                exc_info=True,
            )
            return []

    except Exception as e:
        log.error(
            "Error in retrieve_docs for collection '%s': %s", collection_name, str(e), exc_info=True
        )
        return []


@async_timer
async def query_vectorstore(collection_name, username, query, k, parent_user):
    """
    Query a specific vectorstore collection.
    """
    try:
        start_time = time.time()
        log.info(f"Creating vectorstore for collection: {collection_name}")

        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=f"{CHROMA_FOLDER}/{username}",
            embedding_function=embedder,
        )

        log.info(f"Retrieving docs from {collection_name}")
        docs = await retrieve_docs(vectorstore, query, k, parent_user)

        duration = time.time() - start_time
        log.info(
            f"Retrieved {len(docs)} documents from {collection_name} in {duration:.2f} seconds"
        )
        return docs
    except Exception as e:
        log.error(f"Error querying collection {collection_name}: {e}")
        return []


@async_timer
async def query_collection_and_subfolders(base_collection, username, query, k, parent_user):
    """
    Query a collection and all its subfolders.
    """
    collection_start = time.time()
    log.info(f"[{base_collection}] Starting collection and subfolder query...")

    # Get all collections
    client_start = time.time()
    all_collections, _ = await get_collections_and_check_exists(username)
    client_time = time.time() - client_start
    log.info(f"[{base_collection}] Got collection list in {client_time:.2f}s")

    # Find relevant collections
    collections_to_query = [base_collection]
    subfolder_pattern = f"^{re.escape(base_collection)}FORWARDSLASH"
    collections_to_query.extend(
        [coll_name for coll_name in all_collections if re.match(subfolder_pattern, coll_name)]
    )

    log.info(
        f"[{base_collection}] Found {len(collections_to_query)} collections to query: {collections_to_query}"
    )

    # Query all collections in parallel
    results = await asyncio.gather(
        *[
            query_vectorstore(collection_name, username, query, k, parent_user)
            for collection_name in collections_to_query
        ]
    )

    total_docs = sum(len(docs) for docs in results)
    total_time = time.time() - collection_start
    log.info(
        f"[{base_collection}] Retrieved {total_docs} total documents from {len(collections_to_query)} collections in {total_time:.2f}s"
    )

    return [doc for result in results for doc in result]


async def retrieve_bid_library_docs(
    relevant_query: str, bid_id: str, username: str, k: int
) -> List[Dict[str, str]]:
    """
    Retrieve documents from the bid library.
    """
    if not bid_id:
        return []

    full_query = f"{relevant_query}"
    vectorstore_tender_library = Chroma(
        collection_name='tender_library_' + bid_id,
        persist_directory=f"{CHROMA_FOLDER}/{username}",
        embedding_function=embedder,
    )

    bid_library_docs = await retrieve_docs(
        vectorstore_tender_library, full_query, k, await get_parent_user(username)
    )
    return bid_library_docs


async def retrieve_content_library_docs(
    username: str, datasets: List[str], sub_topic: str, k: int = RETRIEVE_SUBTOPIC_CHUNKS
) -> List[Dict[str, str]]:
    """
    Retrieve documents from the content library.
    """
    log.info(f"Content library datasets being queried: {datasets}")

    all_docs = []
    # Query each dataset
    for dataset in datasets:
        state_docs = await query_vectorstore(
            dataset, username, sub_topic, k, await get_parent_user(username)
        )
        all_docs.extend(state_docs)

    # Remove duplicates based on content and source
    unique_docs = {}
    for doc in all_docs:
        key = (doc['content'], doc['source'])
        if key not in unique_docs:
            unique_docs[key] = doc

    return list(unique_docs.values())


@async_timer
async def retrieve_documents_for_state(state):
    """
    Retrieve relevant documents from collections based on the input state.
    Parallelized version with timing metrics.
    """
    total_start = time.time()
    k = int(state.broadness)
    parent_user = await get_parent_user(state.username)

    # Prepare all queries
    tasks = []
    log.info(f"Preparing queries for datasets: {state.datasets}")

    for dataset in state.datasets:
        tasks.append(
            query_collection_and_subfolders(
                dataset, state.username, state.input_text, k, parent_user
            )
        )

    if state.bid_id:
        log.info(f"Adding tender library query for bid_id: {state.bid_id}")
        tasks.append(
            query_vectorstore(
                'tender_library_' + state.bid_id, state.username, state.input_text, k, parent_user
            )
        )

    # Run all queries in parallel
    parallel_start = time.time()
    log.info(f"Starting parallel execution of {len(tasks)} tasks...")
    all_results = await asyncio.gather(*tasks)
    parallel_time = time.time() - parallel_start
    log.info(f"Completed parallel execution in {parallel_time:.2f}s")

    # Deduplicate results
    dedup_start = time.time()
    log.info("Starting deduplication...")

    seen_keys = set()
    deduped_docs = []
    total_docs = 0

    for result in all_results:
        total_docs += len(result)
        for doc in result:
            key = f"{doc.get('content', '')}::{doc.get('source', '')}"
            if key not in seen_keys:
                seen_keys.add(key)
                clean_doc = {
                    "content": str(doc.get('content', '')),
                    "source": str(doc.get('source', '')),
                }
                deduped_docs.append(clean_doc)

    dedup_time = time.time() - dedup_start
    total_time = time.time() - total_start

    log.info(f"Deduplication completed in {dedup_time:.2f}s")
    log.info(f"Total documents before deduplication: {total_docs}")
    log.info(f"Total unique documents after deduplication: {len(deduped_docs)}")
    log.info(f"Total retrieval process completed in {total_time:.2f}s")

    return deduped_docs


@async_timer
async def retrieve_evidence_chunks(
    username: str, query: str, num_chunks: int = 2
) -> List[Dict[str, Any]]:
    """
    Retrieve evidence chunks from the user's default collection.

    Args:
        username: The username whose collection to search
        query: The search query
        num_chunks: Number of chunks to retrieve (default: 2)

    Returns:
        List of evidence chunks with content and source information
    """
    log.info(f"Retrieving evidence chunks for query: '{query}'")
    parent_user = await get_parent_user(username)

    try:
        # Get all collections
        collection_names, _ = await get_collections_and_check_exists(username)

        # If no collections are found, return empty results
        if not collection_names:
            log.warning(f"No collections found for user {username}")
            return []

        # Priority ordered list of collections to try
        collections_to_try = ['default']

        # Add remaining collections
        for coll in collection_names:
            if coll not in collections_to_try and not coll.startswith('tender_library_'):
                collections_to_try.append(coll)

        # If no suitable collections found
        if not collections_to_try:
            log.warning(f"No suitable collections found for user {username}")
            return []

        # Use the first available collection
        collection_name = collections_to_try[0]
        log.info(f"Using collection {collection_name} for evidence retrieval")

        # Create the vectorstore and retrieve chunks
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=f"{CHROMA_FOLDER}/{username}",
            embedding_function=embedder,
        )

        # Retrieve documents with a smaller number of chunks (just the top ones)
        evidence_docs = await retrieve_docs(vectorstore, query, num_chunks, parent_user)

        # Format the evidence properly
        formatted_evidence = []
        for doc in evidence_docs:
            formatted_evidence.append(
                {"content": doc.get('content', ''), "source": doc.get('source', 'Unknown source')}
            )

        log.info(f"Retrieved {len(formatted_evidence)} evidence chunks")
        return formatted_evidence

    except Exception as e:
        log.error(f"Error retrieving evidence chunks: {str(e)}")
        return []


async def get_collections_and_check_exists(
    username: str,
    collection_name: str = None
) -> Tuple[List[str], bool]:
    """
    Get all collections for a user and optionally check if a specific collection exists.

    Args:
        username (str): The username to get collections for
        collection_name (str, optional): The collection name to check for existence

    Returns:
        Tuple[List[str], bool]: A tuple containing:
            - The list of all collection names
            - A boolean indicating if the specified collection_name exists (always True if collection_name is None)
    """
    try:
        log.info(f"Getting ChromaDB collections for user: {username}")
        chroma_db_directory = f"{CHROMA_FOLDER}/{username}"
        client = chromadb.PersistentClient(path=chroma_db_directory)

        # In ChromaDB v0.6.0+, list_collections() returns collection names directly
        all_collections = client.list_collections()
        log.info(f"Found {len(all_collections)} collections for user {username}")

        # If no collection name specified, just return all collections with existence=True
        if collection_name is None:
            return all_collections, True

        # Check if the specified collection exists
        exists = collection_name in all_collections
        return all_collections, exists

    except Exception as e:
        log.error(f"Error getting ChromaDB collections: {str(e)}")
        return [], False
