"""
This module contains the implementation for api endpoints related to the tender library
"""

import logging
import re
from typing import Dict, List

from fastapi import HTTPException
from bson import ObjectId
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from services.chain import generic_run_chain, load_prompt_from_file, retrieve_bid_library_docs
from config import CHROMA_FOLDER, embedder, bids_collection, llm_tender_library_chat
from utils import get_parent_user

log = logging.getLogger(__name__)


async def find_matching_document_snippets(input_text, username, bid_id) -> List[Dict[str, str]]:
    """
    This function searches the tender library for documents matching the input text.
    It uses vector similarity search to find relevant documents and extracts snippets
    containing the search query.

    Args:
        input_text (str): The text to search for in the documents.
        username (str): The username of the user performing the search.
        bid_id (str): The ID of the bid to search within. If None, skips tender library search.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing a 'snippet' and 'document_name'.
    """
    log.info(
        f"Starting search for input_text: '{input_text}', username: '{username}', bid_id: '{bid_id}'"
    )
    k = 6  # Increased to account for potential filtering
    parent_user = await get_parent_user(username)
    all_docs = []

    if bid_id:
        log.info(f"Querying tender library for bid_id: {bid_id}")
        try:
            vectorstore_tender_library = Chroma(
                collection_name='tender_library_' + bid_id,
                persist_directory=f"{CHROMA_FOLDER}/{parent_user}",
                embedding_function=embedder,
            )
            log.info(f"Vectorstore created successfully for collection: tender_library_{bid_id}")

            bid_library_docs = await retrieve_docs(vectorstore_tender_library, input_text, k)
            log.info(f"Retrieved {len(bid_library_docs)} relevant documents from tender library")
            all_docs.extend(bid_library_docs)
        except Exception as e:
            log.error(f"Error creating or querying vectorstore: {e}")
    else:
        log.warning("No bid_id provided, skipping tender library search")

    # Remove duplicates
    retrieved_docs = list({(doc['snippet'], doc['document_name']) for doc in all_docs})
    log.info(f"Total unique relevant documents retrieved: {len(retrieved_docs)}")

    result = [{"snippet": doc[0], "document_name": doc[1]} for doc in retrieved_docs]
    log.info(f"Returning {len(result)} results")
    return result

async def process_ask_tender_library_question(
    question: str,
    chat_history: List[Dict[str, str]],
    username: str,
    bid_id: str
) -> str:
    """Process a question about tender library documents."""
    log.info("Asking tender library chat question")

    # Validate bid exists
    bid_object_id = ObjectId(bid_id)
    bid = await bids_collection.find_one({"_id": bid_object_id})
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")

    prompt = ChatPromptTemplate.from_template(load_prompt_from_file("ask_tender_library_question"))

    # Try primary method - using the tender library
    try:
        tender_library = bid.get("tender_library", [])
        if not tender_library:
            raise HTTPException(status_code=404, detail="No documents found in the tender library")

        log.info("Using documents from tender library")
        combined_text = "\n\n".join(doc.get("rawtext", "") for doc in tender_library)

        return await generic_run_chain(
            llm_tender_library_chat,
            {
                "context": combined_text,
                "question": question,
                "chat_history": chat_history
            },
            prompt
        )

    # Fall back to retrieval method
    except (HTTPException, Exception) as e:
        log.error(f"Error using tender library: {e}. Falling back to retrieval.")

        try:
            bid_library_docs = await retrieve_bid_library_docs(
                question, bid_id, username, 10
            )
            log.info("Using documents from retrieval")

            context_bid_library = "\n\n".join(
                f"[Source: {doc['source']}]\n{doc['content']}" for doc in bid_library_docs
            )

            return await generic_run_chain(
                llm_tender_library_chat,
                {
                    "context": context_bid_library,
                    "question": question,
                    "chat_history": chat_history
                },
                prompt
            )

        except Exception as retrieval_error:
            log.error(f"Error in retrieval fallback: {retrieval_error}")
            return "Error: Unable to generate a response for your question."

async def retrieve_docs(vectorstore, query: str, k: int) -> List[Dict[str, str]]:
    """
    Retrieve relevant documents from a vectorstore based on a query.
    Args:
        vectorstore: The vectorstore to search in (must have an as_retriever method).
        query (str): The search query.
        k (int): The number of documents to retrieve.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing a 'snippet' and 'document_name'.
    """
    log.info(f"Retrieving docs for query: '{query}', k: {k}")
    try:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
        docs = await retriever.ainvoke(query)
        log.info(f"Retrieved {len(docs)} documents")

        results = []
        for doc in docs:
            snippet = extract_snippet(doc.page_content, query)
            if snippet:  # Only add to results if a valid snippet was found
                doc_name = doc.metadata.get('mongo_unique_id', 'Unknown')
                log.info(f"Document: {doc_name}, Snippet: {snippet}")
                results.append({"snippet": snippet, "document_name": doc_name})

        log.info(f"Found {len(results)} relevant snippets containing the exact search query")
        return results
    except Exception as e:
        log.error(f"Error getting relevant documents: {e}")
        return []


def extract_snippet(text: str, query: str, context_words: int = 12) -> str:
    """
    This function searches for the exact query in the text (case-insensitive) and
    extracts a snippet of text around it. The snippet includes a specified number
    of context words before and after the query.

    Args:
        text (str): The full text to extract the snippet from.
        query (str): The search query to find in the text.
        context_words (int, optional): The number of words to include before and after the query. Defaults to 12.

    Returns:
        str: A snippet of text containing the query, or None if the query is not found.
    """
    # Find the exact query in the text (case-insensitive)
    match = re.search(re.escape(query), text, re.IGNORECASE)
    if not match:
        # If exact query not found, return None
        return None
    # Get the start and end positions of the match
    start, end = match.span()
    # Find the start and end indices for the snippet
    words = text.split()
    word_start = max(0, len(text[:start].split()) - context_words)
    word_end = min(len(words), len(text[:end].split()) + context_words)
    # Extract the snippet
    snippet = ' '.join(words[word_start:word_end])
    # If the snippet is too long, truncate it while keeping the query
    if len(snippet.split()) > 25:
        query_start_word = len(text[:start].split())
        query_end_word = len(text[:end].split())
        # Ensure the query is in the middle of the snippet
        start_index = max(0, query_start_word - context_words)
        end_index = min(len(words), query_end_word + context_words)
        snippet = ' '.join(words[start_index:end_index])

    return f"...{snippet}..."


def remove_exclamation_marks(text):
    return re.sub(r'!!!(.*?)!!!', '', text)


def process_text_with_formatting(paragraph, text):
    text = remove_exclamation_marks(text)
    parts = re.split(r'(\[.*?\])', text)
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            run = paragraph.add_run(part[1:-1])
            run.bold = True
        else:
            if part:
                paragraph.add_run(part)


def count_words(text: str) -> int:
    """
    Count words in text, handling various whitespace and punctuation.

    Args:
        text (str): The text to count words in

    Returns:
        int: Number of words in the text
    """
    # Remove extra whitespace and handle special characters
    text = text.strip()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Split on whitespace and filter out empty strings
    words = [word for word in text.split() if word]
    return len(words)
