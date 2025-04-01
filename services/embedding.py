import logging
import uuid
from bson.binary import Binary
from chromadb.utils import embedding_functions
from config import CHROMA_FOLDER, doc_entry_collection
from langchain_chroma import Chroma
from services.helper import post_to_slack
from services.splitter import split_text

# pylint: disable=too-many-positional-arguments

log = logging.getLogger(__name__)

default_ef = embedding_functions.DefaultEmbeddingFunction()


async def text_to_chromadb(
    text: str,
    user: str,
    collection: str,
    user_name: str,
    mode,
    embedding,
    metadata,
    file_size = 1024000 ,
    file_content: Binary = None,
    log_entry=True,
    format='text',
    unique_id=None
):
    # Log initial function call and parameters
    log.info(f"Starting text_to_chromadb with metadata: {metadata}")
    log.info(f"Unique ID: {unique_id}")

    if unique_id is None:
        unique_id = str(uuid.uuid4())

    """Embed text to chromadb and create index entry in mongodb"""
    log.info("Embedding text to chromadb")
    log.info("--> Loading and embedding text, generating titles")

    # Split the text into chunks using Langchain's CharacterTextSplitter
    chunks = split_text(text)


    # Convert the chunks of text into embeddings to form a knowledge base
    chroma_db_directory = f"{CHROMA_FOLDER}/{user}"
    log.info(f"Creating vectorstore at {chroma_db_directory}")

    vectorstore = Chroma(
        collection,
        embedding_function=embedding,
        persist_directory=chroma_db_directory,
        collection_metadata={
            "metadata": unique_id,
            "unique_id": unique_id,
        },
    )
    # Create metadatas list with both chunk-specific and document metadata
    chunk_metadatas = [
        {
            "mongo_unique_id": unique_id,
            "chunk_index": i,
            **metadata  # Include all passed metadata (including filename, upload_date, uploaded_by)
        }
        for i, _ in enumerate(chunks)
    ]

    # Log first chunk metadata to verify structure
    if chunk_metadatas:
        log.info("First chunk metadata example:")
        log.info(chunk_metadatas[0])

    log.info("Attempting to add texts to Chroma...")

    await Chroma.aadd_texts(
        vectorstore,
        chunks,
        metadatas=chunk_metadatas
    )

    log.info(f"--> Successfully embedded text into ChromaDB {chroma_db_directory}.")

    # Build the document to be inserted into MongoDB
    document = {
        "collection_name": collection,
        "chroma_db_directory": chroma_db_directory,
        "mode": mode,
        "text": text,
        "user": user_name,
        "format": format,
        "meta": metadata,
        "filename": metadata['filename'],
        'profile_name': collection,
        "unique_id": unique_id,
    }

    log.info("Document being inserted into MongoDB:")
    log.info("METADATA FILENAME")
    log.info(metadata['filename'])

    MAX_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    if file_size < MAX_SIZE:
        log.info("filesize ok")
        if mode == 'pdf':
            document["pdf_file_content"] = file_content
            log.info("adding pdf content")
        elif format == 'file':  # Add this condition for non-PDF files
            document["file_content"] = file_content  # Use the same field for storing binary content

    #make sure tender library files dont get inserted into doc collection
    if log_entry:
        log.info("inserting into company lib")
        await doc_entry_collection.insert_one(document)

    log.info("Successfully completed text_to_chromadb function")
    await post_to_slack(
        f"User {user_name} added a new document to collection {collection} with metadata {metadata}"
    )

async def delete_chroma_entry(unique_id: str, user: str, collection_name: str):
    chroma_db_directory = f"{CHROMA_FOLDER}/{user}"
    log.info(
        f"--> Deleting entry from ChromaDB for unique_id: {unique_id} in collection: {collection_name}"
    )

    try:
        # Initialize Chroma client
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=chroma_db_directory,
        )

        # Get all documents in collection before deletion
        all_docs = vectorstore.get()
        log.info(f"--> Total documents in collection before deletion: {len(all_docs['ids'])}")

        # Get matching documents and log details
        matching_docs = vectorstore.get(where={"mongo_unique_id": unique_id}, limit=None)
        log.info(f"--> Found {len(matching_docs['ids'])} matching documents to delete")
        log.info(f"--> Document IDs to delete: {matching_docs['ids']}")

        if not matching_docs['ids']:
            log.warning(f"--> No matching documents found for unique_id: {unique_id}")
            # Log all metadata to help debug
            all_metadata = vectorstore.get()['metadatas']
            log.info("Available metadata in collection:")
            for metadata in all_metadata:
                log.info(f"    {metadata}")
            return

        # Delete the entry from ChromaDB
        vectorstore.delete(ids=matching_docs['ids'])

        # Verify deletion
        remaining_docs = vectorstore.get(where={"mongo_unique_id": unique_id}, limit=None)
        if remaining_docs['ids']:
            log.error(f"--> Deletion failed! {len(remaining_docs['ids'])} documents still remain")
            log.error(f"--> Remaining IDs: {remaining_docs['ids']}")
        else:
            log.info(f"--> Successfully deleted {len(matching_docs['ids'])} chunks from ChromaDB")

        # Log final collection state
        final_docs = vectorstore.get()
        log.info(f"--> Total documents in collection after deletion: {len(final_docs['ids'])}")

    except Exception as e:
        log.error(f"--> Failed to delete ChromaDB entry: {str(e)}", exc_info=True)


async def delete_chroma_folder(profile_name: str, user: str):
    """Delete a document entry from MongoDB and its corresponding ChromaDB"""
    log.info("Deleting ChromaDB entry")
    log.info("--> Deleting entry from MongoDB")
    entry = await doc_entry_collection.find_one({"profile_name": profile_name})
    if entry:
        await doc_entry_collection.delete_one({"profile_name": profile_name})
        log.info(f"--> Successfully deleted entry from MongoDB with profile_name {profile_name}")
    else:
        log.info(f"--> Entry with profile_name {profile_name} not found in MongoDB")

    chroma_db_directory = f"{CHROMA_FOLDER}/{user}"
    log.info(
        f"--> Attempting to delete ChromaDB collection: {profile_name} in directory: {chroma_db_directory}"
    )

    try:
        vectorstore = Chroma(
            collection_name=profile_name,
            persist_directory=chroma_db_directory,
        )

        # Log collection contents before deletion
        all_docs = vectorstore.get()
        log.info(f"--> Collection contains {len(all_docs['ids'])} documents before deletion")
        log.info("--> Collection metadata before deletion:")
        for metadata in all_docs['metadatas']:
            log.info(f"    {metadata}")

        vectorstore.delete_collection()

        # Verify collection deletion
        try:
            test_vectorstore = Chroma(
                collection_name=profile_name,
                persist_directory=chroma_db_directory,
            )
            test_docs = test_vectorstore.get()
            if len(test_docs['ids']) > 0:
                log.error(
                    f"--> Collection deletion failed! {len(test_docs['ids'])} documents still exist"
                )
            else:
                log.info("--> Collection successfully deleted and verified empty")
        except Exception as verify_e:
            log.info(f"--> Collection appears to be deleted (cannot be accessed): {str(verify_e)}")

    except Exception as e:
        log.error(f"--> Failed to delete ChromaDB collection: {str(e)}", exc_info=True)
