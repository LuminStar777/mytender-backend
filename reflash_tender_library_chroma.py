import asyncio
import logging

from langchain_chroma import Chroma
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from config import CHROMA_FOLDER
from config import embedder, bids_collection
from services.embedding import text_to_chromadb
from services.helper import init_logger
from services.splitter import split_text

init_logger(screenlevel=logging.INFO, filename="reflash_log")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


async def save_to_chroma(text, user, collection, metadata, unique_id):
    """Embed text to chromadb and create index entry in mongodb"""

    # Split the text into chunks using Langchain's CharacterTextSplitter

    # ensure text has less than 100,000 characeters, otherwise loop over it and then concatenate:
    def split_large_text(text, max_chunk_size=99000):

        if len(text) <= max_chunk_size:
            return split_text(text)

        all_chunks = []
        for i in range(0, len(text), max_chunk_size):
            chunk = text[i : i + max_chunk_size]
            all_chunks.extend(split_text(chunk))

        return all_chunks

    chunks = split_large_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    chroma_db_directory = f"{CHROMA_FOLDER}/{user}"
    vectorstore = Chroma(
        collection,
        embedding_function=embedder,
        persist_directory=chroma_db_directory,
        collection_metadata={
            "metadata": metadata,
        },
    )
    await Chroma.aadd_texts(
        vectorstore, chunks, metadatas=[{"mongo_unique_id": unique_id} for _ in chunks]
    )
    log.info("Upload successful")


async def process_documents():
    # Count total documents for the progress bar
    total_docs = await bids_collection.count_documents({})

    # Loop through MongoDB documents with a progress bar
    cursor = bids_collection.find()
    async for doc in tqdm_asyncio(cursor, total=total_docs, desc="Processing documents"):
        try:
            tender_library = doc["tender_library"]
            user = doc["original_creator"]
            bid_id = str(doc["_id"])

            # Add progress bar for tender library entries
            for entry in tqdm(tender_library, desc=f"Processing entries for bid {bid_id}"):
                text = entry["rawtext"]
                unique_id = entry["filename"]
                metadata = entry["filename"]
                profile_name = "tender_library_" + bid_id
                await text_to_chromadb(
                    text=text,
                    user=user,
                    collection=profile_name,
                    user_name=user,
                    mode="qa",
                    embedding=embedder,
                    metadata=metadata,
                    unique_id=unique_id,
                    log_entry=False,
                )
                log.info(f"Uploaded {profile_name} with {len(text)} characters")
            # manually delete chroma folder on disk {CHROMA_FOLDER}/{user}

        except Exception as e:
            log.info(e)


if __name__ == "__main__":
    asyncio.run(process_documents())
