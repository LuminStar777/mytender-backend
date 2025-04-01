import asyncio
import logging

from tqdm.asyncio import tqdm_asyncio

from config import doc_entry_collection, embedder
from services.embedding import text_to_chromadb
from services.helper import init_logger

init_logger(screenlevel=logging.INFO, filename="reflash_log")
log = logging.getLogger(__name__)


async def process_documents():
    # Count total documents for the progress bar
    total_docs = await doc_entry_collection.count_documents({})

    # Loop through MongoDB documents with a progress bar
    cursor = doc_entry_collection.find()
    async for doc in tqdm_asyncio(cursor, total=total_docs, desc="Processing documents"):

        metadata = doc.get('meta', '')

        try:
            unique_id = doc['unique_id']
            text = doc['text']
            profile_name = doc['collection_name']
            user = doc['user']
            # manually delete chroma folder on disk {CHROMA_FOLDER}/{user}
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
        except Exception as e:
            log.error(f"Failed upload for {metadata} with error: {e}")
            # delete mongodb entry by mongo_id
            await doc_entry_collection.delete_one({"_id": doc["_id"]})


if __name__ == "__main__":
    asyncio.run(process_documents())
