"""
Test module for creating an empty folder in chroma and then deleting it
"""
import asyncio
import json
import os
import uuid
import logging
import pytest
import chromadb
from chromadb.errors import InvalidCollectionException
from langchain_chroma import Chroma
from api import create_upload_folder, delete_template
from config import CHROMA_FOLDER, embedder
from services.embedding import text_to_chromadb
from services.chain import GraphState, retrieve_documents
from api_modules.company_library import (
    delete_content_library_item)
log = logging.getLogger(__name__)

pytestmark = pytest.mark.asyncio

@pytest.mark.asyncio
async def test_folder_creation_and_deletion():
    # Use a unique folder name to avoid conflicts
    test_folder_name = f"test_folder_{uuid.uuid4().hex[:8]}"
    try:
        # 1. Create a folder
        create_folder_result = await create_upload_folder(
            folder_name=test_folder_name,
            parent_folder=None,
            current_user="adminuser"
        )
        assert create_folder_result["message"] == "Folder created successfully"
        assert "unique_id" in create_folder_result
        assert create_folder_result["collection_name"] == test_folder_name
        # Store the collection name for later use
        collection_name = create_folder_result["collection_name"]
        # 2. Delete the folder
        delete_result = await delete_template(
            profile_name=test_folder_name,
            current_user="adminuser"
        )
        assert delete_result.status_code == 200
        delete_data = delete_result.body.decode()
        assert f"Deleted folder '{test_folder_name}' and all its subfolders." in delete_data
        # Parse the JSON response
        delete_json = json.loads(delete_data)
        chroma_db_directory = f"{CHROMA_FOLDER}/adminuser"
        # 3. Verify Chroma deletion
        chroma_client = chromadb.PersistentClient(path=chroma_db_directory)
        try:
            chroma_client.get_collection(name=collection_name)
            pytest.fail(f"Collection {collection_name} still exists in Chroma")
        except InvalidCollectionException:
            # ValueError is raised when the collection doesn't exist, which is what we want
            log.info(f"Collection {collection_name} successfully deleted from Chroma")
        # Assert that at least one Chroma folder was deleted
        assert delete_json["chroma_folders_deleted"] > 0, "No Chroma folders were deleted"
        log.info(f"Test completed successfully! {delete_json['chroma_folders_deleted']} Chroma folder(s) deleted.")
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")

@pytest.mark.asyncio
async def test_text_embedding_and_deletion():
    test_id = str(uuid.uuid4())
    test_collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    test_user = "test_user"
    try:
        chroma_db_directory = f"{CHROMA_FOLDER}/{test_user}"
        os.makedirs(chroma_db_directory, exist_ok=True)
        await text_to_chromadb(
            text="This is a test document",
            user=test_user,
            collection=test_collection,
            user_name=test_user,
            mode="qa",
            embedding=embedder,
            metadata={
                "filename": "test_document.txt",  # Added required filename
                "test": "metadata"
            },
            unique_id=test_id
        )
        await asyncio.sleep(1)
        client = chromadb.PersistentClient(path=chroma_db_directory)
        collection = client.get_collection(name=test_collection)
        results = collection.get(
            where={"mongo_unique_id": test_id}
        )
        assert len(results['ids']) > 0
        await delete_content_library_item(test_id, test_user)
        await asyncio.sleep(1)
        results_after_deletion = collection.get(
            where={"mongo_unique_id": test_id}
        )
        assert len(results_after_deletion['ids']) == 0
    finally:
        try:
            client = chromadb.PersistentClient(path=chroma_db_directory)
            client.delete_collection(test_collection)
            if os.path.exists(chroma_db_directory) and not os.listdir(chroma_db_directory):
                os.rmdir(chroma_db_directory)
        except Exception as e:
            log.error(f"Failed to delete test collection: {e}")

@pytest.mark.asyncio
async def test_retrieve_documents():
    """
    This test creates temporary Chroma collections (main, feedback, and subfolder),
    adds test documents to these collections, and then verifies that the
    retrieve_documents function correctly retrieves and processes these documents.

    Assertions:
    - Checks if retrieved_docs is not None.
    - Verifies that documents were retrieved (length > 0).
    - Ensures that test documents from all collections (main, feedback, subfolder) are present.
    - Verifies that each retrieved document has a 'source' field.
    """
    # Use a unique folder name to avoid conflicts
    test_folder_name = f"test_folder_{uuid.uuid4().hex[:8]}"
    test_user = "adminuser"
    chroma_db_directory = f"{CHROMA_FOLDER}/{test_user}"

    try:
        # 1. Set up test environment
        unique_id = str(uuid.uuid4())

        # Create test collections
        test_collection = Chroma(
            collection_name=test_folder_name,
            embedding_function=embedder,
            persist_directory=chroma_db_directory,
            collection_metadata={"metadata": unique_id, "unique_id": unique_id},
        )
        feedback_collection = Chroma(
            collection_name=f"{test_folder_name}_feedback",
            embedding_function=embedder,
            persist_directory=chroma_db_directory,
            collection_metadata={"metadata": unique_id, "unique_id": unique_id},
        )
        subfolder_collection = Chroma(
            collection_name=f"{test_folder_name}FORWARDSLASHsubfolder",
            embedding_function=embedder,
            persist_directory=chroma_db_directory,
            collection_metadata={"metadata": unique_id, "unique_id": unique_id},
        )

        # Add some test documents with metadata
        test_collection.add_texts(
            texts=["This is a test document", "Another test document"],
            ids=["test1", "test2"],
            metadatas=[{"mongo_unique_id": "test1"}, {"mongo_unique_id": "test2"}]
        )
        feedback_collection.add_texts(
            texts=["This is a feedback document"],
            ids=["feedback1"],
            metadatas=[{"mongo_unique_id": "feedback1"}]
        )
        subfolder_collection.add_texts(
            texts=["This is a subfolder document", "Another subfolder document"],
            ids=["sub1", "sub2"],
            metadatas=[{"mongo_unique_id": "sub1"}, {"mongo_unique_id": "sub2"}]
        )

        # 2. Create a test GraphState
        state = GraphState(
            choice="some_choice",
            input_text="test document",
            extra_instructions="",
            username=test_user,
            datasets=[test_folder_name],
            broadness=2,
            model="some_model",
            company_name="Test Company",
            bid_id=None,
            post_processing_enabled=False
        )

        # 3. Call retrieve_documents
        updated_state = await retrieve_documents(state)

        # 4. Assert the results
        assert updated_state.retrieved_docs is not None, "Retrieved docs is None"
        assert len(updated_state.retrieved_docs) > 0, "No documents were retrieved"
        # Update assertions to check content of dictionaries
        assert any("test document" in doc['content'] for doc in updated_state.retrieved_docs), "Test document not found in retrieved docs"
        assert any("subfolder document" in doc['content'] for doc in updated_state.retrieved_docs), "Subfolder document not found in retrieved docs"

        # Check if source (mongo_unique_id) is present
        assert all('source' in doc for doc in updated_state.retrieved_docs), "Source missing in some retrieved docs"

        log.info(f"Test completed successfully! {len(updated_state.retrieved_docs)} documents retrieved.")

    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")

    finally:
        # 5. Clean up
        try:
            Chroma(collection_name=test_folder_name, embedding_function=embedder, persist_directory=chroma_db_directory).delete_collection()
            Chroma(collection_name=f"{test_folder_name}_feedback", embedding_function=embedder, persist_directory=chroma_db_directory).delete_collection()
            Chroma(collection_name=f"{test_folder_name}FORWARDSLASHsubfolder", embedding_function=embedder, persist_directory=chroma_db_directory).delete_collection()
            log.info("Test collections deleted successfully")
        except Exception as cleanup_error:
            log.error(f"Error during cleanup: {str(cleanup_error)}")
