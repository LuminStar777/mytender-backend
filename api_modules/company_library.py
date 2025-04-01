"""
Company Library API Module

This module implements API endpoints and related functionality
for managing a company's document library.

Key Features:
- File upload and processing (PDF, DOCX, DOC)
- Collection management (creation, listing, deletion)
- Document retrieval and content display
- Template (folder) management

Main Functions:
- process_get_collections: Retrieve organisation collections
- process_create_upload_folder: Create a new folder (collection) in the library
- process_get_folder_filenames: Get filenames within a specific collection
- process_create_upload_file: Handle file uploads and processing
- process_show_file_content: Retrieve text content of a file
- process_show_file_content_pdf_format: Stream PDF file content
- process_update_text: Update existing document text
- process_create_upload_text: Create and upload new text-based content
- process_delete_template: Delete a template (folder) and its associated data

The module uses these libraries:
- FastAPI
- MongoDB
- ChromaDB for vector storage
- LlamaParse for PDF parsing
- Mammoth for DOC file processing
"""
import io
import logging
import os
import re
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import chromadb
from bson.binary import Binary
from fastapi import (
    HTTPException,
    UploadFile
)
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from pymongo import DESCENDING
from pymongo.results import UpdateResult

from config import (
    embedder,
    doc_entry_collection,
    CHROMA_FOLDER,
    feedback_collection
)
from services.embedding import (
    delete_chroma_entry,
    text_to_chromadb
)
from services.parser import parse_file_content
from services.vector_manager import get_collections_and_check_exists
from utils import (
    get_parent_user,
    UserNotFoundException,
    is_user_type,
)

log = logging.getLogger(__name__)


async def process_get_collections(current_user: str) -> List[str]:
    """
    -This function returns the chroma collections for a user and is used in the Company Library.
    -If the user is an owner (they bought the licenses) we get their own collections
    -If the user is a member, added by the owner we get the owner's collections
    -This is so that users in an organisation share the same company library

    Args:
        current_user (str): The username of the current authenticated user.
    Returns:
        List[str]: A list of collection names.
    """
    try:
        parent_user = await get_parent_user(current_user)
        collections, _ = await get_collections_and_check_exists(parent_user)

        # Filter out entries that contain '_feedback' and tender_library
        collections = [col_name for col_name in collections
                      if "_feedback" not in col_name
                      and not col_name.startswith("tender_library")
                      and not col_name.startswith("case_studies_collection")]
        return collections
    except UserNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))

async def process_get_case_studies(current_user: str) -> List[str]:
    """
    This function returns the case study collections for a user and is used in the Company Library.
    - If the user is an owner (they bought the licenses) we get their own collections
    - If the user is a member, added by the owner we get the owner's collections
    - This is so that users in an organisation share the same company library

    Args:
        current_user (str): The username of the current authenticated user.
    Returns:
        List[str]: A list of case study collection names.
    """
    log.info(f"Getting case studies for user: {current_user}")
    try:
        parent_user = await get_parent_user(current_user)
        log.info(f"Parent user determined: {parent_user}")

        collections, _ = await get_collections_and_check_exists(parent_user)
        log.info(f"Found {len(collections)} total collections")

        # Filter to only include collections that are inside case_studies_collection
        # This will match both direct children and nested folders
        case_study_collections = [
            col_name for col_name in collections
            if col_name.startswith("case_studies_collection")
        ]

        # Remove the case_studies_collection itself if it exists, as we only want its children
        if "case_studies_collection" in case_study_collections:
            case_study_collections.remove("case_studies_collection")

        log.info(f"Filtered to {len(case_study_collections)} case study collections")
        log.debug(f"Case study collections: {case_study_collections}")

        return case_study_collections
    except Exception as e:
        log.error(f"Error retrieving case studies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving case studies: {str(e)}")

async def process_create_upload_folder(
        folder_name: str,
        parent_folder: Optional[str],
        current_user: str) -> Dict[str, str]:
    """
    This function creates a new collection in the Chroma database, representing a folder
    in the user's document structure. It generates a unique ID for the folder and handles
    nested folder structures if a parent folder is specified. It prevents creation of
    duplicate folder names under the same parent folder.

    Args:
        folder_name (str): The name of the folder to be created.
        parent_folder (Optional[str]): The name of the parent folder, if any.
         Defaults to None for root-level folders.
        current_user (str): The username of the current authenticated user.

    Returns:
        Dict[str, str]: A dictionary containing information about the created folder.

    Raises:
        HTTPException: If there's an error in creating the Chroma collection or if
         a folder with the same name already exists.
    """

    unique_id = str(uuid.uuid4())
    parent_user = await get_parent_user(current_user)
    chroma_db_directory = f"{CHROMA_FOLDER}/{parent_user}"

    # Use parent_folder + "FORWARDSLASH" + folder_name for the collection name
    collection_name = f"{parent_folder}FORWARDSLASH{folder_name}" if parent_folder else folder_name

    # Check if collection already exists
    try:
        _, collection_exists = await get_collections_and_check_exists(parent_user, collection_name)

        if collection_exists:
            error_msg = f"A folder named '{folder_name}' already exists in this location"
            log.error(f"Duplicate folder error: {error_msg}")
            # Use JSONResponse instead of HTTPException
            return JSONResponse(
                status_code=400,
                content={"detail": error_msg}
            )

        # Create Chroma collection
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=chroma_db_directory,
            collection_metadata={
                "metadata": unique_id,
                "unique_id": unique_id,
            },
        )
        log.info(f"Vectorstore created: {vectorstore}")

        return {
            "message": "Folder created successfully",
            "unique_id": unique_id,
            "chroma_db_directory": chroma_db_directory,
            "collection_name": collection_name,
            "parent_folder": parent_folder,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Error creating folder: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error creating folder: {str(e)}"}
        )

async def process_create_upload_folder_case_studies(
        folder_name: str,
        parent_folder: Optional[str],
        current_user: str) -> Dict[str, str]:
    """
    This function creates a new collection in the Chroma database, representing a folder
    in the user's document structure.

    Args:
        folder_name (str): The name of the folder to be created.
        parent_folder (Optional[str]): The name of the parent folder, if any.
         Defaults to None for root-level folders.
        current_user (str): The username of the current authenticated user.
    Returns:
        Dict[str, str]: A dictionary containing information about the created folder.

    Raises:
        HTTPException: If there's an error in creating the Chroma collection or if
         a folder with the same name already exists.
    """
    # Check if user is an owner
    if not await is_user_type(current_user, "owner"):
        log.info(f"User {current_user} attempted to create folder but is not an owner")
        raise HTTPException(status_code=403, detail="Only owners can create content library folders")

    unique_id = str(uuid.uuid4())
    parent_user = await get_parent_user(current_user)
    chroma_db_directory = f"{CHROMA_FOLDER}/{parent_user}"

    if parent_folder:
        collection_name = f"{parent_folder}FORWARDSLASH{folder_name}"
    else:
        parent_folder = "case_studies_collection"
        collection_name = f"{parent_folder}FORWARDSLASH{folder_name}"

    # Check if collection already exists
    try:
        _, collection_exists = await get_collections_and_check_exists(parent_user, collection_name)

        if collection_exists:
            error_msg = f"A folder named '{folder_name}' already exists in this location"
            log.error(f"Duplicate folder error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # Create Chroma collection
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=chroma_db_directory,
            collection_metadata={
                "metadata": unique_id,
                "unique_id": unique_id,
            },
        )
        log.info(f"Vectorstore created: {vectorstore}")
        return {
            "message": "Folder created successfully",
            "unique_id": unique_id,
            "chroma_db_directory": chroma_db_directory,
            "collection_name": collection_name,
            "parent_folder": parent_folder,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Error creating folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")

async def process_get_folder_filenames(
        collection_name: str,
        current_user: str) -> List[Dict[str, any]]:
    """
    Retrieve filenames and metadata for documents in a specified collection for a user.

    Args:
        collection_name (str): The name of the collection to query.
        current_user (str): The username of the current authenticated user.

    Returns:
        List[Dict[str, any]]: A list of dictionaries, each containing:
            - meta: Metadata associated with the document.
            - unique_id: Unique identifier of the document.

    Raises:
        HTTPException: If the collection name is not provided (400 Bad Request).
    """
    if not collection_name:
        raise HTTPException(status_code=400, detail="Collection name is required")

    # Fetching documents from the database
    parent_user = await get_parent_user(current_user)
    documents = await doc_entry_collection.find(
        {
            "user": parent_user,
            "collection_name": collection_name,
        },
        {"_id": 0, "meta": 1, "unique_id": 1, "filename": 1},
    ).sort("timestamp", DESCENDING).limit(100).to_list(length=None)

    return documents

async def process_move_file(
    unique_id: str,
    new_folder: str,
    current_user: str,
):
    """
    Move a file from one folder to another by copying it to new location and deleting from old

    Args:
        unique_id (str): Unique identifier of the file to move
        new_folder (str): Destination folder path
        current_user (str): Current authenticated user
    """
    # Check if user is an owner
    if not await is_user_type(current_user, "owner"):
        log.info(f"User {current_user} attempted to move file but is not an owner")
        raise HTTPException(status_code=403, detail="Only owners can move content library files")

    # Get parent user
    parent_user = await get_parent_user(current_user)

    # Get the existing document
    entry = await doc_entry_collection.find_one({"unique_id": unique_id})
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    filename = entry.get("filename")
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename not found in document"
        )

    # Get file extension (if any)
    file_extension = os.path.splitext(filename)[1].lower()

    try:
        # If no extension, use process_create_upload_text
        if not file_extension:
            text = entry.get("text", "")
            mode = entry.get("mode", "plain")
            await process_create_upload_text(
                text=text,
                profile_name=new_folder,
                mode=mode,
                filename=filename,
                current_user=current_user
            )
        else:
            # Handle files with extensions using process_create_upload_file
            file_data = io.BytesIO()
            if entry.get("pdf_file_content"):
                file_data.write(entry["pdf_file_content"])
            elif entry.get("file_content"):
                file_data.write(entry["file_content"])
            else:
                file_data.write(entry["text"].encode())
            file_data.seek(0)
            upload_file = UploadFile(
                filename=filename,
                file=file_data
            )
            # Determine mode based on extension
            if file_extension == '.pdf':
                mode = 'pdf'
            else:
                mode = 'plain'
            await process_create_upload_file(
                file=upload_file,
                profile_name=new_folder,
                mode=mode,
                current_user=current_user
            )

        # Delete from old location
        await delete_content_library_item(unique_id, parent_user)

        return {
            "message": "File moved successfully",
            "profile_name": new_folder
        }
    except Exception as e:
        log.error(f"Error moving file {unique_id} to {new_folder}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to move file: {str(e)}"
        )

async def process_create_upload_file(
        file: UploadFile,
        profile_name: str,
        mode: str,
        current_user: str) -> Dict[str, str]:
    try:

        # Get parent user
        parent_user = await get_parent_user(current_user)
        log.info(f"Processing upload for parent_user: {parent_user}")

        # Validate and create uploads directory
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Generate random filename and get file content
        random_filename = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        log.info(f"File details - name: {file.filename}, extension: {file_extension}")

        # Read and validate file content
        try:
            file_content = await file.read()
            log.info(f"File content read, size: {len(file_content)} bytes")
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file provided")
        except Exception as e:
            log.error(f"Error reading file content: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

        # Parse file content and extract metadata
        try:
            log.info(f"Attempting to parse file content with extension: {file_extension}")
            result = await parse_file_content(file_content, file_extension, random_filename)
            log.info(f"Parse result: {result}")
            if result is None:
                log.error("parse_file_content returned None")
                raise ValueError("File parsing failed - no result returned")

            parsed_content = result.get('parsed_content')
            metadata = result.get('metadata', {})
            log.info(f"Parsed content length: {len(parsed_content) if parsed_content else 'None'}")
            log.info(f"Extracted metadata: {metadata}")

        except ValueError as e:
            log.error(f"ValueError in parse_file_content: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Please upload a PDF, Word, or Excel document. {e}"
            )

        # Create timestamp and enriched metadata
        upload_timestamp = datetime.utcnow().strftime("%d/%m/%Y")
        enriched_metadata = {
            'filename': file.filename,
            'upload_date': upload_timestamp,
            'uploaded_by': current_user,
            'parent_user': parent_user,
            'file_extension': file_extension,
            'profile_name': profile_name,
            **metadata
        }
        log.info(f"Enriched metadata: {enriched_metadata}")

        # Log details before chromadb upload
        log.info(f"Preparing chromadb upload with mode: {mode}")
        log.info(f"Target collection: {profile_name}")

        # Prepare common arguments for text_to_chromadb
        common_args = {
            'text': parsed_content,
            'user': parent_user,
            'collection': profile_name,
            'user_name': parent_user,
            'mode': mode,
            'embedding': embedder,
            'metadata': enriched_metadata,
            'format': "file",
            'unique_id': f"{random_filename}_{file.filename}",
            'file_content': Binary(file_content)  # Store binary content for all file types
        }
        log.info("Common args prepared for chromadb")

        # Upload to specified collection
        log.info("Attempting upload to chromadb...")
        await text_to_chromadb(**common_args)
        log.info("Upload to primary collection successful")

        # Upload to default collection if not already default
        if profile_name != "default":
            common_args['collection'] = "default"
            await text_to_chromadb(**common_args)
            log.info("Upload to default collection successful")

        return {
            "filename": file.filename,
            "profile_name": profile_name,
            "status": "Text successfully uploaded to database",
            "upload_date": upload_timestamp
        }

    except HTTPException as he:
        log.error(f"HTTP error in process_create_upload_file: {str(he)}")
        raise he
    except Exception as e:
        log.error(f"Unexpected error in process_create_upload_file: {str(e)}")
        log.error(f"Error occurred at:", exc_info=True)  # This will log the full traceback
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing the file: {str(e)}"
        )

async def process_show_file_content(
        profile_name: str,
        current_user: str,
        file_name: str) -> str:
    """
    This function queries the document entry collection to find the content
    of a specified file within a folder for the current user.
    It returns the text content of the first matching document.

    Args:
        profile_name (str): The name of the profile (collection) to search in.
        current_user (str): The username of the current authenticated user.
        file_name (str): The name of the file whose content is to be retrieved.

    Returns:
        str: The text content of the first matching document, or an empty string if no document is found.
    """
    # Fetching documents from the database
    parent_user = await get_parent_user(current_user)
    document = (
        await doc_entry_collection.find_one(
            {
                "user": parent_user,
                "collection_name": profile_name,
                "filename": file_name,
            },
            {"_id": 0, "text": 1}
        )
    )

    # Return the text if a document is found, otherwise return an empty string
    return document.get("text", "") if document else ""


async def process_show_file_content_pdf_format(
        profile_name: str,
        current_user: str,
        file_name: str) -> StreamingResponse:
    try:
        log.info(profile_name)
        log.info(current_user)
        log.info(file_name)
        parent_user = await get_parent_user(current_user)
        document = await doc_entry_collection.find_one(
            {
                "user": parent_user,
                "collection_name": profile_name,
                "filename": file_name,
            },
            {"pdf_file_content": 1, "filename": 1, "_id": 0}
        )
        if not document or "pdf_file_content" not in document:
            raise HTTPException(status_code=404, detail="File not found")

        pdf_file_content = document["pdf_file_content"]
        file_stream = io.BytesIO(pdf_file_content)
        file_stream.seek(0)
        return StreamingResponse(
            file_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename={file_name}"}
        )
    except Exception:
        log.error(f"Error in PDF processing: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error processing PDF file")

async def process_update_text(
        id: str,
        text: str,
        profile_name: str,
        mode: str,
        current_user: str) -> UpdateResult:
    parent_user = await get_parent_user(current_user)
    # Ensure that the profile_name is properly formatted
    formattedProfileName = profile_name.strip().replace(" ", "_")

    # Build the update data
    update_data = {
        "text": text,
        "mode": mode,
        "profile_name": formattedProfileName
    }

    # Use id to find the document
    result = await doc_entry_collection.update_one(
        {"unique_id": id, "user": parent_user},
        {"$set": update_data}
    )

    return result


async def process_create_upload_text(
        text: str,
        profile_name: str,
        mode: str,
        filename: Optional[str],
        current_user: str) -> Dict[str, str]:
    parent_user = await get_parent_user(current_user)
    formatted_profile_name = profile_name.strip().replace(" ", "_")
    formatted_filename = filename.strip().replace(" ", "_") if filename else None

    # Create proper metadata dictionary
    metadata = {
        "filename": formatted_filename,
        "upload_date": datetime.now().strftime("%d/%m/%Y"),
        "uploaded_by": current_user
    }

    existing_document = await doc_entry_collection.find_one({
        "profile_name": formatted_profile_name,
        "filename": formatted_filename,
        "user": parent_user
    })
    if existing_document:
        raise HTTPException(
            status_code=409,
            detail="A document with this filename and profile name already exists."
        )

    if mode.lower() == "plain":
        await text_to_chromadb(
            text,
            parent_user,
            profile_name,
            parent_user,
            mode,
            embedder,
            metadata=metadata,  # Pass metadata dictionary instead of just filename
        )
        await text_to_chromadb(
            text,
            parent_user,
            "default",
            parent_user,
            mode,
            embedder,
            metadata=metadata,  # Pass metadata dictionary instead of just filename
        )

    elif mode.lower() == "feedback":
        unique_id = str(uuid.uuid4())
        profile_name = profile_name + "_feedback"
        await text_to_chromadb(
            text,
            parent_user,
            profile_name,
            parent_user,
            mode,
            embedder,
            metadata=unique_id,
        )

        await feedback_collection.insert_one(
            {
                "timestamp": datetime.now(),
                "text": text,
                "current_user": current_user,
                "profile_name": profile_name,
                "metadata": unique_id,
            }
        )


    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    log.info(
        {
            "profile_name": profile_name,
            "status": "Text uploaded to database",
        }
    )

    return {"message": "Text uploaded successfully!"}


async def process_delete_template(
    profile_name: str,
    current_user: str):
    try:
        # if not await is_user_type(current_user, "owner"):
        #     log.info(f"only owners can delete folders")
        #     raise HTTPException(status_code=403, detail="Only owners can delete content library folders")

        parent_user = await get_parent_user(current_user)
        log.info(f"Starting deletion process for profile: {profile_name}, user: {current_user}")

        # Prepare the regex pattern for matching the main folder and its subfolders
        folder_pattern = f"^{re.escape(profile_name)}($|FORWARDSLASH)"
        log.info(folder_pattern)

        # Delete ChromaDB folders
        chroma_deleted = await delete_chroma_folders(folder_pattern, parent_user)

        # Delete MongoDB documents
        mongo_deleted = await delete_mongo_documents(folder_pattern, parent_user)

        total_deleted = chroma_deleted + mongo_deleted

        # If nothing was deleted, it might indicate an issue
        if total_deleted == 0:
            log.warning(f"No items found to delete for folder '{profile_name}'")

        return JSONResponse(content={
            "message": f"Deleted folder '{profile_name}' and all its subfolders.",
            "chroma_folders_deleted": chroma_deleted,
            "mongo_documents_deleted": mongo_deleted,
            "total_deleted": total_deleted
        })
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions as they already have the correct format
        raise http_ex
    except Exception as e:
        # Log the unexpected error
        log.error(f"Error deleting folder '{profile_name}': {str(e)}")
        # Return a 500 error with details that the frontend can display
        raise HTTPException(status_code=500, detail=f"Failed to delete folder: {str(e)}")


async def delete_chroma_folders(folder_pattern: str, user: str) -> int:
    deleted_count = 0
    chroma_db_directory = f"{CHROMA_FOLDER}/{user}"
    log.info(f"Searching for collections in directory: {chroma_db_directory}")
    log.info(f"Using folder pattern: {folder_pattern}")
    try:
        # Get all collections
        collections, _ = await get_collections_and_check_exists(user)

        # Filter collections based on the folder pattern
        matching_collections = [coll_name for coll_name in collections if re.match(folder_pattern, coll_name)]
        log.info(f"Found {len(matching_collections)} matching collections")

        # Initialize Chroma client for deletion
        client = chromadb.PersistentClient(path=chroma_db_directory)

        for collection_name in matching_collections:
            try:
                log.info(f"--> Deleting ChromaDB collection: {collection_name}")
                client.delete_collection(collection_name)
                deleted_count += 1
                log.info(f"Deleted ChromaDB collection: {collection_name}")
            except Exception as e:
                log.info(f"Error deleting ChromaDB collection {collection_name}: {str(e)}")
    except Exception as e:
        log.info(f"Error accessing ChromaDB: {str(e)}")
    log.info(f"Total collections deleted: {deleted_count}")
    return deleted_count


async def delete_mongo_documents(folder_pattern: str, user: str) -> int:
    result = await doc_entry_collection.delete_many({
        "user": user,
        "profile_name": {"$regex": folder_pattern}
    })
    # Log the result
    log.info(f"Deleted {result.deleted_count} documents from the database")
    # Count documents after deletion
    count_after = await doc_entry_collection.count_documents({
        "user": user,
        "profile_name": {"$regex": folder_pattern}
    })
    log.info(f"Found {count_after} documents matching the criteria after deletion")

    return result.deleted_count


async def delete_content_library_item(unique_id: str, user: str):
    """Delete a document entry from MongoDB and its corresponding ChromaDB"""

    log.info("Deleting ChromaDB entry")

    # Delete from MongoDB
    log.info("--> Deleting entry from MongoDB")
    entry = await doc_entry_collection.find_one({"unique_id": unique_id})
    if entry:
        collection_name = entry.get("collection_name")
        await doc_entry_collection.delete_one({"unique_id": unique_id})
        log.info(f"--> Successfully deleted entry from MongoDB with unique_id {unique_id}")
    else:
        log.info(f"--> Entry with unique_id {unique_id} not found in MongoDB")
        return
    await delete_chroma_entry(unique_id, user, collection_name)
    log.info("Deletion process completed")
