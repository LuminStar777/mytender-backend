import json
import zipfile
import os
import io
import logging
from typing import Dict, List
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse

from api_modules.company_library import process_create_upload_folder, process_create_upload_file

log = logging.getLogger(__name__)


async def zip_bulk_upload(
    zip_file: UploadFile, profile_name: str, current_user: str
) -> Dict[str, List[str]]:
    """
    Process a ZIP file upload, creating folders and uploading files while maintaining structure.

    Args:
        zip_file (UploadFile): The uploaded ZIP file
        profile_name (str): Base profile/folder name where files will be uploaded
        current_user (str): Current authenticated user

    Returns:
        Dict[str, List[str]]: Results containing created folders and uploaded files

    Raises:
        HTTPException: If there are issues with the ZIP file or upload process
    """

    created_folders = []
    uploaded_files = []

    try:
        # Read the zip file into memory
        zip_content = await zip_file.read()
        zip_buffer = io.BytesIO(zip_content)

        # Open the zip file
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            # First pass: collect all folders
            folders = set()
            for file_path in zip_ref.namelist():
                # Skip directories themselves and hidden files
                if file_path.endswith('/') or file_path.startswith('__MACOSX'):
                    continue

                # Get directory path if file is in a subdirectory
                dir_path = os.path.dirname(file_path)
                if dir_path:
                    folders.add(dir_path)

            # Second pass: create all folders hierarchically
            for folder in sorted(folders, key=lambda x: x.count('/')):
                folder_parts = folder.split('/')

                # Always skip the first folder level in the ZIP file structure
                if len(folder_parts) > 1:
                    folder_parts = folder_parts[1:]
                elif len(folder_parts) == 1:
                    # If there's only one folder level, we can skip it entirely
                    # since we're extracting into the profile_name folder
                    continue

                current_path = profile_name

                for part in folder_parts:
                    new_path = f"{current_path}FORWARDSLASH{part}" if current_path else part
                    try:
                        result = await process_create_upload_folder(
                            folder_name=part, parent_folder=current_path, current_user=current_user
                        )

                        # Check if result is a JSONResponse (error response)
                        if isinstance(result, JSONResponse):
                            # Extract error details
                            response_body = json.loads(result.body.decode())
                            error_detail = response_body.get('detail', 'Unknown error')

                            raise HTTPException(
                                    status_code=result.status_code,
                                    detail=error_detail
                            )

                        # If we get here, it's a successful dictionary response
                        if result.get("collection_name"):
                            created_folders.append(result["collection_name"])

                    except HTTPException as he:
                        # Only continue for duplicate folder errors
                        if "already exists" in str(he.detail):
                            log.warning(f"Folder already exists, continuing: {str(he.detail)}")
                        else:
                            # Re-raise other exceptions
                            raise he

                    current_path = new_path

            # Third pass: upload all files
            for file_path in zip_ref.namelist():
                # Skip directories and hidden files
                if file_path.endswith('/') or file_path.startswith('__MACOSX'):
                    continue

                # Extract file path parts
                path_parts = file_path.split('/')
                if len(path_parts) > 1:
                    # Skip the root folder
                    file_name = path_parts[-1]
                    # Take only subfolder structure, skipping the first level
                    dir_path = '/'.join(path_parts[1:-1])
                else:
                    # File is at the root of the ZIP
                    file_name = path_parts[0]
                    dir_path = ""

                # Extract file content
                file_content = zip_ref.read(file_path)

                # Determine target folder
                target_folder = profile_name
                if dir_path:
                    target_folder = (
                        f"{profile_name}FORWARDSLASH{dir_path.replace('/', 'FORWARDSLASH')}"
                    )

                # Create UploadFile object
                file_obj = UploadFile(filename=file_name, file=io.BytesIO(file_content))

                # Determine file type/mode
                mode = "plain"
                if file_name.lower().endswith('.pdf'):
                    mode = "pdf"
                elif file_name.lower().endswith('.docx'):
                    mode = "word"
                elif file_name.lower().endswith(('.xlsx', '.xls')):
                    mode = "excel"

                try:
                    await process_create_upload_file(
                        file=file_obj,
                        profile_name=target_folder,
                        mode=mode,
                        current_user=current_user,
                    )
                    uploaded_files.append(f"{target_folder}/{file_name}")
                except Exception as e:
                    log.error(f"Error uploading file {file_path}: {str(e)}")
                    continue

        return {
            "status": "success",
            "created_folders": created_folders,
            "uploaded_files": uploaded_files,
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file provided")
    except HTTPException as he:
        # Re-raise HTTPExceptions (including our custom ones) without modifying them
        # This preserves the original error message like "A folder named 'X' already exists"
        raise he
    except Exception as e:
        log.error(f"Error processing ZIP file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing ZIP file: {str(e)}")
