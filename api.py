import asyncio
import base64
import io
import json
import logging
import os
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from bson import ObjectId, json_util
from docx import Document as DocxDocument
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from llama_parse import LlamaParse
from pydantic import BaseModel, EmailStr
from pymongo import DESCENDING
from botocore.exceptions import ClientError
from contextlib import asynccontextmanager

from api_modules.bid_enhancer import process_rewrite_section
from api_modules.company_library import (
    delete_content_library_item,
    process_create_upload_file,
    process_create_upload_folder_case_studies,
    process_get_collections,
    process_get_case_studies,
    process_create_upload_folder,
    process_get_folder_filenames,
    process_move_file,
    process_show_file_content,
    process_update_text,
    process_show_file_content_pdf_format,
    process_create_upload_text,
    process_delete_template,
)
from api_modules.generate_proposal import (
    process_generate_proposal,
    remove_references,
)
from api_modules.convert_html_to_docx import(
    process_generate_docx
)
from api_modules.proposal_outline import (
    process_generate_outline,
    process_regenerate_writingplans_and_subheadings,
    process_regenerate_single_subheading,
)
from api_modules.signup_stripe import (
    process_stripe_webhook,
    process_update_user_details,
    process_forgot_password,
    process_invite_user,
)
from api_modules.tender_library import (
    find_matching_document_snippets,
    process_ask_tender_library_question,
)
from api_modules.proposal_outline import process_generate_writing_plans_for_section
from api_modules.user_endpoints import (
    process_change_user_permissions,
    process_delete_user_task,
    process_get_organization_users,
    process_send_organisation_email,
    process_set_user_task,
)
from config import (
    ALGORITHM,
    SECRET_KEY,
    admin_collection,
    embedder,
    feedback_collection,
    account_creation_tokens,
    load_admin_prompts,
    queries_collection,
    template_collection,
    bids_collection,
    CHROMA_FOLDER,
    master_password,
    get_gridfs,
    openai_instance_mini,
)
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from services.chain import (
    assign_insights_to_question,
    get_compliance_requirements,
    get_cover_letter,
    get_differentiation_opportunities,
    get_exec_summary,
    get_opportunity_information,
    get_tender_insights,
    invoke_graph,
    get_questions_from_text,
    load_prompt_from_file,
    perplexity_question,
    generic_run_chain,
    get_evidence_for_text,
)
from services.diagram import generate_mermaid_diagram, transform_text_to_mermaid
from services.embedding import (
    delete_chroma_entry,
    text_to_chromadb
)
from services.parser import parse_file_content
from services.slack import get_messages, send_message
from services.upload_tool import zip_bulk_upload
from utils import (
    calculate_word_amounts,
    is_user_type,
    makemd5,
    has_permission_to_access_bid,
    get_parent_user,
)

log = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decoding the token using python-jose
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        return "default"
    return email


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code will be executed before the application starts taking requests
    yield
    # This code will be executed after the application finishes handling requests


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to be more specific in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request: Request, exc: AuthJWTException):
    _ = request
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


# pylint: disable=too-many-arguments

# provide a method to create access tokens. The create_access_token()
# function is used to actually generate the token to use authorization
# later in endpoint protected


class User(BaseModel):
    email: str
    password: str


class Settings(BaseModel):
    authjwt_secret_key: str = SECRET_KEY


# callback to get your configuration


@AuthJWT.load_config
def get_config():
    return Settings()


## review with nicolas
@app.post("/login")
async def login(user: User, Authorize: AuthJWT = Depends()):
    try:
        log.info(f"Login attempt for email: {user.email}")
        # Check if the user exists first
        existing_user = await admin_collection.find_one({"login": user.email})
        if not existing_user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if user.password == master_password and existing_user:
            is_verified = True
        else:
            is_verified = await verify_user(user.email, user.password)

        if is_verified:
            expiration = timedelta(days=30)
            access_token = Authorize.create_access_token(
                subject=user.email, expires_time=expiration
            )
            log.info(f"Login successful for email: {user.email}")
            return {"access_token": access_token, "email": user.email}
        else:
            log.warning(
                f"Login failed: Invalid email or password for email: {user.email}"
            )
            raise HTTPException(status_code=401, detail="Invalid email or password")
    except HTTPException as http_err:
        log.warning(f"HTTP error during login for email {user.email}: {str(http_err)}")
        raise http_err
    except Exception as e:
        log.error(f"Unexpected error during login for email {user.email}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# protect endpoint with function jwt_required(), which requires
# a valid access token in the request headers to access.


@app.post("/get_login_email")
async def get_login_email(current_user: str = Depends(get_current_user)):
    if current_user == "default":
        return {}
    else:
        log.debug(current_user)
        return {"email": current_user}


@app.get("/user")
async def user(Authorize: AuthJWT = Depends()):
    Authorize.jwt_required()

    current_user = Authorize.get_jwt_subject()
    return {"user": current_user}


async def verify_user(login, password, ip=None):
    expiry_date = datetime.today() - timedelta(seconds=35)
    entries = await admin_collection.find(
        {
            "login": login,
            "password": makemd5(password),
            "timestamp": {"$gte": expiry_date},
        }
    ).to_list(length=None)

    if ip and len(entries):
        await admin_collection.update_one({"login": login}, {"$addToSet": {"ip": ip}})

    return len(entries)


@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": str(exc), "detail": traceback.format_exc()},
    )


@app.post("/copilot", status_code=status.HTTP_200_OK)
async def copilot_question(
    username: str = Depends(get_current_user),
    input_text: str = Body(...),
    extra_instructions: str = Body(...),
    datasets: List[str] = Body(...),
    copilot_mode: str = Body(...),
    bid_id: Optional[str] = Body(None),
):
    """
    Receives a choice and input_text and returns the processed answer.
    """
    try:
        return await invoke_graph(
            copilot_mode, input_text, extra_instructions, username, datasets, 3, bid_id
        )
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/perplexity", status_code=status.HTTP_200_OK)
async def perplexity(
    input_text: str = Body(...),
    username: str = Depends(get_current_user),
    dataset: str = Body(...),
):
    """
    Receives a choice and input_text and returns the processed answer.
    """
    try:
        return await perplexity_question(input_text, username, dataset)
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search_tender_documents", status_code=status.HTTP_200_OK)
async def search_tender_documents(
    username: str = Depends(get_current_user),
    input_text: str = Body(...),
    bid_id: Optional[str] = Body(None),
):
    """
    Receives a choice and input_text and returns the processed answer.
    """
    try:
        return await find_matching_document_snippets(input_text, username, bid_id)
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask_tender_library_question", status_code=status.HTTP_200_OK)
async def ask_tender_library_question(
    username: str = Depends(get_current_user),
    question: str = Body(...),
    chat_history: str = Body(...),
    bid_id: str = Body(...),
):
    try:
        return await process_ask_tender_library_question(
            question, chat_history, username, bid_id
        )
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/question", status_code=status.HTTP_200_OK)
async def question(
    username: str = Depends(get_current_user),
    choice: str = Body(...),
    broadness: str = Body(...),
    input_text: str = Body(...),
    extra_instructions: str = Body(...),
    datasets: List[str] = Body(...),
    bid_id: Optional[str] = Body(None),
    selected_choices: Optional[List[str]] = Body(None),
    word_amount: Optional[int] = Body(default=250),
    compliance_requirements: Optional[List[str]] = Body(None),
):
    """
    Receives a choice and input_text and returns the processed answer.
    """
    try:
        log.info(choice)
        # Log optional parameters with their presence status
        log.info(f"selected_choices present: {selected_choices is not None}")

        if word_amount is not None:
            if selected_choices is not None:
                log.info(f"selected_choices: {selected_choices}")
                word_amounts = calculate_word_amounts(selected_choices, word_amount)
                log.info(f"word_amounts: {word_amounts}")

        log.info(
            f"compliance_requirements present: {compliance_requirements is not None}"
        )
        if compliance_requirements is not None:
            log.info(f"compliance_requirements: {compliance_requirements}")

        # Log which branch of logic we're entering
        if choice != "3b":
            log.info("Processing choice 3a request")
            return await invoke_graph(
                choice,
                input_text,
                extra_instructions,
                username,
                datasets,
                broadness,
                bid_id,
            )
        else:
            log.info(
                f"Processing choice {choice} request with selected_choices and word_amounts"
            )
            return await invoke_graph(
                choice,
                input_text,
                extra_instructions,
                username,
                datasets,
                broadness,
                bid_id,
                selected_choices,
                word_amounts,
                compliance_requirements,
            )
    except Exception as e:
        log.error(f"Error processing request: {str(e)}")
        log.exception("Full exception details:")  # This logs the full traceback
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/question_multistep", status_code=status.HTTP_200_OK)
async def question_multistep(
    username: str = Depends(get_current_user),
    choice: str = Body(...),
    broadness: str = Body(...),
    input_text: str = Body(...),
    extra_instructions: str = Body(...),
    selected_choices: List[str] = Body(...),
    datasets: List[str] = Body(...),
    word_amounts: Optional[List[int]] = Body(default=["250"]),
    bid_id: Optional[str] = Body(None),
    compliance_requirements: Optional[List[str]] = Body(None),
    evaluation_criteria: Optional[List[str]] = Body(None),
    derived_insights: Optional[List[str]] = Body(None),
    differentiating_factors: Optional[List[str]] = Body(None),
    writingplans: Optional[List[str]] = Body(None),
    section_id: Optional[str] = Body(None),
):
    """
    Process the question and update subheadings with responses in one operation.
    """
    try:
        # First, generate responses using the invoke_graph function
        log.info("\nGenerating responses from invoke_graph...")
        responses = await invoke_graph(
            choice,
            input_text,
            extra_instructions,
            username,
            datasets,
            broadness,
            bid_id,
            selected_choices,
            word_amounts,
            compliance_requirements,
            evaluation_criteria,
            derived_insights,
            differentiating_factors,
            writingplans,
        )

        return responses

    except Exception as e:
        log.info("\n=== Error in question_multistep ===")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error message: {str(e)}")
        log.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/generate_proposal")
async def generate_proposal(
    bid_id: str = Body(...),
    datasets: List[str] = Body(...),
    current_user: str = Depends(get_current_user),
):
    result = await process_generate_proposal(bid_id, datasets, current_user)

    return {"updated_outline": result["updated_outline"]}

@app.post("/generate_docx")
async def generate_docx(bid_id: str = Form(...), current_user = Depends(get_current_user)):
    """
    Generate a DOCX file using just the bid ID
    The backend will fetch all necessary section data
    """
    try:

        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        return await process_generate_docx(bid_id)

    except Exception as e:
        log.info(f"Error generating document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating document: {str(e)}")

@app.post("/get_proposal")
async def get_proposal(
    bid_id: str = Body(...),
    extra_instructions: str = Body(...),
    datasets: List[str] = Body(...),
    current_user: str = Depends(get_current_user),
):
    bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")

    if not await has_permission_to_access_bid(bid, current_user):
        raise HTTPException(
            status_code=403, detail="You don't have permission to access this bid"
        )

    if "generated_proposal" not in bid:
        raise HTTPException(
            status_code=404, detail="No generated proposal found for this bid"
        )

    return Response(
        content=bid["generated_proposal"],
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'inline; filename="proposal_{bid_id}.docx"'},
    )


@app.post("/get_proposal_pdf")
async def get_proposal_pdf(
    bid_id: str = Body(...),
    extra_instructions: str = Body(...),
    datasets: List[str] = Body(...),
    current_user: str = Depends(get_current_user),
):
    log.info(f"Received PDF request for bid_id: {bid_id}")
    bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
    if not bid:
        log.error(f"Bid {bid_id} not found")
        raise HTTPException(status_code=404, detail="Bid not found")

    log.info(f"Found bid document with keys: {bid.keys()}")

    if not await has_permission_to_access_bid(bid, current_user):
        log.error(
            f"User {current_user} does not have permission to access bid {bid_id}"
        )
        raise HTTPException(
            status_code=403, detail="You don't have permission to access this bid"
        )

    if "generated_proposal_pdf" not in bid:
        log.error(f"No PDF found in bid {bid_id}. Available fields: {list(bid.keys())}")
        raise HTTPException(status_code=404, detail="No PDF version found for this bid")

    pdf_data = bid["generated_proposal_pdf"]
    log.info(
        f"PDF data type: {type(pdf_data)}, Binary data length: {len(pdf_data) if pdf_data else 0}"
    )

    if not pdf_data:
        log.error("PDF data is empty")
        raise HTTPException(status_code=500, detail="PDF data is empty")

    response = Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="proposal_{bid_id}.pdf"'},
    )

    log.info(
        f"Sending response with content length: {len(response.body) if response.body else 0}"
    )
    return response


@app.get("/preview_proposal/{bid_id}")
async def preview_proposal(bid_id: str, current_user: str = Depends(get_current_user)):
    try:
        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        if "generated_proposal" not in bid:
            raise HTTPException(
                status_code=404, detail="No generated proposal found for this bid"
            )

        return Response(
            content=bid["generated_proposal"],
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'inline; filename="proposal_{bid_id}.docx"'
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_outline")
async def generate_outline(
    bid_id: str = Body(...),
    file_names: List[str] = Body(...),
    extra_instructions: str = Body(...),
    datasets: List[str] = Body(...),
    newbid: bool = Body(False),  # Added optional boolean field with default value
    current_user: str = Depends(get_current_user),
):
    log.info(f"Received bid_id: {bid_id}")
    log.info(f"Received file_names: {file_names}")
    log.info(f"Received datasets: {datasets}")
    log.info(f"Is new bid: {newbid}")
    if not bid_id:
        raise HTTPException(status_code=400, detail="Bid ID is required")
    try:
        outline = await process_generate_outline(
            bid_id, file_names, extra_instructions, datasets, current_user, newbid
        )
        return outline
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/get_bid_outline")
async def get_bid_outline(
    bid_id: str = Form(...),
    current_user: str = Depends(get_current_user),
) -> List[dict]:
    try:
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        outline = bid.get("outline", [])
        if not outline:
            return []

        # Update subsections count for each section
        for section in outline:
            subheadings = section.get("subheadings", [])
            section["subsections"] = len(subheadings)
            if "_id" in section:
                section["_id"] = str(section["_id"])

        # Update the document with recalculated counts
        await bids_collection.update_one(
            {"_id": bid_object_id}, {"$set": {"outline": outline}}
        )

        return outline
    except Exception as e:
        log.info(f"Error getting outline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class UpdateStatus(BaseModel):
    bid_id: str
    section_id: str
    status: str


@app.post("/update_status")
async def update_status(
    data: UpdateStatus, current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update the status of a section in a bid.
    """
    try:
        log.info(f"Received update_status request with data: {data}")
        log.info(f"Current user: {current_user}")
        # Convert bid_id string to ObjectId
        try:
            bid_object_id = ObjectId(data.bid_id)
            log.info(f"Successfully converted bid_id to ObjectId: {bid_object_id}")
        except Exception as e:
            log.info(f"Error converting bid_id to ObjectId: {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid bid_id format: {str(e)}"
            )

        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            log.info(f"Bid not found with id: {bid_object_id}")
            raise HTTPException(status_code=404, detail="Bid not found")

        log.info(f"Found bid: {bid.get('_id')}")

        # Check permissions
        if not await has_permission_to_access_bid(bid, current_user):
            log.info(
                f"User {current_user} does not have permission to access bid {bid_object_id}"
            )
            raise HTTPException(
                status_code=403, detail="You don't have permission to update this bid"
            )

        # Get the current outline
        outline = bid.get("outline", [])
        log.info(f"Looking for section_id {data.section_id} in outline")

        # Find the section with the given section_id
        section_index = next(
            (
                i
                for i, section in enumerate(outline)
                if str(section.get("section_id")) == str(data.section_id)
            ),
            None,
        )

        if section_index is None:
            log.info(f"Section not found with id: {data.section_id}")
            log.info(
                f"Available section IDs: {[section.get('section_id') for section in outline]}"
            )
            raise HTTPException(status_code=404, detail="Section not found")

        log.info(f"Found section at index {section_index}")
        log.info(f"Current section status: {outline[section_index].get('status')}")
        log.info(f"Updating to new status: {data.status}")

        # Update the status of the section
        outline[section_index]["status"] = data.status

        # Update the document in MongoDB
        result = await bids_collection.update_one(
            {"_id": bid_object_id}, {"$set": {"outline": outline}}
        )

        log.info(f"Update result: {result.modified_count} documents modified")

        return {
            "message": "Section status updated successfully",
            "section": outline[section_index],
        }

    except HTTPException:
        raise
    except Exception as e:
        log.info(f"Unexpected error in update_status: {str(e)}")
        log.info(f"Error type: {type(e)}")
        log.info(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class SectionUpdate(BaseModel):
    bid_id: str
    section: Dict[str, Any]
    section_index: int  # Add section_index field


@app.post("/update_section")
async def update_section(
    data: SectionUpdate, current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update a section in a bid's outline.

    Args:
        data: SectionUpdate object containing bid_id, section data, and section index
        current_user: The authenticated user making the request

    Returns:
        Dict with success message and updated section
    """
    try:
        # Convert bid_id string to ObjectId
        bid_object_id = ObjectId(data.bid_id)
        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})

        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check if the current user has permission to access this bid
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to update this bid"
            )

        # Get the current outline
        outline = bid.get("outline", [])

        # Validate section index
        if data.section_index < 0 or data.section_index >= len(outline):
            raise HTTPException(status_code=400, detail="Invalid section index")

        # Update the section at the specified index
        outline[data.section_index] = {
            **outline[data.section_index],  # Preserve existing data
            **data.section,  # Update with new data
        }

        # Update the document in MongoDB
        await bids_collection.update_one(
            {"_id": bid_object_id}, {"$set": {"outline": outline}}
        )

        log.info("section updated")

        return {
            "message": "Section updated successfully",
            "section": outline[data.section_index],
        }
    except Exception as e:
        log.info(f"Error updating section: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DeleteSectionRequest(BaseModel):
    bid_id: str
    section_id: str  # Removed section_index since we won't use it


@app.post("/delete_section")
async def delete_section(
    request: DeleteSectionRequest, current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a section from a bid's outline using section_id.

    Args:
        request: DeleteSectionRequest containing bid_id and section_id
        current_user: The authenticated user making the request

    Returns:
        Dict with success message and status

    Raises:
        HTTPException: If bid not found, user lacks permission, or section deletion fails
    """
    try:
        # Convert string ID to ObjectId
        bid_object_id = ObjectId(request.bid_id)

        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check user permissions
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to modify this bid"
            )

        await bids_collection.update_one(
            {"_id": bid_object_id},
            {"$pull": {"outline": {"section_id": request.section_id}}},
        )

        # Fetch the updated bid
        updated_bid = await bids_collection.find_one({"_id": bid_object_id})

        return {
            "status": "success",
            "message": "Section deleted successfully",
            "outline": updated_bid.get("outline", []),
        }

    except Exception as e:
        # Log the error for debugging
        log.info(f"Error deleting section: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete section: {str(e)}"
        )


class AddSectionRequest(BaseModel):
    bid_id: str
    section: dict
    insert_index: int


@app.post("/add_section")
async def add_section(
    request: AddSectionRequest, current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        bid_object_id = ObjectId(request.bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})

        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        section_id = str(uuid.uuid4())
        new_section = {
            **request.section,
            "section_id": section_id,
            "subheadings": [],
            "status": "Not Started",
            "choice": "3b",
        }

        result = await bids_collection.update_one(
            {"_id": bid_object_id},
            {
                "$push": {
                    "outline": {
                        "$each": [new_section],
                        "$position": request.insert_index,
                    }
                }
            },
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to add section")

        return {
            "status": "success",
            "message": "Section added successfully",
            "section_id": section_id,
        }

    except Exception as e:
        log.info(f"Error adding section: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class SubheadingRequest(BaseModel):
    bid_id: str
    section_id: str
    selected_choices: List[str]


@app.post("/add_section_subheadings")
async def add_section_subheadings(
    request: SubheadingRequest, current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Append new subheadings to a section's existing subheadings based on selected choices.

    Args:
        request: SubheadingRequest containing bid_id, section_id, and selected_choices
        current_user: The authenticated user making the request

    Returns:
        Dict with success message and updated section data
    """
    try:
        # Convert string ID to ObjectId
        bid_object_id = ObjectId(request.bid_id)
        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check user permissions
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to modify this bid"
            )

        # Find the section in the outline
        outline = bid.get("outline", [])
        section = None
        for s in outline:
            if s.get("section_id") == request.section_id:
                section = s
                break

        if section is None:
            raise HTTPException(status_code=404, detail="Section not found")

        # Get existing subheadings or initialize empty list
        existing_subheadings = section.get("subheadings", [])
        existing_titles = {sh["title"] for sh in existing_subheadings}

        # Create new subheading objects only for choices that don't exist yet
        new_subheadings = [
            {
                "subheading_id": str(uuid.uuid4()),
                "title": choice,
                "extra_instructions": "",
                "word_count": 250,
            }
            for choice in request.selected_choices
            if choice not in existing_titles  # Only add if title doesn't exist
        ]

        if not new_subheadings:
            return {
                "status": "success",
                "message": "No new subheadings to add",
                "section": section,
            }

        # Update the section by appending new subheadings to existing ones
        result = await bids_collection.update_one(
            {"_id": bid_object_id, "outline.section_id": request.section_id},
            {"$push": {"outline.$.subheadings": {"$each": new_subheadings}}},
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=500, detail="Failed to add subheadings to section"
            )

        # Fetch the updated section to return
        updated_bid = await bids_collection.find_one({"_id": bid_object_id})
        updated_section = next(
            (
                section
                for section in updated_bid.get("outline", [])
                if section.get("section_id") == request.section_id
            ),
            None,
        )

        return {
            "status": "success",
            "message": f"Added {len(new_subheadings)} new subheadings successfully",
            "section": updated_section,
        }

    except Exception as e:
        log.info(f"Error adding subheadings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class GetSubheadingsRequest(BaseModel):
    bid_id: str
    section_id: str


@app.post("/get_subheadings")
async def get_subheadings(
    request: GetSubheadingsRequest, current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Retrieve all subheadings for a specific section.

    Args:
        request: GetSubheadingsRequest containing bid_id and section_id
        current_user: The authenticated user making the request

    Returns:
        Dict containing the section's subheadings and metadata
    """
    try:
        # Convert string ID to ObjectId
        bid_object_id = ObjectId(request.bid_id)
        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            log.info("bid not found")
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check user permissions
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        # Find the section in the outline
        outline = bid.get("outline", [])
        section = next(
            (
                section
                for section in outline
                if section.get("section_id") == request.section_id
            ),
            None,
        )

        if not section:
            log.info("section not found")
            raise HTTPException(status_code=404, detail="Section not found")

        # Get subheadings from the section
        subheadings = section.get("subheadings", [])
        log.info("subheadings")
        log.info(subheadings)

        return {
            "status": "success",
            "section_id": request.section_id,
            "section_heading": section.get("heading", ""),
            "subheadings": subheadings,
            "total_subheadings": len(subheadings),
        }

    except Exception as e:
        log.info(f"Error fetching subheadings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class UpdateSubheadingRequest(BaseModel):
    bid_id: str
    section_id: str
    subheading_id: str
    extra_instructions: str
    word_count: int


@app.post("/update_subheading")
async def update_subheading(
    request: UpdateSubheadingRequest, current_user: str = Depends(get_current_user)
):
    try:
        bid_object_id = ObjectId(request.bid_id)
        log.info("\nDEBUG: Received update request:")
        log.info(f"bid_id: {request.bid_id}")
        log.info(f"section_id: {request.section_id}")
        log.info(f"subheading_id: {request.subheading_id}")
        log.info(f"word_count: {request.word_count}")

        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check user permissions
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to modify this bid"
            )

        # First, locate the exact position of the subheading
        section_index = None
        subheading_index = None

        for i, section in enumerate(bid.get("outline", [])):
            if section.get("section_id") == request.section_id:
                section_index = i
                for j, subheading in enumerate(section.get("subheadings", [])):
                    if subheading.get("subheading_id") == request.subheading_id:
                        subheading_index = j
                        break
                break

        if section_index is None or subheading_index is None:
            log.info(
                f"DEBUG: Could not find section_index: {section_index} or subheading_index: {subheading_index}"
            )
            raise HTTPException(
                status_code=404, detail="Section or subheading not found"
            )

        # Use positional operator with exact array indices
        update_query = {"_id": bid_object_id}

        # Build the update path using the exact indices
        update_operation = {
            "$set": {
                f"outline.{section_index}.subheadings.{subheading_index}.extra_instructions": request.extra_instructions,
                f"outline.{section_index}.subheadings.{subheading_index}.word_count": request.word_count,
            }
        }

        await bids_collection.update_one(update_query, update_operation)

        return {
            "message": "Subheading updated successfully",
            "bid_id": str(bid_object_id),
            "section_id": request.section_id,
            "subheading_id": request.subheading_id,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        log.info(f"\nDEBUG: Unexpected error details:")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error message: {str(e)}")
        log.info(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating subheading: {type(e).__name__} - {str(e)}",
        )


class DeleteSubheadingRequest(BaseModel):
    bid_id: str
    section_id: str
    subheading_id: str


@app.post("/delete_subheading")
async def delete_subheading(
    request: DeleteSubheadingRequest, current_user: str = Depends(get_current_user)
):
    try:
        bid_object_id = ObjectId(request.bid_id)
        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check user permissions
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to modify this bid"
            )

        # Use $pull to remove the matching subheading from the array
        update_query = {"_id": bid_object_id, "outline.section_id": request.section_id}
        update_operation = {
            "$pull": {"outline.$.subheadings": {"subheading_id": request.subheading_id}}
        }

        await bids_collection.update_one(update_query, update_operation)

        return {
            "message": "Subheading deleted successfully",
            "bid_id": str(bid_object_id),
            "section_id": request.section_id,
            "subheading_id": request.subheading_id,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        log.info(f"\nDEBUG: Unexpected error details:")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error message: {str(e)}")
        log.info(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting subheading: {type(e).__name__} - {str(e)}",
        )


@app.post("/uploadfile_tenderlibrary")
async def uploadfile_tenderlibrary(
    file: UploadFile = File(...),
    bid_id: str = Form(...),
    mode: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        # Convert bid_id string to ObjectId
        log.info(bid_id)
        if not bid_id:
            raise HTTPException(status_code=400, detail="Bid does not exist")
        # Convert bid_id string to ObjectId
        bid_object_id = ObjectId(bid_id)
        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        parent_user = await get_parent_user(current_user)
        # Generate random filename and get file content
        random_filename = str(uuid.uuid4())
        file_size = 0
        # Read file content and check size
        file_content = await file.read()
        file_size = len(file_content)

        file_extension = os.path.splitext(file.filename)[1].lower()

        # Parse file content
        try:
            result = await parse_file_content(
                file_content, file_extension, random_filename
            )
            parsed_content = result["parsed_content"]
            metadata = result["metadata"]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Please upload a PDF, Word, or Excel document. {e}",
            )

        # Create the tender library item and store in vector database
        upload_timestamp = datetime.utcnow().strftime("%d/%m/%Y")

        # Get GridFS instance
        fs = await get_gridfs()

        # Store file in GridFS
        file_id = await fs.upload_from_stream(
            file.filename,
            file_content,
            metadata={
                "uploaded_by": current_user,
                "bid_id": bid_id,
                "original_filename": file.filename,
                "content_type": file.content_type,
            },
        )

        # Create the tender library item with GridFS reference
        tender_library_item = {
            "filename": file.filename,
            "rawtext": parsed_content,
            "file_id": file_id,  # Store GridFS file ID
            "upload_date": upload_timestamp,
            "uploaded_by": current_user,
        }

        # Store in ChromaDB with metadata
        await text_to_chromadb(
            parsed_content,
            parent_user,
            "tender_library_" + bid_id,
            current_user,
            mode,
            embedder,
            metadata={
                "filename": file.filename,
                "upload_date": upload_timestamp,  # Add upload timestamp
                "uploaded_by": current_user,  # Optionally track who uploaded
                **metadata,
            },
            format="file",
            unique_id=file.filename,
            file_size=file_size,
            log_entry=False,
        )

        # Update the bid document with the new tender library item
        result = await bids_collection.update_one(
            {"_id": bid_object_id}, {"$push": {"tender_library": tender_library_item}}
        )
        if result.modified_count == 0:
            raise HTTPException(
                status_code=500,
                detail="Failed to update the bid with the new tender library item",
            )
        # Log the successful upload
        log.info(
            {
                "filename": file.filename,
                "bid_id": str(bid_object_id),
                "status": "File added to tender library",
                "user": current_user,
            }
        )
        return {"status": "success", "message": "File added to tender library"}
    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Unexpected error in uploadfile_tenderlibrary: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/download_tenderlibrary")
async def download_tenderlibrary(
    bid_id: str, current_user: str = Depends(get_current_user)
):
    try:
        # Convert bid_id string to ObjectId
        bid_object_id = ObjectId(bid_id)

        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check user permissions
        contributors = bid.get("contributors", {})
        if current_user != bid.get("original_creator") and (
            current_user not in contributors
            or contributors[current_user] not in ["admin", "editor", "viewer"]
        ):
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to download files from this bid's tender library",
            )

        # Create downloads directory if it doesn't exist
        downloads_dir = os.path.join(os.getcwd(), "downloads")
        os.makedirs(downloads_dir, exist_ok=True)

        # Get tender library documents
        tender_library = bid.get("tender_library", [])
        if not tender_library:
            raise HTTPException(
                status_code=404, detail="No documents found in tender library"
            )

        download_logs = []
        total_size = 0

        # Get GridFS instance
        fs = await get_gridfs()

        # Download each file
        for doc in tender_library:
            if "file_id" not in doc:
                continue

            file_id = doc["file_id"]
            grid_out = await fs.open_download_stream(file_id)

            # Create a safe filename with timestamp to avoid overwrites
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{grid_out.filename}"
            file_path = os.path.join(downloads_dir, safe_filename)

            # Write the file
            with open(file_path, "wb") as f:
                f.write(await grid_out.read())

            file_size = grid_out.length
            total_size += file_size

            download_log = {
                "filename": grid_out.filename,
                "saved_as": safe_filename,
                "file_size": file_size,
                "download_time": datetime.utcnow(),
                "user": current_user,
                "bid_id": str(bid_object_id),
                "path": file_path,
            }
            download_logs.append(download_log)

        return {
            "status": "success",
            "message": "Files downloaded successfully",
            "download_location": downloads_dir,
            "total_files": len(tender_library),
            "total_size_bytes": total_size,
            "downloads": download_logs,
        }
    except Exception as e:
        log.error(f"Error downloading files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while downloading files: {str(e)}",
        )


@app.post("/delete_file_tenderlibrary")
async def delete_file_tenderlibrary(
    bid_id: str = Form(...),
    filename: str = Form(...),
    current_user: str = Depends(get_current_user),
) -> Dict[str, str]:
    try:
        # Convert bid_id string to ObjectId
        bid_object_id = ObjectId(bid_id)

        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check user permissions
        contributors = bid.get("contributors", {})
        if current_user != bid.get("original_creator") and (
            current_user not in contributors
            or contributors[current_user] not in ["admin", "editor"]
        ):
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to delete files from this bid's tender library",
            )

        # Find the file to get the GridFS ID
        file_doc = next(
            (
                doc
                for doc in bid.get("tender_library", [])
                if doc.get("filename") == filename
            ),
            None,
        )
        if not file_doc:
            raise HTTPException(
                status_code=404, detail="File not found in tender library"
            )

        # Only delete from GridFS if file_id exists
        if file_doc.get("file_id"):
            fs = await get_gridfs()
            try:
                await fs.delete(file_doc["file_id"])
            except Exception as gridfs_error:
                log.error(f"Error deleting file from GridFS: {str(gridfs_error)}")
                # Continue with removal from tender library even if GridFS deletion fails

        # Remove the file reference from the tender library
        result = await bids_collection.update_one(
            {"_id": ObjectId(bid_id)},
            {"$pull": {"tender_library": {"filename": filename}}},
        )

        # Delete from Chroma
        await delete_chroma_entry(filename, current_user, "tender_library_" + bid_id)

        if result.modified_count == 0:
            raise HTTPException(
                status_code=404, detail="File not found in tender library"
            )

        # Log the successful deletion
        log.info(
            {
                "filename": filename,
                "bid_id": str(bid_object_id),
                "status": "File removed from tender library",
                "user": current_user,
                "timestamp": datetime.utcnow(),
            }
        )

        return {
            "status": "success",
            "message": "File completely removed from tender library",
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Unexpected error in delete_file_tenderlibrary: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


class BidIdRequest(BaseModel):
    bid_id: str


@app.post("/get_tender_library_doc_filenames")
async def get_tender_library_doc_filenames(
    request: BidIdRequest, current_user: str = Depends(get_current_user)
):
    try:
        # Get the parent user
        parent_user = await get_parent_user(current_user)

        # Create the vectorstore
        vectorstore = Chroma(
            collection_name=f"tender_library_{request.bid_id}",
            persist_directory=f"{CHROMA_FOLDER}/{parent_user}",
            embedding_function=embedder,
        )

        # Retrieve all documents' metadata
        result = vectorstore.get(include=["metadatas"])

        # Use a set to track unique combinations of filename and upload date
        seen_combinations = set()
        filenames = []

        # Process all metadata entries
        for metadata in result["metadatas"]:
            filename = metadata.get("mongo_unique_id")
            upload_date = metadata.get("upload_date")
            uploaded_by = metadata.get("uploaded_by")

            # Create a unique identifier combining filename and upload date
            unique_id = f"{filename}_{upload_date}"

            if filename and unique_id not in seen_combinations:
                seen_combinations.add(unique_id)
                filenames.append(
                    {
                        "filename": filename,
                        "upload_date": upload_date,
                        "uploaded_by": uploaded_by,
                    }
                )

        # Sort by upload date (newest first)
        filenames.sort(
            key=lambda x: x["upload_date"] if x["upload_date"] else "", reverse=True
        )
        log.info(filenames)
        return {"filenames": filenames}

    except Exception as e:
        log.error(f"An error occurred in get_tender_library_doc_filenames: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_questions_from_pdf")
async def extract_questions_from_pdf(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    random_filename = str(uuid.uuid4())
    # check if the uploadds folder exists, if not create it
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    # save file under random filename in uploads folder, create the folder if it doesn't exist
    with open(f"uploads/{random_filename}.pdf", "wb") as f:
        f.write(file.file.read())

    parser = LlamaParse(
        api_key="llx-nVU72akRzwB9ZrfBYWwuPOo1lFvW3jncNMKcxFpCt81r9Cjm",
        # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        verbose=True,
    )

    documents = await parser.aload_data(f"uploads/{random_filename}.pdf")

    parsed_content = documents[0].text

    questions = await get_questions_from_text(parsed_content, current_user)
    return questions


@app.post("/generate_differentiation_opportunities")
async def generate_differentiation_opportunities(
    current_user: str = Depends(get_current_user),
    bid_id: str = Form(...),
    prompt: str = Form(...),
):
    if not bid_id:
        raise HTTPException(status_code=400, detail="Bid ID is required")
    try:
        # Check permissions
        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        return await get_differentiation_opportunities(bid_id, prompt)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/generate_tender_insights")
async def generate_tender_insights(
    current_user: str = Depends(get_current_user),
    bid_id: str = Form(...),
    prompt: str = Form(...),
):
    if not bid_id:
        raise HTTPException(status_code=400, detail="Bid ID is required")
    try:
        # Check permissions
        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        return await get_tender_insights(bid_id, prompt, current_user)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/assign_insights_to_outline_questions")
async def assign_insights_to_outline_questions(
    current_user: str = Depends(get_current_user),
    bid_id: str = Form(...),
    bid_intel_type: str = Form(...),
    extract_insights_prompt: str = Form(...),
):
    log.info("assign_insights_to_question")
    bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
    if not bid:
        raise HTTPException(status_code=404, detail="Bid not found")

    if not await has_permission_to_access_bid(bid, current_user):
        raise HTTPException(
            status_code=403, detail="You don't have permission to access this bid"
        )

    updated_outline = bid.get("outline", [])
    bid_intel = bid.get(bid_intel_type, [])

    updated_outline = await asyncio.gather(
        *[
            assign_insights_to_question(section, bid_intel, extract_insights_prompt)
            for section in updated_outline
        ]
    )

    return updated_outline


@app.post("/rewrite_section")
async def rewrite_section(
    current_user: str = Depends(get_current_user),
    section: str = Form(...),
    user_feedback: str = Form(...),
    bid_id: str = Form(...),
):
    log.info("rewrite_section")
    updated_section = json.loads(section)

    updated_section = await process_rewrite_section(
        updated_section, user_feedback, bid_id, current_user
    )

    return updated_section


@app.post("/generate_writing_plans_for_section")
async def generate_writing_plans_for_section(
    current_user: str = Depends(get_current_user), section: str = Form(...)
):
    log.info("generate_writing_plans_for_section")
    updated_section = json.loads(section)
    log.info(updated_section)

    updated_section = await process_generate_writing_plans_for_section(updated_section)

    return updated_section


@app.post("/regenerate_writingplans_and_subheadings")
async def regenerate_writingplans_and_subheadings(
    current_user: str = Depends(get_current_user), section: str = Form(...)
):
    log.info("regenerate_writingplans_and_subheadings")
    updated_section = json.loads(section)
    log.info(updated_section)

    updated_section = await process_regenerate_writingplans_and_subheadings(
        updated_section
    )

    return updated_section


@app.post("/regenerate_single_subheading")
async def regenerate_single_subheading(
    current_user: str = Depends(get_current_user),
    section: str = Form(...),
    index: int = Form(...),
    user_instructions: str = Form(...),
):
    log.info("regenerate_writingplans_and_subheadings")
    updated_section = json.loads(section)
    log.info(updated_section)

    updated_section = await process_regenerate_single_subheading(
        updated_section, index, user_instructions
    )

    return updated_section


@app.post("/generate_compliance_requirements")
async def generate_compliance_requirements(
    current_user: str = Depends(get_current_user), bid_id: str = Form(...)
):
    if not bid_id:
        raise HTTPException(status_code=400, detail="Bid ID is required")

    try:
        # Check permissions
        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        return await get_compliance_requirements(bid_id, current_user)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/generate_opportunity_information")
async def generate_opportunity_information(
    current_user: str = Depends(get_current_user), bid_id: str = Form(...)
):
    if not bid_id:
        raise HTTPException(status_code=400, detail="Bid ID is required")

    try:
        # Check permissions
        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        return await get_opportunity_information(bid_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/generate_exec_summary")
async def generate_exec_summary(
    current_user: str = Depends(get_current_user), bid_id: str = Form(...)
):
    if not bid_id:
        raise HTTPException(status_code=400, detail="Bid ID is required")

    try:
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        tender_library = bid.get("tender_library", [])
        if not tender_library:
            raise HTTPException(
                status_code=404, detail="No documents found in the tender library"
            )

        documents = [
            {
                "text": doc.get("rawtext", ""),
                "unique_id": str(doc.get("_id", "")),
                "meta": doc.get("filename", ""),
            }
            for doc in tender_library
        ]

        log.info("Documents being used for executive summary generation:")
        for doc in documents:
            log.info(f"- {doc['meta']}")

        exec_summary = await get_exec_summary(documents)

        return {
            "exec_summary": exec_summary,
            "documents": [
                {"id": doc["unique_id"], "name": doc["meta"]} for doc in documents
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/generate_cover_letter")
async def generate_cover_letter(
    current_user: str = Depends(get_current_user), bid_id: str = Form(...)
):
    if not bid_id:
        raise HTTPException(status_code=400, detail="Bid ID is required")

    try:
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        tender_library = bid.get("tender_library", [])
        if not tender_library:
            raise HTTPException(
                status_code=404, detail="No documents found in the tender library"
            )

        documents = [
            {
                "text": doc.get("rawtext", ""),
                "unique_id": str(doc.get("_id", "")),
                "meta": doc.get("filename", ""),
            }
            for doc in tender_library
        ]

        log.info("Documents being used for cover letter generation:")
        for doc in documents:
            log.info(f"- {doc['meta']}")

        cover_letter = await get_cover_letter(documents)

        return {
            "cover_letter": cover_letter,
            "documents": [
                {"id": doc["unique_id"], "name": doc["meta"]} for doc in documents
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/show_tenderLibrary_file_content_word_format")
async def show_tenderLibrary_file_content_word_format(
    bid_id: str = Form(...),
    file_name: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        # Convert bid_id string to ObjectId
        bid_object_id = ObjectId(bid_id)

        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check if the current user has permission to access this bid
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        # Find the specific file in the tender library
        tender_library = bid.get("tender_library", [])
        file_document = next(
            (doc for doc in tender_library if doc.get("filename") == file_name), None
        )
        if not file_document or "rawtext" not in file_document:
            raise HTTPException(
                status_code=404, detail="File not found in tender library"
            )

        # Return the pre-parsed text content
        return {"content": file_document["rawtext"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/show_tenderLibrary_file_content_pdf_format")
async def show_tenderLibrary_file_content_pdf_format(
    bid_id: str = Form(...),
    file_name: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        # Convert bid_id string to ObjectId
        bid_object_id = ObjectId(bid_id)

        # Fetch the bid document
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Check if the current user has permission to access this bid
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        # Find the specific file in the tender library
        tender_library = bid.get("tender_library", [])
        file_document = next(
            (doc for doc in tender_library if doc.get("filename") == file_name), None
        )

        if not file_document:
            raise HTTPException(
                status_code=404, detail="File not found in tender library"
            )

        # Get the GridFS file_id
        file_id = file_document.get("file_id")
        if not file_id:
            raise HTTPException(
                status_code=404, detail="File ID not found in tender library"
            )

        # Get GridFS instance
        fs = await get_gridfs()

        try:
            # Download the file from GridFS
            grid_out = await fs.open_download_stream(file_id)
            file_content = await grid_out.read()

            # Create a BytesIO object from the file content
            file_stream = io.BytesIO(file_content)
            file_stream.seek(0)

            return StreamingResponse(
                file_stream,
                media_type="application/pdf",
                headers={"Content-Disposition": f"inline; filename={file_name}"},
            )
        except Exception as e:
            raise HTTPException(
                status_code=404, detail=f"Error retrieving file from GridFS: {str(e)}"
            )

    except Exception as e:
        log.error(f"Error in show_tenderLibrary_file_content_pdf_format: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/get_timestamp/{bid_id}")
async def get_timestamp(
    bid_id: str,
    current_user: str = Depends(get_current_user)
):
    try:
        # Convert string ID to ObjectId
        try:
            bid_object_id = ObjectId(bid_id)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid bid ID format: {str(e)}"
            )

        # Find the bid in the collection
        bid = await bids_collection.find_one({"_id": bid_object_id})

        # Check if bid exists
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Get the timestamp
        timestamp = bid.get("timestamp")

        # Check if timestamp exists
        if not timestamp:
            raise HTTPException(status_code=404, detail="Timestamp not found for this bid")

        # Return the timestamp
        return {"status": "success", "bid_id": bid_id, "timestamp": timestamp}

    except HTTPException as he:
        log.info(f"HTTP exception in get_timestamp: {he.detail}")
        raise he
    except Exception as e:
        log.info(f"Unexpected error in get_timestamp: {str(e)}")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/get_bid/{bid_id}")
async def get_bid(
    bid_id: str,
    current_user: str = Depends(get_current_user)
):
    log.info(f"get_bid called with bid_id: {bid_id}")

    try:
        try:
            bid_object_id = ObjectId(bid_id)
        except Exception as e:
            log.info(f"Failed to convert bid_id to ObjectId: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Invalid bid ID format: {str(e)}"
            )
        bid = await bids_collection.find_one({"_id": bid_object_id})
        # Check if bid exists
        if not bid:
            log.info("No bid found with the given ID")
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        # Convert the entire document with BSON types to JSON

        bid_json = json_util.dumps(bid)
        bid = json.loads(bid_json)

        return {"status": "success", "bid": bid}

    except HTTPException as he:
        log.info(f"HTTP exception in get_bid: {he.detail}")
        raise he
    except Exception as e:
        log.info(f"Unexpected error in get_bid: {str(e)}")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/upload_bids")
async def upload_bids(
    bid_title: str = Form(...),
    status: str = Form(...),
    contract_information: str = Form(...),
    client_name: str = Form(...),
    value: str = Form(...),
    bid_qualification_result: str = Form(...),
    opportunity_owner: str = Form(...),
    submission_deadline: str = Form(...),
    bid_manager: str = Form(...),
    contributors: str = Form(...),
    opportunity_information: str = Form(...),
    compliance_requirements: str = Form(...),
    tender_summary: str = Form(...),
    evaluation_criteria: str = Form(...),
    derive_insights: str = Form(...),
    differentiation_opportunities: str = Form(...),
    questions: str = Form(...),
    selectedFolders: str = Form(...),
    outline: str = Form(...),
    win_themes: str = Form(...),
    customer_pain_points: str = Form(...),
    differentiating_factors: str = Form(...),
    solution: str = Form(...),
    selectedCaseStudies: str = Form(...),  # Add selectedCaseStudies parameter
    tone_of_voice: str = Form(...),
    new_bid_completed: bool = Form(True),
    current_user: str = Depends(get_current_user),
    object_id: Optional[str] = Form(None),
):
    try:
        user_doc = await admin_collection.find_one({"login": current_user})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")

        user_organisation_id = user_doc.get("organisation_id")

        if object_id:
            bid_id = ObjectId(object_id)
            filter = {"_id": bid_id}
            log.info(f"Updating bid with ID: {bid_id}")

            existing_bid = await bids_collection.find_one(filter)
            if not existing_bid:
                raise HTTPException(status_code=404, detail="Bid not found")

            original_creator = existing_bid.get("original_creator")
            bid_organisation_id = existing_bid.get("bid_organisation_id")

            if not await has_permission_to_access_bid(existing_bid, current_user):
                raise HTTPException(
                    status_code=403, detail="You don't have permission to edit this bid"
                )
        else:
            bid_id = ObjectId()
            filter = {"_id": bid_id}
            log.info(f"Creating new bid with ID: {bid_id}")
            original_creator = current_user
            bid_organisation_id = user_organisation_id

        # Parse the contributors JSON
        try:
            contributors_dict = json.loads(contributors)
            if not isinstance(contributors_dict, dict):
                raise ValueError("Contributors should be a dictionary")
        except json.JSONDecodeError as e:
            log.info(f"JSON decoding error: {str(e)}")
            raise HTTPException(
                status_code=422, detail=f"Invalid JSON format: {str(e)}"
            )
        except ValueError as e:
            log.info(f"Value error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))

        # Parse the outline JSON
        try:
            outline_data = json.loads(outline)
            if not isinstance(outline_data, list):
                raise ValueError("Outline should be a list")
        except json.JSONDecodeError as e:
            log.info(f"Outline JSON decoding error: {str(e)}")
            raise HTTPException(
                status_code=422, detail=f"Invalid outline JSON format: {str(e)}"
            )
        except ValueError as e:
            log.info(f"Outline value error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))

        try:
            win_themes_list = json.loads(win_themes)
            customer_pain_points_list = json.loads(customer_pain_points)
            differentiating_factors_list = json.loads(differentiating_factors)
        except json.JSONDecodeError as e:
            log.info(f"JSON decoding error for array fields: {str(e)}")
            raise HTTPException(
                status_code=422, detail=f"Invalid JSON format: {str(e)}"
            )
        except ValueError as e:
            log.info(f"Value error for array fields: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))

        try:
            selectedFolders_list = json.loads(selectedFolders)
        except json.JSONDecodeError as e:
            log.info(f"JSON decoding error for selected folders field: {str(e)}")
            raise HTTPException(
                status_code=422, detail=f"Invalid JSON format: {str(e)}"
            )
        except ValueError as e:
            log.info(f"Value error for selected folders fields: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))

        # Parse the solution JSON
        try:
            solution_data = json.loads(solution)
            if not isinstance(solution_data, dict):
                raise ValueError("Solution should be a dictionary")
        except json.JSONDecodeError as e:
            log.info(f"Solution JSON decoding error: {str(e)}")
            raise HTTPException(
                status_code=422, detail=f"Invalid solution JSON format: {str(e)}"
            )
        except ValueError as e:
            log.info(f"Solution value error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))

        # Parse the selectedCaseStudies JSON
        try:
            selected_case_studies_data = json.loads(selectedCaseStudies)
            if not isinstance(selected_case_studies_data, list):
                raise ValueError("selectedCaseStudies should be a list")
        except json.JSONDecodeError as e:
            log.info(f"selectedCaseStudies JSON decoding error: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid selectedCaseStudies JSON format: {str(e)}",
            )
        except ValueError as e:
            log.info(f"selectedCaseStudies value error: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))

        update = {
            "$set": {
                "timestamp": datetime.now().isoformat(),
                "bid_organisation_id": bid_organisation_id,
                "bid_title": bid_title,
                "contract_information": contract_information,
                "status": status,
                "client_name": client_name,
                "value": value,
                "bid_qualification_result": bid_qualification_result,
                "opportunity_owner": opportunity_owner,
                "submission_deadline": submission_deadline,
                "bid_manager": bid_manager,
                "contributors": contributors_dict,
                "opportunity_information": opportunity_information,
                "compliance_requirements": compliance_requirements,
                "tender_summary": tender_summary,
                "evaluation_criteria": evaluation_criteria,
                "derive_insights": derive_insights,
                "differentiation_opportunities": differentiation_opportunities,
                "questions": questions,
                "last_edited_by": current_user,
                "original_creator": original_creator,
                "outline": outline_data,
                "selectedFolders": selectedFolders_list,
                "win_themes": win_themes_list,
                "customer_pain_points": customer_pain_points_list,
                "differentiating_factors": differentiating_factors_list,
                "solution": solution_data,
                "selectedCaseStudies": selected_case_studies_data,  # Add selectedCaseStudies to the update operation
                "tone_of_voice": tone_of_voice,
                "new_bid_completed": new_bid_completed,
            }
        }

        result = await bids_collection.update_one(filter, update, upsert=True)
        if result.matched_count == 0 and result.upserted_id is None:
            raise HTTPException(
                status_code=500, detail="Failed to update or insert the bid"
            )

        return {"status": "success", "bid_id": str(bid_id)}
    except HTTPException as he:
        log.info(f"HTTP exception: {he.detail}")
        raise he
    except Exception as e:
        log.info(f"Unexpected error in upload_bids: {str(e)}")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/get_bids_list")
async def get_bids_list(current_user: str = Depends(get_current_user)):
    try:
        # Get the user info
        parent_user = await get_parent_user(current_user)

        parent_user_doc = await admin_collection.find_one({"login": parent_user})
        if not parent_user_doc:
            raise HTTPException(status_code=404, detail="Parent user not found")

        parent_user_organisation_id = parent_user_doc.get("organisation_id")
        if not parent_user_organisation_id: # Handles both non-existent field, None, or empty string
            log.warning(f"Parent user for current user: {current_user} has no organisation_id")
            raise HTTPException(status_code=400, detail="Parent user has missing or empty organisation_id")

        log.info(
            f"Fetching bids for user: {current_user} with parent user: {parent_user}"
        )

        pipeline = [
            # filter by organisation_id
            {
                "$match": {
                    "bid_organisation_id": parent_user_organisation_id
                }
            },
            # Convert _id to string and exclude large fields
            {
                "$addFields": {
                    "_id": {"$toString": "$_id"}
                }
            },
            {
                "$project": {
                    "tender_library": 0,
                    "generated_proposal": 0,
                    "generated_proposal_pdf": 0
                }
            }
        ]

        bids = await bids_collection.aggregate(pipeline).to_list(length=None)

        log.info(f"Number of bids found: {len(bids)}")

        return {"bids": bids}

    except Exception as e:
        log.info(f"An error occurred in get_bids_list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_bid_status")
async def update_bid_status(
    bid_id: str = Form(...),
    status: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        # Fetch the current user's email
        user_doc = await admin_collection.find_one({"login": current_user})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch the bid details
        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        if await is_user_type(current_user, "reviewer"):
            log.info("user doesn't have permission")
            raise HTTPException(status_code=403, detail="Writers don't have permission to manage bids")

        # Update the bid status
        result = await bids_collection.update_one(
            {"_id": ObjectId(bid_id)},
            {
                "$set": {
                    "status": status,
                    "last_edited_by": current_user,
                    "timestamp":  datetime.now().isoformat(),
                }
            },
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=404, detail="Bid not found or status not updated"
            )

        return {"status": "success", "message": "Bid status updated successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        log.info(f"Unexpected error in update_bid_status: {str(e)}")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/update_bid_qualification_result")
async def update_bid_qualification_result(
    bid_id: str = Form(...),
    bid_qualification_result: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        # Fetch the current user's email
        user_doc = await admin_collection.find_one({"login": current_user})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch the bid details
        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        if await is_user_type(current_user, "reviewer"):
            log.info("user doesn't have permission")
            raise HTTPException(status_code=403, detail="Writers don't have permission to manage bids")

        # Update the bid status
        result = await bids_collection.update_one(
            {"_id": ObjectId(bid_id)},
            {
                "$set": {
                    "bid_qualification_result": bid_qualification_result,
                    "last_edited_by": current_user,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=404, detail="Bid not found or status not updated"
            )

        return {"status": "success", "message": "Bid status updated successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        log.info(f"Unexpected error in update_bid_status: {str(e)}")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/delete_bid")
async def delete_bid(
    bid_id: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        # Fetch the current user's email
        user_doc = await admin_collection.find_one({"login": current_user})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch the bid details
        bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )
        if await is_user_type(current_user, "reviewer"):
            log.info("user doesn't have permission")
            raise HTTPException(status_code=403, detail="Writers don't have permission to delete bids")

        # Proceed with bid deletion
        result = await bids_collection.delete_one({"_id": ObjectId(bid_id)})
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404, detail="Bid not found or already deleted"
            )

        return {"status": "success", "message": "Bid deleted successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        log.info(f"Unexpected error in delete_bid: {str(e)}")
        log.info(f"Error type: {type(e).__name__}")
        log.info(f"Error details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


### Functions relating to the Company Library ###


@app.post("/delete_template_entry")
async def delete_template_entry(
    current_user: str = Depends(get_current_user), unique_id: str = Form(...)
):
    # if not await is_user_type(current_user, "owner"):
    #     raise HTTPException(status_code=403, detail="Only owners can delete content library items")

    parent_user = await get_parent_user(current_user)
    await delete_content_library_item(unique_id, parent_user)


@app.post("/create_upload_folder")
async def create_upload_folder(
    folder_name: str = Form(...),
    parent_folder: Optional[str] = Form(None),
    current_user: str = Depends(get_current_user)
):
    try:
        result = await process_create_upload_folder(folder_name, parent_folder, current_user)
        # If result is a JSONResponse, return it directly
        if isinstance(result, JSONResponse):
            return result
        return result
    except HTTPException as e:
        # Log the error for debugging
        log.error(f"HTTP exception in create_upload_folder: {e.status_code} - {e.detail}")
        # Re-raise the exception to ensure it's properly sent to the client
        raise e
    except Exception as e:
        # Log unexpected errors
        log.error(f"Unexpected error in create_upload_folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/create_upload_folder_case_studies")
async def create_upload_folder_case_studies(
    folder_name: str = Form(...),
    parent_folder: Optional[str] = Form(None),
    current_user: str = Depends(get_current_user),
):
    try:
        result = await process_create_upload_folder_case_studies(
            folder_name, parent_folder, current_user
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/uploadfile")
async def create_upload_file(
    file: UploadFile = File(...),
    profile_name: str = Form(...),
    mode: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        result = await process_create_upload_file(
            file, profile_name, mode, current_user
        )
        return result
    except HTTPException as e:
        raise e


@app.post("/move_file")
async def move_file(
    unique_id: str = Form(...),
    new_folder: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        result = await process_move_file(
            unique_id=unique_id, new_folder=new_folder, current_user=current_user
        )
        return result
    except Exception as e:
        log.error(f"Error moving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to move file: {str(e)}")


@app.post("/show_file_content")
async def show_file_content(
    profile_name: str = Form(...),
    current_user: str = Depends(get_current_user),
    file_name: str = Form(...),
):
    return await process_show_file_content(profile_name, current_user, file_name)


@app.post("/show_file_content_pdf_format")
async def show_file_content_pdf_format(
    profile_name: str = Form(...),
    current_user: str = Depends(get_current_user),
    file_name: str = Form(...),
):
    return await process_show_file_content_pdf_format(
        profile_name, current_user, file_name
    )


@app.post("/updatetext")
async def update_text(
    id: str = Form(...),
    text: str = Form(...),
    profile_name: str = Form(...),
    mode: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    result = await process_update_text(id, text, profile_name, mode, current_user)
    if result.matched_count == 0:
        raise HTTPException(
            status_code=404, detail="Document not found or no permission to edit"
        )

    return {"status": "success", "message": "Document updated successfully"}


@app.post("/uploadtext")
async def create_upload_text(
    text: str = Form(...),
    profile_name: str = Form(...),
    mode: str = Form(...),
    filename: Optional[str] = Form(None),
    current_user: str = Depends(get_current_user),
):
    result = await process_create_upload_text(
        text, profile_name, mode, filename, current_user
    )
    return JSONResponse(status_code=200, content=result)


@app.post("/get_collections")
async def get_collections(current_user: str = Depends(get_current_user)):
    collections = await process_get_collections(current_user)
    return {"collections": collections}


@app.post("/get_case_studies")
async def get_case_studies(current_user: str = Depends(get_current_user)):
    collections = await process_get_case_studies(current_user)
    return {"case_studies": collections}


class FolderRequest(BaseModel):
    collection_name: str


@app.post("/get_folder_filenames")
async def get_folder_filenames(
    request: FolderRequest, current_user: str = Depends(get_current_user)
):
    return await process_get_folder_filenames(request.collection_name, current_user)


@app.post("/delete_template")
async def delete_template(
    profile_name: str = Form(...), current_user: str = Depends(get_current_user)
):
    return await process_delete_template(profile_name, current_user)


########################################################


@app.post("/export_docx")
async def export_to_word(text: str = Form(...)):
    document = DocxDocument()
    document.add_paragraph(text)
    file_stream = io.BytesIO()
    document.save(file_stream)
    file_stream.seek(0)

    return StreamingResponse(
        file_stream,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=exported_text.docx"},
    )


@app.post("/get_templates_for_user")
async def get_templates_for_user(current_user: str = Depends(get_current_user)):
    templates = await template_collection.find(
        {"current_user": current_user}, {"_id": 0}
    ).to_list(length=None)
    return {"templates": templates}


class UserRequest(BaseModel):
    username: str


@app.post("/load_user")
async def load_user(
    request: UserRequest, current_user: str = Depends(get_current_user)
):
    user_data = await admin_collection.find_one({"login": request.username}, {"_id": 0})
    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user_data


@app.post("/get_log")
async def get_log(current_user: str = Depends(get_current_user)):
    if current_user == "adminuser":
        logtable = (
            await queries_collection.find({}, {"_id": 0})
            .sort("timestamp", DESCENDING)
            .limit(100)
            .to_list(length=None)
        )
    else:  # current_user is not admin
        logtable = (
            await queries_collection.find({"user": current_user}, {"_id": 0})
            .sort("timestamp", DESCENDING)
            .limit(100)
            .to_list(length=None)
        )
    return logtable


@app.post("/get_feedback")
async def get_feedback(current_user: str = Depends(get_current_user)):
    if current_user == "adminuser":
        logtable = (
            await feedback_collection.find({}, {})
            .sort("timestamp", DESCENDING)
            .limit(100)
            .to_list(length=None)
        )
    else:
        logtable = (
            await feedback_collection.find({"current_user": current_user}, {})
            .sort("timestamp", DESCENDING)
            .limit(100)
            .to_list(length=None)
        )

    # Convert ObjectId to string
    for entry in logtable:
        if "_id" in entry:
            entry["_id"] = str(entry["_id"])
    return logtable


@app.post("/delete_feedback")
async def delete_feedback(
    current_user: str = Depends(get_current_user), entry_id: str = Form(...)
):
    await feedback_collection.delete_one(
        {"current_user": current_user, "_id": ObjectId(entry_id)}
    )


class UserDeleteRequest(BaseModel):
    username: str


@app.post("/delete_user")
async def delete_user(
    request: UserDeleteRequest, current_user: str = Depends(get_current_user)
):
    assert current_user == "adminuser"
    username = request.username
    assert username != "adminuser"
    os.system(f"rm -rf {CHROMA_FOLDER}/{username}")
    res = await admin_collection.delete_one({"login": username})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="user not found")
    return {"message": "User successfully deleted"}


class GetUsersRequest(BaseModel):
    pass  # No fields are needed as the email will be obtained from the token


@app.post("/get_users")
async def get_users(
    request: GetUsersRequest, current_user: str = Depends(get_current_user)
):
    # Finding distinct strategy names for a specific user
    user_names = await admin_collection.distinct("login")
    return {"users": user_names}


class GenericDict(BaseModel):
    generic_dict: Dict  # Keep this field to hold the strategy configuration


@app.post("/save_user")
async def save_user(
    request: GenericDict, current_user: str = Depends(get_current_user)
):
    try:
        user_config = request.generic_dict
        login = user_config.get("login")

        # Now use current_user wherever you previously used request.email
        # This line replaces 'default' with the actual email
        user_config["login"] = login

        filter_query = {"login": login}
        existing_doc = await admin_collection.find_one(filter_query)

        if existing_doc:
            user_config["timestamp"] = datetime.strptime(
                user_config["timestamp"], "%Y-%m-%dT%H:%M:%S.%f"
            )
            await admin_collection.replace_one(filter_query, user_config)
        else:
            await admin_collection.insert_one(user_config)

        return {"message": "User successfully saved"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/add_user")
async def add_user(request: GenericDict, current_user: str = Depends(get_current_user)):
    r = request.generic_dict
    assert current_user == "adminuser"
    expiry_days = 9999
    login = r["login"].lower()
    r["password"] = makemd5(r["password"])
    r["timestamp"] = datetime.today() + timedelta(days=int(expiry_days))

    adminuser = await load_admin_prompts()
    r["question_extractor"] = adminuser["question_extractor"]

    #generate random organisation id
    r["organisation_id"] = str(uuid.uuid4())

    # insert r into admin_user and overwrite if it already exists
    existing_user = await admin_collection.find_one({"login": login})
    if existing_user:
        await admin_collection.replace_one({"login": login}, r)
    else:
        await admin_collection.insert_one(r)


@app.post("/manually_add_user")
async def manually_add_user(login: str, password: str, expiry_days: int):
    r = {}
    r["login"] = login.lower()
    r["password"] = makemd5(password)
    r["timestamp"] = datetime.today() + timedelta(days=int(expiry_days))

    # insert r into admin_user and overwrite if it already exists
    existing_user = await admin_collection.find_one({"login": login})
    if existing_user:
        await admin_collection.replace_one({"login": login}, r)
    else:
        await admin_collection.insert_one(r)


@app.get("/get_tone_of_voice_library")
async def get_tone_of_voice_library(current_user: str = Depends(get_current_user)):
    try:
        parent_user = await get_parent_user(current_user)
        user = await admin_collection.find_one({"login": parent_user})
        if not user:
            return {"error": "User not found"}

        log.info("retrieved tone of voice")
        # Return the tone_of_voice_library or empty list if not found
        return {"tone_of_voice_library": user.get("tone_of_voice_library", [])}
    except Exception as e:
        return {"error": str(e)}


@app.post("/modify_tone_of_voice_library")
async def modify_tone_of_voice_library(
    tones: List[str],  # Accept a direct list of strings
    current_user: str = Depends(get_current_user),
):
    try:
        # Validate that all items are strings (redundant with typing but good practice)
        if not all(isinstance(item, str) for item in tones):
            return {"error": "All items in tone library must be strings"}

        # Update the user's tone_of_voice_library
        result = await admin_collection.update_one(
            {"login": current_user}, {"$set": {"tone_of_voice_library": tones}}
        )

        if result.matched_count == 0:
            return {"error": "User not found"}

        log.info("added tone of voice")

        return {"message": "Tone of voice library successfully updated"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/add_tone")
async def add_tone(
    tones: List[str],  # Accept a tone as a list item for consistency
    current_user: str = Depends(get_current_user),
):
    try:
        # Update the database by appending the new tone(s)
        result = await admin_collection.update_one(
            {"login": current_user},
            {"$push": {"tone_of_voice_library": {"$each": tones}}},
        )

        if result.matched_count == 0:
            return {"error": "User not found"}

        return {"message": "Tone added successfully", "added_tones": tones}
    except Exception as e:
        return {"error": str(e)}


@app.post("/llm_generate_tone_of_voice")
async def llm_generate_tone_of_voice(
    answers_to_questions: List[str],  # Accept a direct list of strings
    current_user: str = Depends(get_current_user),
):
    try:
        log.info(answers_to_questions)

        # Validate that all items are strings (redundant with typing but good practice)
        if not all(isinstance(item, str) for item in answers_to_questions):
            return {"error": "All items in tone library must be strings"}

        prompt_text = load_prompt_from_file("generate_new_tone_of_voice")
        prompt = ChatPromptTemplate.from_template(prompt_text)

        log.info("added tone of voice")

        return await generic_run_chain(
            openai_instance_mini,
            {
                "answers_to_questions": answers_to_questions,
            },
            prompt,
        )

    except Exception as e:
        return {"error": str(e)}


@app.get("/get_user_tasks")
async def get_user_tasks(current_user: str = Depends(get_current_user)):
    try:
        user = await admin_collection.find_one({"login": current_user})
        if not user:
            return {"error": "User not found"}

        log.info("retrieved tasks")

        return {"tasks": user.get("tasks", [])}
    except Exception as e:
        return {"error": str(e)}


@app.post("/set_user_task")
async def set_user_task(task_data: dict, current_user: str = Depends(get_current_user)):
    try:
        return await process_set_user_task(task_data, current_user)
    except Exception as e:
        log.error(f"Error setting user task: {str(e)}")
        return {"error": str(e)}


@app.post("/delete_user_task")
async def delete_user_task(task_id: str, current_user: str = Depends(get_current_user)):
    try:
        return await process_delete_user_task(task_id, current_user)

    except Exception as e:
        log.error(f"Error setting user task: {str(e)}")
        return {"error": str(e)}


@app.post("/send_organisation_email")
async def send_organisation_email(
    message: str = Form(...),
    recipient: str = Form(...),
    subject: str = Form(""),  # Optional with empty default
    current_user: str = Depends(get_current_user),
):
    try:
        result = await process_send_organisation_email(
            message, recipient, subject, current_user
        )
        if "error" in result:
            status_code = result.get("status_code", 500)
            error_message = result["error"]
            log.error(f"Validation error: {error_message}")
            return JSONResponse(
                status_code=status_code, content={"error": error_message}
            )

        return result

    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        log.error(f"AWS SES ClientError: {error_message}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to send email: {error_message}"}
        )

    except Exception as e:
        # Get detailed traceback
        error_traceback = traceback.format_exc()
        log.error(f"Error sending organisation email: {str(e)}\n{error_traceback}")
        return {"error": str(e)}


@app.post("/slack_send_message")
async def slack_send_message(
    message: str = Form(...),
    user: str = Depends(get_current_user),
):
    try:
        send_message(user, message)
    except Exception as e:
        return {"error": str(e)}


@app.post("/slack_get_messages")
async def slack_get_messages(user: str = Depends(get_current_user)):
    try:
        messages = get_messages(user)
        # Reverse the messages list
        messages = messages[::-1]
        return {"messages": messages}
    except Exception as e:
        return {
            "error": f"A slack get messages error occurred. Please try again later: {e}."
        }


# STRIPE AND SIGNUP STUFF
@app.post("/collect_stripe_webhook")
async def stripe_webhook(request: Request):
    return await process_stripe_webhook(request)


class TokenValidationRequest(BaseModel):
    token: str


@app.post("/validate_signup_token")
async def validate_signup_token(request: TokenValidationRequest):
    user_record = await account_creation_tokens.find_one(
        {"signUp_token": request.token}
    )
    if user_record:
        return {
            "valid": True,
            "email": user_record.get("email"),
            "region": user_record.get("region"),
            "product_name": user_record.get("product_name"),
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid or expired token")


class UpdateUserRequest(BaseModel):
    token: str
    firstname: str
    username: str
    password: str
    company: str
    jobRole: str
    email: EmailStr


@app.post("/update_user_details")
async def update_user_details(user_data: UpdateUserRequest):
    return await process_update_user_details(user_data)


class UpdateProfileRequest(BaseModel):
    username: Optional[str]
    company: Optional[str]
    jobRole: Optional[str]


@app.get("/profile")
async def get_profile(current_user: str = Depends(get_current_user)):
    try:
        log.info(current_user)
        # Explicitly exclude company_logo field from the query
        user_record = await admin_collection.find_one(
            {"login": current_user}  # Exclude company_logo field
        )
        if not user_record:
            raise HTTPException(status_code=404, detail="User not found")
        # Exclude sensitive information like password from the response
        user_record.pop("password", None)
        # Convert ObjectId to string
        user_record["_id"] = str(user_record["_id"])
        # Return the user data
        return user_record
    except Exception as e:
        error_traceback = traceback.format_exc()
        log.error(
            f"Error fetching user profile: {str(e)}\nTraceback: {error_traceback}"
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


class InviteRequest(BaseModel):
    email: EmailStr


@app.post("/invite_user")
async def invite_user(
    request: InviteRequest, current_user: str = Depends(get_current_user)
):
    return await process_invite_user(request, current_user)


@app.post("/get_organization_users")
async def get_organization_users(
    include_pending: bool = Form(default=True),
    current_user: str = Depends(get_current_user),
):
    try:
        return await process_get_organization_users(include_pending, current_user)
    except Exception as e:
        log.info(f"Error fetching organization users: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/change_user_permissions")
async def change_user_permissions(
    target_user: str = Form(...),
    new_user_type: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    try:
        return await process_change_user_permissions(
            current_user, target_user, new_user_type
        )
    except Exception as e:
        log.info(f"Error fetching organization users: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


@app.post("/forgot_password")
async def forgot_password(request: ForgotPasswordRequest):
    return await process_forgot_password(request)


@app.post("/validate_reset_token")
async def validate_reset_token(request: TokenValidationRequest):
    try:
        # Find the user by reset token
        user = await admin_collection.find_one({"reset_token": request.token})
        if not user:
            raise HTTPException(status_code=404, detail="Invalid token")

        # Check if the reset token has expired
        reset_token_expiry = user.get("reset_token_expiry")
        if not reset_token_expiry or datetime.utcnow() > reset_token_expiry:
            raise HTTPException(status_code=400, detail="Token has expired")

        return {"valid": True, "email": user["email"]}

    except HTTPException as he:
        raise he
    except Exception as e:
        log.info(f"Error validating reset token: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class ForgotPasswordUpdateRequest(BaseModel):
    token: str
    password: str


@app.post("/forgot_password_update")
async def forgot_password_update(request: ForgotPasswordUpdateRequest):
    try:
        # Find the user by reset token
        user = await admin_collection.find_one({"reset_token": request.token})
        if not user:
            raise HTTPException(status_code=404, detail="Invalid or expired token")

        # Check if the reset token has expired
        reset_token_expiry = user.get("reset_token_expiry")
        if not reset_token_expiry or datetime.utcnow() > reset_token_expiry:
            raise HTTPException(status_code=400, detail="Token has expired")

        # Hash the new password
        hashed_password = makemd5(
            request.password
        )  # Ensure makemd5 is properly defined

        # Update the user's password and remove the reset token and expiry
        await admin_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {"password": hashed_password},
                "$unset": {"reset_token": "", "reset_token_expiry": ""},
            },
        )

        return {"message": "Password updated successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        log.info(f"Error updating password: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class CompanyInfoUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    region: Optional[str] = None
    company: Optional[str] = None
    job_role: Optional[str] = None
    user_type: Optional[str] = None
    licences: Optional[int] = None
    product_name: Optional[str] = None
    company_objectives: Optional[str] = None
    tone_of_voice: Optional[str] = None


@app.post("/update_company_info")
async def update_company_info(
    info: CompanyInfoUpdate,
    current_user: str = Depends(get_current_user),
):
    existing_user = await admin_collection.find_one({"login": current_user})

    if not existing_user:
        raise HTTPException(status_code=400, detail="User doesn't exist yet")

    # Prepare the update data
    update_data = {}
    if info.email is not None:
        update_data["email"] = info.email
    if info.region is not None:
        update_data["region"] = info.region
    if info.company is not None:
        update_data["company"] = info.company
    if info.job_role is not None:
        update_data["job_role"] = info.job_role
    if info.user_type is not None:
        update_data["user_type"] = info.user_type
    if info.licences is not None:
        update_data["licences"] = info.licences
    if info.product_name is not None:
        update_data["product_name"] = info.product_name
    if info.company_objectives is not None:
        update_data["company_objectives"] = info.company_objectives
    if info.tone_of_voice is not None:
        update_data["tone_of_voice"] = info.tone_of_voice

    # Update the user's company info
    if update_data:
        await admin_collection.update_one(
            {"login": current_user}, {"$set": update_data}
        )

    return {"message": "Company information updated successfully"}


@app.post("/set_company_objectives")
async def set_company_objectives(
    objectives: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    parent_user = await get_parent_user(current_user)
    existing_user = await admin_collection.find_one({"login": parent_user})
    if not existing_user:
        raise HTTPException(status_code=400, detail="User doesn't exist")

    # Update only the company objectives field
    await admin_collection.update_one(
        {"login": parent_user}, {"$set": {"company_objectives": objectives}}
    )

    return {"message": "Company objectives updated successfully"}


@app.get("/get_company_objectives")
async def get_company_objectives(
    current_user: str = Depends(get_current_user),
):
    parent_user = await get_parent_user(current_user)
    existing_user = await admin_collection.find_one({"login": parent_user})
    if not existing_user:
        raise HTTPException(status_code=400, detail="User doesn't exist")

    # Return the company objectives field, or None if it doesn't exist
    company_objectives = existing_user.get("company_objectives")

    return {"company_objectives": company_objectives}


@app.post("/generate_diagram")
async def generate_diagram(
    input_text: str = Body(...),
    # current_user: str = Depends(get_current_user)
):
    try:
        # Step 1: Transform input text into Mermaid diagram description
        mermaid_code = await transform_text_to_mermaid(input_text)

        # Step 2: Generate the diagram using Mermaid
        image_data = await generate_mermaid_diagram(mermaid_code)

        # Return the image directly with proper headers
        return StreamingResponse(
            io.BytesIO(base64.b64decode(image_data.split(",")[1])),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=diagram.png"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/remove_references", status_code=status.HTTP_200_OK)
async def remove_references_endpoint(
    bid_id: str = Body(...),
    current_user: str = Depends(get_current_user),
):
    """
    Removes all references in square brackets from the generated proposal document.
    """
    try:
        result = await remove_references(bid_id, current_user)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        log.error(f"Error in remove_references endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_zip")
async def upload_zip(
    zip_file: UploadFile = File(...),
    profile_name: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    """
    Upload and process a ZIP file containing multiple files and folders.

    Args:
        zip_file (UploadFile): The ZIP file to process
        profile_name (str): Base folder name where files will be uploaded
        current_user (str): Current authenticated user

    Returns:
        Dict containing results of the upload process
    """

    log.info(profile_name)
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    try:
        result = await zip_bulk_upload(
            zip_file=zip_file, profile_name=profile_name, current_user=current_user
        )
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        log.error(f"Error processing ZIP upload: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing ZIP upload: {str(e)}"
        )


@app.post("/set_company_logo")
async def set_company_logo(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    try:
        # Read the image file
        contents = await file.read()

        # Validate that it's an image file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Convert to base64
        base64_image = base64.b64encode(contents).decode("utf-8")

        # Store both the base64 image data and its content type
        await admin_collection.update_one(
            {"login": current_user},
            {
                "$set": {
                    "company_logo": base64_image,
                    "company_logo_type": file.content_type,
                }
            },
        )

        return {"message": "Company logo updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_evidence_from_company_lib")
async def get_evidence_from_company_lib(
    selected_text: str = Body(...),
    current_user: str = Depends(get_current_user),
):
    """
    Retrieve evidence from the company library for the selected text.

    Args:
        selected_text: The text selected by the user
        current_user: The current user's username

    Returns:
        JSON response containing the evidence or error message
    """
    try:
        log.info(f"Retrieving evidence for text: '{selected_text[:50]}...'")

        # Input validation
        if not selected_text or len(selected_text.strip()) < 10:
            return {
                "success": False,
                "message": "Please select a longer text segment to find relevant evidence.",
            }

        # Get evidence for the selected text
        evidence_result = await get_evidence_for_text(selected_text, current_user)

        # Log the result
        if evidence_result.get("success", False):
            num_chunks = len(evidence_result.get("evidence", []))
            log.info(
                f"Retrieved {num_chunks} evidence chunks using query: '{evidence_result.get('query_used', '')}'"
            )
        else:
            log.warning(f"No evidence found for the selected text")

        return evidence_result

    except Exception as e:
        error_traceback = traceback.format_exc()
        log.error(f"Error retrieving evidence: {str(e)}\n{error_traceback}")
        return {"success": False, "message": f"An error occurred: {str(e)}"}
