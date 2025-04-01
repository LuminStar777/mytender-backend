import logging
import asyncio
import time
import uuid
from typing import List, Dict
from bson import ObjectId
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
from config import bids_collection, llm_outline, llm_outline_fallback, llm_writing_plan
from services.chain import (
    assign_insights_to_question,
    extract_choice,
    get_compliance_requirements,
    get_differentiation_opportunities,
    get_section_compliance_requirements,
    get_tender_insights,
    load_prompt_from_file,
)
from utils import has_permission_to_access_bid
log = logging.getLogger(__name__)

async def process_generate_writing_plans_for_section(section: dict) -> dict:
    updated_section = section
    word_amount = section.get("word_count", "250")
    log.info(updated_section)
    sub_question_headers = []

    for subheading in section["subheadings"]:
        sub_question_headers.append(subheading.get("title", ""))

    log.info("sub question headers")
    log.info(sub_question_headers)

     # Early return if no sub-question headers
    if not sub_question_headers:
        return updated_section

    prompt_text = load_prompt_from_file("generate_writing_plans")
    prompt = ChatPromptTemplate.from_template(prompt_text)

    async def run_chain(model):
        chain = prompt | model | StrOutputParser()
        return await chain.ainvoke(
            {
                "sub_question_headers": sub_question_headers,
                "word_amount": word_amount,
                "question": section.get("question", "")
            }
        )

    try:
        result = await run_chain(llm_writing_plan)
        log.info(result)

        # Split result into lines and remove empty lines
        instructions = [line.strip() for line in result.split('\n') if line.strip()]

        # Zip subheadings with instructions
        for subheading, instruction in zip(updated_section["subheadings"], instructions):
            subheading["extra_instructions"] = instruction

    except Exception as e:
        log.warning(f"Error using llm for writing plan: {e}. Skipping writing plan.")
        # Set empty instructions if there's an error
        for subheading in updated_section["subheadings"]:
            subheading["extra_instructions"] = ""

    return updated_section

async def process_regenerate_writingplans_and_subheadings(section: dict) -> dict:
    updated_section = section
    log.info(updated_section)
    sub_question_headers = []
    question = section.get("question", "")
    log.info(question)
    word_amount = section.get("word_count", "")
    log.info(word_amount)

    for subheading in section["subheadings"]:
        sub_question_headers.append(subheading.get("title", ""))
    log.info("sub question headers")
    log.info(sub_question_headers)

    # Early return if no sub-question headers or no question
    if not sub_question_headers or not question:
        return updated_section

    prompt_text = load_prompt_from_file("regenerate_subheadings")
    prompt = ChatPromptTemplate.from_template(prompt_text)

    async def run_chain(model):
        chain = prompt | model | StrOutputParser()
        response = await chain.ainvoke(
            {"sub_question_headers": sub_question_headers, "question": question, "word_amount": word_amount}
        )
        # Parse the response into headers and talking points
        sections = response.split('\n')
        new_subheadings = []

        for section in sections:
            if '|' in section:
                title, talking_points = section.split('|')
                # Clean up the title and talking points
                title = title.strip()
                talking_points = talking_points.strip()

                new_subheadings.append({
                    "title": title,
                    "extra_instructions": talking_points
                })

        return new_subheadings

    try:
        updated_subheadings = await run_chain(llm_writing_plan)
        updated_section["subheadings"] = updated_subheadings

    except Exception as e:
        log.warning(f"Error using llm for writing plan: {e}. Skipping writing plan.")
        # Keep existing subheadings but with empty instructions if there's an error
        for subheading in updated_section["subheadings"]:
            subheading["extra_instructions"] = ""

    return updated_section

async def process_regenerate_single_subheading(section: dict, index: int, user_instructions: str) -> dict:
    updated_section = section.copy()  # Create a copy to avoid modifying the original
    log.info(updated_section)
    question = section.get("question", "")
    log.info(question)
    subheading = section["subheadings"][index]
    sub_question_header = subheading.get("title", "")
    sub_question_writing_plan = subheading.get("extra_instructions", "")

    prompt_text = load_prompt_from_file("regenerate_single_subheading")
    prompt = ChatPromptTemplate.from_template(prompt_text)

    async def run_chain(model):
        chain = prompt | model | StrOutputParser()
        response = await chain.ainvoke(
            {
                "sub_question_header": sub_question_header,
                "sub_question_writing_plan": sub_question_writing_plan,
                "question": question,
                "user_instructions": user_instructions
            }
        )

        if '|' in response:
            title, talking_points = response.split('|')
            return {
                "title": title.strip(),
                "extra_instructions": talking_points.strip()
            }
        return subheading  # Return original if response format is invalid

    try:
        updated_subheading = await run_chain(llm_writing_plan)
        updated_section["subheadings"][index] = updated_subheading
    except Exception as e:
        log.warning(f"Error using llm for writing plan: {e}. Keeping original subheading.")
        # Keep existing subheading but clear instructions if there's an error
        updated_section["subheadings"][index]["extra_instructions"] = ""

    return updated_section

async def process_generate_outline(
    bid_id: str,
    file_names: list[str],
    extra_instructions: str,
    datasets: list[str],
    current_user: str,
    newbid: bool = False,
) -> List[str]:
    try:
        log.info(f"Starting outline generation for bid {bid_id}")
        start_time = time.time()

        # SECTION 1: Initial Setup and Validation
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")
        if not await has_permission_to_access_bid(bid, current_user):
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this bid"
            )

        # SECTION 2: Document Collection
        tender_library = bid.get("tender_library", [])
        if not tender_library:
            raise HTTPException(status_code=404, detail="No documents found in the tender library")

        documents = [
            {
                "text": doc.get("rawtext", ""),
                "unique_id": str(doc.get("_id", "")),
                "meta": doc.get("filename", ""),
            }
            for doc in tender_library
            if doc["filename"] in file_names
        ]

        # SECTION 3: Run compliance requirements and outline generation concurrently
        compliance_task = get_compliance_requirements(bid_id, current_user)
        outline_task = get_outline(documents)

        requirements, outline_text = await asyncio.gather(
            compliance_task,
            outline_task
        )

        log.info(f"Compliance requirements and outline generated in {time.time() - start_time:.2f} seconds")

        outline_sections = [line.strip() for line in outline_text.split('\n') if line.strip()]
        log.info(outline_text)

        if outline_sections:
            # Get compliance requirements text
            compliance_text = requirements.get("requirements", "")

            # Clear existing outline before adding new one
            await bids_collection.update_one({"_id": bid_object_id}, {"$set": {"outline": []}})

            # SECTION 4: Process Individual Sections
            async def process_section(section: str) -> dict:
                parts = [part.strip() for part in section.split('|')]
                section_name = ' | '.join(parts[:2]).strip()
                question = parts[2].strip() if len(parts) > 2 else ""
                subsections = parts[3].strip().split(';') if len(parts) > 3 else []
                #word_count = len(subheading_list) * 250 if len(subheading_list) > 0 else 250
                word_count = parts[4].strip() if len(parts) > 4 else 250
                subheading_list = [
                    {
                        "subheading_id": str(uuid.uuid4()),
                        "title": subsection.strip(),
                        "extra_instructions": "",
                    }
                    for subsection in subsections
                    if subsection.strip()
                ]

                # Extract compliance requirements for this section
                section_compliance = await get_section_compliance_requirements(
                    compliance_text, section_name.rsplit('|', maxsplit=1)[-1].strip(), question
                )
                choice = await extract_choice(section_name)

                return {
                    "section_id": str(uuid.uuid4()),
                    "heading": section_name,
                    "question": question,
                    "word_count": word_count,
                    "choice": choice,
                    "answer": "",
                    "reviewer": "",
                    "status": "Not Started",
                    "weighting": "",
                    "page_limit": "",
                    "writingplan": "",
                    "subsections": len(subheading_list),
                    "subheadings": subheading_list,
                    "compliance_requirements": section_compliance,
                }

            # SECTION 5: Process All Sections Concurrently
            log.info("Processing outline sections concurrently")
            section_objects = await asyncio.gather(
                *[process_section(section) for section in outline_sections]
            )

            # make the initial outline so that the get_insights function will have an outline to modify
            await bids_collection.update_one(
                {"_id": bid_object_id}, {"$set": {"outline": section_objects}}
            )

            log.info("outline generated successfully")
            log.info(len(section_objects))

            # Run all insight gathering functions in parallel
            tender_insights = await asyncio.gather(
                get_tender_insights(bid_id, "generate_summarise_tender", current_user),
                get_tender_insights(bid_id, "generate_evaluation_criteria", current_user),
                get_tender_insights(bid_id, "generate_derive_insights", current_user),
                get_differentiation_opportunities(bid_id, "generate_differentiation_opportunities")
            )

            # Unpack results with default empty strings
            [summarise_tender, evaluation_criteria, pain_points, differentiation_opportunities] = [
                result or "" for result in tender_insights
            ]

            updated_outline=section_objects

            updated_outline = await asyncio.gather(
                *[assign_insights_to_question(section, evaluation_criteria["summary"], "extract_section_evaluation_criteria")
                for section in updated_outline]
            )
            updated_outline = await asyncio.gather(
                *[assign_insights_to_question(section, pain_points["summary"], "extract_section_derive_insights")
                for section in updated_outline]
            )
            updated_outline = await asyncio.gather(
                *[assign_insights_to_question(section, differentiation_opportunities["summary"], "extract_differentiation_factors")
                for section in updated_outline]
            )

            updated_outline = await asyncio.gather(
                 *[process_generate_writing_plans_for_section(section)
                 for section in updated_outline]
            )

            log.info(len(updated_outline))

             # Update all fields in the database before returning the response
            update_data = {
                "outline": updated_outline,
                "tender_summary": summarise_tender.get("requirements"),
                "derive_insights": pain_points.get("requirements"),
                "evaluation_criteria": evaluation_criteria.get("requirements"),
                "differentiation_opportunities": differentiation_opportunities.get("analysis"),
                "win_themes": evaluation_criteria.get("summary"),
                "customer_pain_points": pain_points.get("summary"),
                "differentiating_factors": differentiation_opportunities.get("summary"),

            }

            if newbid:
                update_data["new_bid_completed"] = True

             # Update all fields in the database
            await bids_collection.update_one(
                {"_id": bid_object_id},
                {"$set": update_data}
            )

            # Create response data with the same values
            response_data = update_data

            return response_data

        return []

    except Exception as e:
        log.info(f"Error in process_generate_outline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10))
async def _try_generate_outline(chain, input_text: str) -> str:
    """
    Helper function that attempts to generate an outline with retries.
    Implements exponential backoff with waits of 4, 8, and 10 seconds.
    """
    return await chain.ainvoke({"input_text": input_text})

async def get_outline(documents: List[Dict[str, str]]) -> str:
    """
    Generates outline for a tender bid based on all provided documents.
    Attempts generation with primary model up to 3 times with exponential backoff
    before falling back to secondary model.
    
    Args:
        documents: List of dictionaries containing document text
        
    Returns:
        str: Generated outline or error message
    """
    combined_text = "\n\n".join([doc['text'] for doc in documents])
    prompt_text = load_prompt_from_file("generate_outline")
    prompt = ChatPromptTemplate.from_template(prompt_text)

    chain = prompt | llm_outline | StrOutputParser()

    try:
        # Try primary model with retries
        result = await _try_generate_outline(chain, combined_text)
    except Exception as e:
        log.warning(f"Error using llm_outline after 3 retries: {e}. Falling back to llm_outline_fallback.")

        # Switch to fallback model
        chain = prompt | llm_outline_fallback | StrOutputParser()
        try:
            result = await _try_generate_outline(chain, combined_text)
        except Exception as e:
            log.error(f"Error generating outline with fallback model: {e}")
            result = "Error: Unable to generate outline."

    # Clean up newlines for consistent formatting
    return result.replace('\n\n', '\n').replace('\n', '\n\n')
