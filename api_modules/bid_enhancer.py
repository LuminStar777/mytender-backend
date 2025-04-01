import logging
from langchain.prompts import ChatPromptTemplate
from config import llm_writing_plan
from services.chain import (
    generic_run_chain,
    load_prompt_from_file,
    retrieve_bid_library_docs
)
from services.vector_manager import retrieve_content_library_docs

log = logging.getLogger(__name__)


async def process_rewrite_section(section: dict, user_feedback: str, bid_id: str, username: str) -> dict:
    updated_section = section.copy()
    log.info(f"Processing section rewrite: {updated_section.get('heading', '')}")

    # Load planning agent prompt
    planning_prompt_text = load_prompt_from_file("rewrite_section_planning_agent")
    planning_prompt = ChatPromptTemplate.from_template(planning_prompt_text)

    # Get planning information
    planning_data = {
        "original_answer": section.get("answer", ""),
        "user_feedback": user_feedback,
        "section_name": section.get("heading", ""),
        "tender_question": section.get("question", "")
    }

    try:
        # Run planning agent
        planning_result = await generic_run_chain(
            llm_writing_plan,
            planning_data,
            planning_prompt
        )
        log.info(f"Planning agent result: {planning_result}")

        # Parse the planning result text
        result_lines = planning_result.splitlines()
        needs_content_library = False
        needs_tender_document = False
        content_library_queries = []
        tender_queries = []
        rewritten_instruction = ""

        # Extract information from the result text
        for line in result_lines:
            line = line.strip()
            if line.startswith("NEEDS_CONTENT_LIBRARY:"):
                needs_content_library = "Yes" in line
            elif line.startswith("NEEDS_TENDER_DOCUMENT:"):
                needs_tender_document = "Yes" in line
            elif line.startswith("CONTENT_LIBRARY_QUERY:"):
                query = line.replace("CONTENT_LIBRARY_QUERY:", "").strip()
                if query:
                    content_library_queries.append(query)
            elif line.startswith("TENDER_QUERY:"):
                query = line.replace("TENDER_QUERY:", "").strip()
                if query:
                    tender_queries.append(query)
            elif line.startswith("REWRITTEN_INSTRUCTION:"):
                rewritten_instruction = line.replace("REWRITTEN_INSTRUCTION:", "").strip()

            # If we've already found REWRITTEN_INSTRUCTION: and this line doesn't start with any known prefix,
            # it's likely part of a multi-line instruction
            elif rewritten_instruction and not any(line.startswith(prefix) for prefix in [
                "NEEDS_CONTENT_LIBRARY:", "NEEDS_TENDER_DOCUMENT:", 
                "CONTENT_LIBRARY_QUERY:", "TENDER_QUERY:"
            ]):
                rewritten_instruction += "\n" + line

        # Retrieve additional information if needed
        content_library_chunks = ""
        tender_document_chunks = ""

        if needs_content_library and content_library_queries:
            all_content_docs = []
            # Call retrieve_content_library_docs for each query
            for i, query in enumerate(content_library_queries):
                log.info(f"Retrieving content library docs for query {i+1}: {query}")
                content_docs = await retrieve_content_library_docs(username, ["default"], query, 2)  # Reduced k to avoid too many results

                # Add query context to each document
                for doc in content_docs:
                    doc['query_context'] = query

                all_content_docs.extend(content_docs)

            # Format the retrieved content
            if all_content_docs:
                content_chunks = []
                for i, doc in enumerate(all_content_docs):
                    chunk = f"DOCUMENT {i+1} (Query: {doc.get('query_context', 'Unknown')}):\n{doc.get('page_content', '')}"
                    content_chunks.append(chunk)

                content_library_chunks = "\n\n".join(content_chunks)

        if needs_tender_document and tender_queries:
            all_tender_docs = []
            # Call retrieve_bid_library_docs for each query
            for i, query in enumerate(tender_queries):
                log.info(f"Retrieving tender document docs for query {i+1}: {query}")
                tender_docs = await retrieve_bid_library_docs(query, bid_id, username, 2)  # Reduced k to avoid too many results

                # Add query context to each document
                for doc in tender_docs:
                    doc['query_context'] = query

                all_tender_docs.extend(tender_docs)

            # Format the retrieved content
            if all_tender_docs:
                tender_chunks = []
                for i, doc in enumerate(all_tender_docs):
                    chunk = f"DOCUMENT {i+1} (Query: {doc.get('query_context', 'Unknown')}):\n{doc.get('page_content', '')}"
                    tender_chunks.append(chunk)

                tender_document_chunks = "\n\n".join(tender_chunks)

        # Load rewrite prompt
        rewrite_prompt_text = load_prompt_from_file("rewrite_prompt")
        rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt_text)

        # Prepare data for rewrite
        rewrite_data = {
            "original_answer": section.get("answer", ""),
            "user_feedback": user_feedback,
            "section_name": section.get("heading", ""),
            "tender_question": section.get("question", ""),
            "rewritten_instruction": rewritten_instruction,
            "win_themes": section.get("evaluation_criteria", ""),
            "customer_painpoints": section.get("customer_pain_points", ""),
            "competitor_differentiating_factor": section.get("differentiating_factors", ""),
            "content_library_chunks": content_library_chunks,
            "tender_document_chunks": tender_document_chunks
        }

        log.info(rewrite_data)

        # Run rewrite
        rewrite_result = await generic_run_chain(
            llm_writing_plan,
            rewrite_data,
            rewrite_prompt
        )

        # Update section with new answer
        updated_section["answer"] = rewrite_result
        log.info(f"Section rewritten successfully: {updated_section.get('heading', '')}")

    except Exception as e:
        log.error(f"Error in rewrite process: {e}")
        log.exception(e)

    return updated_section
