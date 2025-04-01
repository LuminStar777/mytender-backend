"""
This module contains the core functionality for processing and responding to user queries
using a combination of vector stores, language models, and custom processing logic.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from functools import wraps
from typing import List, Dict
from typing import Optional, Any

from bson import ObjectId
from fastapi import HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from config import (
    queries_collection,
    bids_collection,
    RETRIEVE_SUBTOPIC_CHUNKS,
    load_user_config,
    llm_chain_default,
    perplexity,
    llm_tender_insights,
    llm_writing_plan,
    llm_fallback,
    admin_collection,
    llm_post_process,
    llm_post_process_fallback,
    llm_chunk_title,
    llm_compliance,
    llm_section_compliance,
    llm_bid_intel_summary,
    llm_opportunity_information,
    llm_exec_summary,
    llm_cover_letter,
    llm_evidence_rewriting,
    RELEVANCE_THRESHOLD,
)
from services.helper import load_query, post_to_slack, load_prompt
from services.utils import britishize
from services.vector_manager import (
    retrieve_documents_for_state,
    retrieve_docs as vm_retrieve_docs,
    retrieve_bid_library_docs as vm_retrieve_bid_library_docs,
    retrieve_content_library_docs as vm_retrieve_content_library_docs,
    retrieve_evidence_chunks,
)
from utils import get_parent_user

log = logging.getLogger(__name__)
model = llm_chain_default


class GraphState(BaseModel):
    """
    Represents the state of the graph processing workflow.

    Attributes:
        choice (str): The user's choice of processing method.
        input_text (str): The original input text from the user.
        extra_instructions (str): Any additional instructions for processing.
        username (str): The username of the requester.
        datasets (List[str]): The list of datasets to use for retrieval.
        broadness (str): A parameter controlling the breadth of the search.
        selected_choices (Optional[List[str]]): Selected choices for multi-header processing.
        word_amounts (Optional[List[int]]): Word count targets for each section.
        compliance_reqs (Optional[List[str]]): Compliance requirements for each section.
        model (Any): The language model to be used.
        context (Optional[str]): Retrieved context for the query.
        instructions (Optional[str]): Processed instructions.
        question (Optional[str]): The processed question.
        retrieved_docs (Optional[List[Dict[str, str]]]): List of retrieved documents with content and source.
        relevant_docs (Optional[List[Dict[str, str]]]): List of relevant documents with content and source.
        result (Optional[str]): The final result of the processing.
        relevant_prompt (Optional[str]): The relevant prompt template for the chosen processing method.
        post_processed_result (Optional[str]): The post-processed result.
        pre_post_processed_result (Optional[str]): The result before post-processing.
        post_processing_enabled (bool): Flag to enable/disable post-processing.
        custom_instructions (Optional[str]): The custom instructions for the chosen processing method.
        tone_of_voice (Optional[str]): The tone of voice for the chosen processing method.
    """

    choice: str
    input_text: str
    extra_instructions: str
    username: str
    datasets: List[str]
    broadness: Optional[int] = None
    selected_choices: Optional[List[str]] = None
    word_amounts: Optional[List[int]] = None
    compliance_reqs: Optional[List[str]] = None
    evaluation_criteria: Optional[List[str]] = None
    derived_insights: Optional[List[str]] = None
    differentation_factors: Optional[List[str]] = None
    writingplans: Optional[List[str]] = None
    highlighted_documents: Optional[List[str]] = None
    selected_case_studies: Optional[List[str]] = None
    solution: Optional[Dict[str, str]] = None
    model: Any
    context: Optional[str] = None
    instructions: Optional[str] = None
    question: Optional[str] = None
    result: Optional[str] = None
    relevant_prompt: Optional[str] = None
    retrieved_docs: Optional[List[Dict[str, str]]] = None
    relevant_docs: Optional[List[Dict[str, str]]] = None
    company_name: str
    bid_id: Optional[str] = None
    post_processed_result: Optional[str] = None
    pre_post_processed_result: Optional[str] = None
    post_processing_enabled: bool
    custom_instructions: Optional[str] = None
    tone_of_voice: Optional[str] = None


async def invoke_graph(
    choice: str,
    input_text: str,
    extra_instructions: str,
    username: str,
    datasets: List[str],
    broadness: str,
    bid_id: Optional[str] = None,
    selected_choices: Optional[List[str]] = None,
    word_amounts: Optional[List[str]] = None,
    compliance_reqs: Optional[List[str]] = None,  # New parameter
    evaluation_criteria: Optional[List[str]] = None,
    derived_insights: Optional[List[str]] = None,
    differentation_factors: Optional[List[str]] = None,
    writingplans: Optional[List[str]] = None,
    highlighted_documents: Optional[List[str]] = None,
    selected_case_studies: Optional[List[str]] = None,
    solution: Optional[Dict[str, str]] = None,
    post_processing_enabled: bool = True,
) -> str:
    """
    Invokes the graph processing workflow for a given query.

    Args:
        choice (str): The processing method choice.
        input_text (str): The input text to process.
        extra_instructions (str): Any additional instructions.
        username (str): The username of the requester.
        datasets (List[str]): The list of datasets to use for retrieval.
        broadness (str): The broadness parameter for retrieval.
        selected_choices (Optional[List[str]]): Selected choices for multi-header processing.
        word_amounts (Optional[List[str]]): Word count targets for each section.
        compliance_reqs (Optional[List[str]]): Compliance requirements for each section.
        post_processing_enabled (bool): Flag to enable/disable post-processing. Defaults to True.

    Returns:
        str: The final processed result.
    """
    # we need this as members or an organisation use the parent user's folders
    parent_user = await get_parent_user(username)
    user_config = await load_user_config(parent_user)
    company_name = user_config.get('company', '[COMPANY NAME]')

    if bid_id:
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})
        tone_of_voice = bid.get("tone_of_voice", 'Professional')

    else:
        tone_of_voice = 'Professional'

    log.info('TONE OF VOICE:')
    log.info(tone_of_voice)

    model = llm_chain_default
    # Ensure all word amounts are at least 100
    if word_amounts:
        word_amounts = [str(max(100, int(amount))) for amount in word_amounts]

    if not model:
        raise ValueError(f"Model not properly initialized for choice {choice}")

    # Add this condition at the beginning of the function
    if choice == "3a":
        # Process 3a directly without using the graph
        relevant_prompt = load_prompt_from_file(choice)
        prompt = ChatPromptTemplate.from_template(relevant_prompt)
        chain = prompt | model | StrOutputParser()
        result = await chain.ainvoke(
            {
                "question": input_text,
            }
        )
        final_result = post_process_result(result, user_config)
        await log_and_store_result(
            final_result, input_text, extra_instructions, datasets, username, choice, user_config
        )
        return final_result

    # Load the relevant prompt from file
    relevant_prompt = load_prompt_from_file(choice)
    log.info(input_text)

    # If we still don't have a prompt, raise an error
    if not relevant_prompt:
        raise ValueError(f"No valid prompt found for choice '{choice}'")

    workflow = StateGraph(GraphState)

    # Define nodes using RunnablePassthrough
    workflow.add_node("retrieve_documents", RunnablePassthrough() | retrieve_documents)
    workflow.add_node("check_relevance", RunnablePassthrough() | check_relevance)
    workflow.add_node("process_context", RunnablePassthrough() | process_context)
    workflow.add_node("get_instructions", RunnablePassthrough() | get_instructions)
    workflow.add_node("get_question", RunnablePassthrough() | get_question)
    workflow.add_node("process_query", RunnablePassthrough() | process_query)
    workflow.add_node("post_process_3b", RunnablePassthrough() | post_process_3b)

    # Set up the workflow
    workflow.set_entry_point("retrieve_documents")
    workflow.add_edge("retrieve_documents", "check_relevance")
    workflow.add_edge("check_relevance", "process_context")
    workflow.add_edge("process_context", "get_instructions")
    workflow.add_edge("get_instructions", "get_question")
    workflow.add_edge("get_question", "process_query")
    workflow.add_edge("process_query", "post_process_3b")
    workflow.add_edge("post_process_3b", END)

    graph_runnable = workflow.compile()

    initial_state = GraphState(
        choice=choice,
        input_text=input_text,
        extra_instructions=extra_instructions,
        username=parent_user,
        datasets=datasets,
        broadness=broadness,
        selected_choices=selected_choices,
        word_amounts=word_amounts,
        compliance_reqs=compliance_reqs,  # Add to initial state
        evaluation_criteria=evaluation_criteria,
        derived_insights=derived_insights,
        differentation_factors=differentation_factors,
        writingplans=writingplans,
        highlighted_documents=highlighted_documents,
        selected_case_studies=selected_case_studies,
        solution = solution,
        model=model,
        relevant_prompt=relevant_prompt,
        company_name=company_name,
        tone_of_voice=tone_of_voice,  # Add tone_of_voice to initial state
        bid_id=bid_id,
        post_processing_enabled=post_processing_enabled,
        custom_instructions=(choice[1:].replace('_', ' ') if choice.startswith('4') else None),
    )

    result = await graph_runnable.ainvoke(initial_state)
    final_result = post_process_result(result["post_processed_result"], user_config)

    await log_and_store_result(
        final_result, input_text, extra_instructions, datasets, username, choice, user_config
    )

    return final_result


def load_prompt_from_file(choice: str) -> str:
    """
    Loads a prompt from a local text file based on the choice.
    """
    # remove the 1 from copilot prompts
    if choice.startswith('1'):
        choice = choice[1:]

    if choice.startswith('4'):
        # Get the custom prompt file for option 4
        try:
            return load_prompt("custom")
        except Exception as e:
            log.error(f"Error loading custom prompt: {str(e)}")
            raise

    # Use the helper function for all other prompts
    return load_prompt(choice)


async def get_instructions(state: GraphState) -> GraphState:
    """
    Processes and sets the instructions in the GraphState.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        GraphState: The updated state with processed instructions.
    """
    state.instructions = state.extra_instructions
    return state


async def get_question(state: GraphState) -> GraphState:
    """
    Sets the question in the GraphState.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        GraphState: The updated state with the question set.
    """
    state.question = state.input_text
    return state


def process_numbers(context: str, user_config: Dict) -> str:
    """
    Processes numbers in the context based on user configuration.

    Args:
        context (str): The context to process.
        user_config (Dict): The user configuration dictionary.

    Returns:
        str: The processed context with numbers handled according to configuration.
    """
    try:
        if 'numbers_allowed_prefixes' in user_config and user_config['numbers_allowed_prefixes']:
            numbers_allowed_prefixes = user_config['numbers_allowed_prefixes'].split(',')
            context = re.sub(r"\b\d+\b", "[number]", context)
            for prefix in numbers_allowed_prefixes:
                if len(prefix) > 1:
                    context = re.sub(r"\b(" + re.escape(prefix) + r"\d+)\b", r"\1", context)
    except Exception as e:
        log.error(f"Error replacing numbers: {e}")
    return context


def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        start_dt = datetime.now()
        log.info(f"Starting {func.__name__} at {start_dt}")
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            log.info(f"Finished {func.__name__} in {duration:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            log.error(f"Error in {func.__name__} after {duration:.2f} seconds: {str(e)}")
            raise

    return wrapper


@async_timer
async def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieve relevant documents from collections based on the input state.
    Parallelized version with timing metrics.
    """
    state.retrieved_docs = await retrieve_documents_for_state(state)
    return state


@async_timer
async def retrieve_docs(vectorstore, query: str, k: int, parent_user: str) -> List[Dict[str, str]]:
    """
    Retrieve documents from a vectorstore with timing metrics.
    """
    return await vm_retrieve_docs(vectorstore, query, k, parent_user)


async def retrieve_bid_library_docs(
    relevant_query: str, bid_id: str, username: str, k: int
) -> List[Dict[str, str]]:
    return await vm_retrieve_bid_library_docs(relevant_query, bid_id, username, k)


async def retrieve_content_library_docs(
    relevant_query: str, username: str, k: int
) -> List[Dict[str, str]]:
    """
    Retrieve documents from the content library.
    This is a wrapper around the function in vector_manager.py.
    """
    # Call the function from vector_manager with the appropriate parameters
    return await vm_retrieve_content_library_docs(username, ["default"], relevant_query, k)


async def check_relevance(state: GraphState) -> GraphState:
    relevance_prompt = ChatPromptTemplate.from_template(
        load_prompt_from_file("check_for_relevance")
    )

    relevance_chain = relevance_prompt | state.model | StrOutputParser()

    async def check_single_doc(doc):
        result = await relevance_chain.ainvoke({"query": state.input_text, "chunk": doc['content']})
        try:
            relevance_score = float(result.strip())
            if relevance_score >= RELEVANCE_THRESHOLD:
                doc['content'] = f"Relevance value: {relevance_score:.1f}\n\n{doc['content']}"
                return doc
            return None
        except ValueError:
            log.error(f"Invalid relevance score returned: {result}")
            return None

    relevant_docs = await asyncio.gather(*[check_single_doc(doc) for doc in state.retrieved_docs])
    state.relevant_docs = [doc for doc in relevant_docs if doc is not None]
    return state


async def process_context(state: GraphState) -> GraphState:
    context = "\n\n".join(
        [f"[Source: {doc['source']}]\n{doc['content']}" for doc in state.relevant_docs]
    )
    state.context = process_numbers(context, await load_user_config(state.username))
    return state


async def process_query(state: GraphState) -> GraphState:
    """
    Processes the query based on the chosen method.
    """
    prompt = ChatPromptTemplate.from_template(state.relevant_prompt)

    user_config = await load_user_config(state.username)

    tone_of_voice = state.tone_of_voice
    company_objectives = user_config.get('company_objectives', '[COMPANY OBJECTIVES]')

    if state.choice[0] in ["1", "2", "3a", "4"]:
        chain_input = {
            "context": state.context or "",
            "extra_instructions": state.instructions or "",
            "question": state.question or "",
            "company_name": state.company_name,
            "tone_of_voice": tone_of_voice,
            "company_objectives": company_objectives,
            "custom_instructions": state.custom_instructions,
        }
        chain = RunnablePassthrough() | prompt | state.model | StrOutputParser()
        state.result = await chain.ainvoke(chain_input)
    elif state.choice.startswith("3b") and state.post_processing_enabled:
        state.result = await process_multiple_headers(state)
    else:
        raise ValueError("Invalid choice")

    return state


async def check_subtopic_relevance(
    docs: List[Dict[str, str]], question: str, sub_topic: str, model: Any
) -> List[Dict[str, str]]:
    relevance_prompt = ChatPromptTemplate.from_template(
        load_prompt_from_file("check_for_relevance_with_subtopic")
    )

    relevance_chain = relevance_prompt | model | StrOutputParser()

    async def check_single_doc(doc):
        result = await relevance_chain.ainvoke(
            {"question": question, "subtopic": sub_topic, "chunk": doc['content']}
        )
        try:
            relevance_score = float(result.strip())
            if relevance_score >= RELEVANCE_THRESHOLD:
                doc['content'] = f"Relevance value: {relevance_score:.1f}\n\n{doc['content']}"
                return doc
            return None
        except ValueError:
            log.error(f"Invalid relevance score returned: {result}")
            return None

    relevant_docs = await asyncio.gather(*[check_single_doc(doc) for doc in docs])
    return [doc for doc in relevant_docs if doc is not None]


async def process_multiple_headers(state: GraphState) -> str:
    user_config = await load_user_config(state.username)

    async def retrieve_content_library_docs(
        sub_topic: str, k: int = RETRIEVE_SUBTOPIC_CHUNKS
    ) -> List[Dict[str, str]]:
        return await vm_retrieve_content_library_docs(state.username, state.datasets, sub_topic, k)

    async def retrieve_bid_library_docs(
        sub_topic: str, k: int = RETRIEVE_SUBTOPIC_CHUNKS
    ) -> List[Dict[str, str]]:
        if not state.bid_id:
            return []

        full_query = f"{sub_topic} {state.input_text}"
        return await vm_retrieve_bid_library_docs(full_query, state.bid_id, state.username, k)

    async def invoke_chain(
        sub_topic,
        words,
        compliance_req,
        evaluation_crit,
        derived_insight,
        differentiation_factor,
        writingplan,
    ):
        # Retrieve documents from content library and bid library separately
        tone_of_voice = state.tone_of_voice
        company_objectives = user_config.get('company_objectives', '[COMPANY OBJECTIVES]')

        content_library_docs = await retrieve_content_library_docs(sub_topic)
        bid_library_docs = await retrieve_bid_library_docs(sub_topic)
        # Check relevance of content library documents
        relevant_content_docs = await check_subtopic_relevance(
            content_library_docs, state.question, sub_topic, state.model
        )
        # Check relevance of bid library documents
        relevant_bid_docs = await check_subtopic_relevance(
            bid_library_docs, state.question, sub_topic, state.model
        )
        # Create separate contexts without including state.context
        context_content_library = "\n\n".join(
            [f"[Source: {doc['source']}]\n{doc['content']}" for doc in relevant_content_docs]
        )

        context_bid_library = "\n\n".join(
            [f"[Source: {doc['source']}]\n{doc['content']}" for doc in relevant_bid_docs]
        )

        if state.highlighted_documents:
            highlighted_docs_context = "\n\n".join(
                [f"[Source: Highlighted Document]\n{text}" for text in state.highlighted_documents]
            )
            # Combine the existing content library context with highlighted documents
            if highlighted_docs_context:
                context_content_library += "\n\n" + highlighted_docs_context

        selected_case_studies_context = ""

        if state.selected_case_studies and len(state.selected_case_studies) > 0:
            selected_case_studies_context = "\n\n".join(
                [f"[Source: Selected Case Study]\n{text}" for text in state.selected_case_studies]
            )

        chain_input = {
            "case_studies": selected_case_studies_context,
            "context_content_library": context_content_library,
            "context_bid_library": context_bid_library,
            "extra_instructions": state.instructions or "",
            "question": state.question or "",
            "sub_topic": sub_topic,
            "word_amounts": words,
            "compliance_requirements": compliance_req,  # Pass the specific compliance requirements
            "evaluation_criteria": evaluation_crit,
            "derived_insights": derived_insight,
            "differentiation_factors": differentiation_factor,
            "writingplan": writingplan,
            "except_sub_topics": ",".join(
                [choice for choice in (state.selected_choices or []) if choice != sub_topic]
            ),
            "company_name": state.company_name,
            "tone_of_voice": tone_of_voice,
            "business_usp": company_objectives,
        }

        if state.solution:
            chain_input["solution"] = state.solution

        prompt = ChatPromptTemplate.from_template(state.relevant_prompt)
        chain = RunnablePassthrough() | prompt | state.model | StrOutputParser()
        res = await chain.ainvoke(chain_input)
        return f"{sub_topic}:\n\n{res}"

    async def validate_state_requirements(state):
        """
        Validates that all required state attributes are present.
        Returns True if all requirements are met, False otherwise.
        """
        required_attributes = [
            state.selected_choices,
            state.word_amounts,
            state.compliance_reqs,
            state.evaluation_criteria,
            state.derived_insights,
            state.differentation_factors,
            state.writingplans,
        ]
        return all(required_attributes)

    if await validate_state_requirements(state):
        results = await asyncio.gather(
            *[
                invoke_chain(
                    sub_topic,
                    words,
                    compliance_req,
                    evaluation_crit,
                    derived_insght,
                    differentiation_factor,
                    writingplan,
                )
                for sub_topic, words, compliance_req, evaluation_crit, derived_insght, differentiation_factor, writingplan in zip(
                    state.selected_choices,
                    state.word_amounts,
                    state.compliance_reqs,
                    state.evaluation_criteria,
                    state.derived_insights,
                    state.differentation_factors,
                    state.writingplans,
                )
            ]
        )
        return "\n\n".join(results)
    else:
        return "No sub-topics, word amounts, or compliance requirements provided"


def post_process_result(result: str, user_config: Dict) -> str:
    """
    Post-processes the result based on user configuration.

    Args:
        result (str): The result to process.
        user_config (Dict): The user configuration dictionary.

    Returns:
        str: The post-processed result.
    """
    if 'forbidden' in user_config:
        forbidden_words = user_config['forbidden'].split(',')
        for word in forbidden_words:
            if len(word) > 1:
                result = re.sub(r"\b" + word + r"\b", r"[ ]", result)
    return britishize(result)


async def log_and_store_result(
    result: str,
    input_text: str,
    extra_instructions: str,
    datasets: str,
    username: str,
    choice: str,
    user_config: Dict,
) -> None:
    """
    Logs and stores the result of the processing.

    Args:
        result (str): The final result.
        input_text (str): The original input text.
        extra_instructions (str): Any additional instructions.
        datasets (str): The datasets used.
        username (str): The username of the requester.
        choice (str): The processing method choice.
        user_config (Dict): The user configuration dictionary.
    """
    log.info("\n===================== OUTPUT ANSWER: \n" + result[:20] + "...")
    await queries_collection.insert_one(
        {
            "timestamp": datetime.now(),
            "result": result,
            "extra_instructions": extra_instructions,
            "input_text": input_text,
            "datasets": datasets,
            "user": username,
            "choice": choice,
            "template": user_config.get(f"prompt{choice}", choice[1:] if choice[0] == "1" else ""),
        }
    )
    await post_to_slack(
        f"User {username} asked (choice: {choice}, datasets: {datasets}): \n {input_text}"
    )
    await post_to_slack(f"Response: \n {result}")


async def perplexity_question(
    input_text: str,
    username: str,
    dataset: str,
) -> str:
    """
    Processes a question using the perplexity model.

    Args:
        input_text (str): The input text to process.
        username (str): The username of the requester.
        dataset (str): The dataset to use.

    Returns:
        str: The processed response from the perplexity model.
    """

    system = "You are a helpful assistant."
    human = "{input}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | perplexity | StrOutputParser()

    try:
        response = await chain.ainvoke(input_text)
    except Exception as e:
        log.error(f"Error invoking chain: {e}")
        response = f"Error: {e}"

    return response


async def get_questions_from_text(input_text: str, username: str) -> List[str]:
    """
    Extracts questions from the given input text.

    Args:
        input_text (str): The input text to extract questions from.
        username (str): The username of the requester.

    Returns:
        List[str]: A list of extracted questions.
    """

    async def get_question(query):
        return input_text

    user_config = await load_user_config(username)
    prompt_text = user_config["question_extractor"]

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = (
        {
            "question": get_question,
        }
        | prompt
        | model
        | StrOutputParser()
    )

    try:
        result = await chain.ainvoke(input_text)
        result = result.split(',')
    except Exception as e:
        log.error(f"Error invoking chain: {e}")
        result = []

    return result


async def get_differentiation_opportunities(bid_id: str, prompt: str) -> Dict[str, Any]:
    """
    Generates differentiation opportunities by analyzing competitors and company USPs using Perplexity.

    Args:
        bid_id (str): The ID of the bid
        prompt (str): The prompt type to use (opp1 or opp2)

    Returns:
        Dict[str, Any]: Dictionary containing analysis and document info
    """
    try:
        # Load bid and user data
        log.info(
            f"Getting differentiation opportunities for bid {bid_id} with prompt {prompt[0:50]}..."
        )
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})

        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Get user profile data
        user = await admin_collection.find_one({"login": bid.get("original_creator")})
        if not user:
            raise HTTPException(status_code=404, detail="User profile not found")

        # Get tender summary - use first paragraph
        tender_summary = bid.get("tender_summary", "")
        if tender_summary:
            tender_summary = tender_summary.split('\n\n')[0]

        # Get client name
        client_name = bid.get("client_name", "")

        # Get company profile and USPs
        company_profile = user.get("company_profile", "")
        usps = user.get("company_objectives", "")
        company_name = user.get("company_name", "")

        # Load both prompts
        prompt_text1 = load_prompt_from_file("generate_differentiation_opp1")
        prompt_text2 = load_prompt_from_file("generate_differentiation_opp2")

        # Create prompt templates
        prompt1 = ChatPromptTemplate.from_messages(
            [("system", "You are a business analyst."), ("human", prompt_text1)]
        )

        prompt2 = ChatPromptTemplate.from_messages(
            [("system", "You are a strategic business consultant."), ("human", prompt_text2)]
        )

        async def run_chain(prompt_template, input_data):
            chain = prompt_template | perplexity | StrOutputParser()
            return await chain.ainvoke(input_data)

        # First run - competitor analysis
        competitors_analysis = await run_chain(
            prompt1,
            {
                "tender_summary": tender_summary,
                "company_profile": company_profile,
                "client_name": client_name,
            },
        )

        # Save competitors analysis to bid document
        await bids_collection.update_one(
            {"_id": bid_object_id}, {"$set": {"competitors_analysis": competitors_analysis}}
        )

        # Second run - differentiation opportunities
        differentiation = await run_chain(
            prompt2,
            {
                "client_name": client_name,
                "tender_summary": tender_summary,
                "usps": usps,
                "competitors_analysis": competitors_analysis,
                "company_name": company_name,
            },
        )

        async def generate_bid_intel_summary(model, text, chain_prompt):
            chain = chain_prompt | model | StrOutputParser()
            response = await chain.ainvoke({"input_text": text})
            points = [
                point.strip()
                for point in response.split(',')
                if point.strip() and not point.isspace()
            ]
            return points

        summary_prompt_text = load_prompt_from_file("differentiation_opportunities_summary")
        summary_prompt = ChatPromptTemplate.from_template(summary_prompt_text)
        bid_intel_summary = await generate_bid_intel_summary(
            llm_bid_intel_summary, differentiation, summary_prompt
        )
        # Save differentiation analysis to bid document
        await bids_collection.update_one(
            {"_id": bid_object_id}, {"$set": {"differentiation_opportunities": differentiation}}
        )

        return {
            "analysis": differentiation,
            "tender_summary": tender_summary,
            "client_name": client_name,
            "competitors_analysis": competitors_analysis,
            "summary": bid_intel_summary,
        }

    except Exception as e:
        log.error(f"Error in get_differentiation_opportunities: {str(e)}")
        log.error(f"Error type: {type(e).__name__}")
        log.error(f"Error details: {e.args}")
        log.error("Traceback: ", exc_info=True)
        raise


async def get_tender_insights(bid_id: str, prompt_type: str, username: str) -> Dict[str, Any]:
    """
    Generates requirements from tender documents and saves to MongoDB.
    Args:
        bid_id (str): The ID of the bid
        prompt_type (str): Type of prompt to use
        username (str): Username making the request
    Returns:
        Dict[str, Any]: Dictionary containing requirements and document info
    """
    log.info(f"Getting tender insights for bid {bid_id} with prompt {prompt_type[0:50]}...")

    try:
        # Initialize variables and load bid
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

        # Configure prompt settings
        should_generate_bidintel = prompt_type in [
            "generate_derive_insights",
            "generate_evaluation_criteria",
        ]
        relevant_query = load_query("summarise_tender_query")
        summary_prompt = None

        # Load appropriate prompts based on type
        if should_generate_bidintel:
            if prompt_type == "generate_derive_insights":
                summary_prompt = ChatPromptTemplate.from_template(
                    load_prompt_from_file("derive_insights_summary")
                )
                relevant_query = load_query("derive_insights_query")
            elif prompt_type == "generate_evaluation_criteria":
                summary_prompt = ChatPromptTemplate.from_template(
                    load_prompt_from_file("evaluation_criteria_summary")
                )
                relevant_query = load_query("evaluate_tender_query")

        main_prompt = ChatPromptTemplate.from_template(load_prompt_from_file(prompt_type))

        async def run_chain(model, text: str, chain_prompt: ChatPromptTemplate) -> Any:
            chain = chain_prompt | model | StrOutputParser()
            response = await chain.ainvoke({"input_text": text})
            if should_generate_bidintel and chain_prompt == summary_prompt:
                return [point.strip() for point in response.split(',') if point.strip()]
            return response

        # Try getting content from tender library first
        result = summary = None
        try:
            tender_library = bid.get("tender_library", [])
            if not tender_library:
                raise HTTPException(
                    status_code=404, detail="No documents found in the tender library"
                )

            log.info("Using documents from tender library:")
            combined_text = "\n\n".join(doc.get("rawtext", "") for doc in tender_library)

            result = await run_chain(llm_tender_insights, combined_text, main_prompt)
            if result and should_generate_bidintel:
                summary = await run_chain(llm_tender_insights, result, summary_prompt)

        except Exception as e:
            # Fall back to using retrieved chunks
            log.error(f"Error using tender library: {e}. Falling back to retrieval.")
            try:
                bid_library_docs = await retrieve_bid_library_docs(
                    relevant_query, bid_id, username, 10
                )

                log.info("Using documents from retrieval:")
                context_bid_library = "\n\n".join(
                    f"[Source: {doc['source']}]\n{doc['content']}" for doc in bid_library_docs
                )

                result = await run_chain(llm_tender_insights, context_bid_library, main_prompt)
                if result and should_generate_bidintel:
                    summary = await run_chain(llm_tender_insights, result, summary_prompt)

            except Exception as e:
                log.error(f"Error in retrieval fallback: {e}")
                result = "Error: Unable to generate."
                summary = (
                    ["Error: Unable to generate summary."] if should_generate_bidintel else None
                )

        # Build response
        response = {"requirements": result or "Error: No content generated."}
        if should_generate_bidintel:
            response["summary"] = summary or ["Error: No bid intel generated."]

        return response

    except Exception as e:
        log.error(
            f"Error in get_tender_insights: {str(e)} | Type: {type(e).__name__} | Details: {e.args}",
            exc_info=True,
        )
        raise


# assigns the relevant information from the bid intel content extracted to specific sections in the outline
async def assign_insights_to_question(
    section: dict, summary: List, extract_insights_prompt: str
) -> dict:
    """
    Updates a section with relevant insights.
    Args:
        section (dict): The section object from the outline
        summary (List): The list of insights from the bid intel page
        extract_insights_prompt (str): The type of insight prompt to use
    Returns:
        dict: Section with new insights field
    """
    # Get insights for this section
    section_insight = await get_section_insight(section, summary, extract_insights_prompt)

    # Update the relevant field based on the prompt type
    if extract_insights_prompt == "extract_section_derive_insights":
        section["relevant_derived_insights"] = section_insight
    elif extract_insights_prompt == "extract_section_evaluation_criteria":
        section["relevant_evaluation_criteria"] = section_insight
    elif extract_insights_prompt == "extract_differentiation_factors":
        section["relevant_differentiation_factors"] = section_insight
    return section


async def get_compliance_requirements(bid_id: str, username: str) -> Dict[str, Any]:
    """
    Generates compliance requirements from tender documents and saves to MongoDB.

    Args:
        bid_id (str): The ID of the bid

    Returns:
        Dict[str, Any]: Dictionary containing requirements and document info
    """
    try:
        # Load bid documents
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

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
        ]

        log.info("Documents being used for compliance requirements generation:")
        for doc in documents:
            log.debug(f"- {doc['meta']}")

        # Process documents
        combined_text = "\n\n".join([doc['text'] for doc in documents])
        prompt_text = load_prompt_from_file("generate_compliance_requirements")
        prompt = ChatPromptTemplate.from_template(prompt_text)

        async def run_chain(model, context):
            chain = prompt | model | StrOutputParser()
            return await chain.ainvoke({"input_text": context})

        try:
            result = await run_chain(llm_compliance, combined_text)
            log.info(result)

        except Exception as e:
            log.warning(
                f"Error using llm_compliance for compliance requirements: {e}. Falling back to chunking retrieval"
            )
            try:
                relevant_query = load_query("get_compliance_requirements_chunking_query")
                bid_library_docs = await retrieve_bid_library_docs(
                    relevant_query, bid_id, username, 15
                )

                log.info("Using documents from retrieval:")
                context_bid_library = "\n\n".join(
                    f"[Source: {doc['source']}]\n{doc['content']}" for doc in bid_library_docs
                )

                result = await run_chain(llm_compliance, context_bid_library)
                log.info(result)

            except Exception as e:
                log.error(f"Error generating compliance requirements with fallback chunking: {e}")
                result = "Error: Unable to generate compliance requirements."

        # Save to MongoDB
        await bids_collection.update_one(
            {"_id": bid_object_id}, {"$set": {"compliance_requirements": result}}
        )

        return {
            "requirements": result,
            "documents": [{"id": doc['unique_id'], "name": doc['meta']} for doc in documents],
        }

    except Exception as e:
        log.error(f"Error in get_compliance_requirements: {str(e)}")
        raise


async def get_opportunity_information(bid_id: str) -> Dict[str, Any]:
    """
    Generates opportunity information from tender documents and saves to MongoDB.

    Args:
        bid_id (str): The ID of the bid

    Returns:
        Dict[str, Any]: Dictionary containing opportunity information and document info
    """
    try:
        # Load bid documents
        bid_object_id = ObjectId(bid_id)
        bid = await bids_collection.find_one({"_id": bid_object_id})
        if not bid:
            raise HTTPException(status_code=404, detail="Bid not found")

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
        ]

        log.info("Documents being used for opportunity information generation:")
        for doc in documents:
            log.info(f"- {doc['meta']}")

        # Process documents
        combined_text = "\n\n".join([doc['text'] for doc in documents])
        prompt_text = load_prompt_from_file("generate_opportunity_information")
        prompt = ChatPromptTemplate.from_template(prompt_text)

        async def run_chain(model):
            chain = prompt | model | StrOutputParser()
            return await chain.ainvoke({"input_text": combined_text})

        try:
            result = await run_chain(llm_opportunity_information)
        except Exception as e:
            log.warning(
                f"Error using llm_large for opportunity information: {e}. Falling back to llm_fallback."
            )
            try:
                result = await run_chain(llm_fallback)
            except Exception as e:
                log.error(f"Error generating opportunity information with fallback model: {e}")
                result = "Error: Unable to generate opportunity information."

        # Save to MongoDB
        await bids_collection.update_one(
            {"_id": bid_object_id}, {"$set": {"opportunity_information": result}}
        )

        return {
            "opportunity_information": result,
            "documents": [{"id": doc['unique_id'], "name": doc['meta']} for doc in documents],
        }

    except Exception as e:
        log.error(f"Error in get_opportunity_information: {str(e)}")
        raise


async def get_exec_summary(documents: List[Dict[str, str]]) -> str:
    """
    Generates an executive summary for a tender bid based on all provided documents.
    """
    combined_text = "\n\n".join([doc['text'] for doc in documents])
    prompt_text = load_prompt_from_file("generate_exec_summary")
    prompt = ChatPromptTemplate.from_template(prompt_text)

    async def run_chain(model):
        chain = prompt | model | StrOutputParser()
        return await chain.ainvoke({"input_text": combined_text})

    try:
        result = await run_chain(llm_exec_summary)
    except Exception as e:
        log.warning(
            f"Error using llm_large for executive summary: {e}. Falling back to llm_fallback."
        )
        try:
            result = await run_chain(llm_fallback)
        except Exception as e:
            log.error(f"Error generating executive summary with fallback model: {e}")
            result = "Error: Unable to generate executive summary."

    return result.replace('\n\n', '\n').replace('\n', '\n\n')


async def get_cover_letter(documents: List[Dict[str, str]]) -> str:
    """
    Generates a cover letter for a tender bid based on all provided documents.
    """
    combined_text = "\n\n".join([doc['text'] for doc in documents])
    prompt_text = load_prompt_from_file("generate_cover_letter")
    prompt = ChatPromptTemplate.from_template(prompt_text)

    async def run_chain(model):
        chain = prompt | model | StrOutputParser()
        return await chain.ainvoke({"input_text": combined_text})

    try:
        result = await run_chain(llm_cover_letter)
    except Exception as e:
        log.warning(f"Error using llm_large for cover letter: {e}. Falling back to llm_fallback.")
        try:
            result = await run_chain(llm_fallback)
        except Exception as e:
            log.error(f"Error generating cover letter with fallback model: {e}")
            result = f"Error: Unable to generate cover letter."

    return result.replace('\n', '\n\n')


async def post_process_3b(state: GraphState) -> GraphState:
    """
    Post-processes the result for option 3b.
    """
    state.pre_post_processed_result = state.result
    if state.choice.startswith("3b") and state.result and state.post_processing_enabled:
        try:
            post_processing_prompt = load_prompt("post_processing")
            prompt = ChatPromptTemplate.from_template(post_processing_prompt)
            chain = prompt | state.model | StrOutputParser()

            state.post_processed_result = await chain.ainvoke(
                {
                    "final_answer": state.result,
                    "tone_of_voice": state.tone_of_voice,
                    "question": state.question,
                }
            )
        except Exception as e:
            log.error(f"Error in post-processing: {str(e)}")
            state.post_processed_result = state.result
    else:
        state.post_processed_result = state.result

    return state


async def post_process_final_document(text: str, username: str) -> str:
    """
    Post-processes the final document using the post-processing prompt.

    Args:
        text (str): The document text to process.
        username (str): The username for loading user config.

    Returns:
        str: The post-processed document text.
    """
    try:
        log.info("Starting final document post-processing")

        # Load the post-processing prompt
        post_processing_prompt = load_prompt_from_file("post_process_final_document")
        prompt = ChatPromptTemplate.from_template(post_processing_prompt)

        # Create and execute the processing chain
        async def run_chain(model):
            chain = prompt | model | StrOutputParser()
            return await chain.ainvoke({"final_answer": text})

        try:
            processed_text = await run_chain(llm_post_process)
        except Exception as e:
            log.warning(
                f"Error using llm_post_process for final document: {e}. Falling back to llm_post_process_fallback."
            )
            try:
                processed_text = await run_chain(llm_post_process_fallback)
            except Exception as e:
                log.error(f"Error processing final document with fallback model: {e}")
                processed_text = text

        # Apply user-specific post-processing
        user_config = await load_user_config(username)
        final_text = post_process_result(processed_text, user_config)

        log.info("Final document post-processing completed successfully")
        return final_text

    except Exception as e:
        log.error(f"Error in post_process_final_document: {str(e)}")
        # Return original text if processing fails
        return text


async def generate_chunk_title(chunk: str) -> str:
    """
    Generate a descriptive title for a chunk of text using the chunk_title prompt.

    Args:
        chunk (str): The text chunk to generate a title for.

    Returns:
        str: A descriptive title for the chunk.
    """
    try:
        # Load the chunk title prompt
        prompt_text = load_prompt("chunk_title")
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Create and execute the chain
        async def run_chain(model):
            chain = prompt | model | StrOutputParser()
            return await chain.ainvoke(
                {"chunk": chunk[:2000]}
            )  # Limit chunk size for title generation

        try:
            title = await run_chain(llm_chunk_title)
        except Exception as e:
            log.warning(
                f"Error using llm_chunk_title for chunk title: {e}. Falling back to llm_large."
            )
            try:
                title = await run_chain(llm_chunk_title)
            except Exception as e:
                log.error(f"Error generating chunk title with fallback model: {e}")
                return ""

        # Clean and validate the title
        title = title.strip()
        if len(title) > 100:  # Limit title length
            title = title[:97] + "..."

        log.debug(f"Generated title: {title}")
        return title

    except Exception as e:
        log.error(f"Error in generate_chunk_title: {str(e)}")
        return ""


async def get_section_compliance_requirements(
    compliance_text: str, section_title: str, question: str
) -> str:
    """
    Extracts relevant compliance requirements for a specific section from the full compliance text.

    Args:
        compliance_text (str): The full compliance requirements text
        section_title (str): The title of the current section
        question (str): The question/requirement for the current section

    Returns:
        str: Relevant compliance requirements for the section
    """
    if not compliance_text:
        return ""

    prompt_text = load_prompt_from_file("extract_section_compliance")
    prompt = ChatPromptTemplate.from_template(prompt_text)

    async def run_chain(model):
        chain = prompt | model | StrOutputParser()
        return await chain.ainvoke(
            {"compliance_text": compliance_text, "sub_topic": section_title, "question": question}
        )

    try:
        result = await run_chain(llm_compliance)
    except Exception as e:
        log.warning(
            f"Error using llm_compliance for section compliance: {e}. Falling back to llm_fallback."
        )
        try:
            result = await run_chain(llm_fallback)
        except Exception as e:
            log.error(f"Error extracting section compliance with fallback model: {e}")
            result = ""

    return result.strip()


async def get_section_insight(section: str, summmary: List, extract_insights_prompt: str) -> str:
    """
    Extracts relevant insights for a specific section from the bid intel.

    Args:
        section (str): The question we are assigning the insights to
        summmary (List): The list of insights from the bid intel page
        insight_type (str): The type of insight from the bid intel page we are assigning

    Returns:
        str: Relevant compliance requirements for the section
    """
    if not summmary:
        return ""

    prompt_text = load_prompt_from_file(extract_insights_prompt)
    prompt = ChatPromptTemplate.from_template(prompt_text)

    async def run_chain(model):
        chain = prompt | model | StrOutputParser()
        return await chain.ainvoke(
            {
                "insights_list": summmary,
                "sub_topic": section["heading"],
                "question": section["question"],
            }
        )

    try:
        result = await run_chain(llm_fallback)
    except Exception as e:
        log.warning(
            f"Error using llm_fallback for section compliance: {e}. Falling back to llm_section_compliance."
        )
        try:
            result = await run_chain(llm_section_compliance)
        except Exception as e:
            log.error(f"Error extracting section compliance with fallback model: {e}")
            result = ""

    return result.strip()


async def extract_choice(section_name: str) -> str:
    prompt_text = load_prompt_from_file("3b_assignment")
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm_writing_plan | StrOutputParser()
    result = await chain.ainvoke({"section_name": section_name})
    # Add trim to remove any extra whitespace
    return result.strip()


async def generic_run_chain(model, context_dict, prompt):
    """
    Run a chain with a model, dictionary of context variables, and prompt.

    Args:
        model: The LLM model to use
        context_dict: Dictionary containing multiple context keys
        prompt: The prompt template

    Returns:
        The generated output as a string
    """
    chain = prompt | model | StrOutputParser()
    return await chain.ainvoke(context_dict)


async def generate_search_queries_for_evidence(selected_text: str) -> List[str]:
    """
    Generate search queries based on the selected text using the evidence_retrieval prompt.

    Args:
        selected_text: The text selected by the user for evidence retrieval

    Returns:
        List of search queries generated from the prompt
    """
    try:
        # Load the evidence retrieval prompt
        evidence_prompt_template = load_prompt_from_file("evidence_retrieval")
        prompt = ChatPromptTemplate.from_template(evidence_prompt_template)

        # Create the chain
        chain = prompt | llm_fallback | StrOutputParser()

        # Generate queries
        result = await chain.ainvoke({"question": selected_text})

        # Parse the result to extract the search queries
        lines = result.strip().split('\n')
        queries = []

        for line in lines:
            # Look for lines that look like search queries (usually in quotes or clear bullet points)
            stripped_line = line.strip().strip('"\'').strip()
            if (
                stripped_line
                and not stripped_line.startswith('Generated Queries:')
                and len(stripped_line) > 10
            ):
                # Check if the line starts with a number, bullet point, or quotation mark
                if (
                    re.match(r'^[0-9]+[\.\)]', stripped_line)
                    or stripped_line.startswith('-')
                    or stripped_line.startswith('"')
                    or 'query' in line.lower()
                ):
                    # Remove any prefix (numbers, bullets, etc.)
                    clean_query = re.sub(
                        r'^[0-9]+[\.\)]|-|\"|\'|\"|\"|\'|"', '', stripped_line
                    ).strip()
                    if clean_query:
                        queries.append(clean_query)

        # If parsing fails, simply use the selected text as the query
        if not queries:
            log.warning(
                "Failed to parse search queries from the model output. Using selected text instead."
            )
            queries = [selected_text]

        log.info(f"Generated {len(queries)} search queries: {queries}")
        return queries

    except Exception as e:
        log.error(f"Error generating search queries: {str(e)}")
        # Return the selected text as a fallback query
        return [selected_text]


async def postprocess_evidence(selected_text: str, evidence_chunks: List[Dict[str, str]]) -> str:
    """
    Post-process retrieved evidence to better integrate it with the selected text.

    Args:
        selected_text: The text selected by the user
        evidence_chunks: List of evidence chunks with content and source

    Returns:
        Enhanced text with integrated evidence
    """
    try:
        log.info("Post-processing evidence to integrate with text")

        # Load the evidence post-processing prompt
        prompt_text = load_prompt("evidence_postprocess")

        # Prepare the prompt context
        context = {
            "text": selected_text,
            "Retrieved chunks": "\n\n".join(
                [
                    f"Source: {chunk.get('source', 'Unknown')}\nContent: {chunk.get('content', '')}"
                    for chunk in evidence_chunks
                ]
            ),
            "highlighted area in the text": selected_text,  # Using the entire text as highlighted area
        }

        # Create a prompt template
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Run the chain to get enhanced text
        enhanced_text = await generic_run_chain(llm_evidence_rewriting, context, prompt)

        log.info("Successfully post-processed evidence")
        return enhanced_text

    except Exception as e:
        log.error(f"Error post-processing evidence: {str(e)}")
        # Return the original text if post-processing fails
        return selected_text


async def get_evidence_for_text(selected_text: str, username: str) -> Dict[str, Any]:
    """
    Get supporting evidence for the selected text.

    Args:
        selected_text: The text selected by the user
        username: The current user's username

    Returns:
        Dictionary containing evidence information
    """
    try:
        # Input validation - ensure the text is long enough
        if len(selected_text.strip()) < 10:
            return {
                "success": False,
                "message": "Please select a longer text segment to find relevant evidence.",
            }

        # Generate search queries based on the selected text
        queries = await generate_search_queries_for_evidence(selected_text)

        # Use all queries to gather evidence
        all_evidence = []

        # Gather evidence from all queries in parallel
        evidence_results = await asyncio.gather(*[
            retrieve_evidence_chunks(username, query) for query in queries
        ])

        # Combine all evidence chunks, removing duplicates based on content
        seen_contents = set()
        for chunks in evidence_results:
            for chunk in chunks:
                # Use content as a key to avoid duplicates
                content_hash = hash(chunk.get('content', ''))
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_evidence.append(chunk)

        if all_evidence:
            # Post-process the evidence to integrate it with the text
            enhanced_text = await postprocess_evidence(selected_text, all_evidence)

            return {
                "success": True,
                "evidence": all_evidence,
                "query_used": queries[0] if queries else "",  # Add this for backward compatibility
                "all_queries": queries,
                "enhanced_text": enhanced_text,
            }
        else:
            # If no evidence found, return a failure response
            return {
                "success": False,
                "message": "No relevant evidence found in the company library.",
                "queries_tried": queries,
            }

    except Exception as e:
        log.error(f"Error getting evidence: {str(e)}")
        return {"success": False, "message": f"An error occurred: {str(e)}"}
