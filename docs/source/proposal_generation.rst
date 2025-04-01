Proposal Generation
==================

Overview
--------

The Proposal Generation module is a sophisticated component of the mytender.io platform designed to streamline and enhance the bid response process. It leverages RAG technology, tender analysis, and specialized LLMs to generate high-quality proposal content tailored to specific tender requirements.

Architecture
-----------

The proposal generation system consists of several interconnected components:

* **Outline Generation**: Creates structured proposal frameworks
* **Content Generation**: Produces section-specific content
* **Evidence Integration**: Incorporates supporting material
* **Document Assembly**: Formats final proposal documents

Key Components
-------------

1. **Proposal Outlining**
   * Automated tender structure analysis
   * Section organization and hierarchy
   * Word count allocation
   * Compliance mapping

2. **Section Generation**
   * Requirement-driven content creation
   * Evidence-based argumentation
   * Multi-section parallel processing
   * Writing plan implementation

3. **Quality Enhancement**
   * Compliance verification
   * Style consistency enforcement
   * Content optimization
   * Reference management

Core Functionality
----------------

Outline Generation
^^^^^^^^^^^^^^^^

The system automatically generates proposal outlines from tender documents:

.. code-block:: python

   async def process_generate_outline(
       bid_id: str,
       file_names: list[str],
       extra_instructions: str,
       datasets: list[str],
       current_user: str,
       newbid: bool = False,
   ) -> List[str]:
       # Outline generation logic

**Process Flow:**

1. Tender documents are analyzed for structure and requirements
2. LLM generates a hierarchical outline with sections and subsections
3. Word counts are allocated based on section importance
4. Compliance requirements are mapped to relevant sections

**Parameters:**

* ``bid_id``: The tender identifier
* ``file_names``: Specific documents to analyze
* ``extra_instructions``: Additional guidance for outline creation
* ``datasets``: Company knowledge collections to reference
* ``current_user``: User performing the operation
* ``newbid``: Whether this is a new bid creation

**Implementation Details:**

The outline generation uses specialized LLM processing:

.. code-block:: python

   # LLM outline generation with fallback pattern
   async def get_outline(documents: List[Dict[str, str]]) -> str:
       combined_text = "\n\n".join([doc['text'] for doc in documents])
       prompt_text = load_prompt_from_file("generate_outline")
       prompt = ChatPromptTemplate.from_template(prompt_text)
       
       chain = prompt | llm_outline | StrOutputParser()
       
       try:
           # Try primary model with retries
           result = await _try_generate_outline(chain, combined_text)
       except Exception as e:
           # Switch to fallback model
           chain = prompt | llm_outline_fallback | StrOutputParser()
           result = await _try_generate_outline(chain, combined_text)
       
       return result

Writing Plan Generation
^^^^^^^^^^^^^^^^^^^^^

Each section receives a detailed writing plan:

.. code-block:: python

   async def process_generate_writing_plans_for_section(section: dict) -> dict:
       # Writing plan generation logic

**Writing Plan Elements:**

* Content structure guidance
* Key points to address
* Evidence suggestions
* Tone and approach recommendations

Section Content Generation
^^^^^^^^^^^^^^^^^^^^^^^^

The system generates content for individual sections:

.. code-block:: python

   async def process_section(section, selected_folders, current_user, bid_id, selected_case_studies_raw_text):
       """Process a single section with its LLM call"""
       # Section generation logic

**Generation Process:**

1. Section input is assembled from various sources
2. Relevant context is retrieved from knowledge base
3. LLM generates content based on writing plan
4. Content is post-processed for quality

**Context Integration:**

Each section integrates multiple information sources:

* Tender requirements specific to the section
* Compliance requirements
* Evaluation criteria
* Company differentiators
* Case studies and evidence

Proposal Assembly
^^^^^^^^^^^^^^^

Individual sections are assembled into a complete proposal:

.. code-block:: python

   async def process_generate_proposal(
       bid_id: str, selected_folders: List[str], current_user: str
   ) -> dict:
       # Full proposal generation logic

**Assembly Process:**

1. Outline structure defines document organization
2. Section content is incorporated sequentially
3. Formatting is applied according to document standards
4. References and cross-references are resolved

Evidence Integration
^^^^^^^^^^^^^^^^^

The system incorporates supporting evidence:

.. code-block:: python

   async def get_evidence_for_text(selected_text: str, username: str) -> Dict[str, Any]:
       """Get supporting evidence for selected text."""
       # Evidence retrieval and integration logic

**Evidence Types:**

* Case studies
* Credentials
* Past performance examples
* Technical specifications

Technical Implementation
----------------------

State Management
^^^^^^^^^^^^^

The system maintains proposal state through MongoDB:

.. code-block:: python

   # State storage pattern
   await bids_collection.update_one(
       {"_id": ObjectId(bid_id)}, {"$set": {"outline": updated_outline}}
   )

   # State retrieval pattern
   bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
   outline = bid.get("outline", [])

**State Elements:**

* Outline structure
* Section content
* Generation status
* Editing history

Parallel Processing
^^^^^^^^^^^^^^^^^

Content generation leverages asynchronous processing:

.. code-block:: python

   # Process sections in batches to avoid overwhelming the LLM service
   BATCH_SIZE = 50  # Adjust based on your LLM service limits
   sections = []
   
   for i in range(0, len(outline), BATCH_SIZE):
       batch = outline[i : i + BATCH_SIZE]
       batch_tasks = [
           process_section(section, selected_folders, current_user, bid_id, selected_case_studies_raw_text)
           for section in batch
       ]
       batch_results = await asyncio.gather(*batch_tasks)
       sections.extend(batch_results)

**Batch Processing Benefits:**

* Improved throughput for large proposals
* Rate limit management
* Progress tracking
* Failure isolation

Error Handling and Retries
^^^^^^^^^^^^^^^^^^^^^^^^

Robust error handling ensures reliable operation:

.. code-block:: python

   # Section processing with retry logic
   max_retries = 3
   base_delay = 5
   
   for attempt in range(max_retries):
       try:
           answer = await invoke_graph(...)
           return {"answer": answer, "question": question, "section_title": section_title}
       except Exception as e:
           if attempt < max_retries - 1:
               delay = base_delay * (2**attempt)
               await asyncio.sleep(delay)
               continue
           log.error(f"Error processing section: {question} - {str(e)}", exc_info=True)
           raise

**Resilience Features:**

* Exponential backoff
* Attempt tracking
* Detailed error logging
* Graceful failure handling

Graph-Based Processing
^^^^^^^^^^^^^^^^^^^^

The system uses a graph-based processing approach:

.. code-block:: python

   # Graph-based processing in invoke_graph
   workflow = StateGraph(GraphState)
   
   # Define nodes
   workflow.add_node("retrieve_documents", RunnablePassthrough() | retrieve_documents)
   workflow.add_node("check_relevance", RunnablePassthrough() | check_relevance)
   workflow.add_node("process_context", RunnablePassthrough() | process_context)
   # ... additional nodes
   
   # Set up the workflow
   workflow.set_entry_point("retrieve_documents")
   workflow.add_edge("retrieve_documents", "check_relevance")
   workflow.add_edge("check_relevance", "process_context")
   # ... additional edges
   
   graph_runnable = workflow.compile()
   result = await graph_runnable.ainvoke(initial_state)

**Graph Processing Benefits:**

* Modular processing steps
* Explicit workflow definition
* State management across pipeline
* Reusable processing components

Post-Processing
^^^^^^^^^^^^^

Generated content undergoes quality enhancement:

.. code-block:: python

   async def remove_references(bid_id: str, current_user: str) -> dict:
       """Removes all references in square brackets from the generated proposal."""
       # Implementation details

**Post-Processing Operations:**

* Reference formatting
* Style standardization
* Terminology consistency
* Layout optimization

Document Export
^^^^^^^^^^^^^

Final proposals are exported as formatted documents:

.. code-block:: python

   # Document generation and MongoDB storage
   output_buffer = io.BytesIO()
   doc.save(output_buffer)
   output_buffer.seek(0)
   doc_content = output_buffer.getvalue()
   
   await bids_collection.update_one(
       {"_id": ObjectId(bid_id)}, {"$set": {"generated_proposal": Binary(doc_content)}}
   )

**Export Features:**

* DOCX format support
* Binary storage in MongoDB
* Download capability
* Metadata preservation

Key Integration Points
--------------------

Integration with Tender Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Proposal generation leverages tender analysis:

* Compliance requirements guide content creation
* Evaluation criteria inform emphasis
* Tender insights shape messaging
* Structure reflects tender organization

Integration with Company Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The company knowledge base enhances proposals:

* Evidence retrieval from document library
* Case study incorporation
* Terminology alignment with company standards
* Consistent messaging across proposals

Integration with LLM Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Advanced LLM capabilities power generation:

* Content creation leverages RAG architecture
* Context-aware generation maintains relevance
* Multi-step reasoning improves quality
* Specialized models address different needs

Performance Considerations
------------------------

1. **Outline Complexity**: More detailed outlines require additional processing
2. **Proposal Size**: Larger proposals consume more tokens and processing time
3. **Evidence Integration**: Evidence retrieval adds processing overhead
4. **Concurrent Generation**: Multiple proposals may impact system performance

Future Enhancements
-----------------

Potential improvements to the proposal generation system:

1. **Interactive Refinement**: Iterative improvement based on user feedback
2. **Quality Benchmarking**: Comparative scoring against successful proposals
3. **Enhanced Evidence Integration**: More sophisticated evidence incorporation
4. **Template Learning**: Improvement based on successful proposal patterns

For implementation details, see the ``api_modules.generate_proposal`` and ``api_modules.proposal_outline`` modules. 