Tender Processing
================

Overview
--------

The Tender Processing module is responsible for analyzing tender documents, extracting key information, and generating insights to support bid preparation. This component bridges the gap between raw tender documents and actionable bid intelligence.

Architecture
-----------

The tender processing system has several interconnected components:

* **Tender Library**: Document storage for tender-specific files
* **Analysis Pipelines**: Automated intelligence extraction modules
* **Insight Generation**: LLM-based information extraction and synthesis
* **Bid Intelligence**: Structured data extracted from tender documents

Key Components
-------------

1. **Document Management**
   * Tender-specific document organization
   * Specialized parsing for tender formats
   * Version tracking and change management

2. **Intelligence Extraction**
   * Requirements identification
   * Evaluation criteria analysis
   * Opportunity insight extraction
   * Competitive differentiation analysis

3. **Integration Points**
   * Proposal generation guidance
   * Evidence suggestion for claims
   * Compliance verification support
   * Reference material for RAG context

Core Functionality
----------------

Tender Library Management
^^^^^^^^^^^^^^^^^^^^^^^

Tender documents are organized in specialized collections:

.. code-block:: python

   # Tender library collection naming convention
   tender_library_collection = f'tender_library_{bid_id}'

**Implementation Notes:**

* Each tender has a dedicated vector collection
* Specialized parsing emphasizes structured information extraction
* Tender documents are isolated from general company library

Requirements Extraction
^^^^^^^^^^^^^^^^^^^^^

The system automatically extracts compliance requirements:

.. code-block:: python

   async def get_compliance_requirements(bid_id: str, username: str) -> Dict[str, Any]:
       """
       Generates compliance requirements from tender documents and saves to MongoDB.
       """
       # Implementation details

**Process Flow:**

1. Tender documents are analyzed using specialized prompts
2. Requirements are extracted using LLM-guided analysis
3. Structured requirements are stored for reference
4. Requirements guide proposal outline and content

**Parameters:**

* ``bid_id``: The ID of the bid/tender
* ``username``: The user performing the operation

Insight Generation
^^^^^^^^^^^^^^^^

Multiple specialized insights are extracted from tender documents:

.. code-block:: python

   async def get_tender_insights(bid_id: str, prompt_type: str, username: str) -> Dict[str, Any]:
       """
       Generates various types of insights from tender documents.
       """
       # Implementation logic

**Insight Types:**

* **Tender Summary**: Overall synthesis of the opportunity
* **Evaluation Criteria**: Extracted scoring factors
* **Pain Points**: Client challenges identified in the tender
* **Opportunity Information**: Structured opportunity details

**Parameters:**

* ``bid_id``: The tender identifier
* ``prompt_type``: Type of insight to generate
* ``username``: User performing the analysis

Differentiation Analysis
^^^^^^^^^^^^^^^^^^^^^^

The system identifies competitive differentiation opportunities:

.. code-block:: python

   async def get_differentiation_opportunities(bid_id: str, prompt: str) -> Dict[str, Any]:
       """
       Analyzes competitors and identifies differentiation opportunities.
       """
       # Implementation logic

**Analysis Process:**

1. Tender context and company profile are analyzed
2. Competitive landscape is assessed
3. Differentiation opportunities are identified
4. Strategic advantages are highlighted

**Parameters:**

* ``bid_id``: The tender identifier
* ``prompt``: The analysis approach to use

Technical Implementation
----------------------

LLM Selection
^^^^^^^^^^^

Different models are used based on analysis requirements:

.. code-block:: python

   # Model selection examples
   result = await run_chain(llm_tender_insights, combined_text, main_prompt)
   result = await run_chain(llm_compliance, combined_text)
   result = await run_chain(perplexity, input_data)

**Model Configuration:**

* ``llm_tender_insights``: OpenAI model for general tender analysis
* ``llm_compliance``: Specialized model for requirements extraction
* ``perplexity``: Alternative model for competitive analysis
* ``llm_fallback``: Backup model for resilience

Document Processing
^^^^^^^^^^^^^^^^^

Tender documents undergo specialized processing:

.. code-block:: python

   # Document access pattern
   tender_library = bid.get("tender_library", [])
   documents = [
       {
           "text": doc.get("rawtext", ""),
           "unique_id": str(doc.get("_id", "")),
           "meta": doc.get("filename", ""),
       }
       for doc in tender_library
       if doc["filename"] in file_names
   ]

   # Text combination for analysis
   combined_text = "\n\n".join([doc['text'] for doc in documents])

**Processing Considerations:**

* Document selection can target specific files
* Full-text analysis preserves context
* Document metadata maintains traceability

Resilience Patterns
^^^^^^^^^^^^^^^^^

The system implements multiple resilience techniques:

.. code-block:: python

   # Fallback pattern
   try:
       result = await run_chain(llm_tender_insights, combined_text, main_prompt)
   except Exception as e:
       log.warning(f"Error using primary model: {e}. Falling back to retrieval.")
       # Fallback implementation

   # Retry pattern with tenacity
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
   async def _try_generate_outline(chain, input_text: str) -> str:
       return await chain.ainvoke({"input_text": input_text})

**Resilience Strategies:**

* Multiple fallback mechanisms
* Exponential backoff for retries
* Graceful degradation paths
* Detailed error logging

Data Storage
^^^^^^^^^^

Analysis results are stored in MongoDB for reference:

.. code-block:: python

   # Storage pattern
   await bids_collection.update_one(
       {"_id": bid_object_id}, 
       {"$set": {"compliance_requirements": result}}
   )

   # Retrieval pattern
   bid = await bids_collection.find_one({"_id": ObjectId(bid_id)})
   compliance_text = bid.get("compliance_requirements", "")

**Storage Strategy:**

* Results stored directly in bid document
* Field-specific storage for different insight types
* Structured storage for programmatic access

Concurrent Processing
^^^^^^^^^^^^^^^^^^^

Parallel processing improves performance:

.. code-block:: python

   # Concurrent execution using asyncio
   tender_insights = await asyncio.gather(
       get_tender_insights(bid_id, "generate_summarise_tender", current_user),
       get_tender_insights(bid_id, "generate_evaluation_criteria", current_user),
       get_tender_insights(bid_id, "generate_derive_insights", current_user),
       get_differentiation_opportunities(bid_id, "generate_differentiation_opportunities")
   )

**Performance Benefits:**

* Reduced overall processing time
* Independent analysis pathways
* Status tracking for long-running operations

Error Handling
^^^^^^^^^^^^

Comprehensive error handling ensures reliable operation:

.. code-block:: python

   try:
       # Complex operation
   except Exception as e:
       log.error(f"Error in get_tender_insights: {str(e)}")
       log.error(f"Error type: {type(e).__name__}")
       log.error(f"Error details: {e.args}")
       log.error("Traceback: ", exc_info=True)
       raise

**Error Management:**

* Detailed logging of exception details
* Typed exception handling
* Context preservation for debugging
* User-friendly error messages

Prompt Engineering
^^^^^^^^^^^^^^^^

The system uses specialized prompts for different analyses:

.. code-block:: python

   # Prompt loading pattern
   prompt_text = load_prompt_from_file("generate_compliance_requirements")
   prompt = ChatPromptTemplate.from_template(prompt_text)

**Prompt Categories:**

* Requirements extraction prompts
* Evaluation criteria prompts
* Insight generation prompts
* Competitive analysis prompts

Integration Points
----------------

Integration with Proposal Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tender analysis guides proposal creation:

* Requirements inform section compliance
* Evaluation criteria guide content emphasis
* Insights shape value propositions
* Differentiation opportunities influence messaging

Integration with RAG Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tender documents enhance RAG operations:

* Tender-specific retrieval using specialized collections
* Context enrichment with tender insights
* Query augmentation with requirement awareness
* Response validation against compliance parameters

Integration with User Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The system provides structured data for UI components:

* Summary displays for key insights
* Requirement checklists for compliance verification
* Evaluation criteria highlighting
* Differentiation opportunity visualization

Performance Considerations
------------------------

1. **Document Size**: Larger tender documents require more processing time
2. **Analysis Depth**: More comprehensive analyses have higher LLM token usage
3. **Concurrent Analyses**: Multiple parallel operations may impact system responsiveness
4. **Model Selection**: Different models have varying performance characteristics

For implementation details, see the tender-related functions in the ``services.chain`` module and associated API modules. 