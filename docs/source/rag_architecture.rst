RAG Architecture
===============

Overview
--------

The mytender.io platform employs a sophisticated Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware responses for users working with tender documents. This architecture combines vector search capabilities with advanced language models to generate highly relevant content that incorporates an organization's proprietary knowledge.

.. image:: _static/rag_architecture.png
   :width: 800
   :alt: RAG Architecture Diagram

Key Components
-------------

The RAG pipeline consists of seven main stages, each with specific responsibilities and optimizations:

1. Document Ingestion
2. Chunking & Embedding
3. Vector Storage
4. Retrieval
5. Context Processing
6. Response Generation
7. Post Processing

Technical Pipeline Flow
----------------------

1. Document Ingestion
^^^^^^^^^^^^^^^^^^^^

The first stage of the RAG pipeline involves ingesting documents from various sources.

**Process Flow:**

1. User uploads documents through the API endpoint ``/uploadfile``
2. Content is extracted using specialized parsers based on file type
3. Metadata is generated and attached to the document
4. Text is prepared for chunking and embedding

**Supported File Types:**

* **PDF Files**: Processed using ``pymupdf4llm``
* **Word Documents**: Processed using ``MarkItDown``
* **Excel Files**: Processed using ``pandas``
* **Plain Text**: Processed directly

**Key Functions:**

.. code-block:: python

   async def process_create_upload_file(
        file: UploadFile,
        profile_name: str,
        mode: str,
        current_user: str) -> Dict[str, str]:
       # File upload processing, parsing, and metadata extraction
       # Returns a dictionary with upload status information

**Configuration Parameters:**

* ``mode``: Determines processing approach ('pdf', 'plain')
* ``profile_name``: Target collection for document storage
* ``format``: File format indicator ('text', 'file')
* ``metadata``: Document metadata including filename, uploader, timestamp

**Implementation Details:**

The system extracts content and maintains document structure, with specialized handling for tables, headers, and formatting. Binary content is stored in MongoDB for PDFs and other binary formats, with a 10MB size limit applied during processing.

2. Chunking & Embedding
^^^^^^^^^^^^^^^^^^^^^

The extracted content is processed for storage and retrieval.

**Process Flow:**

1. Text is split into manageable chunks with appropriate overlap
2. Each chunk is converted into a vector embedding
3. Metadata is associated with each chunk
4. Chunks are prepared for storage

**Key Functions:**

.. code-block:: python

   async def text_to_chromadb(
       text: str,
       user: str,
       collection: str,
       user_name: str,
       mode,
       embedding,
       metadata,
       file_size = 1024000,
       file_content: Binary = None,
       log_entry=True,
       format='text',
       unique_id=None
   ):
       # Chunks text and creates embeddings in ChromaDB
       # Also logs document metadata in MongoDB

**Chunking Parameters:**

* Chunk size: Configurable based on embedding model limitations
* Chunk overlap: Typically 10-20% to maintain context across chunks
* Chunking strategy: Implemented in ``services.splitter.split_text()``

**Embedding Details:**

* Default model: OpenAI Embeddings (``embedder`` from config)
* Embedding dimensions: 1536 for OpenAI embeddings
* Alternative models: Support for Ollama embeddings in local deployments

**Vector Quality Considerations:**

* Chunks are designed to balance context preservation with retrieval precision
* Each chunk maintains reference to original document via metadata
* Unique IDs associate chunks with their source documents

3. Vector Storage
^^^^^^^^^^^^^^^

The vectorized content is stored for efficient retrieval.

**Storage Architecture:**

* **Primary Store**: ChromaDB collections organized by dataset/profile
* **Metadata Store**: MongoDB collections tracking document properties
* **Storage Path**: Configured in ``CHROMA_FOLDER/{username}``

**Key Functions:**

.. code-block:: python

   # ChromaDB Integration
   vectorstore = Chroma(
       collection_name=collection,
       embedding_function=embedding,
       persist_directory=chroma_db_directory,
       collection_metadata={
           "metadata": unique_id,
           "unique_id": unique_id,
       },
   )

   # MongoDB Document Storage
   document = {
       "collection_name": collection,
       "chroma_db_directory": chroma_db_directory,
       "mode": mode,
       "text": text,
       "user": user_name,
       "format": format,
       "meta": metadata,
       "filename": metadata['filename'],
       'profile_name': collection,
       "unique_id": unique_id,
   }

**Collection Organization:**

* Collections are organized hierarchically with folder/subfolder support
* Default collection: Always contains all documents for backup retrieval
* Special collections: Case studies, tender-specific, and feedback collections
* Folder separation: Uses ``FORWARDSLASH`` delimiter in collection names

**Management Operations:**

* Document deletion: Removes entries from both ChromaDB and MongoDB
* Collection deletion: Removes entire collections with contained documents
* Document migration: Supports moving documents between collections

4. Retrieval
^^^^^^^^^^

When a user query is received, the system retrieves relevant information.

**Retrieval Flow:**

1. User query is processed and prepared for vector search
2. Query is embedded using the same embedding model
3. Vector similarity search is performed across collections
4. Relevance scores determine which documents are included
5. Retrieved content is filtered and ranked

**Key Functions:**

.. code-block:: python

   @async_timer
   async def retrieve_docs(vectorstore, query: str, k: int, parent_user: str) -> List[Dict[str, str]]:
       """Retrieve documents with timing metrics."""
       # Performs retrieval operation with detailed logging
       # Returns list of documents with content and source

   @async_timer
   async def retrieve_documents_for_state(state):
       """Retrieve documents based on state parameters."""
       # Parallelized retrieval across multiple collections
       # Deduplicates results and formats response

**Search Parameters:**

* ``query``: The search text to embed and match against documents
* ``k``: Number of chunks to retrieve (derived from "broadness" setting)
* ``search_type``: Default is "mmr" (Maximum Marginal Relevance)
* ``parent_user``: Organization owner account for shared collections
* ``datasets``: List of collections to search

**Relevance Checking:**

Documents undergo a two-stage relevance process:

1. Initial retrieval based on vector similarity 
2. Secondary LLM-based relevance assessment:

.. code-block:: python

   async def check_relevance(state: GraphState) -> GraphState:
       # Uses LLM to score relevance on 0-10 scale
       # Filters out documents below RELEVANCE_THRESHOLD (default: 6)

**Performance Optimizations:**

* Concurrent collection querying with `asyncio.gather()`
* Timeout protection (45-second maximum for retrieval operations)
* Exponential backoff for retry operations
* Result deduplication to prevent redundant content

5. Context Processing
^^^^^^^^^^^^^^^^^^^

Retrieved documents are processed to create context for the LLM.

**Processing Steps:**

1. Documents are assembled with source attribution
2. Context is restructured for optimal LLM comprehension
3. Special content types receive tailored handling
4. Context size is managed to fit model limitations

**Key Functions:**

.. code-block:: python

   async def process_context(state: GraphState) -> GraphState:
       context = "\n\n".join(
           [f"[Source: {doc['source']}]\n{doc['content']}" for doc in state.relevant_docs]
       )
       state.context = process_numbers(context, await load_user_config(state.username))
       return state

**Specialized Processing:**

* **Numeric Handling**: `process_numbers()` function applies user configuration for numeric content
* **Source Attribution**: Each chunk is labeled with its document source
* **Content Merging**: Relevant chunks are assembled into a unified context

**Context Optimization:**

* Context organization prioritizes most relevant content first
* Source information enables attribution in generated responses
* User configuration allows customization of processing behavior

6. Response Generation
^^^^^^^^^^^^^^^^^^^^

The LLM uses the prepared context to generate a response.

**Generation Flow:**

1. Prompt is constructed with context and query information
2. Appropriate LLM is selected based on task requirements
3. Prompt engineering techniques enhance response quality
4. Response is generated with relevant content incorporation

**Model Selection:**

Multiple LLM models are available based on the task:

.. code-block:: python

   # LLM assignments with explicit naming
   llm_chain_default = openai_instance      # Default model for general processing
   llm_tender_insights = openai_instance    # Model for tender insights
   llm_outline = gemini_15pro_instance      # Model for generating outlines
   llm_fallback = bedrock_claude35          # Fallback model when primary fails
   llm_post_process = openai_instance       # Model for post-processing
   llm_writing_plan = openai_fast_instance  # Model for writing plans

**Generation Parameters:**

* ``model``: LLM instance to use for generation
* ``temperature``: Controls randomness (default: 0 for deterministic outputs)
* ``relevant_prompt``: Template loaded based on task type
* ``choice``: Determines the generation approach and prompt selection

**Prompt Engineering:**

* Task-specific prompts loaded via `load_prompt_from_file()`
* System messages set appropriate LLM behavioral context
* Context truncation and organization based on importance

**Multi-Section Generation:**

For longer content with multiple sections (choice "3b"):

.. code-block:: python

   async def process_multiple_headers(state: GraphState) -> str:
       # Generates content for multiple sections in parallel
       # Each section has dedicated context and parameters

7. Post Processing
^^^^^^^^^^^^^^^^

The response undergoes final refinements before delivery.

**Processing Steps:**

1. Content formatting and structure adjustments
2. Language standardization (UK English)
3. Content filtering for forbidden terms
4. Final quality enhancements

**Key Functions:**

.. code-block:: python

   def post_process_result(result: str, user_config: Dict) -> str:
       # Filters forbidden words
       # Applies British English spelling
       return britishize(result)

   async def post_process_3b(state: GraphState) -> GraphState:
       # Special post-processing for multi-section responses
       # Applies additional formatting and consistency checks

**Post-Processing Options:**

* ``post_processing_enabled``: Boolean to toggle post-processing
* ``forbidden``: User-configurable list of terms to filter
* ``tone_of_voice``: Influences final content style adjustments

**Quality Enhancements:**

* Consistent UK English spelling via `britishize()` function
* Removal of prohibited terminology per user configuration
* Format standardization for output presentation

Error Handling and Resilience
----------------------------

The RAG pipeline includes comprehensive error handling and resilience mechanisms:

**Retry Mechanisms:**

* Decorated functions with `@retry` use exponential backoff
* Fallback models when primary models encounter issues
* Timeout protection for network operations

**Error Logging:**

* Detailed logging with `@async_timer` performance tracking
* Exception capturing with context for troubleshooting
* Graceful degradation when components fail

**User Communication:**

* Clear error messages returned via API
* Custom HTTP exceptions with appropriate status codes
* Fallback content when full processing fails

Specialized Use Cases
-------------------

The RAG architecture supports several specialized use cases beyond basic query-response:

**Tender Processing:**

* Document library management for tender documents
* Automatic extraction of tender requirements
* Context-aware bid response generation
* Compliance verification against requirements

**Proposal Generation:**

* Automatic outline creation from tender documents
* Evidence retrieval for proposal claims
* Section-specific context assembly
* Compliance-aware content generation

**Multi-Stage Operations:**

* Writing plan generation before content creation
* Draft generation with subsequent refinement
* Insight extraction to guide content creation
* Differentiation factor identification

Advanced Customization
--------------------

The RAG pipeline offers several customization points:

**User Configuration:**

* Organization-specific content standards
* Terminology preferences and restrictions
* Collection management policies
* Language and formatting preferences

**Model Selection:**

* Task-specific model assignment
* Performance vs. cost optimization
* Fallback hierarchy configuration
* Local deployment options

**Prompt Engineering:**

* Custom prompt templates
* Task-specific instruction tuning
* Output format control
* Context utilization guidance

Performance Considerations
------------------------

Several factors impact RAG pipeline performance:

**Retrieval Optimization:**

* ``k`` parameter balances breadth vs. specificity
* MMR search promotes diversity in results
* Relevance threshold adjusts precision vs. recall
* Collection selection impacts search domain

**Resource Usage:**

* Document chunking affects storage requirements
* Embedding dimensionality impacts search performance
* LLM token consumption scales with context size
* Concurrent operations management prevents overload

**Latency Management:**

* Asynchronous processing with `asyncio`
* Parallel retrieval from multiple collections
* Caching strategies for frequently accessed content
* Timeout protection for external service calls

Future Enhancements
-----------------

Potential areas for RAG pipeline improvement:

1. **Retrieval Enhancement**
   * Hybrid sparse-dense retrieval combining BM25 with vectors
   * Re-ranking with cross-encoders for improved precision
   * Query expansion for better recall on complex topics

2. **Context Refinement**
   * Content summarization before LLM processing
   * Hierarchical context organization (document→section→chunk)
   * Dynamic context management based on query complexity

3. **Generation Improvements**
   * Response streaming for improved user experience
   * Multi-step reasoning for complex questions
   * Self-consistency checks for factual accuracy

4. **Infrastructure Optimization**
   * Vector database partitioning for larger collections
   * Result caching for frequent queries
   * Adaptive model selection based on query characteristics

API Reference
-----------

For technical implementation details, refer to these core modules:

* ``services.embedding``: Vector creation and storage
* ``services.vector_manager``: Retrieval and search operations
* ``services.chain``: LLM processing and orchestration
* ``api_modules.company_library``: Document management
* ``api_modules.generate_proposal``: Content generation

Consult the API documentation section for detailed endpoint specifications and request formats. 