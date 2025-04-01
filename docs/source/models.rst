Models
======

Overview
--------

The mytender.io platform leverages a diverse ecosystem of Large Language Models (LLMs) to power different aspects of its functionality. This approach enables task-specific optimization, resilience through fallbacks, and performance tuning for different requirements.

Model Architecture
---------------

The system employs a multi-model architecture with:

* **Primary Models**: High-capability models for core tasks
* **Specialized Models**: Task-optimized models for specific functions
* **Fast Models**: Lower-latency models for interactive operations
* **Fallback Models**: Alternative models for resilience

Model Configuration
----------------

The model configuration is centralized in the ``config.py`` file:

.. code-block:: python

   # LLM assignments with explicit naming
   llm_outline_fallback = openai_instance  # Fallback model for outline generation
   llm_compliance = openai_instance        # Model for compliance processing
   llm_section_compliance = openai_instance # Model for section-level compliance extraction
   llm_chain_default = openai_instance     # Default model for general chain processing
   llm_outline = gemini_15pro_instance     # Model for generating outlines
   llm_tender_insights = openai_instance   # Model for tender insights
   llm_tender_library_chat = openai_instance # Model for tender library chat
   llm_fallback = bedrock_claude35         # Fallback model when primary fails
   llm_post_process = openai_instance      # Model for post-processing
   llm_post_process_fallback = bedrock_claude35 # Fallback model for post-processing
   llm_writing_plan = openai_fast_instance # Model for writing plans
   llm_chunk_title = openai_instance_mini  # Model for generating chunk titles
   llm_bid_intel_summary = openai_instance # Model for generating bid intel summary
   llm_opportunity_information = openai_instance # Model for generating opportunity information
   llm_cover_letter = openai_instance      # Model for generating cover letter
   llm_exec_summary = openai_instance      # Model for generating executive summary
   llm_diagram = openai_instance           # Model for generating diagrams
   llm_keywords_for_subtopic = openai_instance_mini # Model for generating keywords for subtopic
   llm_evidence_rewriting = openai_instance # Model for evidence rewriting

Supported Providers
-----------------

The system integrates with multiple model providers:

1. **OpenAI**
   * Primary provider for most system functions
   * Models: GPT-4o, GPT-4o-mini
   * Implementation: `ChatOpenAI` from langchain_openai

2. **Anthropic**
   * Provider for certain specialized tasks
   * Models: Claude 3.5 Sonnet
   * Implementation: `ChatAnthropic` from langchain_anthropic

3. **Google**
   * Provider for outline generation
   * Models: Gemini 1.5 Pro
   * Implementation: `ChatGoogleGenerativeAI` from langchain_google_genai

4. **AWS Bedrock**
   * Provider for fallback operations
   * Models: Claude 3.5 Sonnet, Llama 3
   * Implementation: `ChatBedrock` from langchain_aws

5. **Perplexity**
   * Provider for specialized research
   * Models: Sonar Pro
   * Implementation: `ChatPerplexity` from langchain_community

6. **Groq**
   * Provider for fast inference
   * Models: Mixtral, Llama
   * Implementation: `ChatGroq` from langchain_groq

7. **Ollama**
   * Provider for local deployment
   * Models: Llama 2
   * Implementation: `OllamaLLM` from langchain_ollama

Model Instantiation
-----------------

Models are instantiated with specific configurations:

.. code-block:: python

   # Update OpenAI instances to use rate limiter
   openai_instance = ChatOpenAI(
       model_name=GPT_DEFAULT_MODEL,
       verbose=False,
       temperature=0,
       callback_manager=callback_manager,
       rate_limiter=openai_rate_limiter,
   )

   openai_instance_mini = ChatOpenAI(
       model_name=GPT_FAST_MODEL,
       verbose=False,
       temperature=0,
       callback_manager=callback_manager,
       rate_limiter=openai_rate_limiter,
   )

   claude_instance = ChatAnthropic(
       model="claude-3-5-sonnet-20240620", 
       anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
   )

   gemini_15pro_instance = ChatGoogleGenerativeAI(
       model="gemini-1.5-pro-latest",
       verbose=False,
       temperature=0,
       callback_manager=callback_manager,
       rate_limiter=gemini_rate_limiter,
   )

Rate Limiting
-----------

The system implements provider-specific rate limiters:

.. code-block:: python

   bedrock_rate_limiter = LoggingRateLimiter(
       name="Bedrock",
       requests_per_second=3.0,
       max_bucket_size=5,
       check_every_n_seconds=0.01,
   )

   gemini_rate_limiter = LoggingRateLimiter(
       name="Gemini",
       requests_per_second=33,
       max_bucket_size=50,
       check_every_n_seconds=0.01,
   )

   openai_rate_limiter = LoggingRateLimiter(
       name="OpenAI",
       requests_per_second=33,
       max_bucket_size=50,
       check_every_n_seconds=0.01,
   )

**Rate Limiting Parameters:**

* ``requests_per_second``: Maximum sustained request rate
* ``max_bucket_size``: Burst capacity for temporary spikes
* ``check_every_n_seconds``: Interval for token bucket refills

Model Usage Patterns
------------------

Task-Specific Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different tasks use specialized models:

.. code-block:: python

   # Outline generation with Gemini
   chain = prompt | llm_outline | StrOutputParser()
   
   # Compliance processing with specialized model
   chain = prompt | llm_compliance | StrOutputParser()
   
   # Fast operations with mini models
   chain = prompt | llm_writing_plan | StrOutputParser()

Fallback Patterns
^^^^^^^^^^^^^^^

The system implements fallback patterns for resilience:

.. code-block:: python

   # Try primary model first
   try:
       result = await run_chain(primary_model, input_data)
   except Exception as e:
       # Fall back to alternative model
       result = await run_chain(fallback_model, input_data)

Retry Patterns
^^^^^^^^^^^^

Important operations implement retry logic:

.. code-block:: python

   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
   async def _try_function(chain, input_data):
       return await chain.ainvoke(input_data)

Performance Monitoring
-------------------

Model performance is monitored using several techniques:

.. code-block:: python

   # Timing decorator
   @async_timer
   async def function_with_timing():
       # Implementation

   # Detailed logging
   log.info(f"Starting {func.__name__} at {start_dt}")
   log.info(f"Finished {func.__name__} in {duration:.2f} seconds")

   # LangSmith integration
   tracer = LangChainTracer(project_name=project_name, client=ls_client)
   callback_manager = CallbackManager([tracer])

Embedding Models
--------------

Vector embeddings use dedicated models:

.. code-block:: python

   # Default embedding model
   openai_embedding = OpenAIEmbeddings()
   
   # Alternative local embedding
   ollama_embedding = OllamaEmbeddings()
   
   # Standard embedding assignment
   embedder = OpenAIEmbeddings()

**Embedding Usage:**

* Document chunking and storage
* Query vectorization
* Similarity search
* Relevance ranking

Model Selection Factors
--------------------

Several factors influence model selection:

1. **Task Complexity**
   * Complex reasoning: GPT-4o, Claude
   * Simple tasks: GPT-4o-mini, faster models

2. **Latency Requirements**
   * Interactive features: Faster models
   * Background processing: More capable models

3. **Specialized Capabilities**
   * Document analysis: Models with long context
   * Creative generation: Models with strong composition

4. **Cost Considerations**
   * Token-efficient models for high-volume tasks
   * Premium models for critical operations

Environmental Configuration
-------------------------

Models are configured through environment variables:

.. code-block:: python

   # environment variable configuration
   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com/"
   # check if the LANGCHAIN_PROJECT environment variable is set, if not, set it to "mytender"
   if "LANGCHAIN_PROJECT" not in os.environ:
       os.environ["LANGCHAIN_PROJECT"] = "mytender_local"
   assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"

**Required Environment Variables:**

* ``OPENAI_API_KEY``: OpenAI API key
* ``ANTHROPIC_API_KEY``: Anthropic API key
* ``GOOGLE_API_KEY``: Google AI key
* ``GROQ_API_KEY``: Groq API key
* ``AWS_ACCESS_KEY_ID``: AWS credentials
* ``AWS_SECRET_ACCESS_KEY``: AWS credentials

Chain Construction
----------------

Models are integrated into LangChain chains:

.. code-block:: python

   # Basic chain construction
   prompt = ChatPromptTemplate.from_template(prompt_text)
   chain = prompt | model | StrOutputParser()
   result = await chain.ainvoke(input_data)
   
   # More complex chain with intermediate steps
   chain = (
       {
           "question": get_question,
       }
       | prompt
       | model
       | StrOutputParser()
   )

Prompt Engineering
----------------

Models receive carefully engineered prompts:

.. code-block:: python

   # Prompt loading
   prompt_text = load_prompt_from_file("prompt_name")
   prompt = ChatPromptTemplate.from_template(prompt_text)
   
   # System messages in ChatPromptTemplate
   prompt = ChatPromptTemplate.from_messages([
       ("system", "You are a business analyst."), 
       ("human", prompt_text)
   ])

For detailed API implementations, see the `config.py` file and associated modules. 