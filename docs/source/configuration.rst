Configuration
=============

Overview
--------

The mytender.io platform uses a centralized configuration system that controls system behavior, integrations, and customization options. This document describes the key configuration components, their parameters, and how they affect platform operation.

Configuration Architecture
-----------------------

The configuration system consists of several components:

* **Environment Variables**: External configuration through OS variables
* **Configuration Module**: Centralized settings in ``config.py``
* **User Configuration**: Organization-specific settings in MongoDB
* **Runtime Configuration**: Dynamic settings applied during operation

Core Configuration File
---------------------

The primary configuration is defined in ``config.py``:

.. code-block:: python

   # Key constants
   RETRIEVE_SUBTOPIC_CHUNKS = 20
   RELEVANCE_THRESHOLD = 6  # Threshold for including chunks in results (0-10 scale)
   
   # Model definitions
   GPT_DEFAULT_MODEL = "gpt-4o"
   GPT_FAST_MODEL = "gpt-4o-mini"
   
   # Operational parameters
   CHUNK_SIZE = (200, 2000)
   MAX_TITLE_REQUESTS = 10000

**Configuration Categories:**

* Model assignments
* Operational constants
* Integration parameters
* Storage locations
* Security settings

Environment Variables
-------------------

The system relies on environment variables for sensitive configuration:

.. code-block:: python

   load_dotenv()
   
   # LangChain configuration
   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com/"
   
   # Critical API keys
   assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"
   STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY_LIVE")
   STRIPE_WEBHOOK_KEY = os.environ.get("STRIPE_WEBHOOK_SECRET_LIVE")
   
   # AWS configuration
   AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
   AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
   AWS_SES_REGION_NAME = os.environ.get("AWS_SES_REGION_NAME")
   AWS_SES_REGION_ENDPOINT = os.environ.get("AWS_SES_REGION_ENDPOINT")

**Required Environment Variables:**

1. **API Credentials**
   * ``OPENAI_API_KEY``: OpenAI API key
   * ``ANTHROPIC_API_KEY``: Anthropic API key
   * ``GOOGLE_API_KEY``: Google Generative AI key
   * ``GROQ_API_KEY``: Groq API key
   * ``PPLX_API_KEY``: Perplexity API key

2. **Service Integration**
   * ``MONGO_PASSWORD``: MongoDB authentication
   * ``MONGO_USERNAME``: MongoDB username
   * ``AWS_ACCESS_KEY_ID``: AWS access key
   * ``AWS_SECRET_ACCESS_KEY``: AWS secret key
   * ``STRIPE_SECRET_KEY_LIVE``: Stripe integration

3. **Application Settings**
   * ``MASTER_PASSWORD``: Admin authentication
   * ``LANGCHAIN_API_KEY``: LangSmith integration
   * ``SECRET_KEY``: JWT token signing

Storage Configuration
-------------------

The system uses multiple storage mechanisms with specific configurations:

.. code-block:: python

   # Vector storage
   CHROMA_FOLDER = os.environ.get("CHROMA_FOLDER", 'chroma_db')
   
   # MongoDB collections
   DOC_ENTRIES_COLLECTION = os.environ.get("DOC_ENTRIES_COLLECTION", 'doc_entries_local')
   USER_ADMIN_COLLECTION = os.environ.get("USER_ADMIN_COLLECTION", 'user_admin_local')
   BIDS_COLLECTION = os.environ.get("BIDS_COLLECTION", 'bids_local')
   FILE_COLLECTION = os.environ.get("FILE_COLLECTION", 'files_local')
   
   # Database connection
   HOSTNAME = "44.208.84.199"
   MONGO_DB = "spark"  # The database you want to connect to
   MONGO_AUTH_DB = "admin"  # Database where the user is authenticated
   connection_string = (
       f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{HOSTNAME}:27017/{MONGO_DB}"
       f"?authSource={MONGO_AUTH_DB}"
   )

**Storage Parameters:**

* ``CHROMA_FOLDER``: Location of vector database files
* Collection names for various data types
* MongoDB connection parameters
* GridFS configuration for binary storage

MongoDB Collections
^^^^^^^^^^^^^^^^^

The system uses several MongoDB collections:

.. code-block:: python

   # Initialize collections
   mongo_client = AsyncIOMotorClient(connection_string)
   mongo_db = mongo_client[MONGO_DB]
   
   admin_collection = mongo_db[USER_ADMIN_COLLECTION]
   doc_entry_collection = mongo_db[DOC_ENTRIES_COLLECTION]
   queries_collection = mongo_db["queries"]
   feedback_collection = mongo_db["feedback"]
   template_collection = mongo_db["templates"]
   bids_collection = mongo_db[BIDS_COLLECTION]
   account_creation_tokens = mongo_db["account_creation_tokens"]

**Collection Purposes:**

* ``admin_collection``: User profiles and organization settings
* ``doc_entry_collection``: Document metadata and binary content
* ``queries_collection``: Query history and results
* ``feedback_collection``: User feedback for improvements
* ``bids_collection``: Tender and proposal information
* ``template_collection``: Reusable content templates

User Configuration
----------------

Organization-specific settings are stored in MongoDB:

.. code-block:: python

   async def load_user_config(login):
       """Load user configuration"""
       user_config = await admin_collection.find_one({"login": login})
       return user_config

**User Configuration Parameters:**

* ``company``: Organization name
* ``company_objectives``: Business USPs and objectives
* ``forbidden``: Words to filter from responses
* ``company_profile``: Organization profile information
* ``numbers_allowed_prefixes``: Number handling configuration
* Prompt customizations for various operations

System-Wide Settings
------------------

Global behavior is controlled through specific parameters:

1. **Vector Retrieval Configuration**
   * ``RETRIEVE_SUBTOPIC_CHUNKS``: Number of chunks for subqueries
   * ``RELEVANCE_THRESHOLD``: Minimum relevance score (0-10)
   * ``CHUNK_SIZE``: Document chunking parameters

2. **Security Configuration**
   * ``SECRET_KEY``: JWT token signing key
   * ``ALGORITHM``: JWT signing algorithm ("HS256")
   * Authentication parameters

3. **Rate Limiting**
   * Provider-specific RPS limits
   * Burst capacity settings
   * Timeout configurations

Performance Configuration
----------------------

Performance-related settings control system behavior:

.. code-block:: python

   # Rate limiters
   bedrock_rate_limiter = LoggingRateLimiter(
       name="Bedrock",
       requests_per_second=3.0,  # Reduced from 50 to 3 RPS (180 RPM)
       max_bucket_size=5,        # Reduced burst capacity
       check_every_n_seconds=0.01, # Keep fast checks
   )
   
   # Batch processing
   BATCH_SIZE = 50  # Adjust based on your LLM service limits

**Performance Parameters:**

* Provider-specific rate limits
* Batch processing sizes
* Timeout durations for operations
* Retry configurations

Dynamic Configuration
------------------

Some settings are determined at runtime:

.. code-block:: python

   # Dynamic collection path
   tender_library_collection = f'tender_library_{bid_id}'
   
   # Dynamic path computation
   chroma_db_directory = f"{CHROMA_FOLDER}/{parent_user}"
   
   # Conditional model selection
   if prompt_type == "generate_derive_insights":
       summary_prompt = ChatPromptTemplate.from_template(
           load_prompt_from_file("derive_insights_summary")
       )
       relevant_query = load_query("derive_insights_query")

Configuration Loading
------------------

Configuration is loaded using specific patterns:

.. code-block:: python

   # User configuration loading
   user_config = await load_user_config(parent_user)
   company_name = user_config.get('company', '[COMPANY NAME]')
   
   # Prompt loading
   prompt_text = load_prompt_from_file("generate_outline")
   prompt = ChatPromptTemplate.from_template(prompt_text)
   
   # Dynamic query loading
   relevant_query = load_query("summarise_tender_query")

Future Configuration Enhancements
------------------------------

Planned improvements to the configuration system:

1. **Enhanced Environment Management**
   * Better separation of development and production settings
   * Simplified local development configuration
   * Containerized deployment configuration

2. **Configuration Validation**
   * Runtime validation of configuration values
   * Automatic error detection for misconfiguration
   * Configuration documentation generation

3. **Dynamic Reconfiguration**
   * Hot-reloading of certain configuration parameters
   * Admin interface for configuration management
   * Tenant-specific configuration isolation

For detailed implementation, see the ``config.py`` file and associated modules. 