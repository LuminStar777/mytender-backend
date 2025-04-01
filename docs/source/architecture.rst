Architecture
============

This page documents the architecture of the mytender.io platform.

System Overview
--------------

The mytender.io platform is built as a microservices-based system that leverages advanced language models and retrieval-augmented generation (RAG) to create high-quality tender responses. The architecture consists of several key components:

1. **API Layer**: Handles incoming requests and routes them to appropriate services
2. **Authentication & User Management**: Manages user sessions and permissions
3. **Vector Database Layer**: Stores embeddings for efficient semantic search
4. **Content Orchestration**: Coordinates the generation of content based on prompts and retrieved information
5. **Prompt Management System**: Handles specialized prompts for different use cases

Component Diagram
----------------

Below is a simplified component diagram of the mytender.io platform:

.. code-block::

    ┌───────────────┐      ┌───────────────────┐      ┌─────────────────┐
    │               │      │                   │      │                 │
    │   Frontend    │◄────►│    API Gateway    │◄────►│  Authentication │
    │               │      │                   │      │                 │
    └───────────────┘      └───────────────────┘      └─────────────────┘
                                    ▲
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
        ┌───────────▼─────┐ ┌───────▼────────┐ ┌───▼──────────────┐
        │                 │ │                │ │                  │
        │  Bid Library    │ │  Content       │ │  Content         │
        │  Management     │ │  Management    │ │  Generation      │
        │                 │ │                │ │                  │
        └─────────────────┘ └────────────────┘ └──────────────────┘
                 ▲                  ▲                   ▲
                 │                  │                   │
        ┌────────┴──────┐  ┌────────┴───────┐  ┌───────┴──────────┐
        │               │  │                │  │                  │
        │   Vector DB   │  │   Document     │  │  LLM Service     │
        │   (ChromaDB)  │  │   Processing   │  │                  │
        │               │  │                │  │                  │
        └───────────────┘  └────────────────┘  └──────────────────┘

Key Services
-----------

API Gateway
^^^^^^^^^^^

The API Gateway serves as the entry point for client applications. It:

- Routes requests to appropriate microservices
- Handles request validation
- Implements rate limiting 
- Manages API versioning

Authentication Service
^^^^^^^^^^^^^^^^^^^^^

The Authentication Service manages:

- User authentication using JWT tokens
- User registration and profile management
- Role-based authorization

Content Management
^^^^^^^^^^^^^^^^^

The Content Management Service handles:

- Document upload and processing
- Vector embedding generation
- Metadata management
- Content library organization

Vector Database
^^^^^^^^^^^^^^

The Vector Database (built on ChromaDB) stores:

- Document embeddings for semantic search
- Metadata about documents
- Reference information for retrieval

Content Generation
^^^^^^^^^^^^^^^^^

The Content Generation Service:

- Processes user queries
- Retrieves relevant context from the vector database
- Applies appropriate prompts based on the query type
- Orchestrates LLM calls for content generation
- Post-processes generated content

Technology Stack
---------------

The system uses the following technologies:

- **Backend**: Python with FastAPI
- **Vector Database**: ChromaDB
- **Language Models**: OpenAI's GPT models, Anthropic Claude
- **Embedding Models**: OpenAI embedding models
- **Document Processing**: Langchain
- **Authentication**: JWT-based auth
- **Deployment**: Docker, AWS

Data Flow
---------

1. **Document Ingestion**:
   - Documents are uploaded through the API
   - Documents are processed and chunked
   - Chunks are embedded and stored in the vector database

2. **Query Processing**:
   - User submits a query with parameters
   - Relevant documents are retrieved from the vector database
   - Context is generated and fed into the appropriate prompt
   - LLM generates a response
   - Response is post-processed and returned to the user 