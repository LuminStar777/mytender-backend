RAG System
==========

This page documents the Retrieval-Augmented Generation (RAG) system used in the Spark AI Chatbot.

Overview
--------

The Retrieval-Augmented Generation (RAG) system combines the power of large language models with information retrieval to generate factual, accurate, and contextually relevant responses. Instead of relying solely on the knowledge contained within the language model, the RAG system enriches prompts with retrieved documents that are relevant to the user's query.

.. image:: _static/rag_architecture.png
   :width: 100%
   :alt: RAG Architecture Diagram

Core Components
--------------

The RAG implementation consists of several key components:

Document Ingestion Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before retrieval can occur, documents must be processed through an ingestion pipeline:

1. **Document Loading**: Supports various file formats (PDF, DOCX, TXT, etc.)
2. **Chunking**: Documents are split into manageable chunks using optimal chunking strategies
3. **Embedding Generation**: Each chunk is transformed into a vector representation
4. **Vector Storage**: Embeddings and their metadata are stored in ChromaDB

Query Processing
^^^^^^^^^^^^^^^

When a user submits a query:

1. **Query Understanding**: The query is analyzed to determine the appropriate retrieval strategy
2. **Query Embedding**: The query is converted to the same vector space as the document chunks
3. **Vector Search**: Similar vectors are retrieved based on relevance to the query
4. **Reranking**: Retrieved chunks are reranked based on additional criteria

RAG Orchestration
^^^^^^^^^^^^^^^^

The orchestration layer manages the integration of retrieved content with prompt generation:

1. **Context Selection**: Chooses the most relevant chunks based on the query
2. **Context Augmentation**: Formats the retrieved context for prompt insertion
3. **Prompt Selection**: Determines which prompt template to use (e.g., 3b, 3b_technical, etc.)
4. **Variable Binding**: Binds retrieved context and user parameters to prompt variables
5. **LLM Invocation**: Calls the language model with the augmented prompt

Advanced RAG Techniques
---------------------

The system implements several advanced RAG techniques:

Multi-Stage Retrieval
^^^^^^^^^^^^^^^^^^^^

For complex queries, a multi-stage retrieval approach is used:

1. **Initial Broad Retrieval**: Retrieves documents with high recall
2. **Query Decomposition**: Breaks down complex queries into sub-queries
3. **Evidence Collection**: Retrieves specific evidence for each sub-query
4. **Evidence Integration**: Combines evidence into a coherent context

Relevance Checking
^^^^^^^^^^^^^^^^^

Not all retrieved documents are equally relevant. The system implements:

1. **Semantic Relevance Scoring**: Uses embeddings to score chunk similarity
2. **Content Filtering**: Filters out irrelevant or low-quality chunks
3. **Diversity Sampling**: Ensures retrieved chunks cover different aspects of the query
4. **Threshold-Based Inclusion**: Uses relevance thresholds to determine inclusion

Implementation Details
--------------------

Code Structure
^^^^^^^^^^^^^

The core RAG functionality is implemented in the following modules:

- ``services/vector_manager.py``: Handles vector storage and retrieval
- ``services/chain.py``: Implements the orchestration logic
- ``services/helper.py``: Provides utility functions for document processing
- ``services/utils.py``: Contains general utility functions

Key Functions
^^^^^^^^^^^

These key functions implement the RAG pipeline:

- ``retrieve_documents``: Retrieves documents from vector storage
- ``check_relevance``: Evaluates the relevance of retrieved documents
- ``process_context``: Formats retrieved documents into prompt context
- ``process_multiple_headers``: Handles multi-topic content generation with RAG

Configuration
^^^^^^^^^^^

The RAG system can be configured with the following parameters:

- ``RETRIEVE_SUBTOPIC_CHUNKS``: Number of chunks to retrieve per sub-topic
- ``RELEVANCE_THRESHOLD``: Minimum relevance score for inclusion
- ``MAX_TOKENS_PER_CHUNK``: Maximum token size for each chunk during retrieval

Performance Considerations
------------------------

The RAG system balances several performance considerations:

- **Latency vs. Quality**: Tuning retrieval parameters for response time
- **Context Length Limits**: Managing prompt size with LLM context windows
- **Vector Search Optimization**: Using efficient indexing for fast retrieval
- **Caching Strategies**: Caching common queries and embeddings

Future Improvements
-----------------

Planned enhancements to the RAG system include:

- **Hybrid Search**: Combining vector search with keyword search
- **Adaptive Retrieval**: Dynamically adjusting retrieval strategy based on query type
- **Entity-Based Retrieval**: Using named entity recognition to improve retrieval
- **Conversational Context**: Maintaining context across multiple conversational turns 