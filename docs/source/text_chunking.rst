Text Chunking Strategy
====================

This page documents the text chunking strategy used in the Spark AI Chatbot for processing documents before vectorization.

Overview
--------

When documents are uploaded to the system, they need to be broken down into smaller, manageable chunks before being embedded and stored in the vector database. The Spark AI Chatbot uses a semantic chunking approach to ensure that chunks maintain contextual coherence.

Chunking Implementation
----------------------

The system uses the ``MarkdownSplitter`` from the ``semantic_text_splitter`` library, which is specifically designed to break text into semantically meaningful chunks that preserve context.

.. code-block:: python

   from semantic_text_splitter import MarkdownSplitter
   from config import CHUNK_SIZE
   
   def split_text(text):
       splitter = MarkdownSplitter(CHUNK_SIZE)
       return splitter.chunks(text)

Configuration
------------

The chunking strategy is configured through the ``CHUNK_SIZE`` parameter defined in the system's configuration:

.. code-block:: python

   CHUNK_SIZE = (200, 2000)

This tuple configuration specifies:

1. **Minimum Chunk Size**: 200 tokens - ensures chunks aren't too small to lose context
2. **Maximum Chunk Size**: 2000 tokens - ensures chunks aren't too large for efficient processing

Benefits of Semantic Chunking
----------------------------

Unlike simple character or token-based chunking methods, the ``MarkdownSplitter`` has several advantages:

1. **Respects Markdown Structure**: Preserves document structure like headings, lists, and paragraphs
2. **Context Preservation**: Tries to keep related content together in the same chunk
3. **Semantic Boundaries**: Splits at natural semantic boundaries rather than arbitrary character counts
4. **Content Optimization**: Creates chunks that are optimized for semantic search and retrieval

Implementation in Document Processing
------------------------------------

When a document is uploaded through the system's API endpoints, the content passes through the following process:

1. **Document Parsing**: The document (PDF, DOCX, etc.) is parsed to extract text
2. **Text Splitting**: The extracted text is split using the ``split_text()`` function
3. **Embedding Generation**: Each chunk is converted to a vector representation
4. **Vector Storage**: The embeddings are stored in ChromaDB with appropriate metadata

.. code-block:: python

   # From the embedding.py module
   chunks = split_text(text)
   
   # Create metadatas list with both chunk-specific and document metadata
   chunk_metadatas = [
       {
           "mongo_unique_id": unique_id,
           "chunk_index": i,
           **metadata  # Include all passed metadata (filename, upload_date, uploaded_by)
       }
       for i, _ in enumerate(chunks)
   ]
   
   # Add texts and their embeddings to Chroma
   await Chroma.aadd_texts(
       vectorstore,
       chunks,
       metadatas=chunk_metadatas
   )

Impact on Retrieval Quality
--------------------------

The quality of chunking directly impacts the performance of the RAG system:

- **Too Large Chunks**: Can contain too much irrelevant information, diluting the relevance
- **Too Small Chunks**: May lose important context needed for proper understanding
- **Poorly Segmented Chunks**: Can break apart related information, making retrieval less effective

The semantic chunking approach used in the system helps mitigate these issues by creating chunks that preserve semantic meaning and context while remaining within optimal size limits for the language models. 