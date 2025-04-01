Services
========

This section documents the core services that power the mytender.io platform.

Embedding Service
----------------

The embedding service handles the processing of documents into vector embeddings.

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
   )

   async def delete_chroma_entry(unique_id: str, user: str, collection_name: str)
   
   async def delete_chroma_folder(profile_name: str, user: str)

Parser Service
-------------

The parser service extracts content from different file types.

.. code-block:: python

   async def parse_file_content(
       file_content: bytes, 
       file_extension: str, 
       random_filename: str
   ) -> Dict[str, Union[str, int]]
   
   async def parse_pdf(file_path: str) -> str
   
   async def parse_word(file_path: str) -> str
   
   def parse_excel(file_path: str) -> tuple[str, int]

Vector Manager
-------------

The vector manager service handles retrieval operations.

.. code-block:: python

   @async_timer
   async def retrieve_docs(vectorstore, query: str, k: int, parent_user: str) -> List[Dict[str, str]]
   
   @async_timer
   async def query_vectorstore(collection_name, username, query, k, parent_user)
   
   @async_timer
   async def query_collection_and_subfolders(base_collection, username, query, k, parent_user)
   
   async def retrieve_bid_library_docs(
       relevant_query: str, 
       bid_id: str, 
       username: str, 
       k: int
   ) -> List[Dict[str, str]]
   
   async def retrieve_content_library_docs(
       username: str, 
       datasets: List[str], 
       sub_topic: str, 
       k: int = RETRIEVE_SUBTOPIC_CHUNKS
   ) -> List[Dict[str, str]]

Chain Service
------------

The chain service orchestrates the RAG workflow.

.. code-block:: python

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
       compliance_reqs: Optional[List[str]] = None,
       # More parameters...
   ) -> str
   
   async def get_instructions(state: GraphState) -> GraphState
   
   async def get_question(state: GraphState) -> GraphState
   
   @async_timer
   async def retrieve_documents(state: GraphState) -> GraphState
   
   async def check_relevance(state: GraphState) -> GraphState
   
   async def process_context(state: GraphState) -> GraphState
   
   async def process_query(state: GraphState) -> GraphState 