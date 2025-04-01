Company Library
==============

Overview
--------

The Company Library module is a core component of the mytender.io platform that manages organization-specific document storage and retrieval. It enables users to upload, organize, search, and manage proprietary documents that serve as the knowledge base for RAG operations.

Architecture
-----------

The Company Library is built on a hierarchical data structure:

* **Collections**: Top-level organizational units (folders)
* **Documents**: Individual files within collections
* **Chunks**: Segmented and embedded portions of documents for vector search

Key Components
-------------

1. **Document Storage**
   * ChromaDB for vector storage
   * MongoDB for metadata and binary content
   * Hierarchical folder organization

2. **Vector Embedding**
   * Document chunking for optimal retrieval
   * Vector representations using OpenAI embeddings
   * Metadata association for source attribution

3. **Access Control**
   * Organization-level document sharing
   * Role-based permissions (owner vs. member)
   * Content security through user isolation

Core Functionality
----------------

Document Upload and Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library supports multiple document types with specialized processing:

.. code-block:: python

   async def process_create_upload_file(
       file: UploadFile,
       profile_name: str,
       mode: str,
       current_user: str) -> Dict[str, str]:
       # Upload processing logic

**Parameters:**

* ``file``: The file to upload (PDF, Word, Excel, etc.)
* ``profile_name``: Target collection name
* ``mode``: Processing mode ('pdf', 'plain')
* ``current_user``: Username performing the upload

**Implementation Notes:**

* Files undergo specialized parsing based on file type
* Documents are chunked and embedded automatically
* Content is stored in both the target collection and a default collection
* User permissions are verified during upload

File Type Support:

* PDF files: Processed with ``pymupdf4llm``
* Word documents: Processed with ``MarkItDown``
* Excel files: Processed with ``pandas``
* Plain text: Processed directly

Collection Management
^^^^^^^^^^^^^^^^^^^

Collections provide organizational structure for documents:

.. code-block:: python

   async def process_create_upload_folder(
       folder_name: str,
       parent_folder: Optional[str],
       current_user: str) -> Dict[str, str]:
       # Folder creation logic

**Parameters:**

* ``folder_name``: Name of the new folder
* ``parent_folder``: Optional parent folder for nesting
* ``current_user``: Username performing the operation

**Hierarchical Organization:**

* Collections can be nested using the ``FORWARDSLASH`` convention
* Example: ``parent_folderFORWARDSLASHchild_folder``
* Access follows organizational membership rules

Document Retrieval
^^^^^^^^^^^^^^^^

Documents can be retrieved for viewing or processing:

.. code-block:: python

   async def process_get_folder_filenames(
       collection_name: str,
       current_user: str) -> List[Dict[str, any]]:
       # Document listing logic

   async def process_show_file_content(
       profile_name: str,
       current_user: str,
       file_name: str) -> str:
       # Content retrieval logic

**Retrieval Options:**

* Text-based retrieval for editing
* PDF streaming for document viewing
* Metadata-only retrieval for listings

Content Management
^^^^^^^^^^^^^^^

The library provides operations for content manipulation:

.. code-block:: python

   async def process_update_text(
       id: str,
       text: str,
       profile_name: str,
       mode: str,
       current_user: str) -> UpdateResult:
       # Content update logic

   async def process_move_file(
       unique_id: str,
       new_folder: str,
       current_user: str):
       # File movement logic

   async def delete_content_library_item(
       unique_id: str,
       user: str):
       # Item deletion logic

**Content Operations:**

* Document updates maintain vector synchronization
* Content movement preserves accessibility
* Deletion removes content from both ChromaDB and MongoDB

Case Studies Management
^^^^^^^^^^^^^^^^^^^^^

Special collections for case studies have dedicated handling:

.. code-block:: python

   async def process_get_case_studies(current_user: str) -> List[str]:
       # Case studies retrieval logic

   async def process_create_upload_folder_case_studies(
       folder_name: str,
       parent_folder: Optional[str],
       current_user: str) -> Dict[str, str]:
       # Case study folder creation logic

**Case Study Features:**

* Organized in dedicated ``case_studies_collection`` namespace
* Specialized retrieval for proposal generation
* Enhanced metadata for case study attribution

Technical Implementation
----------------------

Folder Structure
^^^^^^^^^^^^^^

Collections use a flattened hierarchy through naming conventions:

* Root collection: ``collection_name``
* Nested collection: ``parent_collectionFORWARDSLASHchild_collection``
* Special collections: Prefixed names (e.g., ``case_studies_collection``)

Document Storage
^^^^^^^^^^^^^

Documents are stored in multiple systems:

* **ChromaDB**: Stores vector embeddings and primary search index
* **MongoDB**: Stores document metadata and binary content
* **Filesystem**: Temporary storage during processing

The binary content handling depends on file type and size:

.. code-block:: python

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

   # Size-contingent binary storage
   MAX_SIZE = 10 * 1024 * 1024  # 10MB limit
   if file_size < MAX_SIZE:
       if mode == 'pdf':
           document["pdf_file_content"] = file_content
       elif format == 'file':
           document["file_content"] = file_content

Permissions and Access Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Access control is enforced at multiple levels:

* **Organizational Boundaries**: Users can only access their organization's documents
* **Role Restrictions**: Only owners can modify the content library
* **Parent User Resolution**: Documents are associated with organization owners

Example permission check:

.. code-block:: python

   if not await is_user_type(current_user, "owner"):
       raise HTTPException(status_code=403, 
                          detail="Only owners can upload to content library")

   parent_user = await get_parent_user(current_user)

Error Handling
^^^^^^^^^^^^

The library implements comprehensive error handling:

* HTTP exceptions with appropriate status codes
* Detailed error messages for client debugging
* Transaction-like operations for data consistency

Example error handling pattern:

.. code-block:: python

   try:
       # Operation logic
   except HTTPException as he:
       # Re-raise HTTP exceptions directly
       raise he
   except Exception as e:
       log.error(f"Unexpected error: {str(e)}")
       raise HTTPException(
           status_code=500,
           detail=f"An unexpected error occurred: {str(e)}"
       )

API Endpoints
-----------

The Company Library exposes several API endpoints:

* ``/get_collections``: List available collections
* ``/create_upload_folder``: Create a new collection
* ``/get_folder_filenames``: List documents in a collection
* ``/uploadfile``: Upload a new document
* ``/show_file_content``: Retrieve document content
* ``/update_text``: Update document content
* ``/delete_template``: Delete a collection
* ``/move_file``: Move a document between collections

Performance Considerations
------------------------

Several factors affect company library performance:

1. **Document Size**: Larger documents require more processing time
2. **Collection Structure**: Nested collections have minor performance overhead
3. **Concurrent Operations**: Multiple uploads may impact system responsiveness
4. **Vector Store Size**: Very large collections may require optimization

Integrations
----------

The Company Library integrates with other system components:

* **RAG Pipeline**: Supplies documents for context retrieval
* **Proposal Generation**: Provides evidence for claim support
* **Case Studies**: Offers organizational success stories
* **Tender Processing**: Supports document comparison and analysis

For implementation details, see the ``api_modules.company_library`` module. 