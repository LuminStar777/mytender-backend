mytender.io API Endpoints
=======================

This document provides a comprehensive reference to all API endpoints available in the mytender.io platform. It covers authentication, user management, bid processing, proposal generation, and all other API functions.

Authentication
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /login``
     - Authenticates a user and returns JWT tokens for authorization
   * - ``POST /get_login_email``
     - Returns the email associated with the current authenticated user
   * - ``GET /user``
     - Returns the authenticated user information
   * - ``POST /refresh``
     - Refreshes the JWT access token
   * - ``POST /forgot_password``
     - Sends a password reset link to user's email

User Management
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /add_user``
     - Creates a new user account
   * - ``POST /save_user``
     - Updates an existing user account
   * - ``POST /get_users``
     - Returns a list of all users
   * - ``POST /get_organization_users``
     - Returns users filtered by organization ID
   * - ``POST /profile``
     - Returns the current user's profile information
   * - ``POST /update_user_details``
     - Updates profile information for a user
   * - ``POST /update_company_info``
     - Updates organization-level information
   * - ``POST /change_user_permissions``
     - Modifies user role and permissions
   * - ``POST /delete_user``
     - Removes a user account
   * - ``POST /invite_user``
     - Sends invitation to join the platform

Bid Management
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /upload_bids``
     - Creates a new bid or updates an existing one
   * - ``POST /get_bids_list``
     - Returns a list of all bids for the user's organization
   * - ``GET /get_bid/{bid_id}``
     - Returns complete details for a specific bid
   * - ``DELETE /delete_bid/{bid_id}``
     - Deletes a bid permanently
   * - ``POST /update_bid_status``
     - Updates the status of a bid
   * - ``POST /update_bid_qualification_result``
     - Updates bid qualification result
   * - ``POST /get_bid_outline``
     - Returns the outline structure of a bid
   * - ``POST /update_section``
     - Updates a section within a bid's outline
   * - ``POST /add_section``
     - Adds a new section to a bid's outline
   * - ``POST /update_subheading``
     - Updates a subheading within a bid section
   * - ``POST /add_subheading``
     - Adds a new subheading to a bid section
   * - ``POST /get_section_content``
     - Returns the content of a bid section
   * - ``POST /rewrite_section``
     - Rewrites a bid section with AI enhancement

Content Libraries
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /get_collections``
     - Returns available content collections
   * - ``POST /get_case_studies``
     - Returns available case studies
   * - ``POST /create_upload_folder``
     - Creates a new content folder
   * - ``POST /create_upload_folder_case_studies``
     - Creates a new case studies folder
   * - ``POST /create_upload_file``
     - Uploads a file to the content library
   * - ``POST /create_upload_text``
     - Adds text content to the library
   * - ``POST /show_file_content``
     - Returns content of a file
   * - ``POST /show_file_content_pdf_format``
     - Returns PDF content as viewable format
   * - ``POST /update_text``
     - Updates text content in the library
   * - ``POST /get_folder_filenames``
     - Lists files within a folder
   * - ``POST /move_file``
     - Moves a file between folders
   * - ``POST /delete_content_library_item``
     - Removes an item from the content library

Tender Document Management
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /upload_tender_documents``
     - Uploads documents for a tender
   * - ``POST /get_tender_documents``
     - Returns tender documents for a bid
   * - ``POST /find_matching_document_snippets``
     - Searches for relevant content in tender docs
   * - ``POST /ask_tender_library_question``
     - Asks AI to answer questions based on tender docs

Proposal Generation
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /generate_proposal``
     - Generates a complete proposal for a bid
   * - ``POST /generate_outline``
     - Creates an initial proposal outline structure
   * - ``POST /generate_writing_plans_for_section``
     - Creates writing plans for a specific section
   * - ``POST /generate_cover_letter``
     - Generates a cover letter for a bid
   * - ``POST /regenerate_writingplans_and_subheadings``
     - Recreates writing plans and subheadings
   * - ``POST /regenerate_single_subheading``
     - Regenerates a specific subheading
   * - ``POST /remove_references``
     - Removes citation references from proposal
   * - ``POST /generate_docx``
     - Converts proposal to DOCX format

AI Assistance
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /question``
     - Asks a general question to the AI
   * - ``POST /copilot``
     - Interacts with AI copilot for bid writing assistance
   * - ``POST /get_compliance_requirements``
     - Gets compliance requirements from documents
   * - ``POST /get_opportunity_information``
     - Extracts opportunity information from documents
   * - ``POST /get_differentiation_opportunities``
     - Identifies differentiation opportunities for a bid
   * - ``POST /get_tender_insights``
     - Gets AI insights on tender documents
   * - ``POST /get_exec_summary``
     - Generates an executive summary
   * - ``POST /assign_insights_to_question``
     - Maps tender insights to specific questions

Templates
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /get_templates_for_user``
     - Returns templates available for a user
   * - ``POST /save_template``
     - Saves a new template
   * - ``POST /delete_template``
     - Deletes a template

Feedback and Logging
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /feedback``
     - Submits user feedback
   * - ``POST /log_query``
     - Logs a user query for analytics

Other Utilities
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``POST /stripe_webhook``
     - Handles Stripe payment webhooks
   * - ``POST /generate_mermaid_diagram``
     - Creates Mermaid diagrams from text descriptions
   * - ``POST /transform_text_to_mermaid``
     - Converts text to Mermaid diagram syntax

Authentication and Authorization
-------------------------------

Most endpoints require JWT authentication. Authorization flow works as follows:

1. User authenticates with email/password via ``/login``
2. Server returns JWT access token and refresh token
3. Access token is included in the Authorization header for subsequent requests
4. Expired tokens can be refreshed using the ``/refresh`` endpoint

Permission levels are enforced based on the user's ``userType`` field:

* **owner** - Organization owner with full access
* **admin** - Administrator with management access
* **writer** - Can create and edit bids
* **reviewer** - Can view and provide feedback

For bid-specific operations, the ``has_permission_to_access_bid`` function verifies:

1. The bid belongs to the user's organization
2. The user has appropriate role-based permissions
3. The user is listed as a contributor if required 