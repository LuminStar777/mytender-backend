mytender.io MongoDB Schema Documentation
==============================

This document explains the schema structure for bid documents stored in the MongoDB database. Each field is documented with its purpose, usage context, and where it is set or updated in the codebase.

Bid Document Structure
---------------------

Below is the structure of a bid document in the MongoDB database:

.. code-block:: javascript

    {
        "_id" : ObjectId("67e676ecb27fa9254524f8ae"),
        "bid_manager" : " ",
        "bid_organisation_id" : "9999-9999-9999",
        "bid_qualification_result" : " ",
        "bid_title" : "Knight Frank The Lumen and Partnership House",
        "client_name" : " ",
        "compliance_requirements" : "Based on the provided tender documents...",
        // Other fields...
    }

Fields Overview
--------------

Core Bid Information
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``_id``
     - MongoDB unique identifier for the bid
     - Created automatically when a new bid is inserted; used in API routes like ``/get_bid/{bid_id}`` to identify specific bids
   * - ``bid_organisation_id``
     - Organization ID that owns this bid
     - Used for filtering bids by organization; set from the user's organization ID in ``upload_bids``
   * - ``bid_title``
     - Title of the bid/tender
     - Displayed in the UI bid list; set via form input in ``upload_bids``
   * - ``timestamp``
     - Last modification timestamp
     - Updated whenever the bid is modified; used to track when changes were made
   * - ``original_creator``
     - User who initially created the bid
     - Preserved when bid is updated; used for permission checking
   * - ``last_edited_by``
     - User who last edited the bid
     - Updated on each modification; tracks who made recent changes

Bid Status and Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``status``
     - Current status of the bid (e.g., "Draft", "Submitted")
     - Used for filtering and displaying status in UI; updated via ``update_bid_status`` endpoint
   * - ``bid_qualification_result``
     - Qualification status of the bid
     - Indicates if the bid passed qualification; updated via ``update_bid_qualification_result``
   * - ``value``
     - Monetary value of the bid
     - Financial tracking; set in ``upload_bids`` form

Client and Opportunity Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``client_name``
     - Name of the client organization
     - Used in proposal generation and UI displays; set via form in ``upload_bids``
   * - ``opportunity_owner``
     - Person responsible for the opportunity
     - Internal tracking; set via form in ``upload_bids``
   * - ``opportunity_information``
     - Detailed information about the opportunity
     - Used in proposal generation; set via form in ``upload_bids``
   * - ``contract_information``
     - Details about the contract
     - Used for legal reference; set via form in ``upload_bids``
   * - ``submission_deadline``
     - Deadline for bid submission
     - Used for timing and notifications; set via form in ``upload_bids``

Bid Content and Structure
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``outline``
     - Structure of the proposal (sections and subsections)
     - Core field for organizing proposal content; accessed via ``get_bid_outline`` and modified via multiple endpoints
   * - ``compliance_requirements``
     - Required compliance items from the tender
     - Used to ensure bid meets all requirements; set via form in ``upload_bids``
   * - ``tender_summary``
     - Summary of the tender details
     - Used for reference and proposal generation; set via form in ``upload_bids``
   * - ``evaluation_criteria``
     - Criteria for bid evaluation
     - Used to target proposal strengths; set via form in ``upload_bids``
   * - ``questions``
     - Questions from or about the tender
     - Used to clarify requirements; set via form in ``upload_bids``

Strategy and Differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``win_themes``
     - Themes that help win the bid
     - Strategic positioning; stored as a JSON array in ``upload_bids``
   * - ``customer_pain_points``
     - Issues the customer is trying to solve
     - Used to target specific customer needs; stored as a JSON array in ``upload_bids``
   * - ``differentiating_factors``
     - Factors that differentiate from competitors
     - Used in proposal generation; stored as a JSON array in ``upload_bids``
   * - ``differentiation_opportunities``
     - Opportunities to stand out
     - Used for strategic content development; set via form in ``upload_bids``
   * - ``derive_insights``
     - Insights from tender analysis
     - Used for strategic content; set via form in ``upload_bids``
   * - ``tone_of_voice``
     - Tone guidance for proposal writing
     - Used for content generation consistency; set via form in ``upload_bids``

Collaboration and References
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``bid_manager``
     - Person managing the bid
     - Contact point for bid; set via form in ``upload_bids``
   * - ``contributors``
     - Dictionary of users contributing to the bid
     - Used for permissions and tracking; stored as a JSON object in ``upload_bids``
   * - ``selectedFolders``
     - Folders selected for this bid
     - Used for organizing related content; stored as a JSON array in ``upload_bids``
   * - ``selectedCaseStudies``
     - Case studies referenced in the bid
     - Used for evidence support in proposals; stored as JSON in ``upload_bids``
   * - ``solution``
     - Proposed solution details
     - Core information for proposal; stored as JSON in ``upload_bids``

Additional Fields
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``new_bid_completed``
     - Flag indicating if initial bid setup is complete
     - Used for UI workflow; set via form in ``upload_bids``
   * - ``tender_library``
     - Stored tender documents (excluded from list queries)
     - Reference materials for the bid process
   * - ``generated_proposal``
     - Generated proposal content (excluded from list queries)
     - Output of the proposal generation
   * - ``generated_proposal_pdf``
     - PDF version of the proposal (excluded from list queries)
     - Downloadable document for submission

User Admin Collection
--------------------

The user_admin collection stores user accounts, organization details, and system preferences. These records are used for authentication, permission control, and user-specific settings.

User Admin Document Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is the structure of a user_admin document:

.. code-block:: javascript

    {
        "_id" : ObjectId("67daa07b3194a696bf247ec8"),
        "email" : "alex@mytender.io",
        "login" : "runningman",
        "password" : "a53108f7543b75adbb34afc035d4cdf6",
        "company" : "alex",
        "jobRole" : "bid winner",
        "stripe_customer_id" : "cus_RyGdQWVa0Ra5Ne",
        "timestamp" : ISODate("2052-08-03T10:44:06.017+0000"),
        "organisation_id" : "9d34767e-c658-4433-b995-26c1a3a153d7",
        "region" : "GB",
        "product_name" : "Standard",
        "userType" : "owner",
        "licenses" : NumberInt(0),
        "forbidden" : "",
        "numbers_allowed_prefixes" : "",
        // Other fields...
    }

User Admin Fields Overview
^^^^^^^^^^^^^^^^^^^^^^^^

Authentication and Identity
""""""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``_id``
     - MongoDB unique identifier for the user
     - Used throughout the system to reference users
   * - ``email``
     - User's email address
     - Used for login, notifications, and as a unique identifier; set during user registration
   * - ``login``
     - Username for authentication
     - Used in login process and for identifying users in the system; referenced in ``verify_user`` function
   * - ``password``
     - Hashed password for authentication
     - Stored as MD5 hash; used in login verification in ``verify_user`` function
   * - ``timestamp``
     - Account creation or last update time
     - Tracks when the user record was created or updated

Organization and Role
""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``organisation_id``
     - Unique identifier for the user's organization
     - Used to group users and bids; referenced in bid access control via ``bid_organisation_id``
   * - ``company``
     - Company/organization name
     - Displayed in UI and used in proposal generation via ``load_user_config``
   * - ``jobRole``
     - User's role in their company
     - Used for informational purposes and in some UI elements
   * - ``userType``
     - Role in the system (owner, admin, writer, reviewer)
     - Determines permissions; used in access control functions
   * - ``region``
     - Geographic region/country code
     - Used for localization and may affect proposal formatting

Subscription and Billing
"""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``stripe_customer_id``
     - Stripe payment system customer ID
     - Used for billing management and subscription tracking
   * - ``product_name``
     - Subscription level/product tier
     - Determines feature access and limits; used in feature gating logic
   * - ``licenses``
     - Number of user licenses for the organization
     - Controls how many users can be added to the organization

Content Generation Settings
"""""""""""""""""""""""""

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Purpose
     - Usage Context
   * - ``forbidden``
     - Words not to be used in generated content
     - A comma-separated list processed in ``post_process_result`` function
   * - ``numbers_allowed_prefixes``
     - Prefixes that allow numbers to be retained
     - A comma-separated list used in ``process_numbers`` function to control numerical formatting
   * - ``company_objectives``
     - Company goals and mission statement
     - Used for content generation to align with company strategy; accessed in ``process_query`` function
   * - ``company_profile``
     - Description of the company
     - Used when generating differentiation opportunities in ``get_differentiation_opportunities``
   * - ``tone_of_voice``
     - Organization's preferred writing tone
     - Used to maintain consistent messaging in generated content

Schema Management
----------------

Field Updates
^^^^^^^^^^^

Most fields are updated through the ``/upload_bids`` endpoint, which handles both creation and updates to existing bids. Some specific fields have dedicated update endpoints:

- ``/update_bid_status`` - Updates the ``status`` field
- ``/update_bid_qualification_result`` - Updates the ``bid_qualification_result`` field
- ``/update_section`` - Updates sections within the ``outline`` field
- ``/update_subheading`` - Updates subheadings within sections in the ``outline`` field

User records are primarily updated through:

- ``/save_user`` - Updates existing user records
- ``/add_user`` - Creates new user records
- ``/update_user_details`` - Updates user profile information
- ``/update_company_info`` - Updates organization-level information

Data Access
^^^^^^^^^^

Bids are retrieved through several endpoints:

- ``/get_bids_list`` - Returns a list of all bids for an organization (excludes large fields like ``tender_library``)
- ``/get_bid/{bid_id}`` - Returns a complete bid document by ID
- ``/get_bid_outline`` - Returns only the outline structure of a bid

User data is accessed through:

- ``/load_user`` - Retrieves user profile data
- ``/get_users`` - Retrieves all users in an organization
- ``/get_organization_users`` - Retrieves users filtered by organization
- ``/profile`` - Retrieves the current user's profile

Permission Model
^^^^^^^^^^^^^^^

Access to bid documents is controlled by organization ID and contributor status:

- Users can only access bids from their organization (``bid_organisation_id`` matches user's ``organisation_id``)
- Contributors listed in the ``contributors`` field have specific edit permissions
- The ``original_creator`` has special permissions
- User types (owner, admin, writer, reviewer) defined in ``userType`` have different levels of access

The ``has_permission_to_access_bid`` function enforces these permissions throughout the API. 