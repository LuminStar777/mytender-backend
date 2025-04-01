API Reference
============

This page will document the API endpoints for the mytender.io platform.

Routes
------

The API provides various endpoints for interacting with the chatbot system. Below are the key endpoints:

Authentication
^^^^^^^^^^^^^

.. code-block:: python

   @app.post("/login")
   async def login(user: User, Authorize: AuthJWT = Depends())

User Management
^^^^^^^^^^^^^^

.. code-block:: python

   @app.get("/user")
   async def user(Authorize: AuthJWT = Depends())

   @app.post("/get_users")
   async def get_users(request: GetUsersRequest, current_user: str = Depends(get_current_user))

Chatbot Interaction
^^^^^^^^^^^^^^^^^

.. code-block:: python

   @app.post("/copilot", status_code=status.HTTP_200_OK)
   async def copilot_question(
       username: str = Depends(get_current_user),
       input_text: str = Body(...),
       extra_instructions: str = Body(...),
       datasets: List[str] = Body(...),
       copilot_mode: str = Body(...),
       bid_id: Optional[str] = Body(None),
   )

   @app.post("/question", status_code=status.HTTP_200_OK)
   async def question(
       username: str = Depends(get_current_user),
       choice: str = Body(...),
       broadness: str = Body(...),
       input_text: str = Body(...),
       extra_instructions: str = Body(...),
       datasets: List[str] = Body(...),
       bid_id: Optional[str] = Body(None),
       selected_choices: Optional[List[str]] = Body(None),
       word_amount: Optional[int] = Body(default=250),
       compliance_requirements: Optional[List[str]] = Body(None),
   )

Document Management
^^^^^^^^^^^^^^^^^

.. code-block:: python

   @app.post("/uploadfile")
   async def create_upload_file(
       file: UploadFile = File(...),
       profile_name: str = Form(...),
       mode: str = Form(...),
       current_user: str = Depends(get_current_user),
   )

   @app.post("/delete_template")
   async def delete_template(
       profile_name: str = Form(...), 
       current_user: str = Depends(get_current_user)
   ) 