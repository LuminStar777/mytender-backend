from datetime import datetime
import logging
import uuid
from fastapi import HTTPException
from config import admin_collection, account_creation_tokens
from utils import SENDER, create_ses_client, is_same_org, is_user_type

log = logging.getLogger(__name__)


async def process_get_organization_users ( include_pending: bool, current_user: str):
    try:
        user_record = await admin_collection.find_one({"login": current_user})
        if not user_record:
            raise HTTPException(status_code=404, detail="Current user not found")
        organisation_id = user_record.get("organisation_id")
        if not organisation_id:
            raise HTTPException(
                status_code=400, detail="Organisation ID not found for current user"
            )
        # Fetch users from the admin_collection
        admin_users = await admin_collection.find({"organisation_id": organisation_id}).to_list(
            length=None
        )

        all_users = [
            {
                "email": user["email"], 
                "username": user.get("login"), 
                "role": user.get("userType"),
                "company_logo": user.get("company_logo", "")  # Include company logo field
            }
            for user in admin_users
        ]

        # Only fetch pending users if include_pending is True
        if include_pending:
            # Fetch pending users from the account_creation_tokens collection
            pending_users = await account_creation_tokens.find(
                {"organisation_id": organisation_id}
            ).to_list(length=None)

            # Add pending users to the results
            all_users += [
                {
                    "email": user["email"], 
                    "username": "Request Pending", 
                    "role": "Pending",
                    "company_logo": user.get("company_logo", "")
                }
                for user in pending_users
            ]

        return all_users
    except HTTPException as he:
        raise he

async def process_change_user_permissions(current_user: str, target_user: str, new_user_type: str) -> dict:
    """
    Change user permissions for a target user within the same organization.
    
    Args:
        current_user (str): Username of the user making the change (must be owner)
        target_user (str): Username of the user whose permissions will be changed
        new_user_type (str): New user type to assign ("admin", "editor", etc.)
        
    Returns:
        dict: Result of the operation
        
    Raises:
        HTTPException: If permission requirements are not met
    """
    if not await is_user_type(current_user, "owner"):
        raise HTTPException(status_code=403, detail="Only account owners can change user permissions")

    if not await is_same_org(current_user, target_user):
        raise HTTPException(status_code=403, detail="Users must be in the same organisation")

    # Prevent changing your own permissions
    if current_user == target_user:
        raise HTTPException(status_code=403, detail="Cannot change your own permissions")

    # Validate the new_user_type
    allowed_user_types = ["owner", "member", "reviewer"]
    if new_user_type not in allowed_user_types:
        raise HTTPException(status_code=400, detail=f"Invalid user type. Must be one of: {', '.join(allowed_user_types)}")

    # Update the target user's permissions
    result = await admin_collection.update_one(
        {"login": target_user},
        {"$set": {"userType": new_user_type}}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Target user not found or no changes made")

    log.info(f"Changed user permissions for {target_user} to {new_user_type} by {current_user}")

    return {
        "success": True,
        "message": f"User permissions for {target_user} updated to {new_user_type}",
        "target_user": target_user,
        "new_user_type": new_user_type
    }

async def process_set_user_task(
    task_data: dict,
    current_user: str
):
    # Validate required fields
    required_fields = ["name", "bid_id", "index"]
    for field in required_fields:
        if field not in task_data:
            return {"error": f"Missing required field: {field}"}

    # Extract target_user from task_data if provided, otherwise use current_user
    target_user = task_data.get("target_user", current_user)

    # Create task object with required fields
    task = {
        "_id": str(uuid.uuid4()),  # Generate a unique ID for the task
        "name": task_data["name"],
        "bid_id": task_data["bid_id"],
        "index": task_data["index"],
        "created_at": datetime.utcnow(),
        "created_by": current_user,  # Track who created the task
        "priority": task_data.get("priority", "normal")  # Optional priority field
    }

    # Add task to target user's tasks array
    result = await admin_collection.update_one(
        {"login": target_user},
        {"$push": {"tasks": task}}
    )

    if result.modified_count == 0:
        return {"error": f"Failed to add task or user '{target_user}' not found"}

    log.info(f"Added new task for user {target_user} by {current_user}")
    return {"success": True, "message": f"Task added successfully for {target_user}", "task": task}


async def process_delete_user_task(
    task_id: str,
    current_user: str
):
    # Validate that task_id is provided
    if not task_id:
        return {"error": "Missing required field: task_id"}

    # Remove the task from user's tasks array
    result = await admin_collection.update_one(
        {"login": current_user},
        {"$pull": {"tasks": {"_id": task_id}}}
    )

    if result.modified_count == 0:
        return {"error": "Failed to delete task, task not found or user not authorized"}

    log.info(f"Deleted task {task_id} for user {current_user}")
    return {"success": True, "message": "Task deleted successfully"}

async def process_send_organisation_email(
    message: str,
    recipient: str,
    subject: str,
    current_user: str
):
    # Find the sender by email
    sender = await admin_collection.find_one({"login": current_user})
    if not sender:
        return {"error": "Sender user not found", "status_code": 404}

    # Find the recipient by email
    receiver = await admin_collection.find_one({"email": recipient})
    if not receiver:
        return {"error": "Recipient user not found", "status_code": 404}

    # Format the email
    sender_name = sender.get("name", sender.get("login", "A team member"))
    SUBJECT = subject or f"New message from {sender_name}"

    BODY_TEXT = (f"Hello {receiver.get('name', receiver.get('login', ''))}!\n\n"
                f"Message from {sender_name}:\n\n"
                f"{message}\n\n"
                f"Best regards,\nmytender.io")

    BODY_HTML = f"""<html>
    <head></head>
    <body>
        <p>Hello {receiver.get('name', receiver.get('login', ''))}!</p>
        <p>Message from {sender_name}:</p>
        <p>{message}</p>
        <p>Best regards,<br/>mytender.io</p>
    </body>
    </html>
    """

    # Create SES client and send the email
    client = create_ses_client()
    response = client.send_email(
        Destination={'ToAddresses': [recipient]},
        Message={
            'Body': {
                'Html': {'Charset': "UTF-8", 'Data': BODY_HTML},
                'Text': {'Charset': "UTF-8", 'Data': BODY_TEXT},
            },
            'Subject': {'Charset': "UTF-8", 'Data': SUBJECT},
        },
        Source=SENDER,
    )

    log.info(f"Organisation email sent! Message ID: {response['MessageId']}")
    return {"success": True, "message": "Email sent successfully", "message_id": response['MessageId']}
