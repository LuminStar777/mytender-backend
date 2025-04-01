"""
-This file includes utility functions that need to be used multiple times 
across the different apiModules.
-For example get_chroma_directory and get_organisation_contributors will be included here
"""
import hashlib
from typing import List
import boto3
from fastapi import HTTPException
from config import (
    admin_collection,
    AWS_SES_REGION_NAME,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
)

class UserNotFoundException(Exception):
    """Raised when a user is not found in the admin collection."""

def makemd5(key_string):
    """
    Used for encrypting password
    """
    new_key_string = key_string.encode("utf-8")
    return hashlib.md5(new_key_string).hexdigest()

async def get_parent_user(current_user: str) -> str:
    """
    Retrieve parent user. This is used to get the correct chroma directory and 
    mongo documents.

    This function looks up the user in the admin collection and determines
    the parebt yser. If the user has a parent user, it uses
    the parent's directory instead. ie. the user was invited by an organisation

    Args:
        current_user (str): The username of the current user.

    Returns:
        str: The path to the Chroma directory for the user.

    Raises:
        UserNotFoundException: If the user is not found in the admin collection.
    """
    user_doc = await admin_collection.find_one({"login": current_user})
    if not user_doc:
        raise UserNotFoundException("User not found")

    parent_user = user_doc.get("parent_username")
    if not parent_user:
        return current_user
    else:
        return parent_user

async def has_permission_to_access_bid(bid: dict, user: str) -> bool:

    parent_user = await get_parent_user(user)

    parent_user_doc = await admin_collection.find_one({"login": parent_user})
    if not parent_user_doc:
        raise HTTPException(status_code=404, detail="Parent user not found")

    parent_user_organisation_id = parent_user_doc.get("organisation_id")
    if not parent_user_organisation_id: # Handles both non-existent field, None, or empty string
        raise HTTPException(status_code=400, detail="Parent user has missing or empty organisation_id")

    bid_organisation_id = bid.get("bid_organisation_id")

    if parent_user_organisation_id != bid_organisation_id:
        return False

    return True

async def is_user_type(user: str, user_type) -> bool:
    user_doc = await admin_collection.find_one({"login": user})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    userType = user_doc.get("userType")
    if userType == user_type:
        return True

    return False

async def is_same_org(user1: str, user2: str) -> bool:

    user1_doc = await admin_collection.find_one({"login": user1})
    if not user1_doc:
        raise HTTPException(status_code=404, detail="User not found")

    user1_org = user1_doc.get("organisation_id")

    user2_doc = await admin_collection.find_one({"login": user2})
    if not user2_doc:
        raise HTTPException(status_code=404, detail="User not found")

    user2_org = user2_doc.get("organisation_id")

    if user1_org == user2_org:
        return True

    return False

def calculate_word_amounts(selected_choices: List[str], section_word_amount: int) -> List[int]:
    # If no choices, return default word amount
    if len(selected_choices) == 0:
        return [250]

    # Calculate word amounts per subsection
    num_choices = len(selected_choices)

    # Calculate base word count per subsection
    base_word_count = int(section_word_amount) // num_choices

    # Calculate remaining words to distribute
    remaining_words = int(section_word_amount) % num_choices

    # Create word amounts list with even distribution
    word_amounts = [base_word_count] * num_choices

    # Distribute remaining words one by one
    for i in range(remaining_words):
        word_amounts[i] += 1

    return word_amounts

SENDER = 'info@mytender.io'
# Create a new SES resource and specify a region.


def create_ses_client():
    """
    Creates and returns a boto3 SES client.

    Returns:
        boto3.client: A boto3 SES client configured with the specified AWS credentials and region.
    """
    return boto3.client(
        'ses',
        region_name=AWS_SES_REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
