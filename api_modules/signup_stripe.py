"""
User Management and Email Handling Module
This module provides functionality for user signup/management and Stripe webhook processing

Stripe webhook handling:
process_stripe_webhook: Processes incoming Stripe webhook events.
handle_checkout_session_completed: Handles completed checkout sessions.

User signup and invitation:
create_signup_token: Creates a signup token for new users.
store_token_in_database: Stores signup tokens in the account_creation_tokens collection.
send_invite_email: Sends invitation emails to new users.
process_update_user_details: Updates user details upon signup completion.
process_invite_user: Allows owners to invite new members to their organization.

Password management:
process_forgot_password: Handles password reset requests.

Email utilities:
create_ses_client: Creates an Amazon SES client for sending emails.

External libraries used:
Stripe for payment processing
Amazon SES for email sending
Jinja2 for email template rendering
"""
import logging
import traceback
import os
import uuid
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from datetime import datetime, timedelta
import stripe

from botocore.exceptions import ClientError
from pydantic import BaseModel, EmailStr
from jinja2 import Environment, FileSystemLoader
from fastapi import (
    HTTPException,
    Request,
)
from config import (
    STRIPE_SECRET_KEY,
    STRIPE_WEBHOOK_KEY,
    account_creation_tokens,
    admin_collection,
    load_admin_prompts

)
from utils import SENDER, create_ses_client, makemd5
# pylint: disable=too-many-positional-arguments

stripe.api_key = STRIPE_SECRET_KEY
log = logging.getLogger(__name__)

# Webhook secret
endpoint_secret = STRIPE_WEBHOOK_KEY

async def process_stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            await handle_checkout_session_completed(session)
        else:
            log.warning(f'Unhandled event type: {event["type"]}')
        return {"status": "success"}
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        log.error(f"Invalid payload or signature: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def handle_checkout_session_completed(session):
    email = session.get('customer_details', {}).get('email')
    region = session.get('customer_details', {}).get('address', {}).get('country')
    customer_id = session.get('customer', 'Unknown')  # Get the Stripe Customer ID

    # Retrieve the Checkout Session's line items
    try:
        checkout_session_id = session.get('id')
        line_items = stripe.checkout.Session.list_line_items(checkout_session_id)
    except stripe.error.StripeError as e:
        log.error(f"Error retrieving line items for session {checkout_session_id}: {str(e)}")
        return

    # Determine the product name and licenses
    if line_items and line_items['data']:
        product_name = line_items['data'][0].get('description', 'Unknown')  # or use 'name' if it exists
    else:
        product_name = 'Unknown'

    # Set the number of licenses based on the product name
    if product_name == 'Standard':
        licenses = 0
    elif product_name == 'Advanced':
        licenses = 5
    #elif product_name.lower() == 'enterprise':
    #    licenses = 20
    else:
        licenses = 0  # Default value if the product name doesn't match any known products

    if email:
        token = await create_signup_token(email, region, product_name, customer_id, licenses)
        await send_invite_email(email, token)

async def create_signup_token(email, region, product_name, customer_id, licenses):
    token = str(uuid.uuid4())
    await store_token_in_database(email, token, region, product_name, customer_id, licenses)
    return token

async def store_token_in_database(email, token, region, product_name, customer_id, licenses):
    expiry_days = 9999

    # Generate a unique organisation_id using UUID
    organisation_id = str(uuid.uuid4())

    r = {
        "email": email,
        "signUp_token": token,
        "timestamp": datetime.today() + timedelta(days=int(expiry_days)),
        "region": region,
        "product_name": product_name,
        "stripe_customer_id": customer_id,  # Store Stripe Customer ID
        "licenses": licenses,  # Store the number of licenses
        "userType": "owner",
        "organisation_id": organisation_id  # Add unique organisation_id
    }
    try:
        log.info(f"Checking if user with email {email} exists.")
        existing_user = await account_creation_tokens.find_one({"email": email})

        if existing_user:
            log.info(f"User with email {email} found. Replacing token.")
            await account_creation_tokens.replace_one({"email": email}, r)
        else:
            log.info(f"User with email {email} not found. Inserting new record.")
            await account_creation_tokens.insert_one(r)

        log.info(f"Token for user {email} stored successfully.")
    except Exception as e:
        log.error(f"Error storing token for user {email}: {str(e)}")


async def send_invite_email(email: str, token: str):
    APP_URL = os.getenv('APP_URL', 'http://localhost:5173')
    invite_url = f"{APP_URL}/signup?token={token}"
    SUBJECT = "You're invited to join mytender.io!"
    # Extract username from email
    username = email.split('@')[0]
    log.info(username)
    BODY_TEXT = (f"Hello {username}!\n\n"
                 f"You have been invited to join our platform. "
                 f"To create an account, please visit:\n"
                 f"{invite_url}\n\n"
                 f"Best regards,\nmytender.io")

    # Set up Jinja2 environment
    current_dir = Path(__file__).resolve().parent
    templates_dir = current_dir / 'templates'
    env = Environment(loader=FileSystemLoader(templates_dir))

    # Prepare image variables
    images_dir = current_dir / 'images'
    image_vars = {
        'logo_url': 'cid:mytender.io_badge.png',
        'thumbnail': 'cid:thumbnail.png',
        'user_check_icon': 'cid:user-check-solid-48.png',
        'user_voice_icon': 'cid:user-voice-solid-48.png',
        'book_bookmark_icon': 'cid:book-bookmark-solid-48.png',
        'pen_icon': 'cid:pen-solid-48.png',
        'chat_icon': 'cid:chat-solid-48.png',
        'search_icon': 'cid:search-solid-48.png',
        'file_pdf_icon': 'cid:file-pdf-solid-48.png',
        'file_icon': 'cid:file-solid-48.png',
        'support_chat': 'cid:supportchat.png',
        'newsletter': 'cid:newsletter.png',
        'linkedin': 'cid:linkedin.png'
    }
    # Prepare features data
    features = [
        {
            'icon': image_vars['pen_icon'],
            'title': 'Writing Enhancement',
            'description': 'Highlight responses within the Q&A Generator and improve them with our prompts which will appear in a drop-down list in the Bid Pilot.'
        },
        {
            'icon': image_vars['book_bookmark_icon'],
            'title': 'Content Library',
            'description': 'Upload data to your Content Library and select the folder you want to use for your responses so that the AI gives relevant answers.'
        },
        {
            'icon': image_vars['search_icon'],
            'title': 'AI Web Scraping',
            'description': 'Set the toggle in Bid Pilot to Internet Search and you can ask questions using our web scraping AI model, to pull data through from the internet.'
        },
        {
            'icon': image_vars['file_pdf_icon'],
            'title': 'View PDFs',
            'description': 'When inside the Content Library, click on documents for the ability to view text or PDFs uploaded. You can also edit uploaded data here.'
        },
        {
            'icon': image_vars['chat_icon'],
            'title': 'Custom Prompting',
            'description': 'Highlight responses within the Q&A Generator and make custom changes with the custom prompt, located at the bottom of the Bid Pilot.'
        },
        {
            'icon': image_vars['file_icon'],
            'title': 'Submission Collateral',
            'description': 'After compiling your answers in the Q/A answer sheet, you can use the + sign to create executive summaries and cover letters.'
        },
    ]

    # Load and render HTML template
    template = env.get_template('invite_email.html')
    html_content = template.render(invite_url=invite_url, username=username, features=features, **image_vars)

    # Create a MIME multipart message
    msg = MIMEMultipart('related')
    msg['Subject'] = SUBJECT
    msg['From'] = SENDER
    msg['To'] = email

    # Attach HTML part
    msg_alternative = MIMEMultipart('alternative')
    msg.attach(msg_alternative)

    part_text = MIMEText(BODY_TEXT, 'plain')
    part_html = MIMEText(html_content, 'html')

    msg_alternative.attach(part_text)
    msg_alternative.attach(part_html)

    # Attach images
    for image_file in images_dir.glob('*'):
        with open(image_file, 'rb') as img:
            mime_img = MIMEImage(img.read())
            mime_img.add_header('Content-ID', f'<{image_file.name}>')
            msg.attach(mime_img)

    # Attach PDF
    pdf_files = {
        'library_setup_guide.pdf': 'library_setup_guide.pdf',
        'customer_support.pdf': 'customer_support.pdf'
    }
    for pdf_name, content_id in pdf_files.items():
        pdf_path = current_dir / 'pdfs' / pdf_name
        with open(pdf_path, 'rb') as pdf:
            mime_pdf = MIMEApplication(pdf.read(), _subtype='pdf')
            mime_pdf.add_header('Content-Disposition', 'attachment', filename=pdf_name)
            mime_pdf.add_header('Content-ID', f'<{content_id}>')
            msg.attach(mime_pdf)

    try:
        client = create_ses_client()
        response = client.send_raw_email(
            Source=SENDER,
            Destinations=[email],
            RawMessage={'Data': msg.as_string()}
        )
    except ClientError as e:
        log.error(f"Failed to send invite email: {e.response['Error']['Message']}")
    else:
        log.info(f"Invite email sent! Message ID: {response['MessageId']}")


class UpdateUserRequest(BaseModel):
    token: str
    username: str
    password: str
    company: str
    jobRole: str
    email: EmailStr

async def process_update_user_details(user_data: UpdateUserRequest):
    try:
        log.info("Received request to update user details.")
        log.info(f"Received token: {user_data.token}")

        # Find the record by token
        user_record = await account_creation_tokens.find_one({"signUp_token": user_data.token})
        if not user_record:
            log.info("Token not found or expired.")
            raise HTTPException(status_code=404, detail="Invalid or expired token")

        existing_user = await admin_collection.find_one({"email": user_data.email})
        if existing_user:
            log.info(f"Email '{user_data.email}' is already taken.")
            raise HTTPException(status_code=400, detail="Email already taken")

        existing_user = await admin_collection.find_one({"login": user_data.username})
        if existing_user:
            log.info(f"Username '{user_data.username}' is already taken.")
            raise HTTPException(status_code=400, detail="Username already taken")

        adminuseraccount = await load_admin_prompts()
        hashed_password = makemd5(user_data.password)

        new_user_data = {
            "firstname": user_data.firstname,
            "email": user_data.email,
            "login": user_data.username,
            "password": hashed_password,
            "company": user_data.company,
            "jobRole": user_data.jobRole,
            "stripe_customer_id": user_record["stripe_customer_id"],
            "timestamp": user_record["timestamp"],
            "organisation_id": user_record["organisation_id"],
            "region": user_record["region"],
            "product_name": user_record["product_name"],
            "userType": user_record.get("userType", "member"),
            "licenses": user_record["licenses"],
            "forbidden": adminuseraccount["forbidden"],
            "numbers_allowed_prefixes": adminuseraccount["numbers_allowed_prefixes"],
            "question_extractor": adminuseraccount["question_extractor"],
        }

        # Add parent_username only if it exists in user_record
        if "parent_username" in user_record:
            new_user_data["parent_username"] = user_record["parent_username"]

        await admin_collection.insert_one(new_user_data)
        await account_creation_tokens.delete_one({"signUp_token": user_data.token})

        log.info("User details updated successfully.")
        return {"status": "success", "message": "User details updated successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        log.info(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class InviteRequest(BaseModel):
    email: EmailStr  # Email of the invitee

async def process_invite_user(request: InviteRequest, current_user):
    try:
        log.info(current_user)
        # Fetch the current user record to get the organisation_id, userType, and licenses
        user_record = await admin_collection.find_one({"login": current_user})
        if not user_record:
            raise HTTPException(status_code=404, detail="Current user not found")

        # Check if the user is an owner
        if user_record.get("userType") != "owner":
            raise HTTPException(status_code=403, detail="Permission denied: User is not an owner")

        organisation_id = user_record.get("organisation_id")
        licenses = user_record.get("licenses", 0)
        if not organisation_id:
            raise HTTPException(status_code=400, detail="Organisation ID not found for current user")

        parent_username = user_record.get("login")
        if not parent_username:
            raise HTTPException(status_code=400, detail="Login not found for current user")

        # Ensure the invitee has the same region and product name as the inviter
        region = user_record.get("region")
        product_name = user_record.get("product_name")
        if not region or not product_name:
            raise HTTPException(status_code=400, detail="Region or product name not found for current user")

        # Check if licenses are available
        if licenses <= 0:
            raise HTTPException(status_code=400, detail="No available licenses")

        # Check if the user with the given email already exists
        existing_user = await admin_collection.find_one({"email": request.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email already exists")

        existing_pending = await account_creation_tokens.find_one({"email": request.email})
        if existing_pending:
            raise HTTPException(status_code=400, detail="A pending request already exists for this email")

        # Generate a unique signup token for the invitee
        invitee_token = str(uuid.uuid4())
        expiry_days = 999  # Set a shorter expiry for invite tokens
        expiry_date = datetime.today() + timedelta(days=expiry_days)

        # Store the invitee's token and other details in the database
        invitee_data = {
            "email": request.email,
            "signUp_token": invitee_token,
            "timestamp": expiry_date,
            "organisation_id": organisation_id,
            "parent_username": parent_username,
            "region": region,  # Use the same region as the inviter
            "product_name": product_name,  # Use the same product name as the inviter
            "userType": "member",  # Default user type for invitees
            "stripe_customer_id": "not_account_onwer",
            "licenses": 0
        }

        # Replace the existing record with the new one if it exists, otherwise insert a new record
        await account_creation_tokens.replace_one({"email": request.email}, invitee_data, upsert=True)

        # Deduct one license
        new_licenses = licenses - 1
        await admin_collection.update_one(
            {"login": current_user},
            {"$set": {"licenses": new_licenses}}
        )

        # Send the invite email
        await send_invite_email(request.email, invitee_token)

        return {"status": "success", "message": "Invitation sent successfully"}

    except HTTPException as he:
        log.info(f"HTTP error: {str(he)}")
        raise he
    except Exception as e:
        # Log the full stack trace for the exception
        error_message = f"Error inviting user: {str(e)}"
        log.info(error_message)
        traceback.print_exc()  # This will log.info the full traceback in the console
        raise HTTPException(status_code=500, detail="Internal server error")

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

async def process_forgot_password(request: ForgotPasswordRequest):
    try:
        # Find the user by email
        user = await admin_collection.find_one({"email": request.email})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Generate a unique token
        reset_token = str(uuid.uuid4())
        expiry_time = datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour

        # Store the reset token and its expiry in the database
        await admin_collection.update_one(
            {"email": request.email},
            {"$set": {"reset_token": reset_token, "reset_token_expiry": expiry_time}}
        )

        # Send email with reset link
        APP_URL = os.getenv('APP_URL', 'http://localhost:5173')

        reset_link = f"{APP_URL}/reset_password?token={reset_token}"
        SUBJECT = "Password Reset Request"
        BODY_TEXT = (f"Hello!\n\n"
                     f"You requested a password reset. Click the link below to reset your password:\n"
                     f"{reset_link}\n\n"
                     f"If you did not request this change, please ignore this email.\n\n"
                     f"Best regards,\nmytender.io")
        BODY_HTML = f"""<html>
        <head></head>
        <body>
          <h1>Password Reset Request</h1>
          <p>Hello!</p>
          <p>You requested a password reset.</p>
          <p>To reset your password, click the link below:</p>
          <p><a href="{reset_link}">Reset Password</a></p>
          <p>If you did not request this change, please ignore this email.</p>
          <p>Best regards,<br>mytender.io</p>
        </body>
        </html>
        """

        try:
            client=create_ses_client()
            response = client.send_email(
                Destination={'ToAddresses': [request.email]},
                Message={
                    'Body': {
                        'Html': {'Charset': "UTF-8", 'Data': BODY_HTML},
                        'Text': {'Charset': "UTF-8", 'Data': BODY_TEXT},
                    },
                    'Subject': {'Charset': "UTF-8", 'Data': SUBJECT},
                },
                Source=SENDER,
            )
        except ClientError as e:
            log.error(f"Failed to send password reset email: {e.response['Error']['Message']}")
        else:
            log.info(f"Password reset email sent! Message ID: {response['MessageId']}")

        return {"message": "Password reset email sent successfully"}
    except Exception as e:
        log.info(f"Error processing forgot password request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
