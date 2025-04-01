User Signup and Account Creation
===============================

This document outlines the complete user signup and account creation process in the mytender.io platform, particularly focusing on the integration with Stripe for payment processing.

Signup and Payment Flow
----------------------

The user signup process involves several steps that integrate Stripe payments with user account creation:

1. **Initial Checkout Process**
   
   When a user selects a pricing plan on the website, the application creates a Stripe checkout session with:
   
   * The selected pricing plan (Standard, Advanced, etc.)
   * Applicable discounts or trial periods
   * Success and cancellation redirect URLs
   
   This is handled by the ``create_checkout_session`` function in the website views.

2. **Stripe Payment**
   
   The user completes payment through the Stripe checkout interface. During this process:
   
   * User provides email, payment information, and billing details
   * Stripe processes the payment and creates a customer record
   * Upon successful payment, the user is redirected to a success page

3. **Webhook Processing**
   
   After successful payment, Stripe sends a webhook notification (``checkout.session.completed``) to the application's webhook endpoint (``/collect_stripe_webhook``). The webhook process:
   
   * Verifies the webhook signature using ``STRIPE_WEBHOOK_KEY``
   * Extracts customer information (email, region, Stripe Customer ID)
   * Determines the product tier and number of licenses based on the purchased plan

4. **Account Initialization**
   
   The ``handle_checkout_session_completed`` function initializes the account creation process:
   
   * Generates a unique signup token via ``create_signup_token``
   * Creates an organization ID (UUID) for the new account
   * Stores account details in the ``account_creation_tokens`` collection
   * Sets the user as an organization owner with appropriate license count

5. **Email Invitation**
   
   An invitation email is sent to the user's email address through ``send_invite_email``:
   
   * Contains a personalized link with the signup token
   * Includes information about platform features
   * Uses HTML templates with embedded images for a professional appearance

6. **Account Completion**
   
   When the user clicks the link in the invitation email, they are directed to a signup form where they:
   
   * Enter additional information (username, password, company, job role)
   * Submit the form, which is processed by ``process_update_user_details``
   * The system verifies the token and checks for duplicate usernames/emails
   * Creates a permanent user record in the ``admin_collection`` with all user details
   * Removes the temporary token from the ``account_creation_tokens`` collection

Organization Management
----------------------

After account creation, organization owners can manage their team accounts:

1. **License Management**

   * Each product tier provides a specific number of user licenses
   * Organization owners can see available licenses in their account
   * Licenses are decremented when new team members are invited

2. **Team Member Invitation**

   Organization owners can invite team members through the ``process_invite_user`` function:
   
   * The owner provides the email address of the invitee
   * The system verifies license availability
   * A new signup token is generated and stored with the organization ID
   * An invitation email is sent to the team member
   * When the invitation is accepted, the new user is associated with the same organization

3. **User Types and Permissions**

   * **Owner**: The initial account creator with full administrative rights
   * **Member**: Team members invited by the owner with standard permissions

Technical Implementation
-----------------------

The implementation uses several key components:

1. **Stripe Integration**

   * ``stripe.checkout.Session.create`` for payment processing
   * ``stripe.Webhook.construct_event`` for secure webhook handling
   * Customer ID storage for ongoing subscription management

2. **Token-Based Verification**

   * UUID tokens ensure secure account creation links
   * Tokens are time-limited for security (though currently set to 9999 days)
   * Token validation prevents unauthorized account creation

3. **Email Communication**

   * Amazon SES for reliable email delivery
   * HTML templates with embedded images for professional communication
   * Personalized content based on user information and product details

4. **Database Storage**

   Two primary collections manage the process:
   
   * ``account_creation_tokens``: Temporary storage for signup tokens and initial user data
   * ``admin_collection``: Permanent storage for complete user accounts

Security Considerations
----------------------

The signup process includes several security measures:

1. **Webhook Signature Verification**

   Stripe webhook signatures are verified using a secret key to prevent unauthorized requests.

2. **Token-Based Authentication**

   Account creation requires a valid token, preventing direct account creation without payment.

3. **Duplicate Prevention**

   The system checks for existing emails and usernames to prevent duplicate accounts.

4. **Password Hashing**

   User passwords are hashed using MD5 before storage (though a more secure algorithm would be recommended).

Future Improvements
------------------

Potential enhancements to the signup process:

1. **Enhanced Password Security**

   Replace MD5 with more secure hashing algorithms (bcrypt, Argon2).

2. **More Granular Permissions**

   Expand user types beyond owner/member for more flexible access control.

3. **Better Token Expiration**

   Implement more reasonable token expiration times with clear renewal processes.

4. **Two-Factor Authentication**

   Add optional 2FA during the signup process for enhanced security. 