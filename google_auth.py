# FILE: google_auth.py
import os
import logging # Added for debugging
from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.config import Config
import secrets
from datetime import timedelta
from typing import Optional
from pymongo.collection import Collection # Ensure this is used if needed for type hints, else remove
from auth_utils import create_access_token

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # You can set this in main.py too

router = APIRouter()

# Configuration for OAuth
config = Config() # You might not need this if directly configuring oauth object.
oauth = OAuth(config) # If config object is empty, this is okay.

# --- Environment Variables for OAuth and Frontend ---
GOOGLE_CLIENT_ID = "33766377350-ebnusnahk1gh5b6ji5l2vansma2h11jb.apps.googleusercontent.com"  # Replace with your Google client ID
GOOGLE_CLIENT_SECRET = "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"  # Replace with your Google client secret
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000") # Your Next.js frontend URL
FRONTEND_URL = os.getenv("FRONTEND_URL", "skill-visa-three.vercel.app") # Your Next.js frontend URL

# The REDIRECT_URI that Google calls back to *your backend*.
# This MUST be registered in Google Cloud Console as an "Authorized redirect URI".
# For your local setup, this is typically http://localhost:8000/auth/google/callback
# This global variable isn't directly used in the `authorize_redirect` below,
# as `request.url_for` dynamically generates it, but it's good for documentation.
# BACKEND_REDIRECT_URI_COMMENT = "http://localhost:8000/auth/google/callback"


if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    logger.error("GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET environment variables are not set.")
    # Depending on your setup, you might want to raise an exception here or handle it.
    # For now, it will proceed and likely fail at runtime if these are missing.

# Configure Google OAuth
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

@router.get("/login/google")
async def login_via_google(request: Request):
    # Generate a secure random state for CSRF protection
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state

    # This redirect_uri is where Google will send the user back *to your backend*
    # after they authenticate with Google.
    # `request.url_for("auth_via_google")` will correctly generate something like
    # "http://localhost:8000/auth/google/callback" if your backend runs on port 8000.
    backend_callback_uri = request.url_for("auth_via_google")
    logger.info(f"Generated backend callback URI for Google: {backend_callback_uri}")

    return await oauth.google.authorize_redirect(request, str(backend_callback_uri), state=state)

@router.get("/auth/google/callback")
async def auth_via_google(request: Request):
    from db import get_users_collection # Assuming db.py has this function
    from datetime import datetime

    try:
        # Verify state to prevent CSRF attacks
        session_state = request.session.get("oauth_state")
        query_state = request.query_params.get("state")

        if not session_state or session_state != query_state:
            logger.warning(f"Invalid state parameter. Session: '{session_state}', Query: '{query_state}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state parameter"
            )

        # Get token from Google
        logger.info("Attempting to authorize access token from Google.")
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")
        logger.info(f"Received userinfo from Google: {user_info.get('email') if user_info else 'None'}")


        if not user_info or not user_info.get("email"):
            logger.error("Could not fetch user information or email from Google.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not fetch user information from Google"
            )

        # Extract user info
        email = user_info["email"]
        name = user_info.get("name", "")
        picture = user_info.get("picture", "")

        # Check if user exists in database, if not create a new one
        users_collection = get_users_collection()
        if users_collection is None:
            logger.error("Database service unavailable (users_collection is None).")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service unavailable"
            )

        user = users_collection.find_one({"email": email})

        if not user:
            logger.info(f"User '{email}' not found. Creating new user.")
            user_data = {
                "email": email,
                "name": name,
                "picture": picture,
                "role": "user",  # Default role
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            }
            users_collection.insert_one(user_data)
        else:
            logger.info(f"User '{email}' found. Updating last login.")
            users_collection.update_one(
                {"email": email},
                {"$set": {"last_login": datetime.utcnow()}}
            )

        # Create access token
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )
        logger.info(f"Access token created for user '{email}'.")

        # Redirect to frontend with token using an ABSOLUTE URL
        final_redirect_url_success = f"{FRONTEND_URL}/auth/auth-success?token={access_token}"
        logger.info(f"Redirecting to frontend success page: {final_redirect_url_success}")
        return RedirectResponse(url=final_redirect_url_success)

    except OAuthError as error:
        error_message = error.error or "Unknown OAuth error" # Ensure error_message has a value
        final_redirect_url_oauth_error = f"{FRONTEND_URL}/auth-error?error={error_message}"
        logger.error(f"OAuthError during Google auth: {error_message}. Redirecting to: {final_redirect_url_oauth_error}", exc_info=True)
        return RedirectResponse(url=final_redirect_url_oauth_error)
    except Exception as e:
        # It's good to log the full exception for debugging
        logger.exception(f"An unexpected error occurred during Google auth for request: {request.url}")
        error_message_generic = str(e)
        final_redirect_url_exception = f"{FRONTEND_URL}/auth-error?error=An unexpected error occurred: {error_message_generic}"
        return RedirectResponse(url=final_redirect_url_exception)