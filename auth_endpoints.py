from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Dict, Any
from datetime import datetime, timedelta
from pymongo.collection import Collection
from auth_utils import create_access_token, get_current_user

router = APIRouter()

@router.post("/token", response_model=Dict[str, str])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    from db import get_users_collection
    
    users_collection = get_users_collection()
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable"
        )
    
    # For non-OAuth login (e.g., for testing or internal use)
    user = users_collection.find_one({"email": form_data.username})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a real application, you would verify the password here
    # For simplicity, we're skipping password verification
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    # Update last login time
    from datetime import datetime
    users_collection.update_one(
        {"email": user["email"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=Dict[str, Any])
async def read_users_me(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get information about the currently logged-in user."""
    # Remove sensitive information
    user_info = {
        "email": current_user.get("email"),
        "name": current_user.get("name"),
        "role": current_user.get("role", "user"),
        "picture": current_user.get("picture")
    }
    return user_info