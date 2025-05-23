from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pymongo.collection import Collection
from typing import Optional

# JWT configuration
SECRET_KEY = "YOUR_SECRET_KEY"  # Replace with a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    from db import get_users_collection
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    users_collection = get_users_collection()
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable"
        )
    
    user = users_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    
    return user

def verify_access(user: Dict[str, Any], required_uploader_email: str) -> bool:
    """
    Verify if the authenticated user has access to the uploader's data.
    Admin users can access any data. Normal users can only access their own data.
    """
    user_email = user.get("email", "").lower()
    user_role = user.get("role", "user")
    
    # Admin users can access any uploader's data
    if user_role == "admin":
        return True
    
    # if user_role == "user":
    #     return True
    
    # Normal users can only access their own data
    return user_email == required_uploader_email.lower()