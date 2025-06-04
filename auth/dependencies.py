from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from database import get_db_connection
from repositories.user_repository import UserRepository
from models.user import User
from .jwt_handler import JWTHandler

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    conn = Depends(get_db_connection)
) -> User:
    """Get current authenticated user from JWT token"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token
        payload = JWTHandler.verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        # Check token type
        if payload.get("type") != "access":
            raise credentials_exception
        
        # Get user ID from token
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Get user from database
        user_repo = UserRepository(conn)
        user = user_repo.get_user_by_id(user_id)
        
        if user is None:
            raise credentials_exception
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception:
        raise credentials_exception

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current admin user"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

# Optional authentication (doesn't raise error if no token)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    conn = Depends(get_db_connection)
) -> Optional[User]:
    """Get current user if token is provided, otherwise return None"""
    
    if not credentials:
        return None
    
    try:
        payload = JWTHandler.verify_token(credentials.credentials)
        if payload is None or payload.get("type") != "access":
            return None
        
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        
        user_repo = UserRepository(conn)
        user = user_repo.get_user_by_id(user_id)
        
        if user and user.is_active:
            return user
        
        return None
        
    except Exception:
        return None 