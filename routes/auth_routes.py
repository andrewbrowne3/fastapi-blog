from fastapi import APIRouter, Depends, HTTPException, status
from datetime import timedelta
import logging

from database import get_db_connection
from repositories.user_repository import UserRepository
from services.user_service import UserService
from schemas.user_schemas import UserLogin, LoginResponse, Token, TokenRefresh, UserResponse
from auth.jwt_handler import JWTHandler
from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])

def get_user_service(conn = Depends(get_db_connection)) -> UserService:
    """Dependency to get UserService instance"""
    user_repo = UserRepository(conn)
    return UserService(user_repo)

@router.post("/login", response_model=LoginResponse)
async def login(
    login_data: UserLogin,
    user_service: UserService = Depends(get_user_service)
):
    """Authenticate user and return JWT tokens"""
    try:
        # Authenticate user using email instead of username
        user = user_service.authenticate_user_by_email(login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create tokens
        access_token = JWTHandler.create_access_token(
            data={"sub": user.id, "username": user.username}
        )
        refresh_token = JWTHandler.create_refresh_token(
            data={"sub": user.id, "username": user.username}
        )
        
        return LoginResponse(
            user=UserResponse(**user.to_dict()),
            tokens=Token(
                access_token=access_token,
                refresh_token=refresh_token
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    conn = Depends(get_db_connection)
):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        payload = JWTHandler.verify_token(token_data.refresh_token)
        if payload is None or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        username = payload.get("username")
        
        if not user_id or not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Verify user still exists and is active
        user_repo = UserRepository(conn)
        user = user_repo.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        access_token = JWTHandler.create_access_token(
            data={"sub": user.id, "username": user.username}
        )
        new_refresh_token = JWTHandler.create_refresh_token(
            data={"sub": user.id, "username": user.username}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user = Depends(get_current_user)
):
    """Get current user information"""
    return UserResponse(**current_user.to_dict())

@router.post("/logout")
async def logout():
    """Logout user (client should delete tokens)"""
    return {"message": "Successfully logged out"} 