from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import logging

from database import get_db_connection
from repositories.user_repository import UserRepository
from repositories.blog_repository import BlogRepository
from services.user_service import UserService
from services.blog_service import BlogService
from schemas.user_schemas import UserCreate, UserUpdate, UserResponse, PasswordChange, QuestionnaireData, QuestionnaireResponse
from schemas.blog_schemas import BlogCreate, BlogUpdate, BlogResponse, BlogListResponse, BlogSaveRequest
from auth.dependencies import get_current_user, get_current_admin_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])

def get_user_service(conn = Depends(get_db_connection)) -> UserService:
    """Dependency to get UserService instance"""
    user_repo = UserRepository(conn)
    return UserService(user_repo)

def get_blog_service(conn = Depends(get_db_connection)) -> BlogService:
    """Dependency to get BlogService instance"""
    blog_repo = BlogRepository(conn)
    return BlogService(blog_repo)

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
):
    """Create a new user (public endpoint)"""
    try:
        user = user_service.create_user(user_data)
        return UserResponse(**user.to_dict())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@router.get("/", response_model=List[UserResponse])
async def get_users(
    limit: int = 100,
    offset: int = 0,
    user_service: UserService = Depends(get_user_service),
    current_user = Depends(get_current_admin_user)  # Only admins can list all users
):
    """Get all users with pagination (admin only)"""
    try:
        users = user_service.get_all_users(limit=limit, offset=offset)
        return [UserResponse(**user.to_dict()) for user in users]
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service),
    current_user = Depends(get_current_user)
):
    """Get user by ID (users can only see their own profile unless admin)"""
    try:
        # Users can only see their own profile unless they're admin
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        user = user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return UserResponse(**user.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    user_service: UserService = Depends(get_user_service),
    current_user = Depends(get_current_user)
):
    """Update user (users can only update their own profile unless admin)"""
    try:
        # Users can only update their own profile unless they're admin
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        # Non-admin users can't change admin status
        if not current_user.is_admin and user_data.is_admin is not None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot change admin status"
            )
        
        user = user_service.update_user(user_id, user_data)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return UserResponse(**user.to_dict())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service),
    current_user = Depends(get_current_admin_user)  # Only admins can delete users
):
    """Delete user (admin only)"""
    try:
        success = user_service.delete_user(user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )

@router.post("/{user_id}/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    user_id: int,
    password_data: PasswordChange,
    user_service: UserService = Depends(get_user_service),
    current_user = Depends(get_current_user)
):
    """Change user password (users can only change their own password)"""
    try:
        # Users can only change their own password
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        success = user_service.change_password(
            user_id, 
            password_data.current_password, 
            password_data.new_password
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return {"message": "Password changed successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change password for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

@router.post("/{user_id}/questionnaire", response_model=QuestionnaireResponse)
async def submit_questionnaire(
    user_id: int,
    questionnaire_data: QuestionnaireData,
    user_service: UserService = Depends(get_user_service),
    current_user = Depends(get_current_user)
):
    """Submit questionnaire data for a user"""
    try:
        # Users can only submit their own questionnaire
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        user = user_service.submit_questionnaire(user_id, questionnaire_data)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return QuestionnaireResponse(
            message="Questionnaire submitted successfully",
            completed=user.questionnaire_completed,
            completed_at=user.questionnaire_completed_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit questionnaire for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit questionnaire"
        )

@router.get("/{user_id}/questionnaire", response_model=QuestionnaireData)
async def get_questionnaire(
    user_id: int,
    user_service: UserService = Depends(get_user_service),
    current_user = Depends(get_current_user)
):
    """Get questionnaire data for a user"""
    try:
        # Users can only see their own questionnaire unless admin
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        user = user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return QuestionnaireData(
            industry=user.industry,
            job_title=user.job_title,
            experience_level=user.experience_level,
            content_goals=user.content_goals,
            target_audience=user.target_audience,
            preferred_tone=user.preferred_tone,
            content_frequency=user.content_frequency,
            topics_of_interest=user.topics_of_interest,
            writing_style_preference=user.writing_style_preference,
            blog_length_preference=user.blog_length_preference,
            include_images=user.include_images,
            include_data_visualizations=user.include_data_visualizations,
            seo_focus=user.seo_focus
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get questionnaire for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve questionnaire"
        )

@router.get("/{user_id}/questionnaire/status")
async def get_questionnaire_status(
    user_id: int,
    user_service: UserService = Depends(get_user_service),
    current_user = Depends(get_current_user)
):
    """Get questionnaire completion status for a user"""
    try:
        # Users can only see their own status unless admin
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        user = user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "completed": user.questionnaire_completed,
            "completed_at": user.questionnaire_completed_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get questionnaire status for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve questionnaire status"
        )

# Blog Management Endpoints

@router.get("/{user_id}/blogs", response_model=BlogListResponse)
async def get_user_blogs(
    user_id: int,
    limit: int = 100,
    offset: int = 0,
    blog_service: BlogService = Depends(get_blog_service),
    current_user = Depends(get_current_user)
):
    """Get all blogs for a specific user"""
    try:
        # Users can only see their own blogs unless admin
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        blogs = blog_service.get_user_blogs(user_id, limit, offset)
        total = blog_service.count_user_blogs(user_id)
        
        return BlogListResponse(
            blogs=[BlogResponse(**blog.to_dict()) for blog in blogs],
            total=total,
            limit=limit,
            offset=offset
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get blogs for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve blogs"
        )

@router.post("/{user_id}/blogs", response_model=BlogResponse, status_code=status.HTTP_201_CREATED)
async def create_user_blog(
    user_id: int,
    blog_data: BlogCreate,
    blog_service: BlogService = Depends(get_blog_service),
    current_user = Depends(get_current_user)
):
    """Create a new blog for a user"""
    try:
        # Users can only create blogs for themselves
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        blog = blog_service.create_blog(user_id, blog_data)
        return BlogResponse(**blog.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create blog for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create blog"
        )

@router.get("/{user_id}/blogs/{blog_id}", response_model=BlogResponse)
async def get_user_blog(
    user_id: int,
    blog_id: int,
    blog_service: BlogService = Depends(get_blog_service),
    current_user = Depends(get_current_user)
):
    """Get a specific blog for a user"""
    try:
        # Users can only see their own blogs unless admin
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        blog = blog_service.get_blog_by_id(blog_id, user_id)
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found"
            )
        
        return BlogResponse(**blog.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get blog {blog_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve blog"
        )

@router.put("/{user_id}/blogs/{blog_id}", response_model=BlogResponse)
async def update_user_blog(
    user_id: int,
    blog_id: int,
    blog_data: BlogUpdate,
    blog_service: BlogService = Depends(get_blog_service),
    current_user = Depends(get_current_user)
):
    """Update a specific blog for a user"""
    try:
        # Users can only update their own blogs
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        blog = blog_service.update_blog(blog_id, user_id, blog_data)
        if not blog:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found"
            )
        
        return BlogResponse(**blog.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update blog {blog_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update blog"
        )

@router.delete("/{user_id}/blogs/{blog_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_blog(
    user_id: int,
    blog_id: int,
    blog_service: BlogService = Depends(get_blog_service),
    current_user = Depends(get_current_user)
):
    """Delete a specific blog for a user"""
    try:
        # Users can only delete their own blogs
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        success = blog_service.delete_blog(blog_id, user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Blog not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete blog {blog_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete blog"
        ) 