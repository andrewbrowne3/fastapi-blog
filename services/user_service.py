from typing import List, Optional
import logging
from models.user import User
from repositories.user_repository import UserRepository
from schemas.user_schemas import UserCreate, UserUpdate, QuestionnaireData

logger = logging.getLogger(__name__)

class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repo = user_repository

    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user with validation"""
        # Check if username already exists
        if self.user_repo.username_exists(user_data.username):
            raise ValueError(f"Username '{user_data.username}' already exists")
        
        # Check if email already exists
        if self.user_repo.email_exists(user_data.email):
            raise ValueError(f"Email '{user_data.email}' already exists")
        
        # Create user object
        user = User(
            username=user_data.username,
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            is_admin=user_data.is_admin
        )
        
        # Hash password
        user.set_password(user_data.password)
        
        # Save to database
        return self.user_repo.create_user(user)

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.user_repo.get_user_by_id(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.user_repo.get_user_by_username(username)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.user_repo.get_user_by_email(email)

    def get_all_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get all users with pagination"""
        return self.user_repo.get_all_users(limit, offset)

    def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """Update user with validation"""
        # Get current user
        current_user = self.user_repo.get_user_by_id(user_id)
        if not current_user:
            return None
        
        # Prepare updates dictionary
        updates = {}
        
        # Validate username if provided
        if user_data.username is not None:
            if self.user_repo.username_exists(user_data.username, exclude_user_id=user_id):
                raise ValueError(f"Username '{user_data.username}' already exists")
            updates['username'] = user_data.username
        
        # Validate email if provided
        if user_data.email is not None:
            if self.user_repo.email_exists(user_data.email, exclude_user_id=user_id):
                raise ValueError(f"Email '{user_data.email}' already exists")
            updates['email'] = user_data.email
        
        # Add other fields
        if user_data.first_name is not None:
            updates['first_name'] = user_data.first_name
        if user_data.last_name is not None:
            updates['last_name'] = user_data.last_name
        if user_data.is_active is not None:
            updates['is_active'] = user_data.is_active
        if user_data.is_admin is not None:
            updates['is_admin'] = user_data.is_admin
        
        return self.user_repo.update_user(user_id, updates)

    def delete_user(self, user_id: int) -> bool:
        """Delete user"""
        return self.user_repo.delete_user(user_id)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = self.user_repo.get_user_by_username(username)
        if not user:
            return None
        
        if not user.verify_password(password):
            return None
        
        if not user.is_active:
            return None
        
        return user

    def authenticate_user_by_email(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = self.user_repo.get_user_by_email(email)
        if not user:
            return None
        
        if not user.verify_password(password):
            return None
        
        if not user.is_active:
            return None
        
        return user

    def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """Change user password with current password verification"""
        user = self.user_repo.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Verify current password
        if not user.verify_password(current_password):
            raise ValueError("Current password is incorrect")
        
        # Hash new password
        new_password_hash = User.hash_password(new_password)
        
        # Update password in database
        return self.user_repo.update_password(user_id, new_password_hash)

    def activate_user(self, user_id: int) -> Optional[User]:
        """Activate user account"""
        return self.user_repo.update_user(user_id, {'is_active': True})

    def deactivate_user(self, user_id: int) -> Optional[User]:
        """Deactivate user account"""
        return self.user_repo.update_user(user_id, {'is_active': False})

    def make_admin(self, user_id: int) -> Optional[User]:
        """Make user an admin"""
        return self.user_repo.update_user(user_id, {'is_admin': True})

    def update_questionnaire(self, user_id: int, questionnaire_data: QuestionnaireData) -> Optional[User]:
        """Update user questionnaire data"""
        # Get current user
        current_user = self.user_repo.get_user_by_id(user_id)
        if not current_user:
            return None
        
        # Convert Pydantic model to dict
        questionnaire_dict = questionnaire_data.dict(exclude_unset=True)
        
        # Update questionnaire data
        return self.user_repo.update_questionnaire(user_id, questionnaire_dict)

    def remove_admin(self, user_id: int) -> Optional[User]:
        """Remove admin privileges from user"""
        return self.user_repo.update_user(user_id, {'is_admin': False}) 