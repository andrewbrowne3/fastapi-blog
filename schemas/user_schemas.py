from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_admin: bool = False

    @validator('username')
    def username_validation(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(v) > 50:
            raise ValueError('Username must be less than 50 characters')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()

    @validator('password')
    def password_validation(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None

    @validator('username')
    def username_validation(cls, v):
        if v is not None:
            if len(v) < 3:
                raise ValueError('Username must be at least 3 characters long')
            if len(v) > 50:
                raise ValueError('Username must be less than 50 characters')
            if not v.replace('_', '').replace('-', '').isalnum():
                raise ValueError('Username must contain only alphanumeric characters, hyphens, and underscores')
            return v.lower()
        return v

# Questionnaire Schemas
class QuestionnaireData(BaseModel):
    industry: Optional[str] = None
    job_title: Optional[str] = None
    experience_level: Optional[str] = None
    content_goals: Optional[List[str]] = []
    target_audience: Optional[str] = None
    preferred_tone: Optional[str] = None
    content_frequency: Optional[str] = None
    topics_of_interest: Optional[List[str]] = []
    writing_style_preference: Optional[str] = None
    blog_length_preference: Optional[str] = None
    include_images: bool = True
    include_data_visualizations: bool = False
    seo_focus: bool = True

    @validator('experience_level')
    def validate_experience_level(cls, v):
        if v and v not in ['Beginner', 'Intermediate', 'Advanced', 'Expert']:
            raise ValueError('Experience level must be one of: Beginner, Intermediate, Advanced, Expert')
        return v

    @validator('preferred_tone')
    def validate_tone(cls, v):
        valid_tones = ['Professional', 'Casual', 'Technical', 'Conversational', 'Formal', 'Friendly']
        if v and v not in valid_tones:
            raise ValueError(f'Preferred tone must be one of: {", ".join(valid_tones)}')
        return v

    @validator('blog_length_preference')
    def validate_blog_length(cls, v):
        if v and v not in ['Short (500-800 words)', 'Medium (800-1500 words)', 'Long (1500+ words)']:
            raise ValueError('Blog length preference must be Short, Medium, or Long')
        return v

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: datetime
    updated_at: Optional[datetime]
    # Questionnaire fields
    industry: Optional[str] = None
    job_title: Optional[str] = None
    experience_level: Optional[str] = None
    content_goals: Optional[List[str]] = []
    target_audience: Optional[str] = None
    preferred_tone: Optional[str] = None
    content_frequency: Optional[str] = None
    topics_of_interest: Optional[List[str]] = []
    writing_style_preference: Optional[str] = None
    blog_length_preference: Optional[str] = None
    include_images: bool = True
    include_data_visualizations: bool = False
    seo_focus: bool = True
    questionnaire_completed: bool = False
    questionnaire_completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

    @validator('new_password')
    def password_validation(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

# JWT Token Schemas
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenRefresh(BaseModel):
    refresh_token: str

class LoginResponse(BaseModel):
    user: UserResponse
    tokens: Token

# Questionnaire Response Schema
class QuestionnaireResponse(BaseModel):
    message: str
    questionnaire_completed: bool
    completed_at: Optional[datetime] = None 