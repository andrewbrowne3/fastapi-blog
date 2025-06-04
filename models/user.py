from datetime import datetime
from typing import Optional, Dict, Any, List
from passlib.context import CryptContext
import json

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User:
    def __init__(
        self,
        id: Optional[int] = None,
        username: str = "",
        email: str = "",
        password_hash: str = "",
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        is_active: bool = True,
        is_admin: bool = False,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        # Questionnaire fields
        industry: Optional[str] = None,
        job_title: Optional[str] = None,
        experience_level: Optional[str] = None,
        content_goals: Optional[List[str]] = None,
        target_audience: Optional[str] = None,
        preferred_tone: Optional[str] = None,
        content_frequency: Optional[str] = None,
        topics_of_interest: Optional[List[str]] = None,
        writing_style_preference: Optional[str] = None,
        blog_length_preference: Optional[str] = None,
        include_images: bool = True,
        include_data_visualizations: bool = False,
        seo_focus: bool = True,
        questionnaire_completed: bool = False,
        questionnaire_completed_at: Optional[datetime] = None
    ):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.first_name = first_name
        self.last_name = last_name
        self.is_active = is_active
        self.is_admin = is_admin
        self.created_at = created_at
        self.updated_at = updated_at
        
        # Questionnaire fields
        self.industry = industry
        self.job_title = job_title
        self.experience_level = experience_level
        self.content_goals = content_goals or []
        self.target_audience = target_audience
        self.preferred_tone = preferred_tone
        self.content_frequency = content_frequency
        self.topics_of_interest = topics_of_interest or []
        self.writing_style_preference = writing_style_preference
        self.blog_length_preference = blog_length_preference
        self.include_images = include_images
        self.include_data_visualizations = include_data_visualizations
        self.seo_focus = seo_focus
        self.questionnaire_completed = questionnaire_completed
        self.questionnaire_completed_at = questionnaire_completed_at

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        """Verify a password against the hash"""
        return pwd_context.verify(password, self.password_hash)

    def set_password(self, password: str):
        """Set a new password (hashes it automatically)"""
        self.password_hash = self.hash_password(password)

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding password_hash)"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # Questionnaire data
            "industry": self.industry,
            "job_title": self.job_title,
            "experience_level": self.experience_level,
            "content_goals": self.content_goals,
            "target_audience": self.target_audience,
            "preferred_tone": self.preferred_tone,
            "content_frequency": self.content_frequency,
            "topics_of_interest": self.topics_of_interest,
            "writing_style_preference": self.writing_style_preference,
            "blog_length_preference": self.blog_length_preference,
            "include_images": self.include_images,
            "include_data_visualizations": self.include_data_visualizations,
            "seo_focus": self.seo_focus,
            "questionnaire_completed": self.questionnaire_completed,
            "questionnaire_completed_at": self.questionnaire_completed_at
        }

    @classmethod
    def from_db_row(cls, row) -> 'User':
        """Create User instance from database row"""
        if not row:
            return None
        
        # Parse JSON fields
        content_goals = []
        topics_of_interest = []
        
        if row.get('content_goals'):
            try:
                content_goals = json.loads(row['content_goals']) if isinstance(row['content_goals'], str) else row['content_goals']
            except (json.JSONDecodeError, TypeError):
                content_goals = []
                
        if row.get('topics_of_interest'):
            try:
                topics_of_interest = json.loads(row['topics_of_interest']) if isinstance(row['topics_of_interest'], str) else row['topics_of_interest']
            except (json.JSONDecodeError, TypeError):
                topics_of_interest = []
        
        return cls(
            id=row['id'],
            username=row['username'],
            email=row['email'],
            password_hash=row['password_hash'],
            first_name=row.get('first_name'),
            last_name=row.get('last_name'),
            is_active=row.get('is_active', True),
            is_admin=row.get('is_admin', False),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at'),
            # Questionnaire fields
            industry=row.get('industry'),
            job_title=row.get('job_title'),
            experience_level=row.get('experience_level'),
            content_goals=content_goals,
            target_audience=row.get('target_audience'),
            preferred_tone=row.get('preferred_tone'),
            content_frequency=row.get('content_frequency'),
            topics_of_interest=topics_of_interest,
            writing_style_preference=row.get('writing_style_preference'),
            blog_length_preference=row.get('blog_length_preference'),
            include_images=row.get('include_images', True),
            include_data_visualizations=row.get('include_data_visualizations', False),
            seo_focus=row.get('seo_focus', True),
            questionnaire_completed=row.get('questionnaire_completed', False),
            questionnaire_completed_at=row.get('questionnaire_completed_at')
        )

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>" 