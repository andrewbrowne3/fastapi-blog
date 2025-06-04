from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from models.blog import Blog
from repositories.blog_repository import BlogRepository
from schemas.blog_schemas import BlogCreate, BlogUpdate, BlogSaveRequest

logger = logging.getLogger(__name__)

class BlogService:
    def __init__(self, blog_repository: BlogRepository):
        self.blog_repo = blog_repository

    def create_blog(self, user_id: int, blog_data: BlogCreate) -> Blog:
        """Create a new blog for a user"""
        try:
            blog = Blog(
                user_id=user_id,
                title=blog_data.title,
                content=blog_data.content,
                html_content=blog_data.html_content,
                topic=blog_data.topic,
                target_audience=blog_data.target_audience,
                tone=blog_data.tone,
                status=blog_data.status,
                thread_id=blog_data.thread_id,
                metadata=blog_data.metadata
            )
            
            return self.blog_repo.create_blog(blog)
            
        except Exception as e:
            logger.error(f"Error creating blog for user {user_id}: {e}")
            raise

    def save_blog_from_generation(self, user_id: int, blog_data: BlogSaveRequest) -> Blog:
        """Save a blog from the generation process"""
        try:
            blog = Blog(
                user_id=user_id,
                title=blog_data.title,
                content=blog_data.content,
                html_content=blog_data.html_content,
                topic=blog_data.topic,
                target_audience=blog_data.target_audience,
                tone=blog_data.tone,
                status="draft",  # Always save as draft initially
                thread_id=blog_data.thread_id,
                metadata=blog_data.metadata
            )
            
            return self.blog_repo.create_blog(blog)
            
        except Exception as e:
            logger.error(f"Error saving blog for user {user_id}: {e}")
            raise

    def get_user_blogs(self, user_id: int, limit: int = 100, offset: int = 0) -> List[Blog]:
        """Get all blogs for a specific user"""
        try:
            return self.blog_repo.get_blogs_by_user_id(user_id, limit, offset)
        except Exception as e:
            logger.error(f"Error getting blogs for user {user_id}: {e}")
            raise

    def get_blog_by_id(self, blog_id: int, user_id: int = None) -> Optional[Blog]:
        """Get a blog by ID, optionally checking user ownership"""
        try:
            blog = self.blog_repo.get_blog_by_id(blog_id)
            
            # If user_id is provided, check ownership
            if blog and user_id is not None and blog.user_id != user_id:
                return None
                
            return blog
        except Exception as e:
            logger.error(f"Error getting blog {blog_id}: {e}")
            raise

    def update_blog(self, blog_id: int, user_id: int, blog_data: BlogUpdate) -> Optional[Blog]:
        """Update a blog (only if user owns it)"""
        try:
            # First check if the blog exists and belongs to the user
            existing_blog = self.get_blog_by_id(blog_id, user_id)
            if not existing_blog:
                return None
            
            # Prepare update data (only include non-None values)
            updates = {}
            for field, value in blog_data.dict(exclude_unset=True).items():
                if value is not None:
                    updates[field] = value
            
            if not updates:
                return existing_blog
            
            return self.blog_repo.update_blog(blog_id, updates)
            
        except Exception as e:
            logger.error(f"Error updating blog {blog_id} for user {user_id}: {e}")
            raise

    def delete_blog(self, blog_id: int, user_id: int) -> bool:
        """Delete a blog (only if user owns it)"""
        try:
            # First check if the blog exists and belongs to the user
            existing_blog = self.get_blog_by_id(blog_id, user_id)
            if not existing_blog:
                return False
            
            return self.blog_repo.delete_blog(blog_id)
            
        except Exception as e:
            logger.error(f"Error deleting blog {blog_id} for user {user_id}: {e}")
            raise

    def get_blog_by_thread_id(self, thread_id: str, user_id: int = None) -> Optional[Blog]:
        """Get a blog by thread ID, optionally checking user ownership"""
        try:
            blog = self.blog_repo.get_blog_by_thread_id(thread_id)
            
            # If user_id is provided, check ownership
            if blog and user_id is not None and blog.user_id != user_id:
                return None
                
            return blog
        except Exception as e:
            logger.error(f"Error getting blog by thread_id {thread_id}: {e}")
            raise

    def count_user_blogs(self, user_id: int) -> int:
        """Count total blogs for a user"""
        try:
            return self.blog_repo.count_user_blogs(user_id)
        except Exception as e:
            logger.error(f"Error counting blogs for user {user_id}: {e}")
            raise

    def publish_blog(self, blog_id: int, user_id: int) -> Optional[Blog]:
        """Publish a blog (change status to published)"""
        try:
            updates = {
                "status": "published",
                "published_at": datetime.utcnow()
            }
            return self.update_blog(blog_id, user_id, BlogUpdate(**updates))
        except Exception as e:
            logger.error(f"Error publishing blog {blog_id} for user {user_id}: {e}")
            raise

    def archive_blog(self, blog_id: int, user_id: int) -> Optional[Blog]:
        """Archive a blog (change status to archived)"""
        try:
            updates = {"status": "archived"}
            return self.update_blog(blog_id, user_id, BlogUpdate(**updates))
        except Exception as e:
            logger.error(f"Error archiving blog {blog_id} for user {user_id}: {e}")
            raise 