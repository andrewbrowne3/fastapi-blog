from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import json

from models.blog import Blog

logger = logging.getLogger(__name__)

class BlogRepository:
    def __init__(self, db_connection):
        self.db = db_connection

    def create_blog(self, blog: Blog) -> Blog:
        """Create a new blog in the database"""
        try:
            with self.db.cursor() as cursor:
                query = """
                    INSERT INTO blogs (user_id, title, content, html_content, topic, 
                                     target_audience, tone, status, thread_id, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at, updated_at
                """
                now = datetime.utcnow()
                metadata_json = json.dumps(blog.metadata) if blog.metadata else None
                
                cursor.execute(query, (
                    blog.user_id, blog.title, blog.content, blog.html_content,
                    blog.topic, blog.target_audience, blog.tone, blog.status,
                    blog.thread_id, metadata_json, now, now
                ))
                
                result = cursor.fetchone()
                self.db.commit()
                
                blog.id = result['id']
                blog.created_at = result['created_at']
                blog.updated_at = result['updated_at']
                
                return blog
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating blog: {e}")
            raise

    def get_blog_by_id(self, blog_id: int) -> Optional[Blog]:
        """Get a blog by its ID"""
        try:
            with self.db.cursor() as cursor:
                query = "SELECT * FROM blogs WHERE id = %s"
                cursor.execute(query, (blog_id,))
                row = cursor.fetchone()
                
                return Blog.from_db_row(row) if row else None
                
        except Exception as e:
            logger.error(f"Error getting blog {blog_id}: {e}")
            raise

    def get_blogs_by_user_id(self, user_id: int, limit: int = 100, offset: int = 0) -> List[Blog]:
        """Get all blogs for a specific user"""
        try:
            with self.db.cursor() as cursor:
                query = """
                    SELECT * FROM blogs 
                    WHERE user_id = %s 
                    ORDER BY updated_at DESC 
                    LIMIT %s OFFSET %s
                """
                cursor.execute(query, (user_id, limit, offset))
                rows = cursor.fetchall()
                
                return [Blog.from_db_row(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting blogs for user {user_id}: {e}")
            raise

    def update_blog(self, blog_id: int, updates: Dict[str, Any]) -> Optional[Blog]:
        """Update a blog with the provided data"""
        try:
            with self.db.cursor() as cursor:
                # Build dynamic update query
                set_clauses = []
                values = []
                
                for key, value in updates.items():
                    if key == 'metadata' and value is not None:
                        set_clauses.append(f"{key} = %s")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = %s")
                        values.append(value)
                
                if not set_clauses:
                    return self.get_blog_by_id(blog_id)
                
                # Add updated_at
                set_clauses.append("updated_at = %s")
                values.append(datetime.utcnow())
                values.append(blog_id)
                
                query = f"""
                    UPDATE blogs 
                    SET {', '.join(set_clauses)}
                    WHERE id = %s
                    RETURNING *
                """
                
                cursor.execute(query, values)
                row = cursor.fetchone()
                self.db.commit()
                
                return Blog.from_db_row(row) if row else None
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating blog {blog_id}: {e}")
            raise

    def delete_blog(self, blog_id: int) -> bool:
        """Delete a blog by its ID"""
        try:
            with self.db.cursor() as cursor:
                query = "DELETE FROM blogs WHERE id = %s"
                cursor.execute(query, (blog_id,))
                self.db.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting blog {blog_id}: {e}")
            raise

    def get_blog_by_thread_id(self, thread_id: str) -> Optional[Blog]:
        """Get a blog by its thread ID"""
        try:
            with self.db.cursor() as cursor:
                query = "SELECT * FROM blogs WHERE thread_id = %s"
                cursor.execute(query, (thread_id,))
                row = cursor.fetchone()
                
                return Blog.from_db_row(row) if row else None
                
        except Exception as e:
            logger.error(f"Error getting blog by thread_id {thread_id}: {e}")
            raise

    def count_user_blogs(self, user_id: int) -> int:
        """Count the total number of blogs for a user"""
        try:
            with self.db.cursor() as cursor:
                query = "SELECT COUNT(*) as count FROM blogs WHERE user_id = %s"
                cursor.execute(query, (user_id,))
                result = cursor.fetchone()
                
                return result['count'] if result else 0
                
        except Exception as e:
            logger.error(f"Error counting blogs for user {user_id}: {e}")
            raise 