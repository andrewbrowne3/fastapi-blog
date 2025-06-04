from typing import List, Optional
import logging
import json
from models.user import User

logger = logging.getLogger(__name__)

class UserRepository:
    def __init__(self, connection):
        self.connection = connection

    def create_user(self, user: User) -> User:
        """Create a new user in the database"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, first_name, last_name, is_active, is_admin)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, username, email, password_hash, first_name, last_name, 
                             is_active, is_admin, created_at, updated_at,
                             industry, job_title, experience_level, content_goals, target_audience,
                             preferred_tone, content_frequency, topics_of_interest, writing_style_preference,
                             blog_length_preference, include_images, include_data_visualizations, seo_focus,
                             questionnaire_completed, questionnaire_completed_at
                """, (
                    user.username, user.email, user.password_hash,
                    user.first_name, user.last_name, user.is_active, user.is_admin
                ))
                
                row = cursor.fetchone()
                self.connection.commit()
                
                return User.from_db_row(row)
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error creating user: {e}")
            raise

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, username, email, password_hash, first_name, last_name,
                           is_active, is_admin, created_at, updated_at,
                           industry, job_title, experience_level, content_goals, target_audience,
                           preferred_tone, content_frequency, topics_of_interest, writing_style_preference,
                           blog_length_preference, include_images, include_data_visualizations, seo_focus,
                           questionnaire_completed, questionnaire_completed_at
                    FROM users WHERE id = %s
                """, (user_id,))
                
                row = cursor.fetchone()
                return User.from_db_row(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {e}")
            raise

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, username, email, password_hash, first_name, last_name,
                           is_active, is_admin, created_at, updated_at,
                           industry, job_title, experience_level, content_goals, target_audience,
                           preferred_tone, content_frequency, topics_of_interest, writing_style_preference,
                           blog_length_preference, include_images, include_data_visualizations, seo_focus,
                           questionnaire_completed, questionnaire_completed_at
                    FROM users WHERE username = %s
                """, (username,))
                
                row = cursor.fetchone()
                return User.from_db_row(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user by username {username}: {e}")
            raise

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, username, email, password_hash, first_name, last_name,
                           is_active, is_admin, created_at, updated_at,
                           industry, job_title, experience_level, content_goals, target_audience,
                           preferred_tone, content_frequency, topics_of_interest, writing_style_preference,
                           blog_length_preference, include_images, include_data_visualizations, seo_focus,
                           questionnaire_completed, questionnaire_completed_at
                    FROM users WHERE email = %s
                """, (email,))
                
                row = cursor.fetchone()
                return User.from_db_row(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {e}")
            raise

    def get_all_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get all users with pagination"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, username, email, password_hash, first_name, last_name,
                           is_active, is_admin, created_at, updated_at,
                           industry, job_title, experience_level, content_goals, target_audience,
                           preferred_tone, content_frequency, topics_of_interest, writing_style_preference,
                           blog_length_preference, include_images, include_data_visualizations, seo_focus,
                           questionnaire_completed, questionnaire_completed_at
                    FROM users
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                rows = cursor.fetchall()
                return [User.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            raise

    def update_user(self, user_id: int, updates: dict) -> Optional[User]:
        """Update user with given updates"""
        if not updates:
            return self.get_user_by_id(user_id)
        
        try:
            # Build dynamic update query
            set_clauses = []
            values = []
            
            # Basic user fields
            basic_fields = ['username', 'email', 'first_name', 'last_name', 'is_active', 'is_admin']
            
            # Questionnaire fields
            questionnaire_fields = [
                'industry', 'job_title', 'experience_level', 'target_audience',
                'preferred_tone', 'content_frequency', 'writing_style_preference',
                'blog_length_preference', 'include_images', 'include_data_visualizations',
                'seo_focus', 'questionnaire_completed', 'questionnaire_completed_at'
            ]
            
            # JSON fields that need special handling
            json_fields = ['content_goals', 'topics_of_interest']
            
            for field, value in updates.items():
                if field in basic_fields + questionnaire_fields:
                    set_clauses.append(f"{field} = %s")
                    values.append(value)
                elif field in json_fields:
                    set_clauses.append(f"{field} = %s")
                    values.append(json.dumps(value) if value is not None else None)
            
            if not set_clauses:
                return self.get_user_by_id(user_id)
            
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            values.append(user_id)
            
            with self.connection.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE users 
                    SET {', '.join(set_clauses)}
                    WHERE id = %s
                    RETURNING id, username, email, password_hash, first_name, last_name,
                             is_active, is_admin, created_at, updated_at,
                             industry, job_title, experience_level, content_goals, target_audience,
                             preferred_tone, content_frequency, topics_of_interest, writing_style_preference,
                             blog_length_preference, include_images, include_data_visualizations, seo_focus,
                             questionnaire_completed, questionnaire_completed_at
                """, values)
                
                row = cursor.fetchone()
                self.connection.commit()
                
                return User.from_db_row(row) if row else None
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            raise

    def update_questionnaire(self, user_id: int, questionnaire_data: dict) -> Optional[User]:
        """Update user questionnaire data"""
        try:
            # Prepare questionnaire updates
            updates = questionnaire_data.copy()
            updates['questionnaire_completed'] = True
            updates['questionnaire_completed_at'] = 'CURRENT_TIMESTAMP'
            
            # Handle JSON fields
            if 'content_goals' in updates:
                updates['content_goals'] = json.dumps(updates['content_goals'])
            if 'topics_of_interest' in updates:
                updates['topics_of_interest'] = json.dumps(updates['topics_of_interest'])
            
            set_clauses = []
            values = []
            
            for field, value in updates.items():
                if field == 'questionnaire_completed_at':
                    set_clauses.append(f"{field} = CURRENT_TIMESTAMP")
                else:
                    set_clauses.append(f"{field} = %s")
                    values.append(value)
            
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            values.append(user_id)
            
            with self.connection.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE users 
                    SET {', '.join(set_clauses)}
                    WHERE id = %s
                    RETURNING id, username, email, password_hash, first_name, last_name,
                             is_active, is_admin, created_at, updated_at,
                             industry, job_title, experience_level, content_goals, target_audience,
                             preferred_tone, content_frequency, topics_of_interest, writing_style_preference,
                             blog_length_preference, include_images, include_data_visualizations, seo_focus,
                             questionnaire_completed, questionnaire_completed_at
                """, values)
                
                row = cursor.fetchone()
                self.connection.commit()
                
                return User.from_db_row(row) if row else None
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error updating questionnaire for user {user_id}: {e}")
            raise

    def delete_user(self, user_id: int) -> bool:
        """Delete user by ID"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
                deleted_count = cursor.rowcount
                self.connection.commit()
                
                return deleted_count > 0
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error deleting user {user_id}: {e}")
            raise

    def update_password(self, user_id: int, new_password_hash: str) -> bool:
        """Update user password"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE users 
                    SET password_hash = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (new_password_hash, user_id))
                
                updated_count = cursor.rowcount
                self.connection.commit()
                
                return updated_count > 0
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error updating password for user {user_id}: {e}")
            raise

    def username_exists(self, username: str, exclude_user_id: Optional[int] = None) -> bool:
        """Check if username already exists"""
        try:
            with self.connection.cursor() as cursor:
                if exclude_user_id:
                    cursor.execute(
                        "SELECT 1 FROM users WHERE username = %s AND id != %s",
                        (username, exclude_user_id)
                    )
                else:
                    cursor.execute("SELECT 1 FROM users WHERE username = %s", (username,))
                
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking username existence: {e}")
            raise

    def email_exists(self, email: str, exclude_user_id: Optional[int] = None) -> bool:
        """Check if email already exists"""
        try:
            with self.connection.cursor() as cursor:
                if exclude_user_id:
                    cursor.execute(
                        "SELECT 1 FROM users WHERE email = %s AND id != %s",
                        (email, exclude_user_id)
                    )
                else:
                    cursor.execute("SELECT 1 FROM users WHERE email = %s", (email,))
                
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking email existence: {e}")
            raise 