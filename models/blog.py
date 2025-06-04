from datetime import datetime
from typing import Optional, Dict, Any
import json

class Blog:
    def __init__(
        self,
        id: Optional[int] = None,
        user_id: int = None,
        title: str = "",
        content: str = "",
        html_content: str = "",
        topic: str = "",
        target_audience: str = "general audience",
        tone: str = "professional",
        status: str = "draft",  # draft, published, archived
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        published_at: Optional[datetime] = None
    ):
        self.id = id
        self.user_id = user_id
        self.title = title
        self.content = content
        self.html_content = html_content
        self.topic = topic
        self.target_audience = target_audience
        self.tone = tone
        self.status = status
        self.thread_id = thread_id
        self.metadata = metadata or {}
        self.created_at = created_at
        self.updated_at = updated_at
        self.published_at = published_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert blog to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "html_content": self.html_content,
            "topic": self.topic,
            "target_audience": self.target_audience,
            "tone": self.tone,
            "status": self.status,
            "thread_id": self.thread_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "published_at": self.published_at
        }

    @classmethod
    def from_db_row(cls, row) -> 'Blog':
        """Create Blog instance from database row"""
        if not row:
            return None
        
        # Parse JSON metadata
        metadata = {}
        if row.get('metadata'):
            try:
                metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        
        return cls(
            id=row['id'],
            user_id=row['user_id'],
            title=row['title'],
            content=row['content'],
            html_content=row.get('html_content', ''),
            topic=row.get('topic', ''),
            target_audience=row.get('target_audience', 'general audience'),
            tone=row.get('tone', 'professional'),
            status=row.get('status', 'draft'),
            thread_id=row.get('thread_id'),
            metadata=metadata,
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at'),
            published_at=row.get('published_at')
        )

    def __repr__(self):
        return f"<Blog(id={self.id}, title='{self.title}', user_id={self.user_id})>" 