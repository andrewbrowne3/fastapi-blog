from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class BlogBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    html_content: Optional[str] = ""
    topic: Optional[str] = ""
    target_audience: str = "general audience"
    tone: str = "professional"
    status: str = Field(default="draft", pattern="^(draft|published|archived)$")
    thread_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class BlogCreate(BlogBase):
    pass

class BlogUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    html_content: Optional[str] = None
    topic: Optional[str] = None
    target_audience: Optional[str] = None
    tone: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(draft|published|archived)$")
    thread_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BlogResponse(BlogBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class BlogListResponse(BaseModel):
    blogs: List[BlogResponse]
    total: int
    limit: int
    offset: int

class BlogSaveRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    html_content: Optional[str] = ""
    topic: Optional[str] = ""
    target_audience: str = "general audience"
    tone: str = "professional"
    thread_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {} 