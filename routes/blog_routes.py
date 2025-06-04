from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel
import time
import uuid
import json
import os
import requests
import sqlite3
from datetime import datetime

# Import the actual blog generation functionality
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blog_fastapi import (
    ReActState, BlogRequest, ImageRequest, ImageSuggestion,
    GoogleImageSearchRequest, GoogleImageResult, GoogleImageSearchResponse,
    GoogleImageSectionRequest, create_blog_graph, get_llm
)

router = APIRouter(prefix="/blog", tags=["blog"])

# Create the graph instance for blog generation
graph = create_blog_graph()

# Pydantic models (these should eventually be moved to schemas)
class ReActState(BaseModel):
    topic: str
    current_step: int = 0
    max_steps: int = 30
    thoughts: List[str] = []
    actions: List[str] = []
    observations: List[str] = []
    research_data: List[Dict[str, Any]] = []
    blog_sections: Dict[str, str] = {}
    react_trace: List[Dict[str, Any]] = []
    final_blog: str = ""
    html_mode: bool = False
    is_complete: bool = False
    thread_id: str = ""
    llm_provider: str = "cloud"
    model_name: Optional[str] = None
    num_sections: int = 3
    target_audience: str = "general audience"
    tone: str = "professional"

class BlogRequest(BaseModel):
    topic: str
    html_mode: bool = False
    thread_id: str = None
    llm_provider: Literal["cloud", "local"] = "cloud"
    model_name: Optional[str] = None
    num_sections: int = 3
    target_audience: str = "general audience"
    tone: str = "professional"

class ImageRequest(BaseModel):
    content: str
    section_title: str = ""
    style: str = "professional"
    size: str = "1024x1024"

class ImageSuggestion(BaseModel):
    prompt: str
    description: str
    placement: str

class GoogleImageSearchRequest(BaseModel):
    query: str
    num_results: int = 10

class GoogleImageResult(BaseModel):
    title: str
    link: str
    thumbnail: str = ""
    width: int = 0
    height: int = 0
    source: str = ""

class GoogleImageSearchResponse(BaseModel):
    query: str
    results: List[GoogleImageResult]
    total_results: int

class GoogleImageSectionRequest(BaseModel):
    sections: List[Dict[str, str]]

# Add blog saving models
class SavedBlog(BaseModel):
    id: Optional[int] = None
    title: str
    content: str
    thread_id: str
    topic: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class SaveBlogRequest(BaseModel):
    title: str
    content: str
    thread_id: str
    topic: str

class UpdateBlogRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

# Database functions for blog saving
def init_blog_db():
    """Initialize the SQLite database for saved blogs"""
    conn = sqlite3.connect('saved_blogs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_blogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_blog_to_db(blog_data: SaveBlogRequest) -> int:
    """Save a blog to the database"""
    init_blog_db()
    conn = sqlite3.connect('saved_blogs.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO saved_blogs (title, content, thread_id, topic)
        VALUES (?, ?, ?, ?)
    ''', (blog_data.title, blog_data.content, blog_data.thread_id, blog_data.topic))
    blog_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return blog_id

def get_saved_blogs() -> List[SavedBlog]:
    """Get all saved blogs"""
    init_blog_db()
    conn = sqlite3.connect('saved_blogs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM saved_blogs ORDER BY updated_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    blogs = []
    for row in rows:
        blogs.append(SavedBlog(
            id=row[0],
            title=row[1],
            content=row[2],
            thread_id=row[3],
            topic=row[4],
            created_at=row[5],
            updated_at=row[6]
        ))
    return blogs

def get_saved_blog_by_id(blog_id: int) -> Optional[SavedBlog]:
    """Get a saved blog by ID"""
    init_blog_db()
    conn = sqlite3.connect('saved_blogs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM saved_blogs WHERE id = ?', (blog_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return SavedBlog(
            id=row[0],
            title=row[1],
            content=row[2],
            thread_id=row[3],
            topic=row[4],
            created_at=row[5],
            updated_at=row[6]
        )
    return None

def update_saved_blog(blog_id: int, update_data: UpdateBlogRequest) -> bool:
    """Update a saved blog"""
    init_blog_db()
    conn = sqlite3.connect('saved_blogs.db')
    cursor = conn.cursor()
    
    # Build update query dynamically
    updates = []
    values = []
    
    if update_data.title is not None:
        updates.append("title = ?")
        values.append(update_data.title)
    
    if update_data.content is not None:
        updates.append("content = ?")
        values.append(update_data.content)
    
    if updates:
        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(blog_id)
        
        query = f"UPDATE saved_blogs SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, values)
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    conn.close()
    return False

def delete_saved_blog(blog_id: int) -> bool:
    """Delete a saved blog"""
    init_blog_db()
    conn = sqlite3.connect('saved_blogs.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM saved_blogs WHERE id = ?', (blog_id,))
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return success

# Note: The actual implementation functions (get_llm, create_blog_graph, etc.) 
# would need to be imported from the main blog_fastapi.py file or moved to separate modules
# For now, I'll create placeholder endpoints that reference the main functionality

@router.post("/")
def generate_blog(request: BlogRequest):
    """Generate a blog using ReAct pattern with full state persistence"""
    start_time = time.time()
    
    # Generate or use provided thread_id
    thread_id = request.thread_id or f"blog-{uuid.uuid4()}"
    
    # Create configuration for this thread
    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 50
    }
    
    # Check if we're resuming an existing thread
    try:
        existing_state = graph.get_state(config)
        if existing_state.values:
            # Resume from existing state
            initial_state = ReActState(**existing_state.values)
            print(f"Resuming thread {thread_id} from step {initial_state.current_step}")
        else:
            # Start new thread
            initial_state = ReActState(
                topic=request.topic,
                html_mode=request.html_mode,
                thread_id=thread_id,
                llm_provider=request.llm_provider,
                model_name=request.model_name,
                num_sections=request.num_sections,
                target_audience=request.target_audience,
                tone=request.tone
            )
            print(f"Starting new thread {thread_id}")
    except:
        # Start new thread if no existing state
        initial_state = ReActState(
            topic=request.topic,
            html_mode=request.html_mode,
            thread_id=thread_id,
            llm_provider=request.llm_provider,
            model_name=request.model_name,
            num_sections=request.num_sections,
            target_audience=request.target_audience,
            tone=request.tone
        )
        print(f"Starting new thread {thread_id}")
    
    # Run the graph
    final_result = graph.invoke(initial_state, config)
    
    # Convert the result to ReActState if it's not already
    if isinstance(final_result, dict):
        final_state = ReActState(**final_result)
    else:
        final_state = final_result
    
    processing_time = time.time() - start_time
    
    return {
        "blog": final_state.final_blog,
        "thread_id": thread_id,
        "steps_completed": final_state.current_step,
        "react_trace": final_state.react_trace,
        "processing_time": f"{processing_time:.2f} seconds",
        "is_complete": final_state.is_complete,
        "format": "HTML" if request.html_mode else "Markdown",
        "target_audience": final_state.target_audience,
        "tone": final_state.tone
    }

@router.post("/stream")
async def generate_blog_stream(request: BlogRequest):
    """Stream the ReAct process in real-time with state persistence"""
    
    def generate():
        start_time = time.time()
        
        # Generate or use provided thread_id
        thread_id = request.thread_id or f"blog-stream-{uuid.uuid4()}"
        
        # Create configuration for this thread
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 50
        }
        
        # Check if we're resuming an existing thread
        try:
            existing_state = graph.get_state(config)
            if existing_state.values:
                # Resume from existing state
                current_state = ReActState(**existing_state.values)
                yield f"data: {json.dumps({'type': 'resume', 'thread_id': thread_id, 'current_step': current_state.current_step})}\n\n"
            else:
                # Start new thread
                current_state = ReActState(
                    topic=request.topic,
                    html_mode=request.html_mode,
                    thread_id=thread_id,
                    llm_provider=request.llm_provider,
                    model_name=request.model_name,
                    num_sections=request.num_sections,
                    target_audience=request.target_audience,
                    tone=request.tone
                )
                yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id, 'topic': request.topic})}\n\n"
        except:
            # Start new thread if no existing state
            current_state = ReActState(
                topic=request.topic,
                html_mode=request.html_mode,
                thread_id=thread_id,
                llm_provider=request.llm_provider,
                model_name=request.model_name,
                num_sections=request.num_sections,
                target_audience=request.target_audience,
                tone=request.tone
            )
            yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id, 'topic': request.topic})}\n\n"
        
        try:
            # Stream the graph execution
            for chunk in graph.stream(current_state, config, stream_mode="values"):
                if chunk:
                    # Convert chunk to ReActState if it's a dict
                    if isinstance(chunk, dict):
                        chunk_state = ReActState(**chunk)
                    else:
                        chunk_state = chunk
                    
                    # Stream current state updates
                    state_update = {
                        "type": "state_update",
                        "current_step": chunk_state.current_step,
                        "is_complete": chunk_state.is_complete,
                        "sections_written": len([s for s in chunk_state.react_trace if s.get('action') == 'WRITE_SECTION']),
                        "sections_needed": chunk_state.num_sections,
                        "sections_remaining": max(0, chunk_state.num_sections - len([s for s in chunk_state.react_trace if s.get('action') == 'WRITE_SECTION'])),
                        "thread_id": thread_id,
                        "target_audience": chunk_state.target_audience,
                        "tone": chunk_state.tone
                    }
                    yield f"data: {json.dumps(state_update)}\n\n"
                    
                    # If we have a new step, stream it
                    if chunk_state.react_trace and len(chunk_state.react_trace) > len(current_state.react_trace):
                        new_step = chunk_state.react_trace[-1]
                        safe_data = {
                            "type": "step",
                            "step": new_step.get("step", 0),
                            "thought": new_step.get("thought", "")[:500],  # Limit length
                            "action": new_step.get("action", ""),
                            "action_input": new_step.get("action_input", "")[:200],
                            "observation": new_step.get("observation", "")[:500],
                            "thread_id": thread_id,
                            "target_audience": new_step.get("target_audience", "general audience"),
                            "tone": new_step.get("tone", "professional")
                        }
                        yield f"data: {json.dumps(safe_data)}\n\n"
                    
                    current_state = chunk_state
            
            # Final result
            processing_time = time.time() - start_time
            final_data = {
                "type": "complete",
                "blog": current_state.final_blog,
                "thread_id": thread_id,
                "steps_completed": current_state.current_step,
                "processing_time": f"{processing_time:.2f} seconds",
                "is_complete": current_state.is_complete,
                "target_audience": current_state.target_audience,
                "tone": current_state.tone
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error", 
                "error": f"Generation failed: {str(e)}",
                "thread_id": thread_id
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@router.post("/resume/{thread_id}")
def resume_blog_generation(thread_id: str):
    """Resume blog generation from a specific thread"""
    return {
        "message": f"Resume blog generation for thread {thread_id}",
        "thread_id": thread_id
    }

@router.get("/state/{thread_id}")
def get_blog_state(thread_id: str):
    """Get the current state of a blog generation thread"""
    return {
        "message": f"Get state for thread {thread_id}",
        "thread_id": thread_id
    }

@router.get("/history/{thread_id}")
def get_blog_history(thread_id: str):
    """Get the complete history of a blog generation thread"""
    return {
        "message": f"Get history for thread {thread_id}",
        "thread_id": thread_id
    }

@router.post("/suggest-images")
async def suggest_images(request: ImageRequest):
    """Suggest images for blog content"""
    return {
        "message": "Image suggestion endpoint",
        "request": request.dict()
    }

@router.post("/generate-image")
async def generate_image(request: ImageRequest):
    """Generate image using DALL-E"""
    return {
        "message": "Image generation endpoint",
        "request": request.dict()
    }

def search_google_images(query: str, num_results: int = 10) -> Dict[str, Any]:
    """
    Search for images using Google Custom Search API
    """
    try:
        # Get API credentials from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not api_key or not search_engine_id:
            raise HTTPException(
                status_code=500, 
                detail="Google API credentials not configured. Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables."
            )
        
        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"
        
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "searchType": "image",
            "num": min(num_results, 10),  # Google API limits to 10 results per request
            "safe": "active",
            "imgSize": "medium",
            "imgType": "photo"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        items = data.get("items", [])
        
        for item in items:
            # Extract image information
            image_result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                "width": item.get("image", {}).get("width", 0),
                "height": item.get("image", {}).get("height", 0),
                "source": item.get("displayLink", "")
            }
            results.append(image_result)
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Google API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Google Images: {str(e)}")

def generate_smart_image_query(content: str, section_title: str = "") -> str:
    """
    Generate an optimized search query for images based on content and section title
    """
    try:
        # Fallback: extract key terms from section title and content
        import re
        
        # Combine section title and content
        text = f"{section_title} {content}".lower()
        
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        meaningful_words = [word for word in words if word not in stop_words]
        
        # Take first 2-3 most relevant words
        if meaningful_words:
            return ' '.join(meaningful_words[:3])
        else:
            return section_title or "professional business"
            
    except Exception as e:
        print(f"Error generating smart query: {e}")
        # Fallback to section title or generic term
        return section_title or "professional business"

@router.post("/search-google-images")
async def search_google_images_endpoint(request: GoogleImageSearchRequest):
    """Search Google Images"""
    try:
        result = search_google_images(request.query, request.num_results)
        return {
            "success": True,
            "query": result["query"],
            "results": [
                {
                    "id": f"google-{i}",
                    "title": img["title"],
                    "link": img["link"],
                    "thumbnail": img["thumbnail"],
                    "width": img["width"],
                    "height": img["height"],
                    "source": img["source"]
                }
                for i, img in enumerate(result["results"])
            ],
            "total_results": result["total_results"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Google Images: {str(e)}")

@router.post("/search-images-for-sections")
async def search_images_for_sections(request: GoogleImageSectionRequest):
    """Search images for multiple blog sections"""
    try:
        results = {}
        
        for section in request.sections:
            section_title = section.get("title", "")
            section_content = section.get("content", "")
            
            # Generate smart search query for this section
            search_query = generate_smart_image_query(section_content, section_title)
            
            try:
                # Search for images for this section
                search_result = search_google_images(search_query, 5)  # Get 5 images per section
                
                results[section_title] = {
                    "query": search_query,
                    "images": [
                        {
                            "id": f"google-{section_title}-{i}",
                            "title": img["title"],
                            "link": img["link"],
                            "thumbnail": img["thumbnail"],
                            "width": img["width"],
                            "height": img["height"],
                            "source": img["source"]
                        }
                        for i, img in enumerate(search_result["results"])
                    ],
                    "total_results": search_result["total_results"]
                }
                
            except Exception as section_error:
                print(f"Error searching images for section '{section_title}': {section_error}")
                results[section_title] = {
                    "query": search_query,
                    "images": [],
                    "total_results": 0,
                    "error": str(section_error)
                }
        
        return {
            "success": True,
            "results": results,
            "sections_processed": len(request.sections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching images for sections: {str(e)}")

@router.get("/models")
def get_available_models():
    """Get available LLM models"""
    return {
        "cloud_models": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307"
        ],
        "local_models": [
            "llama3.2",
            "llama3.1",
            "mistral"
        ]
    }

# Blog saving endpoints
@router.post("/saved")
def save_blog(request: SaveBlogRequest):
    """Save a blog to the database"""
    try:
        blog_id = save_blog_to_db(request)
        return {
            "success": True,
            "message": "Blog saved successfully",
            "blog_id": blog_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving blog: {str(e)}")

@router.get("/saved")
def get_saved_blogs_endpoint():
    """Get all saved blogs"""
    try:
        blogs = get_saved_blogs()
        return {
            "success": True,
            "blogs": blogs,
            "total": len(blogs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving saved blogs: {str(e)}")

@router.get("/saved/{blog_id}")
def get_saved_blog_endpoint(blog_id: int):
    """Get a saved blog by ID"""
    try:
        blog = get_saved_blog_by_id(blog_id)
        if not blog:
            raise HTTPException(status_code=404, detail="Blog not found")
        return {
            "success": True,
            "blog": blog
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving blog: {str(e)}")

@router.put("/saved/{blog_id}")
def update_saved_blog_endpoint(blog_id: int, request: UpdateBlogRequest):
    """Update a saved blog"""
    try:
        success = update_saved_blog(blog_id, request)
        if not success:
            raise HTTPException(status_code=404, detail="Blog not found or no changes made")
        return {
            "success": True,
            "message": "Blog updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating blog: {str(e)}")

@router.delete("/saved/{blog_id}")
def delete_saved_blog_endpoint(blog_id: int):
    """Delete a saved blog"""
    try:
        success = delete_saved_blog(blog_id)
        if not success:
            raise HTTPException(status_code=404, detail="Blog not found")
        return {
            "success": True,
            "message": "Blog deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting blog: {str(e)}") 