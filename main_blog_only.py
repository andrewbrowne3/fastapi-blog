from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the standalone blog router
from blog_router_standalone import router as blog_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Blog API - Blog Generation Only",
    description="A FastAPI blog generation application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3007", "http://127.0.0.1:3000", "http://127.0.0.1:3007", "https://blog.andrewbrowne.org"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include only blog router
app.include_router(blog_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "Blog Generation API is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "title": "Blog Generation API",
        "version": "1.0.0",
        "endpoints": {
            "blog": {
                "generate": "POST /api/blog/",
                "stream": "POST /api/blog/stream",
                "resume": "POST /api/blog/resume/{thread_id}",
                "state": "GET /api/blog/state/{thread_id}",
                "history": "GET /api/blog/history/{thread_id}",
                "models": "GET /api/blog/models",
                "suggest_images": "POST /api/blog/suggest-images",
                "generate_image": "POST /api/blog/generate-image",
                "search_images": "POST /api/blog/search-google-images"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_blog_only:app",
        host="0.0.0.0",
        port=4001,
        reload=True,
        log_level="info"
    ) 