from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your routes
from routes import user_router, auth_router, blog_router
from database import db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Blog API with Database and JWT Auth",
    description="A FastAPI blog application with PostgreSQL database and JWT authentication",
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

# Include routers
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(blog_router)

@app.get("/")
async def root():
    return {
        "message": "Blog API with Database and JWT Auth is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        
        return {
            "status": "healthy",
            "database": "connected",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "title": "Blog API with Database and JWT Auth",
        "version": "1.0.0",
        "endpoints": {
            "authentication": {
                "login": "POST /auth/login",
                "refresh": "POST /auth/refresh",
                "me": "GET /auth/me",
                "logout": "POST /auth/logout"
            },
            "users": {
                "create": "POST /users/",
                "list": "GET /users/ (admin only)",
                "get": "GET /users/{user_id}",
                "update": "PUT /users/{user_id}",
                "delete": "DELETE /users/{user_id} (admin only)",
                "change_password": "POST /users/{user_id}/change-password"
            },
            "blog": {
                "generate": "POST /blog/",
                "stream": "POST /blog/stream",
                "resume": "POST /blog/resume/{thread_id}",
                "state": "GET /blog/state/{thread_id}",
                "history": "GET /blog/history/{thread_id}",
                "models": "GET /blog/models",
                "suggest_images": "POST /blog/suggest-images",
                "generate_image": "POST /blog/generate-image",
                "search_images": "POST /blog/search-google-images"
            }
        },
        "features": [
            "JWT Authentication",
            "User Management",
            "Role-based Access Control",
            "Blog Generation with ReAct Pattern",
            "Real-time Streaming",
            "Image Generation and Search",
            "PostgreSQL Database",
            "Connection Pooling",
            "Password Hashing",
            "Input Validation"
        ]
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting up Blog API with Database and JWT Auth...")
    logger.info("📊 Database connection pool initialized")
    logger.info("🔐 JWT authentication enabled")
    logger.info("📝 Blog generation endpoints available")
    logger.info("🌐 CORS enabled for frontend access")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down Blog API...")
    db.close_all_connections()
    logger.info("📊 Database connections closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_with_db:app",
        host="0.0.0.0",
        port=4000,
        reload=True,
        log_level="info"
    ) 