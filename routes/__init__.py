from .user_routes import router as user_router
from .auth_routes import router as auth_router
from .blog_routes import router as blog_router
 
__all__ = ['user_router', 'auth_router', 'blog_router'] 