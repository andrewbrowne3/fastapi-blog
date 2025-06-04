# FastAPI Blog with Database and JWT Authentication

A robust FastAPI blog application with PostgreSQL database, JWT authentication, and user management.

## 🚀 Features

- **JWT Authentication** with access and refresh tokens
- **User Management** with role-based access control
- **PostgreSQL Database** with connection pooling
- **Blog Generation** using ReAct pattern with LLM
- **Image Generation** with DALL-E integration
- **Google Images Search** for blog content
- **Real-time Streaming** of blog generation process
- **Clean Architecture** with routers, services, and repositories
- **Input Validation** with Pydantic schemas
- **Password Hashing** with bcrypt
- **Database Migrations** for schema management

## 📁 Project Structure

```
fastapi_blog/
├── main_with_db.py          # Main FastAPI application
├── database.py              # Database connection and pooling
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
├── models/                  # Data models
│   ├── __init__.py
│   └── user.py             # User model with password hashing
├── repositories/            # Database access layer
│   ├── __init__.py
│   └── user_repository.py  # User database operations
├── services/               # Business logic layer
│   ├── __init__.py
│   └── user_service.py     # User business logic
├── schemas/                # Pydantic schemas for validation
│   ├── __init__.py
│   └── user_schemas.py     # User request/response schemas
├── auth/                   # JWT authentication
│   ├── __init__.py
│   ├── jwt_handler.py      # JWT token creation/verification
│   └── dependencies.py     # FastAPI auth dependencies
├── routes/                 # API endpoints
│   ├── __init__.py
│   ├── auth_routes.py      # Authentication endpoints
│   ├── user_routes.py      # User management endpoints
│   └── blog_routes.py      # Blog generation endpoints
└── migrations/             # Database migrations
    ├── 001_create_users_table.sql
    └── run_migrations.py   # Migration runner script
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
cd fastapi_blog
pip install -r requirements.txt
```

### 2. Set Up PostgreSQL Database

Install PostgreSQL and create a database:

```sql
-- Connect to PostgreSQL as superuser
CREATE DATABASE blog_db;
CREATE USER blog_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE blog_db TO blog_user;
```

### 3. Configure Environment Variables

Update your `.env` file with your database credentials:

```env
# Database Configuration
DB_HOST=localhost
DB_NAME=blog_db
DB_USER=blog_user
DB_PASSWORD=your_password
DB_PORT=5432

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# API Keys (existing)
OPENAI_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
```

### 4. Run Database Migrations

```bash
cd migrations
python run_migrations.py
```

This will:
- Create the users table
- Set up indexes and triggers
- Optionally create a default admin user

### 5. Start the API Server

```bash
# Development mode with auto-reload
python main_with_db.py

# Or using uvicorn directly
uvicorn main_with_db:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔐 API Endpoints

### Authentication
- `POST /api/auth/login` - Login and get JWT tokens
- `POST /api/auth/refresh` - Refresh access token
- `GET /api/auth/me` - Get current user info
- `POST /api/auth/logout` - Logout (client deletes tokens)

### User Management
- `POST /api/users/` - Create user (public)
- `GET /api/users/` - Get all users (admin only)
- `GET /api/users/{user_id}` - Get user (own profile or admin)
- `PUT /api/users/{user_id}` - Update user (own profile or admin)
- `DELETE /api/users/{user_id}` - Delete user (admin only)
- `POST /api/users/{user_id}/change-password` - Change password

### Blog Generation
- `POST /api/blog/` - Generate blog
- `POST /api/blog/stream` - Stream blog generation
- `POST /api/blog/resume/{thread_id}` - Resume generation
- `GET /api/blog/state/{thread_id}` - Get generation state
- `GET /api/blog/history/{thread_id}` - Get generation history
- `GET /api/blog/models` - Get available models
- `POST /api/blog/suggest-images` - Suggest images
- `POST /api/blog/generate-image` - Generate image with DALL-E
- `POST /api/blog/search-google-images` - Search Google Images

### System
- `GET /` - API welcome message
- `GET /health` - Health check with database status
- `GET /api/info` - API information and endpoints

## 🔑 Authentication Usage

### 1. Create a User
```bash
curl -X POST "http://localhost:8000/api/users/" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "SecurePass123",
    "first_name": "John",
    "last_name": "Doe"
  }'
```

### 2. Login
```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "SecurePass123"
  }'
```

Response:
```json
{
  "user": {
    "id": 1,
    "username": "john_doe",
    "email": "john@example.com",
    "is_active": true,
    "is_admin": false
  },
  "tokens": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
  }
}
```

### 3. Make Authenticated Requests
```bash
curl -X GET "http://localhost:8000/api/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 🏗️ Architecture

### Clean Architecture Layers

1. **Routes Layer** (`routes/`) - HTTP endpoints and request handling
2. **Services Layer** (`services/`) - Business logic and validation
3. **Repository Layer** (`repositories/`) - Database access and queries
4. **Models Layer** (`models/`) - Data models and domain logic
5. **Schemas Layer** (`schemas/`) - Input validation and serialization

### Key Design Patterns

- **Repository Pattern** - Separates data access from business logic
- **Dependency Injection** - FastAPI's built-in DI for clean testing
- **Router Pattern** - Organized endpoint grouping
- **Service Layer** - Encapsulated business rules
- **JWT Authentication** - Stateless token-based auth

## 🔒 Security Features

- **Password Hashing** with bcrypt
- **JWT Tokens** with expiration
- **Role-based Access Control** (admin vs user)
- **SQL Injection Protection** with parameterized queries
- **Input Validation** with Pydantic
- **CORS Configuration** for frontend access

## 🧪 Testing

### Test Database Connection
```bash
curl http://localhost:8000/health
```

### Test Authentication Flow
```bash
# 1. Create user
curl -X POST "http://localhost:8000/api/users/" \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "email": "test@example.com", "password": "Test123456"}'

# 2. Login
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "Test123456"}'

# 3. Use token to access protected endpoint
curl -X GET "http://localhost:8000/api/auth/me" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 🚀 Production Deployment

### Environment Variables for Production
```env
JWT_SECRET_KEY=your-very-long-random-secret-key-for-production
DB_HOST=your-production-db-host
DB_PASSWORD=your-secure-db-password
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main_with_db:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📝 Next Steps

1. **Integrate Blog Logic** - Move the existing blog generation logic from `blog_fastapi.py` into the new router structure
2. **Add Blog Posts Table** - Create database table for storing generated blogs
3. **User Blog Association** - Link blog posts to users
4. **Blog Permissions** - Implement blog-level permissions
5. **API Rate Limiting** - Add rate limiting for API endpoints
6. **Logging & Monitoring** - Enhanced logging and metrics
7. **Testing Suite** - Comprehensive unit and integration tests

## 🤝 Contributing

1. Follow the existing architecture patterns
2. Add proper error handling and logging
3. Include input validation with Pydantic
4. Write tests for new features
5. Update documentation

## 📄 License

This project is licensed under the MIT License. 