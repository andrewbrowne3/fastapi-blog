# FastAPI Blog Generator

A FastAPI-based blog generation service that uses the ReAct (Reasoning and Acting) framework with LangChain and Claude AI to create comprehensive blog posts with images.

## Features

- **ReAct Framework**: Uses Thought → Action → Observation pattern for intelligent blog generation
- **Real-time Streaming**: Stream the blog generation process in real-time via Server-Sent Events
- **Image Integration**: Automatically includes relevant images in blog posts
- **Multiple Output Formats**: Supports both Markdown and HTML output
- **Search Integration**: Uses Tavily search for gathering current information
- **Comprehensive Sections**: Generates multi-section blog posts with proper structure

## Prerequisites

- Python 3.8+
- Claude API key (Anthropic)
- Tavily API key (for search functionality)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fastapi_blog
```

2. Create a virtual environment:
```bash
python -m venv fastapienv
source fastapienv/bin/activate  # On Windows: fastapienv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your actual API keys

```

**Note**: Never commit your actual API keys to version control. Use the `env.example` file as a template.

## Usage

### Starting the Server

```bash
uvicorn blog_fastapi:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Generate Blog (Synchronous)
```bash
POST /blog
Content-Type: application/json

{
    "topic": "The Future of Artificial Intelligence",
    "html_mode": false
}
```

#### Generate Blog (Streaming)
```bash
POST /blog/stream
Content-Type: application/json

{
    "topic": "The Future of Artificial Intelligence",
    "html_mode": true
}
```

### Example Usage

```python
import requests

# Synchronous generation
response = requests.post("http://localhost:8000/blog", json={
    "topic": "Climate Change Solutions",
    "html_mode": False
})

blog_data = response.json()
print(blog_data["blog"])
```

## ReAct Framework Actions

The system uses the following actions in its reasoning process:

- **SEARCH**: Search for information about a topic
- **IMAGE_SEARCH**: Find relevant images for the blog post
- **ANALYZE**: Analyze gathered information to extract key points
- **WRITE_SECTION**: Write a specific section of the blog
- **COMPILE_BLOG**: Combine all sections into the final blog
- **FINISH**: Complete the task

## Project Structure

```
fastapi_blog/
├── blog_fastapi.py          # Main FastAPI application
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── test_*.py               # Test files
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## Docker Support

Build and run with Docker:

```bash
docker build -t fastapi-blog .
docker run -p 8000:8000 -e CLAUDE_API_KEY=your-key -e TAVILY_API_KEY=your-key fastapi-blog
```

## Testing

Run the test suite:

```bash
python -m pytest test_*.py -v
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `CLAUDE_API_KEY` | Anthropic Claude API key | Yes |
| `TAVILY_API_KEY` | Tavily search API key | Yes |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. 