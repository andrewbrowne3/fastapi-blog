# FastAPI Blog Generator

An AI-powered blog generation service using FastAPI, LangChain, and ReAct agents.

## Features

- AI-powered blog generation with multiple LLM providers (OpenAI, Anthropic, Local Ollama)
- Web search integration via Tavily API
- ReAct (Reasoning + Acting) agent for intelligent content creation
- Fallback knowledge base when search APIs are unavailable
- RESTful API with FastAPI

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   cd fastapi_blog
   ```

2. **Set up environment variables:**
   Create a `.env` file:
   ```bash
   # Required: Get from https://tavily.com
   TAVILY_API_KEY=your_tavily_api_key_here
   
   # Optional: For better AI responses
   ANTHROPIC_API_KEY=your_anthropic_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```

3. **Run with Docker:**
   ```bash
   docker-compose up --build
   ```

### Option 2: Local Development

1. **Create virtual environment:**
   ```bash
   python3 -m venv fastapienv
   source fastapienv/bin/activate  # On Windows: fastapienv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file with your API keys (see above)

4. **Run the server:**
   ```bash
   uvicorn blog_fastapi:app --host 0.0.0.0 --port 7999 --reload
   ```

## API Usage

### Generate Blog Post

**POST** `http://localhost:7999/blog`

```json
{
  "topic": "Artificial Intelligence in Healthcare",
  "llm_provider": "local",
  "model_name": "llama3.2",
  "num_sections": 3,
  "target_audience": "general",
  "tone": "informative"
}
```

**Response:**
```json
{
  "blog_content": "# Artificial Intelligence in Healthcare\n\n...",
  "status": "success"
}
```

### Parameters

- `topic` (required): The blog topic
- `llm_provider`: "openai", "anthropic", or "local" (default: "local")
- `model_name`: Model to use (default: "llama3.2")
- `num_sections`: Number of sections (default: 3)
- `target_audience`: Target audience (default: "general")
- `tone`: Writing tone (default: "informative")

## API Keys Setup

### 1. Tavily API Key (Required for web search)
- Visit [tavily.com](https://tavily.com)
- Sign up and get your API key
- Add to `.env`: `TAVILY_API_KEY=your_key_here`

### 2. Anthropic API Key (Optional)
- Visit [console.anthropic.com](https://console.anthropic.com)
- Get your API key
- Add to `.env`: `ANTHROPIC_API_KEY=your_key_here`

### 3. OpenAI API Key (Optional)
- Visit [platform.openai.com](https://platform.openai.com)
- Get your API key
- Add to `.env`: `OPENAI_API_KEY=your_key_here`

## Local LLM Setup (Ollama)

For local LLM support:

1. **Install Ollama:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull a model:**
   ```bash
   ollama pull llama3.2
   ```

3. **Start Ollama service:**
   ```bash
   ollama serve
   ```

## Troubleshooting

### Common Issues

1. **"Connection refused" error:**
   - Ensure the server is running on port 7999
   - Check if another service is using the port

2. **"Invalid API key" errors:**
   - Verify your API keys in the `.env` file
   - Check if keys have expired

3. **"Module not found" errors:**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

4. **Timeout errors:**
   - Local LLMs may be slow; increase timeout
   - Consider using cloud APIs for faster responses

### Testing the API

```bash
curl -X POST "http://localhost:7999/blog" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "topic": "Test Topic",
    "llm_provider": "local",
    "model_name": "llama3.2",
    "num_sections": 1
  }'
```

## Architecture

- **FastAPI**: Web framework and API server
- **LangChain**: LLM orchestration and chaining
- **LangGraph**: State management for ReAct agents
- **Tavily**: Web search API
- **Ollama**: Local LLM inference

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

