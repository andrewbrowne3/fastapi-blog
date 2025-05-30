from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
import os
import json
import re
import time
import uuid
import sqlite3
from typing import List, Dict, Any, Optional, Literal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReActState(BaseModel):
    topic: str
    current_step: int = 0
    max_steps: int = 30
    thoughts: List[str] = []
    actions: List[str] = []
    observations: List[str] = []
    research_data: List[Dict[str, Any]] = []
    blog_sections: Dict[str, str] = {}
    react_trace: List[Dict[str, Any]] = []  # Store thought/action/observation cycles
    final_blog: str = ""
    html_mode: bool = False  # New field for HTML output mode
    is_complete: bool = False
    thread_id: str = ""  # Add thread_id for session management
    llm_provider: str = "cloud"  # Track which LLM provider is being used

class Input(BaseModel):
    topic: str

class BlogRequest(BaseModel):
    topic: str
    html_mode: bool = False  # New field for HTML output mode
    thread_id: str = None  # Optional thread_id for resuming
    llm_provider: Literal["cloud", "local"] = "cloud"  # Choose between cloud (Claude) or local (Ollama)

# Make Claude API key optional - only required for cloud provider
claude_api_key = os.getenv("CLAUDE_API_KEY")

def get_llm(provider: str = "cloud"):
    """Get the appropriate LLM based on provider choice"""
    if provider == "local":
        try:
            return OllamaLLM(model="llama3.2")
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize Ollama. Make sure Ollama is running locally with llama3.2 model. Error: {str(e)}"
            )
    elif provider == "cloud":  # cloud provider (Claude)
        if not claude_api_key:
            raise HTTPException(
                status_code=400,
                detail="CLAUDE_API_KEY environment variable is required for cloud provider"
            )
        return ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            api_key=claude_api_key,
            temperature=0.7,
            max_tokens=4000
        )
    else:
        raise ValueError(f"Invalid LLM provider: {provider}. Must be 'cloud' or 'local'")

# Create ChatPromptTemplate chains with proper message separation
react_template = ChatPromptTemplate.from_messages([
    ("system", """You are a blog writing assistant using the ReAct framework.
    
Task: Create a comprehensive blog post about "{topic}" with multiple sections, each containing at least one relevant image.

Follow this exact format for each step:
Thought: [Your reasoning about what to do next]
Action: [One of: SEARCH, IMAGE_SEARCH, ANALYZE, WRITE_SECTION, COMPILE_BLOG, FINISH]
Action Input: [Specific input for the action]

Available Actions (ONLY use these):
- SEARCH: Search for information about a topic
- IMAGE_SEARCH: Search for relevant images for the blog post
- ANALYZE: Analyze gathered information to extract key points
- WRITE_SECTION: Write a specific section of the blog (automatically includes relevant images)
- COMPILE_BLOG: Combine all sections into final blog
- FINISH: Complete the task

CRITICAL RULES:
- NEVER use actions like SET_ENV_VARIABLE, CONFIGURE, or any other actions not listed above
- All environment variables are already configured - just use SEARCH directly
- If you need to search for information, use SEARCH action immediately
- Write at least 3-4 distinct sections using WRITE_SECTION
- Each WRITE_SECTION automatically includes a relevant image
- After writing {max_sections} sections, you MUST use COMPILE_BLOG to create the final blog
- After COMPILE_BLOG, you MUST use FINISH to complete the task

Current progress: {written_sections} sections written
{progress_message}"""),
    ("human", "{context}\n\nCurrent step {current_step}:")
])

analysis_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert content analyst. Extract key insights from search results for blog writing."),
    ("human", "Analyze these search results and extract 3-5 key points for a blog about \"{topic}\":\n\n{search_results}\n\nProvide a numbered list of key points.")
])

section_template = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled blog writer. Create engaging, informative content sections with proper structure and flow."),
    ("human", "Write a detailed blog section about: {section_topic}\n\nContext: This is for a blog post about \"{main_topic}\"\nUse the information gathered so far to write an engaging, informative section.\n\n{image_context}\n\nMake sure to include the provided image naturally within the section content.")
])

compile_template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional blog editor. Compile sections into cohesive, well-structured blog posts with proper formatting."),
    ("human", "Compile these sections into a cohesive blog post about \"{topic}\":\n\n{sections}{image_context}\n\nCreate a well-structured blog with:\n1. Engaging title\n2. Introduction\n3. Main content sections with relevant images where appropriate\n4. Conclusion\n\n{format_instruction}\n\nIf images are available, include them strategically throughout the content with descriptive alt text.")
])

# Initialize checkpointer for short-term memory
def get_checkpointer():
    """Initialize SQLite checkpointer for state persistence"""
    os.makedirs("checkpoints", exist_ok=True)
    conn = sqlite3.connect("checkpoints/blog_checkpoints.db", check_same_thread=False)
    return SqliteSaver(conn)

def get_search_tool():
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    # TavilySearchResults reads TAVILY_API_KEY from environment automatically
    return TavilySearchResults(max_results=5)

def react_agent(state: ReActState, stream_callback=None) -> ReActState:
    """Main ReAct agent that follows Thought -> Action -> Observation pattern"""
    
    if state.current_step >= state.max_steps or state.is_complete:
        return state
    
    # Get the appropriate LLM for this state
    llm = get_llm(state.llm_provider)
    
    # Create dynamic chain with the selected LLM
    react_chain = react_template | llm | StrOutputParser()
    
    # Build context from previous steps
    context = build_react_context(state)
    
    # Count written sections
    written_sections = len([step for step in state.react_trace if step.get('action') == 'WRITE_SECTION'])
    
    # Determine progress message
    if written_sections >= 3:
        progress_message = "You have written enough sections. Next step should be COMPILE_BLOG."
    else:
        progress_message = f"Continue writing sections. Need {3 - written_sections} more sections."
    
    # Use ChatPromptTemplate with proper variables
    response = react_chain.invoke({
        "topic": state.topic,
        "max_sections": max(3, written_sections),
        "written_sections": written_sections,
        "progress_message": progress_message,
        "context": context,
        "current_step": state.current_step + 1
    })
    
    # Parse the ReAct response
    thought, action, action_input = parse_react_response(response)
    
    # Store the thought and action immediately (before observation)
    step_data = {
        "step": state.current_step + 1,
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "observation": "Processing..."  # Placeholder while executing
    }
    state.react_trace.append(step_data)
    state.current_step += 1
    
    # Stream the thought and action immediately if callback provided
    if stream_callback:
        stream_callback(step_data)
    
    # Execute the action and get observation
    observation = execute_action(action, action_input, state)
    
    # Update the observation in the stored step
    state.react_trace[-1]["observation"] = observation
    
    # Stream the updated step with observation if callback provided
    if stream_callback:
        stream_callback(state.react_trace[-1])
    
    # Check if task is complete
    if action == "FINISH" or "final blog" in observation.lower():
        state.is_complete = True
    
    return state

def parse_react_response(response: str) -> tuple[str, str, str]:
    """Parse ReAct formatted response into components"""
    thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', response, re.DOTALL)
    action_match = re.search(r'Action:\s*(.*?)(?=Action Input:|$)', response, re.DOTALL)
    input_match = re.search(r'Action Input:\s*(.*?)(?=\n|$)', response, re.DOTALL)
    
    thought = thought_match.group(1).strip() if thought_match else "No thought provided"
    action = action_match.group(1).strip() if action_match else "SEARCH"
    action_input = input_match.group(1).strip() if input_match else ""
    
    return thought, action, action_input

def execute_action(action: str, action_input: str, state: ReActState) -> str:
    """Execute the specified action and return observation"""
    
    # Get the appropriate LLM for this state
    llm = get_llm(state.llm_provider)
    
    if action == "SEARCH":
        try:
            search_tool = get_search_tool()
            results = search_tool.invoke({"query": action_input})
            if isinstance(results, list) and results:
                formatted_results = "\n".join([f"- {r.get('title', '')}: {r.get('content', '')[:200]}..." 
                                             for r in results[:3]])
                return f"Search results for '{action_input}':\n{formatted_results}"
            return f"No results found for '{action_input}'"
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    elif action == "IMAGE_SEARCH":
        # Search for relevant images using Unsplash or similar API
        try:
            # For now, we'll generate placeholder image URLs based on the search term
            # You can replace this with actual image search API calls
            search_term = action_input.replace(" ", "+")
            image_suggestions = [
                f"https://images.unsplash.com/photo-1234567890?q={search_term}&w=800&h=400&fit=crop",
                f"https://images.unsplash.com/photo-1234567891?q={search_term}&w=600&h=300&fit=crop",
                f"https://images.unsplash.com/photo-1234567892?q={search_term}&w=400&h=200&fit=crop"
            ]
            return f"Found relevant images for '{action_input}':\n" + "\n".join([f"- {img}" for img in image_suggestions])
        except Exception as e:
            return f"Image search failed: {str(e)}"
    
    elif action == "ANALYZE":
        # Analyze previous search results to extract key points
        search_results = get_recent_search_results(state)
        if search_results:
            # Use dynamic LLM for analysis
            analysis_chain = analysis_template | llm | StrOutputParser()
            response = analysis_chain.invoke({
                "topic": state.topic,
                "search_results": search_results
            })
            return f"Analysis complete. Key points identified:\n{response}"
        return "No search results to analyze"
    
    elif action == "WRITE_SECTION":
        # Write a specific section based on the action input
        # First, search for images for this specific section
        section_images = []
        try:
            search_term = action_input.replace(" ", "+")
            # Generate more realistic Unsplash photo IDs based on the section topic
            base_ids = [
                "1506905925173", "1441974231531", "1500648767791", "1519389950473", 
                "1542281286", "1506905925173", "1441974231531", "1500648767791"
            ]
            photo_id = base_ids[hash(action_input) % len(base_ids)]
            section_images = [
                f"https://images.unsplash.com/photo-{photo_id}?q={search_term}&w=800&h=400&fit=crop&auto=format"
            ]
        except:
            pass
        
        image_context = ""
        if section_images:
            if state.html_mode:
                image_context = f"\n\nInclude this image in the section:\n<img src=\"{section_images[0]}\" alt=\"{action_input}\" style=\"width: 100%; max-width: 600px; height: auto; margin: 20px 0;\">\n"
            else:
                image_context = f"\n\nInclude this image in the section:\n![{action_input}]({section_images[0]})\n"
        
        # Use dynamic LLM for section writing
        section_chain = section_template | llm | StrOutputParser()
        response = section_chain.invoke({
            "section_topic": action_input,
            "main_topic": state.topic,
            "image_context": image_context
        })
        return f"Section written:\n{response}"
    
    elif action == "COMPILE_BLOG":
        # Compile all written sections into final blog
        sections = get_written_sections(state)
        images = get_found_images(state)
        
        if sections:
            # Determine output format based on html_mode
            format_instruction = "Format with proper markdown headings and flow." if not state.html_mode else """
Format as clean, semantic HTML with:
- Proper HTML structure (h1, h2, h3 for headings)
- Paragraphs in <p> tags
- Lists as <ul>/<ol> with <li> items
- Bold text in <strong> tags
- Italic text in <em> tags
- Images with <img> tags and proper alt text
- No need for full HTML document structure (no <html>, <head>, <body> tags)
- Just the content portion ready to be inserted into a webpage"""
            
            image_context = f"\n\nAvailable images to include:\n{images}" if images else ""
            
            # Use dynamic LLM for compilation
            compile_chain = compile_template | llm | StrOutputParser()
            response = compile_chain.invoke({
                "topic": state.topic,
                "sections": sections,
                "image_context": image_context,
                "format_instruction": format_instruction
            })
            
            state.final_blog = response
            format_type = "HTML" if state.html_mode else "Markdown"
            return f"Blog compiled successfully in {format_type} format. Final blog ready."
        return "No sections available to compile"
    
    elif action == "FINISH":
        if state.final_blog:
            return f"Task completed. Blog post about '{state.topic}' is ready."
        return "Task marked as finished but no final blog available"
    
    else:
        return f"Unknown action: {action}. Please use SEARCH, IMAGE_SEARCH, ANALYZE, WRITE_SECTION, COMPILE_BLOG, or FINISH"

def build_react_context(state: ReActState) -> str:
    """Build comprehensive context from ALL previous steps"""
    if not state.react_trace:
        return "This is the first step. Start by searching for information about the topic."
    
    # Organize information by action type for better context
    searches = []
    analyses = []
    sections = []
    images = []
    
    for step in state.react_trace:
        action = step.get('action', '')
        observation = step.get('observation', '')
        
        if action == 'SEARCH' and observation != "Processing...":
            searches.append(f"Search: {step.get('action_input', '')} -> {observation[:300]}...")
        elif action == 'ANALYZE' and observation != "Processing...":
            analyses.append(f"Analysis: {observation[:300]}...")
        elif action == 'WRITE_SECTION' and observation != "Processing...":
            sections.append(f"Section '{step.get('action_input', '')}': {observation[:200]}...")
        elif action == 'IMAGE_SEARCH' and observation != "Processing...":
            images.append(f"Images found: {observation[:200]}...")
    
    context_parts = []
    
    if searches:
        context_parts.append(f"Research conducted:\n" + "\n".join(searches[-5:]))  # Last 5 searches
    
    if analyses:
        context_parts.append(f"Key insights discovered:\n" + "\n".join(analyses[-3:]))  # Last 3 analyses
    
    if sections:
        context_parts.append(f"Sections completed:\n" + "\n".join(sections))
    
    if images:
        context_parts.append(f"Images found:\n" + "\n".join(images[-3:]))  # Last 3 image searches
    
    # Add recent activity summary
    recent_steps = state.react_trace[-3:] if len(state.react_trace) >= 3 else state.react_trace
    if recent_steps:
        recent_activity = []
        for step in recent_steps:
            recent_activity.append(f"Step {step.get('step', '')}: {step.get('action', '')} - {step.get('thought', '')[:100]}...")
        context_parts.append(f"Recent activity:\n" + "\n".join(recent_activity))
    
    return "\n\n".join(context_parts) if context_parts else "No previous context available."

def get_recent_search_results(state: ReActState) -> str:
    """Extract recent search results from ReAct trace"""
    search_results = []
    for step in reversed(state.react_trace):
        if step['action'] == 'SEARCH' and 'Search results' in step['observation']:
            search_results.append(step['observation'])
            if len(search_results) >= 2:  # Get last 2 search results
                break
    return "\n\n".join(search_results)

def get_written_sections(state: ReActState) -> str:
    """Extract written sections from ReAct trace"""
    sections = []
    for step in state.react_trace:
        if step['action'] == 'WRITE_SECTION' and 'Section written:' in step['observation']:
            sections.append(step['observation'].replace('Section written:\n', ''))
    return "\n\n".join(sections)

def get_found_images(state: ReActState) -> str:
    """Extract found images from ReAct trace"""
    images = []
    for step in state.react_trace:
        if step['action'] == 'IMAGE_SEARCH' and 'Found relevant images' in step['observation']:
            images.append(step['observation'].replace('Found relevant images for ', ''))
    return "\n".join(images)

def should_continue(state: ReActState) -> str:
    """Determine if ReAct loop should continue"""
    if state.is_complete or state.current_step >= state.max_steps:
        return END
    return "react_agent"

# Create the ReAct workflow with checkpointer
def create_blog_graph():
    """Create the blog generation graph with checkpointer"""
    workflow = StateGraph(ReActState)
    
    # Add the main ReAct agent node
    workflow.add_node("react_agent", react_agent)
    
    # Set up the flow
    workflow.add_edge(START, "react_agent")
    workflow.add_conditional_edges(
        "react_agent",
        should_continue,
        {
            "react_agent": "react_agent",  # Continue the loop
            END: END  # End the workflow
        }
    )
    
    # Compile with checkpointer for full state persistence
    checkpointer = get_checkpointer()
    return workflow.compile(checkpointer=checkpointer)

# Create the graph instance
graph = create_blog_graph()

@app.post("/blog")
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
        "recursion_limit": 50  # Increase from default 25 to 50
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
                llm_provider=request.llm_provider
            )
            print(f"Starting new thread {thread_id}")
    except:
        # Start new thread if no existing state
        initial_state = ReActState(
            topic=request.topic,
            html_mode=request.html_mode,
            thread_id=thread_id,
            llm_provider=request.llm_provider
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
        "format": "HTML" if request.html_mode else "Markdown"
    }

@app.post("/blog/stream")
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
            "recursion_limit": 50  # Increase from default 25 to 50
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
                    llm_provider=request.llm_provider
                )
                yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id, 'topic': request.topic})}\n\n"
        except:
            # Start new thread if no existing state
            current_state = ReActState(
                topic=request.topic,
                html_mode=request.html_mode,
                thread_id=thread_id,
                llm_provider=request.llm_provider
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
                        "thread_id": thread_id
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
                            "thread_id": thread_id
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
                "is_complete": current_state.is_complete
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

# New endpoints for managing checkpointed sessions

@app.post("/blog/resume/{thread_id}")
def resume_blog_generation(thread_id: str):
    """Resume blog generation from a specific thread"""
    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 50  # Increase from default 25 to 50
    }
    
    try:
        # Get current state
        state_snapshot = graph.get_state(config)
        if not state_snapshot.values:
            return {"error": "Thread not found", "thread_id": thread_id}
        
        current_state = ReActState(**state_snapshot.values)
        
        if current_state.is_complete:
            return {
                "message": "Blog generation already complete",
                "thread_id": thread_id,
                "blog": current_state.final_blog,
                "steps_completed": current_state.current_step
            }
        
        # Continue from where we left off
        final_result = graph.invoke(current_state, config)
        
        # Convert the result to ReActState if it's not already
        if isinstance(final_result, dict):
            final_state = ReActState(**final_result)
        else:
            final_state = final_result
        
        return {
            "blog": final_state.final_blog,
            "thread_id": thread_id,
            "steps_completed": final_state.current_step,
            "react_trace": final_state.react_trace,
            "is_complete": final_state.is_complete,
            "resumed_from_step": current_state.current_step
        }
        
    except Exception as e:
        return {"error": f"Failed to resume: {str(e)}", "thread_id": thread_id}

@app.get("/blog/state/{thread_id}")
def get_blog_state(thread_id: str):
    """Get the current state of a blog generation thread"""
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    try:
        state_snapshot = graph.get_state(config)
        if not state_snapshot.values:
            return {"error": "Thread not found", "thread_id": thread_id}
        
        current_state = ReActState(**state_snapshot.values)
        
        return {
            "thread_id": thread_id,
            "topic": current_state.topic,
            "current_step": current_state.current_step,
            "is_complete": current_state.is_complete,
            "sections_written": len([s for s in current_state.react_trace if s.get('action') == 'WRITE_SECTION']),
            "react_trace": current_state.react_trace,
            "final_blog": current_state.final_blog if current_state.is_complete else None
        }
        
    except Exception as e:
        return {"error": f"Failed to get state: {str(e)}", "thread_id": thread_id}

@app.get("/blog/history/{thread_id}")
def get_blog_history(thread_id: str):
    """Get the full history of a blog generation thread"""
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    try:
        history = list(graph.get_state_history(config))
        
        if not history:
            return {"error": "Thread not found", "thread_id": thread_id}
        
        return {
            "thread_id": thread_id,
            "total_checkpoints": len(history),
            "history": [
                {
                    "checkpoint_id": h.config["configurable"]["checkpoint_id"],
                    "step": h.metadata.get("step", 0),
                    "created_at": h.created_at,
                    "current_step": h.values.get("current_step", 0) if h.values else 0,
                    "is_complete": h.values.get("is_complete", False) if h.values else False
                }
                for h in history
            ]
        }
        
    except Exception as e:
        return {"error": f"Failed to get history: {str(e)}", "thread_id": thread_id}

@app.get("/models")
def get_available_models():
    """Get available models for both local (Ollama) and cloud (Claude) providers"""
    models = {
        "local": ["gemma3:12b", "llama3.2"],
        "cloud": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022", 
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229"
        ]
    }
    
    # Try to get Ollama models if available
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            ollama_models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # First column is the model name
                    if model_name and not model_name.startswith('NAME'):
                        ollama_models.append(model_name)
            if ollama_models:
                models["local"] = ollama_models
    except Exception as e:
        # Keep the hardcoded models if Ollama is not available
        pass
    
    return models
