from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel
import time
import uuid
import json
import os
import requests

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