from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel
import os
import json
import re
import time
from typing import List, Dict, Any

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
    max_steps: int = 12  # Increased for image search steps
    thoughts: List[str] = []
    actions: List[str] = []
    observations: List[str] = []
    research_data: List[Dict[str, Any]] = []
    blog_sections: Dict[str, str] = {}
    react_trace: List[Dict[str, str]] = []
    images: List[Dict[str, str]] = []  # Store found images
    final_blog: str = ""
    is_complete: bool = False

class BlogRequest(BaseModel):
    topic: str
    include_images: bool = True

# Initialize LLM
claude_api_key = os.getenv("CLAUDE_API_KEY")
if not claude_api_key:
    raise ValueError("CLAUDE_API_KEY environment variable is required")

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=claude_api_key,
    temperature=0.7,
    max_tokens=4000
)

def get_search_tool():
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    return TavilySearchResults(api_key=tavily_api_key)

def get_image_search_tool():
    """Get Tavily search tool configured for images"""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    return TavilySearchResults(
        api_key=tavily_api_key,
        search_depth="basic",
        include_images=True,
        include_answer=False,
        max_results=5
    )

def react_agent(state: ReActState) -> ReActState:
    """Enhanced ReAct agent with image search capabilities"""
    
    if state.current_step >= state.max_steps or state.is_complete:
        return state
    
    # Build context from previous steps
    context = build_react_context(state)
    
    # Enhanced ReAct prompt with image search
    react_prompt = f"""You are a blog writing assistant using the ReAct framework.
    
Task: Create a comprehensive HTML blog post about "{state.topic}" with relevant images.

Follow this exact format for each step:
Thought: [Your reasoning about what to do next]
Action: [One of: SEARCH, SEARCH_IMAGES, ANALYZE, WRITE_SECTION, COMPILE_BLOG, FINISH]
Action Input: [Specific input for the action]

Available Actions:
- SEARCH: Search for information about a topic
- SEARCH_IMAGES: Search for relevant images for the blog
- ANALYZE: Analyze gathered information to extract key points
- WRITE_SECTION: Write a specific section of the blog
- COMPILE_BLOG: Combine all sections into final HTML blog with images
- FINISH: Complete the task

Previous context:
{context}

Current step {state.current_step + 1}:
"""

    response = llm.invoke([{"role": "user", "content": react_prompt}])
    
    # Parse the ReAct response
    thought, action, action_input = parse_react_response(response.content)
    
    # Execute the action and get observation
    observation = execute_action(action, action_input, state)
    
    # Store this cycle
    state.react_trace.append({
        "step": state.current_step + 1,
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "observation": observation
    })
    
    state.current_step += 1
    
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
    
    elif action == "SEARCH_IMAGES":
        try:
            image_search_tool = get_image_search_tool()
            results = image_search_tool.invoke({"query": f"{action_input} images"})
            
            if isinstance(results, list) and results:
                images_found = []
                for r in results[:3]:  # Get top 3 image results
                    if 'images' in r and r['images']:
                        for img in r['images'][:2]:  # Max 2 images per result
                            images_found.append({
                                'url': img,
                                'alt': f"Image related to {action_input}",
                                'caption': f"Visual representation of {action_input}"
                            })
                
                # Store images in state
                state.images.extend(images_found[:4])  # Max 4 images total
                
                if images_found:
                    image_list = "\n".join([f"- {img['url']} (Alt: {img['alt']})" for img in images_found])
                    return f"Found {len(images_found)} relevant images for '{action_input}':\n{image_list}"
                else:
                    return f"No images found for '{action_input}'"
            return f"No image results found for '{action_input}'"
        except Exception as e:
            return f"Image search failed: {str(e)}"
    
    elif action == "ANALYZE":
        # Analyze previous search results to extract key points
        search_results = get_recent_search_results(state)
        if search_results:
            analysis_prompt = f"""Analyze these search results and extract 3-5 key points for a blog about "{state.topic}":
            
{search_results}

Provide a numbered list of key points."""
            
            response = llm.invoke([{"role": "user", "content": analysis_prompt}])
            return f"Analysis complete. Key points identified:\n{response.content}"
        return "No search results to analyze"
    
    elif action == "WRITE_SECTION":
        # Write a specific section based on the action input
        section_prompt = f"""Write a detailed blog section about: {action_input}
        
Context: This is for a blog post about "{state.topic}"
Use the information gathered so far to write an engaging, informative section.
Format as HTML with proper tags (h2, h3, p, strong, em, ul, li).
"""
        response = llm.invoke([{"role": "user", "content": section_prompt}])
        return f"Section written:\n{response.content}"
    
    elif action == "COMPILE_BLOG":
        # Compile all written sections into final HTML blog with images
        sections = get_written_sections(state)
        images_html = get_images_html(state)
        
        if sections:
            compile_prompt = f"""Compile these sections into a cohesive HTML blog post about "{state.topic}":

SECTIONS:
{sections}

AVAILABLE IMAGES:
{images_html}

Create a well-structured HTML blog with:
1. Engaging h1 title
2. Introduction paragraph
3. Main content sections with h2/h3 headings
4. Strategically placed images with proper img tags, alt text, and captions
5. Conclusion paragraph

Format as clean, semantic HTML with:
- Proper HTML structure (h1, h2, h3 for headings)
- Paragraphs in <p> tags
- Lists as <ul>/<ol> with <li> items
- Bold text in <strong> tags
- Italic text in <em> tags
- Images with <img> tags, alt attributes, and <figure>/<figcaption> for captions
- No need for full HTML document structure (no <html>, <head>, <body> tags)
- Just the content portion ready to be inserted into a webpage

Integrate the images naturally throughout the content where they make sense contextually.
"""
            
            response = llm.invoke([{"role": "user", "content": compile_prompt}])
            state.final_blog = response.content
            return f"HTML blog compiled successfully with {len(state.images)} images integrated. Final blog ready."
        return "No sections available to compile"
    
    elif action == "FINISH":
        if state.final_blog:
            return f"Task completed. HTML blog post about '{state.topic}' is ready with images."
        return "Task marked as finished but no final blog available"
    
    else:
        return f"Unknown action: {action}. Please use SEARCH, SEARCH_IMAGES, ANALYZE, WRITE_SECTION, COMPILE_BLOG, or FINISH"

def build_react_context(state: ReActState) -> str:
    """Build context string from previous ReAct steps"""
    if not state.react_trace:
        return "No previous steps."
    
    context_parts = []
    for step in state.react_trace[-3:]:  # Show last 3 steps for context
        context_parts.append(f"""Step {step['step']}:
Thought: {step['thought']}
Action: {step['action']} - {step['action_input']}
Observation: {step['observation'][:200]}...
""")
    
    return "\n".join(context_parts)

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

def get_images_html(state: ReActState) -> str:
    """Format images for HTML integration"""
    if not state.images:
        return "No images available"
    
    images_html = []
    for i, img in enumerate(state.images):
        images_html.append(f"""Image {i+1}:
<figure>
  <img src="{img['url']}" alt="{img['alt']}" style="max-width: 100%; height: auto;">
  <figcaption>{img['caption']}</figcaption>
</figure>""")
    
    return "\n\n".join(images_html)

def should_continue(state: ReActState) -> str:
    """Determine if ReAct loop should continue"""
    if state.is_complete or state.current_step >= state.max_steps:
        return "finish"
    return "continue"

# Create ReAct workflow
workflow = StateGraph(ReActState)
workflow.add_node("react_agent", react_agent)
workflow.set_entry_point("react_agent")
workflow.add_conditional_edges(
    "react_agent",
    should_continue,
    {
        "continue": "react_agent",  # Loop back to continue ReAct cycle
        "finish": "__end__"
    }
)

graph = workflow.compile()

@app.post("/blog")
def generate_blog(request: BlogRequest):
    """Generate HTML blog with images using ReAct pattern"""
    initial_state = ReActState(topic=request.topic)
    final_state = graph.invoke(initial_state)
    
    return {
        "blog": final_state.final_blog,
        "react_trace": final_state.react_trace,
        "steps_taken": final_state.current_step,
        "images_found": len(final_state.images),
        "format": "HTML"
    }

@app.post("/blog/stream")
def generate_blog_stream(request: BlogRequest):
    """Stream the ReAct process in real-time"""
    def generate():
        state = ReActState(topic=request.topic)
        
        while not state.is_complete and state.current_step < state.max_steps:
            # Run one ReAct cycle
            state = react_agent(state)
            
            # Stream the latest step
            if state.react_trace:
                latest_step = state.react_trace[-1]
                yield f"data: {json.dumps(latest_step)}\n\n"
            
            time.sleep(0.1)  # Small delay for streaming effect
        
        # Send final result
        yield f"data: {json.dumps({
            'final_blog': state.final_blog, 
            'complete': True, 
            'format': 'HTML',
            'images_found': len(state.images)
        })}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3004) 