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
# Add DALL-E imports
import openai
from openai import OpenAI
import base64
import requests
from io import BytesIO

# Add Google Images search imports
import os
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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
    model_name: Optional[str] = None  # Add this line
    num_sections: int = 3  # Number of sections to write before compiling
    target_audience: str = "general audience"  # Target audience for the blog
    tone: str = "professional"  # Tone of the blog (professional, casual, friendly, etc.)

class Input(BaseModel):
    topic: str

class BlogRequest(BaseModel):
    topic: str
    html_mode: bool = False
    thread_id: str = None
    llm_provider: Literal["cloud", "local"] = "cloud"
    model_name: Optional[str] = None  # Add this line
    num_sections: int = 3  # Number of sections to write before compiling
    target_audience: str = "general audience"  # Target audience for the blog
    tone: str = "professional"  # Tone of the blog (professional, casual, friendly, etc.)

# Add DALL-E related models
class ImageRequest(BaseModel):
    content: str
    section_title: str = ""
    style: str = "professional"
    size: str = "1024x1024"

class ImageSuggestion(BaseModel):
    prompt: str
    description: str
    placement: str  # "header", "section", "inline"

# Add Google Images search models
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
    sections: List[Dict[str, str]]  # List of {"title": "...", "content": "..."}

# Add OpenAI client initialization
openai_client = None
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        print("✅ OpenAI client initialized successfully")
    else:
        print("⚠️ OPENAI_API_KEY not found in environment variables")
except Exception as e:
    print(f"❌ OpenAI client initialization failed: {e}")

# Make Claude API key optional - only required for cloud provider
claude_api_key = os.getenv("CLAUDE_API_KEY")

# Load Tavily API key from environment (loaded via dotenv)
tavily_api_key = os.getenv("TAVILY_API_KEY")

def get_llm(provider: str = "cloud", model_name: str = None):
    """Get the appropriate LLM based on provider choice and optional model name"""
    if provider == "local":
        # Use specified model or default to llama3.2
        local_model = model_name or "llama3.2"
        try:
            # Always use localhost since we're using --network=host
            return OllamaLLM(model=local_model, base_url="http://localhost:11434")
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize Ollama with model '{local_model}'. Make sure Ollama is running locally with the specified model. Error: {str(e)}"
            )
    elif provider == "cloud":
        # Use specified model or default to claude-3-7-sonnet-20250219
        cloud_model = model_name or "claude-3-7-sonnet-20250219"
        return ChatAnthropic(
            model=cloud_model,
    api_key=claude_api_key,
    temperature=0.7,
    max_tokens=4000
)
    else:
        raise ValueError(f"Invalid LLM provider: {provider}. Must be 'cloud' or 'local'")

# Create ChatPromptTemplate chains with proper message separation
react_template = ChatPromptTemplate.from_messages([
    ("system", """You are a blog writing assistant that follows the ReAct (Reasoning and Acting) framework.

Available actions:
- SEARCH: Search for information about a topic
- IMAGE_SEARCH: Find relevant images for the blog
- ANALYZE: Analyze search results and plan content
- WRITE_SECTION: Write a specific section of the blog
- EDIT: Edit and humanize content to make it sound more natural
- STYLE: Add professional CSS styling to the blog (only for HTML mode)
- COMPILE_BLOG: Combine all sections into final blog post
- FINISH: Complete the task

Important guidelines:
- Use EDIT action to improve sections and make them sound more human-written
- After compiling the blog, always use EDIT to humanize the final content
- For HTML blogs, use STYLE after EDIT to add professional CSS styling
- Only use FINISH after the blog has been compiled, edited, and styled (if HTML mode)

Follow this format:
Thought: [your reasoning about what to do next]
Action: [choose one action from the list above]
Action Input: [specific input for the action]"""),
    ("human", "Topic: {topic}\n\nPrevious steps:\n{context}\n\nProgress: {progress}\n\nWhat should you do next?")
])

analysis_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert content analyst. Extract key insights from search results for blog writing."),
    ("human", "Analyze these search results and extract 3-5 key points for a blog about \"{topic}\":\n\n{search_results}\n\nProvide a numbered list of key points.")
])

section_template = ChatPromptTemplate.from_messages([
    ("system", """You are a skilled blog writer. Create engaging, informative content sections with proper structure and flow.

Target Audience: {target_audience}
Tone: {tone}

Tailor your writing style based on the target audience and tone:
- Adjust complexity and terminology for the audience level
- Use language patterns that match the specified tone
- Include examples and explanations appropriate for the audience
- Maintain consistency with the overall blog tone"""),
    ("human", "Write a detailed blog section about: {section_topic}\n\nContext: This is for a blog post about \"{main_topic}\" targeting {target_audience} with a {tone} tone.\nUse the information gathered so far to write an engaging, informative section.\n\n{image_context}\n\nMake sure to include the provided image naturally within the section content and tailor the writing style to the specified audience and tone.")
])

compile_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert blog compiler. Create a cohesive, well-structured blog post from the provided sections.

Target Audience: {target_audience}
Tone: {tone}

Tailor the compilation based on the target audience and tone:
- Ensure consistent voice and style throughout
- Adjust transitions and flow for the specified audience level
- Maintain the specified tone across all sections
- Use appropriate language complexity for the target audience"""),
    ("human", "Compile these sections into a cohesive blog post about \"{topic}\" for {target_audience} with a {tone} tone:\n\n{sections}\n\nAvailable images:\n{images}\n\nCreate a well-structured blog with:\n1. Engaging title\n2. Introduction\n3. Main content sections with relevant images where appropriate\n4. Conclusion\n\nFormat: {format}\n\nIf images are available, include them strategically throughout the content with descriptive alt text. Ensure the final blog maintains consistency in voice, tone, and audience appropriateness.")
])

# Enhanced editor template for final blog review
editor_template = ChatPromptTemplate.from_messages([
    ("system", """You are a professional blog editor specializing in making AI-generated content sound authentically human-written.

Target Audience: {target_audience}
Tone: {tone}

Tailor your editing based on the target audience and tone:

**Audience-Specific Adjustments:**
- For "general audience": Use accessible language, avoid jargon, explain technical terms
- For "professionals": Include industry terminology, assume domain knowledge
- For "beginners": Use simple explanations, step-by-step guidance, encouraging language
- For "experts": Use advanced concepts, technical depth, assume expertise
- For "students": Educational tone, learning-focused, include examples and explanations

**Tone-Specific Adjustments:**
- "Professional": Formal language, authoritative voice, structured approach
- "Casual": Conversational style, contractions, relaxed language
- "Friendly": Warm, approachable, encouraging, personal touches
- "Authoritative": Confident assertions, expert positioning, decisive language
- "Conversational": Direct address to reader, questions, informal style
- "Educational": Teaching-focused, clear explanations, structured learning

For final blog editing, focus on:
- **Flow & Transitions**: Ensure smooth transitions between sections
- **Human Voice**: Add personality, opinions, and conversational elements appropriate to the tone
- **Engagement**: Include rhetorical questions, direct reader address matching the audience level
- **Variety**: Mix sentence lengths and structures appropriate to the tone
- **Authenticity**: Remove AI-like phrases and adjust formality to match tone
- **Storytelling**: Add anecdotes or examples relevant to the target audience
- **Emotional Connection**: Include relatable experiences for the specific audience
- **Natural Language**: Use contractions, casual phrases, and colloquialisms as appropriate for tone
- **Personal Touch**: Add voice elements that match the specified tone
- **Readability**: Break up long paragraphs, add subheadings if needed for the audience

IMPORTANT: Return ONLY the edited blog content. Do not include any meta-commentary, explanations, or notes about the editing process. Just output the clean, final blog post tailored to the specified audience and tone."""),
    ("human", "Edit this blog post to make it sound authentically human-written for {target_audience} with a {tone} tone:\n\n{content}\n\nReturn only the final edited blog content with no additional commentary.")
])

# CSS styling template for professional blog appearance
style_template = ChatPromptTemplate.from_messages([
    ("system", """You are a professional web designer specializing in creating beautiful, modern blog layouts with CSS.

Target Audience: {target_audience}
Tone: {tone}

Tailor the styling based on the target audience and tone:
- For professional audiences: Clean, corporate styling with subtle colors
- For casual audiences: More vibrant, relaxed styling with friendly elements
- For beginners: Clear, simple layouts with good readability
- For experts: Sophisticated, minimalist designs with advanced typography

Create a complete HTML document with embedded CSS that includes:
- **Modern Typography**: Clean, readable fonts (Google Fonts) appropriate for the tone
- **Responsive Design**: Mobile-friendly layout
- **Professional Color Scheme**: Colors that match the tone and audience
- **Proper Spacing**: Comfortable margins, padding, line-height
- **Visual Hierarchy**: Clear headings, subheadings, and content structure
- **Image Styling**: Responsive images with proper spacing
- **Hover Effects**: Subtle interactive elements appropriate for the audience
- **Clean Layout**: Centered content with max-width for readability

IMPORTANT: Return a complete HTML document with embedded CSS. Include the blog content within a properly styled HTML structure. Do not include any meta-commentary."""),
    ("human", "Style this blog content into a beautiful, professional HTML page for {target_audience} with a {tone} tone:\n\n{content}\n\nReturn only the complete HTML document with inline CSS styling that matches the specified audience and tone.")
])

# Initialize checkpointer for short-term memory
def get_checkpointer():
    """Initialize SQLite checkpointer for state persistence"""
    os.makedirs("checkpoints", exist_ok=True)
    conn = sqlite3.connect("checkpoints/blog_checkpoints.db", check_same_thread=False)
    return SqliteSaver(conn)

def get_search_tool():
    if not tavily_api_key:
        print("⚠️ TAVILY_API_KEY not found - search will use fallback mode")
        return None
    try:
        # Test Tavily connection first
        import requests
        test_response = requests.post(
            'https://api.tavily.com/search',
            json={'api_key': tavily_api_key, 'query': 'test', 'max_results': 1},
            timeout=10
        )
        if test_response.status_code == 200:
            print("✅ Tavily API connection successful")
            return TavilySearchResults(max_results=5)
        else:
            print(f"⚠️ Tavily API test failed with status {test_response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Tavily API connection failed: {str(e)} - using fallback search")
        return None

def generate_fallback_search_results(query: str) -> str:
    """Generate synthetic search results when Tavily API is unavailable"""
    
    # Knowledge base for common topics
    knowledge_base = {
        'running': [
            "Running Benefits: Regular running improves cardiovascular health, strengthens muscles, and boosts mental well-being. Studies show runners have lower risk of heart disease and improved longevity.",
            "Running for Weight Loss: Running burns approximately 100 calories per mile and increases metabolism. Consistent running combined with proper nutrition leads to sustainable weight management.",
            "Running Mental Health: Running releases endorphins, reduces stress hormones, and improves mood. Many runners report decreased anxiety and better sleep quality."
        ],
        'artificial intelligence': [
            "AI Applications: Artificial intelligence is transforming industries through machine learning, natural language processing, and computer vision. Applications include healthcare diagnostics, autonomous vehicles, and personalized recommendations.",
            "AI Ethics: As AI becomes more prevalent, ethical considerations include bias in algorithms, privacy concerns, and job displacement. Responsible AI development focuses on fairness and transparency.",
            "AI Future Trends: Emerging AI trends include generative AI, edge computing, and AI-human collaboration. These developments promise to enhance productivity and create new opportunities."
        ],
        'technology': [
            "Technology Innovation: Modern technology advances include cloud computing, mobile devices, and Internet of Things (IoT). These innovations are reshaping how we work, communicate, and live.",
            "Digital Transformation: Organizations are adopting digital technologies to improve efficiency, customer experience, and competitive advantage. This includes automation, data analytics, and digital platforms.",
            "Technology Impact: Technology has revolutionized education, healthcare, and business operations. Benefits include increased accessibility, improved outcomes, and global connectivity."
        ],
        'health': [
            "Health Benefits: Maintaining good health through proper nutrition, regular exercise, and adequate sleep reduces disease risk and improves quality of life. Preventive care is essential for long-term wellness.",
            "Mental Health: Mental health awareness has increased, highlighting the importance of stress management, social connections, and professional support when needed. Mental wellness is as important as physical health.",
            "Healthy Lifestyle: A healthy lifestyle includes balanced nutrition, regular physical activity, adequate hydration, and avoiding harmful substances. Small consistent changes lead to significant health improvements."
        ],
        'business': [
            "Business Strategy: Successful businesses focus on customer value, market differentiation, and operational efficiency. Strategic planning helps organizations adapt to changing market conditions.",
            "Business Innovation: Innovation drives business growth through new products, services, and processes. Companies that embrace innovation are better positioned for long-term success.",
            "Business Leadership: Effective leadership involves clear communication, team empowerment, and strategic vision. Good leaders inspire others and create positive organizational culture."
        ]
    }
    
    # Find relevant knowledge based on query keywords
    query_lower = query.lower()
    relevant_results = []
    
    for topic, results in knowledge_base.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            relevant_results.extend(results)
    
    # If no specific match, provide general results based on common keywords
    if not relevant_results:
        common_keywords = ['benefits', 'advantages', 'importance', 'tips', 'guide', 'how to', 'best practices']
        if any(keyword in query_lower for keyword in common_keywords):
            relevant_results = [
                f"Key Benefits: {query.title()} offers numerous advantages including improved outcomes, enhanced efficiency, and positive impact on well-being.",
                f"Best Practices: Successful implementation of {query} requires proper planning, consistent execution, and regular evaluation of results.",
                f"Expert Insights: Research shows that {query} can lead to significant improvements when approached systematically with clear goals and measurable outcomes."
            ]
        else:
            relevant_results = [
                f"Overview: {query.title()} is an important topic with various applications and benefits across different contexts.",
                f"Key Considerations: When exploring {query}, it's important to consider multiple perspectives and evidence-based approaches.",
                f"Practical Applications: {query.title()} can be applied in various ways to achieve positive outcomes and meaningful results."
            ]
    
    # Format results
    formatted_results = []
    for i, result in enumerate(relevant_results[:3]):  # Limit to 3 results
        formatted_results.append(f"- Result {i+1}: {result}")
    
    return "\n".join(formatted_results)

def generate_react_step(topic: str, context: str, progress_message: str, llm) -> tuple:
    """Generate a single ReAct step (thought, action, action_input)"""
    
    # Create the ReAct chain
    react_chain = react_template | llm | StrOutputParser()
    
    # Generate the response
    response = react_chain.invoke({
        "topic": topic,
        "context": context,
        "progress": progress_message
    })
    
    # Parse the response to extract thought, action, and input
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
    action_match = re.search(r'Action:\s*(\w+)', response)
    input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', response, re.DOTALL)
    
    thought = thought_match.group(1).strip() if thought_match else "No thought provided"
    action = action_match.group(1) if action_match else "SEARCH"
    action_input = input_match.group(1).strip() if input_match else ""
    
    return thought, action, action_input

def react_agent(state: ReActState) -> ReActState:
    """Execute one step of the ReAct agent"""
    
    # Get the appropriate LLM for this state
    llm = get_llm(state.llm_provider, state.model_name)
    
    # Count written sections and check compilation/editing status
    written_sections = len([step for step in state.react_trace if step.get('action') == 'WRITE_SECTION'])
    compiled = any(step.get('action') == 'COMPILE_BLOG' for step in state.react_trace)
    edited = any(step.get('action') == 'EDIT' and 'final blog' in step.get('observation', '').lower() for step in state.react_trace)
    styled = any(step.get('action') == 'STYLE' for step in state.react_trace)
    
    # Determine progress message based on current state
    if compiled and edited and state.html_mode and not styled:
        progress_message = "Blog compiled and edited! Now use STYLE to add professional CSS styling before finishing."
    elif compiled and edited and (not state.html_mode or styled):
        progress_message = "Blog completed! Use FINISH to complete the blog generation."
    elif compiled and not edited:
        progress_message = "Blog compiled! Now use EDIT with 'final blog' as input to humanize the content before finishing."
    elif written_sections >= state.num_sections:
        progress_message = "You have written enough sections. Next step should be COMPILE_BLOG to combine all sections."
    else:
        progress_message = f"Continue writing sections. You need {state.num_sections - written_sections} more sections before compiling."
    
    # Build context from previous steps
    context = build_react_context(state)
    
    # Generate thought, action, and action_input
    thought, action, action_input = generate_react_step(state.topic, context, progress_message, llm)
    
    # Store the step (observation will be added after execution)
    step = {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "observation": ""
    }
    state.react_trace.append(step)
    
    # Execute the action and get observation
    observation = execute_action(state, action, action_input, llm)
    
    # Update the observation in the stored step
    state.react_trace[-1]["observation"] = observation
    
    # Increment current step
    state.current_step += 1
    
    # Check if task is finished
    if action == "FINISH":
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

def execute_action(state: ReActState, action: str, action_input: str, llm) -> str:
    """Execute the specified action and return the result"""
    
    if action == "SEARCH":
        try:
            search_tool = get_search_tool()
            if search_tool:
                # Use Tavily search
                results = search_tool.invoke({"query": action_input})
                if isinstance(results, list) and results:
                    formatted_results = "\n".join([f"- {r.get('title', '')}: {r.get('content', '')[:200]}..." 
                                                 for r in results[:3]])
                    return f"Search results for '{action_input}':\n{formatted_results}"
                return f"No results found for '{action_input}'"
            else:
                # Fallback: Generate synthetic search results based on the query
                print(f"Using fallback search for: {action_input}")
                fallback_results = generate_fallback_search_results(action_input)
                return f"Search results for '{action_input}' (using knowledge base):\n{fallback_results}"
        except Exception as e:
            print(f"Search error: {str(e)}")
            # Fallback: Generate synthetic search results
            fallback_results = generate_fallback_search_results(action_input)
            return f"Search results for '{action_input}' (fallback mode):\n{fallback_results}"
    
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
            "target_audience": state.target_audience,
            "tone": state.tone,
            "image_context": image_context
        })
        return f"Section written:\n{response}"
    
    elif action == "EDIT":
        # Check if we're editing the final compiled blog
        if ("final blog" in action_input.lower() or 
            "finalize" in action_input.lower() or 
            "compile blog" in action_input.lower() or
            "blog post" in action_input.lower() or
            not action_input.strip()):
            # Get the compiled blog from state
            if state.final_blog:
                editor_chain = editor_template | llm | StrOutputParser()
                response = editor_chain.invoke({
                    "content": state.final_blog,
                    "target_audience": state.target_audience,
                    "tone": state.tone
                })
                # Update the final blog with edited version
                state.final_blog = response
                return f"Final blog edited and humanized successfully. The content now sounds more natural and engaging. Ready to finish."
            else:
                return "No compiled blog available to edit. Please compile the blog first using COMPILE_BLOG."
        else:
            # Edit specific content provided
            editor_chain = editor_template | llm | StrOutputParser()
            response = editor_chain.invoke({
                "content": action_input,
                "target_audience": state.target_audience,
                "tone": state.tone
            })
            return f"Content edited and humanized:\n\n{response}"
    
    elif action == "STYLE":
        # Apply CSS styling to the final blog (HTML mode only)
        if not state.html_mode:
            return "STYLE action is only available in HTML mode. Please use HTML mode to apply CSS styling."
        
        if state.final_blog:
            style_chain = style_template | llm | StrOutputParser()
            response = style_chain.invoke({
                "content": state.final_blog,
                "target_audience": state.target_audience,
                "tone": state.tone
            })
            # Update the final blog with styled version
            state.final_blog = response
            return f"Professional CSS styling applied successfully! The blog now has a modern, responsive design with beautiful typography and layout."
        else:
            return "No compiled blog available to style. Please compile and edit the blog first."
    
    elif action == "COMPILE_BLOG":
        # Get all written sections
        sections = []
        images = []
        
        for step in state.react_trace:
            if step.get('action') == 'WRITE_SECTION':
                sections.append(step.get('observation', ''))
            elif step.get('action') == 'IMAGE_SEARCH':
                images.extend(step.get('observation', '').split('\n'))
        
        if sections:
            compile_chain = compile_template | llm | StrOutputParser()
            
            # Prepare sections text
            sections_text = "\n\n".join([f"Section {i+1}:\n{section}" for i, section in enumerate(sections)])
            
            # Prepare images text
            images_text = "\n".join([img for img in images if img.strip()]) if images else "No images found"
            
            response = compile_chain.invoke({
                "sections": sections_text,
                "images": images_text,
                "topic": state.topic,
                "format": "HTML" if state.html_mode else "Markdown",
                "target_audience": state.target_audience,
                "tone": state.tone
            })
            
            state.final_blog = response
            format_type = "HTML" if state.html_mode else "Markdown"
            return f"Blog compiled successfully in {format_type} format. Next step: use EDIT to humanize the content before finishing."
        
        return "No sections available to compile. Please write some sections first."
    
    elif action == "FINISH":
        if state.final_blog:
            # Check if styling is required and completed for HTML mode
            if state.html_mode:
                styled = any(step.get('action') == 'STYLE' for step in state.react_trace)
                if not styled:
                    return "Cannot finish - HTML blog needs CSS styling. Please use STYLE action first to apply professional styling."
            
            return f"Blog generation completed successfully!\n\n{state.final_blog}"
        else:
            return "Cannot finish - no blog content available. Please compile the blog first."
    
    else:
        return f"Unknown action: {action}"

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
            "sections_needed": current_state.num_sections,
            "sections_remaining": max(0, current_state.num_sections - len([s for s in current_state.react_trace if s.get('action') == 'WRITE_SECTION'])),
            "react_trace": current_state.react_trace,
            "final_blog": current_state.final_blog if current_state.is_complete else None,
            "target_audience": current_state.target_audience,
            "tone": current_state.tone
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
    """Get available models for both local and cloud providers"""
    models = {
        "local": [],
        "cloud": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022", 
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229"
        ]
    }
    
    # Try to get local models from Ollama
    try:
        import requests
        # Always use localhost since we're using --network=host
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_models = response.json()
            models["local"] = [model["name"] for model in ollama_models.get("models", [])]
    except:
        # If Ollama is not available, provide some common model names
        models["local"] = ["llama3.2", "gemma3:12b"]
    
    return models

# Analyze content for relevant images using OpenAI
def analyze_content_for_images(content: str, section_title: str = "") -> List[ImageSuggestion]:
    """
    Deeply analyze blog content and suggest highly relevant images with contextual DALL-E prompts
    """
    try:
        # Use OpenAI for sophisticated content analysis
        client = OpenAI()  # Uses OPENAI_API_KEY from environment
        
        # Enhanced analysis prompt for better context understanding
        analysis_prompt = f"""
        You are an expert content analyst and visual designer. Analyze this blog content deeply to suggest highly relevant, contextual images.

        BLOG CONTENT:
        {content}

        SECTION TITLE (if specific section): {section_title}

        ANALYSIS REQUIREMENTS:
        1. Extract key themes, concepts, and specific topics discussed
        2. Identify concrete examples, case studies, or scenarios mentioned
        3. Note the target audience and tone of the content
        4. Consider what visual elements would best support comprehension
        5. Think about what would engage readers and enhance understanding

        For each image suggestion, create:
        - A detailed, contextual DALL-E prompt that directly relates to the content
        - Include specific visual elements that reflect the actual topics discussed
        - Consider the tone (professional, casual, technical, etc.)
        - Specify composition, style, colors that match the content theme
        - Make it relevant to the specific concepts, not just generic

        EXAMPLE OF GOOD vs BAD:
        BAD: "A professional illustration about technology"
        GOOD: "A detailed illustration showing a healthcare professional using an AI diagnostic tool on a tablet, with patient data visualizations and medical charts in the background, modern hospital setting, clean blue and white color scheme, professional medical photography style"

        Respond in this exact JSON format:
        {{
            "suggestions": [
                {{
                    "prompt": "highly detailed, contextual DALL-E prompt that directly relates to specific content discussed",
                    "description": "specific description of what this image shows and why it's relevant",
                    "placement": "header|section|inline",
                    "relevance_score": 0.9,
                    "key_concepts": ["concept1", "concept2", "concept3"]
                }}
            ]
        }}

        Generate 2-3 suggestions that are HIGHLY SPECIFIC to the actual content discussed.
        """
        
        response = client.chat.completions.create(
            model="gpt-4",  # Use GPT-4 for better analysis
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert content analyst and visual designer who creates highly contextual, relevant image suggestions. You deeply understand content themes and create specific, targeted DALL-E prompts that directly support the written material."
                },
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,  # Lower temperature for more focused analysis
            max_tokens=1500
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        suggestions = []
        for item in result.get("suggestions", []):
            suggestions.append(ImageSuggestion(
                prompt=item["prompt"],
                description=item["description"],
                placement=item["placement"]
            ))
        
        return suggestions
        
    except Exception as e:
        print(f"Error in enhanced content analysis: {e}")
        # Return contextual fallback based on content keywords
        return create_fallback_suggestions(content, section_title)

def create_fallback_suggestions(content: str, section_title: str = "") -> List[ImageSuggestion]:
    """
    Create fallback image suggestions based on keyword analysis when AI analysis fails
    """
    # Extract key terms and themes from content
    content_lower = content.lower()
    
    # Define theme-based prompts
    themes = {
        'ai': "A sophisticated AI neural network visualization with glowing nodes and connections, futuristic blue and purple color scheme, high-tech digital art style",
        'healthcare': "Modern healthcare professionals collaborating with digital technology, clean medical environment, professional photography style with soft lighting",
        'business': "Professional business team in a modern office environment, collaborative workspace, natural lighting, corporate photography style",
        'technology': "Cutting-edge technology interface with holographic displays and data visualizations, sleek modern design, blue and white color palette",
        'education': "Diverse students engaged in interactive learning with digital tools, bright classroom environment, educational photography style",
        'finance': "Financial data visualization with charts and graphs on modern displays, professional trading floor atmosphere, blue and green color scheme",
        'environment': "Sustainable technology and green energy solutions, solar panels and wind turbines, natural landscape, environmental photography",
        'food': "Fresh, healthy ingredients artfully arranged, natural lighting, food photography style with vibrant colors",
        'travel': "Scenic destination with cultural landmarks, golden hour lighting, travel photography style with rich colors",
        'fitness': "Active lifestyle with modern fitness equipment, energetic atmosphere, sports photography with dynamic lighting"
    }
    
    # Find matching themes
    detected_themes = []
    for theme, prompt in themes.items():
        if theme in content_lower or any(keyword in content_lower for keyword in [theme + 's', theme + 'ing']):
            detected_themes.append((theme, prompt))
    
    # Create suggestions based on detected themes
    suggestions = []
    if detected_themes:
        for i, (theme, prompt) in enumerate(detected_themes[:2]):
            suggestions.append(ImageSuggestion(
                prompt=f"{prompt}, related to {section_title if section_title else 'the main topic'}, high quality, detailed",
                description=f"Contextual illustration related to {theme} and the content discussed",
                placement="section" if i == 0 else "inline"
            ))
    else:
        # Generic but contextual fallback
        suggestions.append(ImageSuggestion(
            prompt=f"Professional illustration representing the concepts discussed in '{section_title if section_title else 'this content'}', modern design, clean composition, relevant visual metaphors, high quality",
            description="Conceptual illustration supporting the main ideas",
            placement="header"
        ))
    
    return suggestions

# Generate image using DALL-E
def generate_image_with_dalle(prompt: str, size: str = "1024x1024") -> Dict[str, Any]:
    """
    Generate an image using DALL-E based on the provided prompt
    """
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        
        return {
            "success": True,
            "image": {
                "url": image_url,
                "prompt": prompt,
                "size": size
            }
        }
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Download and encode image as base64
def download_and_encode_image(image_url: str) -> str:
    """
    Download an image from URL and encode it as base64
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Encode as base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"Error downloading image: {e}")
        return ""

@app.post("/blog/suggest-images")
async def suggest_images(request: ImageRequest):
    """
    Analyze content and suggest relevant images
    """
    try:
        # Use OpenAI directly instead of passing LLM
        suggestions = analyze_content_for_images(request.content, request.section_title)
        
        return {
            "success": True,
            "suggestions": [
                {
                    "prompt": s.prompt,
                    "description": s.description,
                    "placement": s.placement
                }
                for s in suggestions
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error suggesting images: {str(e)}")

@app.post("/blog/generate-image")
async def generate_image(request: ImageRequest):
    """
    Generate an image based on deep content analysis and user preferences
    """
    try:
        # First analyze the content to get contextual suggestions
        suggestions = analyze_content_for_images(request.content, request.section_title)
        
        # Use the best suggestion as base, or create enhanced prompt
        if suggestions:
            base_prompt = suggestions[0].prompt
        else:
            # Fallback: create contextual prompt from content and section
            base_prompt = f"Professional illustration related to {request.section_title if request.section_title else 'the main topic'}"
        
        # Enhance the prompt with style preferences and ensure quality
        style_enhancements = {
            "professional": "clean, corporate, modern design, high quality, detailed",
            "creative": "artistic, imaginative, vibrant colors, creative composition, high quality",
            "minimalist": "clean, simple, minimal design, elegant, high quality",
            "technical": "detailed, precise, technical illustration, informative, high quality",
            "friendly": "warm, approachable, inviting, colorful, high quality",
            "modern": "contemporary, sleek, cutting-edge design, high quality"
        }
        
        style_addition = style_enhancements.get(request.style, "professional, high quality, detailed")
        
        # Create the final enhanced prompt
        if request.section_title:
            enhanced_prompt = f"{base_prompt}, specifically for section about '{request.section_title}', {style_addition}"
        else:
            enhanced_prompt = f"{base_prompt}, {style_addition}"
        
        # Ensure prompt is not too long (DALL-E has limits)
        if len(enhanced_prompt) > 400:
            enhanced_prompt = enhanced_prompt[:400] + "..."
        
        print(f"Generated contextual DALL-E prompt: {enhanced_prompt}")
        
        # Generate image
        result = generate_image_with_dalle(enhanced_prompt, request.size)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Image generation failed: {result['error']}")
        
        # Download and encode the image
        base64_image = download_and_encode_image(result["image"]["url"])
        
        return {
            "success": True,
            "image": result["image"],
            "base64_image": base64_image,
            "enhanced_prompt": enhanced_prompt,  # Return the enhanced prompt for debugging
            "section_title": request.section_title
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# Google Images Search Functions
def search_google_images(query: str, num_results: int = 10) -> GoogleImageSearchResponse:
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
            image_result = GoogleImageResult(
                title=item.get("title", ""),
                link=item.get("link", ""),
                thumbnail=item.get("image", {}).get("thumbnailLink", ""),
                width=item.get("image", {}).get("width", 0),
                height=item.get("image", {}).get("height", 0),
                source=item.get("displayLink", "")
            )
            results.append(image_result)
        
        return GoogleImageSearchResponse(
            query=query,
            results=results,
            total_results=len(results)
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Google API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Google Images: {str(e)}")

def generate_smart_image_query(content: str, section_title: str = "") -> str:
    """
    Generate an optimized search query for images based on content and section title
    """
    try:
        # Use OpenAI to generate a smart search query
        if openai_client:
            prompt = f"""
            Based on this blog content and section title, generate a concise, effective Google Images search query (2-4 words max) that would find the most relevant, professional images.

            Section Title: {section_title}
            Content: {content[:500]}...

            Focus on:
            - Key concepts and themes
            - Visual elements that would enhance understanding
            - Professional, high-quality imagery
            - Avoid overly specific or niche terms

            Return only the search query, nothing else.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating effective image search queries. Generate concise, professional search terms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            query = response.choices[0].message.content.strip().strip('"').strip("'")
            return query
        else:
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

@app.post("/blog/search-google-images")
async def search_google_images_endpoint(request: GoogleImageSearchRequest):
    """
    Search for images using Google Custom Search API
    """
    try:
        result = search_google_images(request.query, request.num_results)
        return {
            "success": True,
            "query": result.query,
            "results": [
                {
                    "id": f"google-{i}",
                    "title": img.title,
                    "link": img.link,
                    "thumbnail": img.thumbnail,
                    "width": img.width,
                    "height": img.height,
                    "source": img.source
                }
                for i, img in enumerate(result.results)
            ],
            "total_results": result.total_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Google Images: {str(e)}")

@app.post("/blog/search-images-for-sections")
async def search_images_for_sections(request: GoogleImageSectionRequest):
    """
    Search for images for multiple blog sections automatically
    """
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
                            "title": img.title,
                            "link": img.link,
                            "thumbnail": img.thumbnail,
                            "width": img.width,
                            "height": img.height,
                            "source": img.source
                        }
                        for i, img in enumerate(search_result.results)
                    ],
                    "total_results": search_result.total_results
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
