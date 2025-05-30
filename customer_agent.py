from typing import TypedDict

from fastapi import FastAPI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph
from pydantic import BaseModel

app = FastAPI()


# Step 1: Define state structure
class SimpleState(TypedDict):
    prompt: str
    response: str


# Step 2: Initialize LLaMA 3.2 model from Ollama
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434", timeout=120)

# Step 3: LangChain prompt setup
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "{prompt}")]
)
chain = prompt_template | llm | StrOutputParser()


# Step 4: Define node
def respond(state: SimpleState) -> SimpleState:
    result = chain.invoke({"prompt": state["prompt"]})
    state["response"] = result
    return state


# Step 5: LangGraph setup
graph = StateGraph(SimpleState)
graph.add_node("respond", respond)
graph.set_entry_point("respond")
simple_graph = graph.compile()


# Step 6: FastAPI schema
class PromptInput(BaseModel):
    prompt: str


# Step 7: FastAPI endpoint
@app.post("/ask")
async def ask(input: PromptInput):
    state = {"prompt": input.prompt, "response": ""}
    final = simple_graph.invoke(state)
    return {"response": final["response"]}
