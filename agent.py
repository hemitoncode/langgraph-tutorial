import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# 1. Define a simple tool
@tool
def search_tool(query: str) -> str:
    """Searches the web for a given query."""
    # In a real app, this would use a search API
    return f"Result for '{query}': Example search result."

available_tools = [search_tool]

# 2. Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# 3. Define the LLM node
def run_model(state: AgentState) -> dict:
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # Bind tools to the model
    model_with_tools = model.bind_tools(available_tools)
    
    result = model_with_tools.invoke(state["messages"])
    return {"messages": [result]}

# 4. Define the tool node
tool_node = ToolNode(available_tools)

# 5. Define the graph logic (conditional edge)
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue" # Go to tools
    return "end" # End the process

# 6. Build the graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm", run_model)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": END})
graph_builder.add_edge("tools", "llm") # Return to LLM after tool use

# 7. Compile and invoke
app = graph_builder.compile()

# Run the agent
inputs = {"messages": [HumanMessage(content="What is the result of searching for 'LangGraph simple agent'?")]}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("---")
