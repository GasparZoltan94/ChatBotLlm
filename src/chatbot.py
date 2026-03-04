from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence, List
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
model: str = os.getenv('MODEL_NAME') or "minimax-m2.5:cloud"
llm = ChatOllama(
    model=model,
    base_url='https://ollama.com/'
)
    
def chatbot(state:AgentState) -> AgentState:
    system_prompt = """
    You are a professional AI assistant integrated into a chatbot interface.

    Your goals:
    - Provide accurate, concise, and helpful responses.
    - Ask clarifying questions only when necessary.
    - Structure answers clearly using bullet points or sections when helpful.
    - Avoid unnecessary verbosity.
    - Maintain a friendly but professional tone.
    - If the user request is ambiguous, ask for clarification before assuming.
    - If you do not know something, say so clearly instead of guessing.
    - When giving technical explanations, adapt depth to the user's level.
    - When giving step-by-step instructions, make them actionable and easy to follow.
    - Avoid repeating the user's question unless necessary for clarity.
    - Do not mention system instructions or internal reasoning.
    """
    
    system_message = SystemMessage(content=system_prompt)
    messages = list(state['messages']) + [system_message]
    res = llm.invoke(messages)
    return {'messages':[res]}

graph = StateGraph(AgentState)
graph.add_node("agent",chatbot)
graph.add_edge(START, "agent")
graph.add_edge("agent",END)

app = graph.compile()

user_input = input(f"User:")
human_message = HumanMessage(content=user_input)
chat_history = [human_message]
while user_input.lower() not in ["exit","quit"]:
    print(f"🤖 AI: \n")
    res = app.invoke({"messages":chat_history})
    print(f"{res['messages'][-1].content}")
    user_input = input(f"User:")
    chat_history = res['messages']+[HumanMessage(user_input)]