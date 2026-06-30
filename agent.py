"""
Weather & News Agent (LangChain + Mistral + Tavily)
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware.types import wrap_tool_call
from tavily import TavilyClient


# ============================================================
# Tools
# ============================================================

@tool
def get_news(city: str) -> str:
    """Get the latest news for a city."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "error: TAVILY_API_KEY not set"

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(
            query=f"latest news about {city}",
            max_results=5,
            search_depth="advanced",
        )
    except Exception as e:
        return f"error: news search failed ({e})"

    results = response.get("results", [])
    if not results:
        return f"No recent news found for {city}."

    lines = []
    for item in results:
        title = item.get("title", "No title")
        url = item.get("url", "")
        content = item.get("content", "")[:150]
        lines.append(f"{title}\n{url}\n{content}")
    return "\n\n".join(lines)


# ============================================================
# LLM + human-approval middleware
# ============================================================

for var in ("MISTRAL_API_KEY", "WEATHER_API_KEY", "TAVILY_API_KEY"):
    if not os.getenv(var):
        print(f"Warning: {var} is not set — related tool will fail.")

llm = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))


@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before executing a tool."""
    print(f"\n{request.tool.name} wants to run with: {request.tool.args}")
    answer = input("Proceed? (y/n): ").strip().lower()
    if answer in ("y", "yes"):
        return handler(request)
    return "Tool execution cancelled by user"


agent = create_agent(
    model=llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful assistant that can answer questions about the weather and news.",
    middleware=[human_approval],
)

# ============================================================
# Chat loop
# ============================================================

print("Agent ready (type 'exit' to quit)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    try:
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        answer = result["messages"][-1].content
    except Exception as e:
        answer = f"error: {e}"

    print("\nAgent:", answer, "\n")