"""
Weather & News Agent (LangChain + Mistral + Tavily)

Fixes vs. original:
- `create_agent` was accidentally aliased to the `tool` decorator -> the real
  agent builder is imported instead.
- Human-approval middleware was defined but never attached to the agent.
- OpenWeatherMap error check compared str(cod) to an int (always True) -> fixed.
- Added .get()-based safe parsing + network error handling for both tools.
- Output is now limited to: the approval prompt + the agent's final answer
  (no extra framework/debug noise).
- Added a Rich spinner (animated icon) while the agent is thinking / calling
  tools, plus emoji icons for weather conditions, news, and approvals.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent          # was wrongly aliased to `tool` before
from langchain.agents.middleware.types import wrap_tool_call
from tavily import TavilyClient
from rich.console import Console
from rich.markdown import Markdown

console = Console()

WEATHER_ICONS = {
    "clear": "☀️", "cloud": "☁️", "rain": "🌧️", "drizzle": "🌦️",
    "thunderstorm": "⛈️", "snow": "❄️", "mist": "🌫️", "fog": "🌫️", "haze": "🌫️",
}


def weather_icon(desc: str) -> str:
    desc = desc.lower()
    for key, icon in WEATHER_ICONS.items():
        if key in desc:
            return icon
    return "🌡️"


# ============================================================
# Tools
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city in India."""
    api_key = os.getenv("WHEATHER_API_KEYS")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    try:
        data = requests.get(url, timeout=10).json()
    except requests.RequestException as e:
        return f"error: could not reach weather API ({e})"

    if str(data.get("cod")) != "200":
        return f"error: {data.get('message', 'could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    icon = weather_icon(desc)
    return f"{icon} {city}: {temp}°C, {desc}"


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
        return f"📰 No recent news found for {city}."

    lines = []
    for item in results:
        title = item.get("title", "No title")
        url = item.get("url", "")
        content = item.get("content", "")[:150]
        lines.append(f"📰 {title}\n{url}\n{content}")
    return "\n\n".join(lines)


# ============================================================
# LLM + human-approval middleware
# ============================================================

for var in ("MISTRAL_API_KEY", "WEATHER_API_KEY", "TAVILY_API_KEY"):
    if not os.getenv(var):
        console.print(f"[yellow]⚠️  {var} is not set — related tool will fail.[/yellow]")

llm = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))


@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before executing a tool."""
    console.print(f"\n🔧 [bold]{request.tool.name}[/bold] wants to run with: {request.tool.args}")
    answer = console.input("   Proceed? [bold green](y)[/bold green]/[bold red]n[/bold red]: ").strip().lower()
    if answer in ("y", "yes"):
        return handler(request)
    return "Tool execution cancelled by user"


agent = create_agent(
    model=llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful assistant that can answer questions about the weather and news.",
    middleware=[human_approval],   # was previously never attached
)

# ============================================================
# Chat loop
# ============================================================

console.print("[bold cyan]🤖 Agent ready[/bold cyan]  (type 'exit' to quit)\n")

while True:
    user_input = console.input("[bold]You:[/bold] ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    with console.status("[cyan]🤔 thinking...[/cyan]", spinner="dots"):
        try:
            result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"⚠️ error: {e}"

    console.print("\n[bold magenta]🤖 Agent:[/bold magenta]")
    console.print(Markdown(answer))
    console.print()