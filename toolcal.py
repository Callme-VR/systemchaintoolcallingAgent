from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print


# -------------------------
# Create Custom Tool
# -------------------------
@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of the input text.
    """
    return len(text)


# -------------------------
# Initialize LLM
# -------------------------
llm = ChatMistralAI(
    model="mistral-large-latest"
)


# -------------------------
# Bind Tool to LLM
# -------------------------
llm_with_tools = llm.bind_tools([get_text_length])


# -------------------------
# User Query
# -------------------------
query = "how many characters are in 'hi there'"

messages = [
    HumanMessage(content=query)
]


# -------------------------
# First LLM Call
# -------------------------
ai_response = llm_with_tools.invoke(messages)

print("[bold green]AI Response:[/bold green]")
print(ai_response)
print()


# -------------------------
# Check if Tool Was Called
# -------------------------
if ai_response.tool_calls:

    tool_call = ai_response.tool_calls[0]

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print("[bold yellow]Tool Call Detected:[/bold yellow]")
    print(tool_call)
    print()

    # Execute Tool
    if tool_name == "get_text_length":
        tool_result = get_text_length.invoke(tool_args)

    # Add AI response and tool result to message history
    messages.append(ai_response)

    messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

    # -------------------------
    # Second LLM Call
    # -------------------------
    final_response = llm_with_tools.invoke(messages)

    print("[bold cyan]Final Answer:[/bold cyan]")
    print(final_response.content)

else: 
    print("[bold red]No tool was called.[/bold red]")
    print(ai_response.content)
