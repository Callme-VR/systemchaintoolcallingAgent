from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_results=4)

model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_tokens=400
)

prompt = ChatPromptTemplate.from_template(
    """
    System: You are a helpful search agent that summarizes web search results.
    Summarize the following content in a structured format with bullet points and headers.

    Content:
    {content}
    """
)

chain = prompt | model | StrOutputParser()

query = "what is the latest news in ai 2026"
news_result = search_tool.invoke(query)

# pull just the text content, drop urls/scores/metadata
combined_content = "\n\n".join(item["content"] for item in news_result)

result = chain.invoke({"content": combined_content})

print(result)