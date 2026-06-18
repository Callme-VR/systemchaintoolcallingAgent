from dotenv import load_dotenv
from langchain_core import output_parsers
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# set up the prompt template

prompt=ChatPromptTemplate.from_messages(
    [("human", "Explain {topic} in simple words")]
)

# models

model=ChatMistralAI(
     model_name="mistral-small-latest"
)

# output parser

output_parsers=StrOutputParser()

# this is runnable sequence with all above components


chain=prompt|model|output_parsers


import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

result=chain.invoke({"topic": "Ai Agents"})
print(result)