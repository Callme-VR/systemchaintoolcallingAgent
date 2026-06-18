from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

model = ChatMistralAI(model="mistral-small-latest")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are helpful assistant who explain code in simple terms"),
    ("human", "explain the following code in simple words\n{code}")
])

seq = code_prompt | model | parser

seq2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "explaination": RunnableLambda(lambda code: {"code": code}) | explain_prompt | model | parser
    }
)

chain = seq | seq2

result = chain.invoke({
    "topic": "please write a code in python of Three Sum"
})

print(result["code"])
print(result["explaination"])