from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

groq_api=os.getenv("test_groq")
llm=ChatGroq(
    api_key=groq_api,
    model_name="openai/gpt-oss-20b"
)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | llm | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})
print(result)
