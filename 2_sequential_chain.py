from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()
os.environ['LANGCHAIN_PROJECT']='SEQUENTIAL APP'
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

groq_api=os.getenv("test_groq")
llm=ChatGroq(
    api_key=groq_api,
    model_name="openai/gpt-oss-20b"
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

config={
    'run_name':'sequential_chain',
    'tags':['llm app','report generator','summarization'],
    'metadata':{'model':"openai/gpt-oss-20b",'model_temp':0.7,'parser':"stroutputparser"}
}
result = chain.invoke({'topic': 'Unemployment in India'},config=config)

print(result)
