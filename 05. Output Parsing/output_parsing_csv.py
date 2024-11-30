import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import CommaSeparatedListOutputParser

load_dotenv()
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html#langchain_ollama.chat_models.ChatOllama
# https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html

base_url = "http://localhost:11434"
model = 'llama3.2:latest'

llm = ChatOllama(
    base_url=base_url,
    model=model,
    #temperature=0.8,
    num_predict=256
)
# print(llm.invoke("Tell me a joke about Washington DC"))
##
parser = CommaSeparatedListOutputParser()
print(parser.get_format_instructions())
joke_prompt = PromptTemplate(
    template='''
    Answer the user query with a list of values.   Here is your formating instructions.
    {format_instructions}

    Query: {query}
    Answer:''',
    input_variables=['query'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
myChain = joke_prompt | llm | parser
output = myChain.invoke({'query': 'generate my website seo keywords. I have content about the NLP and LLM.'})
print(output)



