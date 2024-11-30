import os

from click import prompt
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
# from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate,
#                                     ChatPromptTemplate)
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

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
## Create a Pydantic class for a Joke
class Joke(BaseModel):
    """Joke to tell"""
    setup: str = Field("The setup of the Joke")
    punchline: str = Field(description="The punchline of the Joke")
    rating: Optional[int] = Field(description="The rating of a the Joke.   1 to 10")
##
structured_llm = llm.with_structured_output(Joke)
output = structured_llm.invoke('Tell me a joke about cats')
print(output)



