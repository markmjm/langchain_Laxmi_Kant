import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate,ChatPromptTemplate)
load_dotenv()
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

#https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html#langchain_ollama.chat_models.ChatOllama
#https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html

base_url = "http://localhost:11434"
model = 'llama3:latest'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
    num_predict = 256
)
system =  SystemMessagePromptTemplate.from_template("You are a helpful {language} translator. Translate the user sentence to {language}.")
human = HumanMessagePromptTemplate.from_template("I love {food} with {drink}.")
messages = [system, human]
template = ChatPromptTemplate(messages)
question = template.invoke({'language': 'French', 'food': 'Pizza', 'drink': 'Pepsi'})
response = llm.invoke(question)
print(response.content)
print(response.response_metadata)