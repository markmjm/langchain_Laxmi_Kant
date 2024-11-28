
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

#https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html#langchain_ollama.chat_models.ChatOllama

base_url = "http://localhost:11434"
model = 'llama3:latest'
# model = 'sheldon'
# model = 'sherlock'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
    num_predict = 256
)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
response = llm.invoke(messages)
print(response.content)
print(response.response_metadata)

# response = ''
# for chunk in llm.stream('How to make a good stuffed cabbage?'):
#  response = response + chunk.content
#  print(response)