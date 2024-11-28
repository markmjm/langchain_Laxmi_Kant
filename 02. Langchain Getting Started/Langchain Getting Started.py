
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")


base_url = "http://localhost:11434"
model = 'llama3:latest'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
    num_predict = 256
)

response = llm.invoke('how to make a good kunafeh?. answer in 5 sentences?')
print(response.content)
print(response.response_metadata)