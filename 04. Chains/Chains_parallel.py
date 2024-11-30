import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate,
                                    ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from sklearn.utils.estimator_checks import check_no_attributes_set_in_init

load_dotenv()
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html#langchain_ollama.chat_models.ChatOllama
# https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html

base_url = "http://localhost:11434"
model = 'llama3:latest'

llm = ChatOllama(
    base_url=base_url,
    model=model,
    temperature=0.8,
    num_predict=256
)
system = SystemMessagePromptTemplate.from_template(
    "You are {school} teacher. You answer in short sentences.'")
human = HumanMessagePromptTemplate.from_template("tell me about the {topics} in {points} points")
messages = [system, human]
template = ChatPromptTemplate(messages)
#
# first runnable - or chain ... you can see the response on langsmith
fact_chain = template | llm | StrOutputParser()
output = fact_chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 5})
print(output)
print(f"{'*'*5}\n{'*'*5}")
#
# 2nd runnable - or chain ... in parallel --- you can see the response on langsmith
question = HumanMessagePromptTemplate.from_template('Write a peom on {topics} in {sentences} lines')
messages = [system, question]
template = ChatPromptTemplate(messages)
poem_chain = template | llm | StrOutputParser()
output = poem_chain.invoke({'school': 'primary', 'topics': 'solar system', 'sentences': 5})
print(output)
print(f"{'*'*5}\n{'*'*5}")
#
#
#
chain = RunnableParallel(fact = fact_chain, poem =poem_chain)
output = chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 8, 'sentences': 8})
print(output)



