import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate,
                                    ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser

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
first_chain = template | llm | StrOutputParser()
# #####################################
# #####################################
# 2nd runnable - or chain - Pass on the output of first chain to the next  ... you can see the response on langsmith
second_prompt = ChatPromptTemplate.from_template(
    '''
    Analyze the following text: {response1}
    You need to tell me that how difficult it is to understand.
    Answer in one sentence only.
    '''
)
second_chain = {"response1": first_chain} | second_prompt | llm | StrOutputParser()
# #####################################
# #####################################
# 3rd runnable - or chain - Pass the output of 2nd chain to the third prompt
third_prompt = ChatPromptTemplate.from_template(
    '''
    Analyze the following text: {response2}
    tell me number of words in this response.
    You should respond as follows only.  
    Number of words in:{response2}:  1234
    '''
)
#  you can see the response from the first and second chains in LangSmith
# now compose the chain
composed_chain = {"response2": second_chain} | third_prompt | llm | StrOutputParser()
composed_response = composed_chain.invoke({'school': 'primary', 'topics': 'solar system', 'points': 5})
print(composed_response)