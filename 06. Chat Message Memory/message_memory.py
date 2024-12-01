import os
from dotenv import load_dotenv
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_ollama import ChatOllama

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
    # temperature=0.8,
    num_predict=256
)


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")


system = SystemMessagePromptTemplate.from_template("You are helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")

messages = [system, MessagesPlaceholder(variable_name='history'), human]

prompt = ChatPromptTemplate(messages=messages)

chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(chain, get_session_history,
                                                   input_messages_key='input',
                                                   history_messages_key='history')


def chat_with_llm(session_id, input):
    output = runnable_with_history.invoke(
        {'input': input},
        config={'configurable': {'session_id': session_id}}
    )

    return output


user_id = "Santa Clause"
about = "My name is Santa Clause. I am the CEO for Santa Clause Enterprises."
response = chat_with_llm(user_id, about)
print(f"{response}\n{'*'*5}\n{'*'*5}")
about = "what is my name?"
response = chat_with_llm(user_id, about)
print(f"{response}\n{'*'*5}\n{'*'*5}")
#
#clean up the DB.  Otherwise, the LLM persona gets very annoyed.
# Test Clean up
history = get_session_history(user_id)
history.get_messages()
history.clear()
print(f"history after clear: {history.get_messages()}")
about = "what is my name?"
response = chat_with_llm(user_id, about)
print(f"{response}\n{'*'*5}\n{'*'*5}")