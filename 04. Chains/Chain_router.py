import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate,
                                    ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import chain

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
prompt = """Given the user review below, classify it as either being about `Positive` or `Negative`.
            Do not respond with more than one word.

            Review: {review}
            Classification:"""

template = ChatPromptTemplate.from_template(prompt)
myChain = template | llm | StrOutputParser()
# output = myChain.invoke({'review': review})
# print(output)
#
#
# route positive responses one way and negative to another.
# Positive
positive_prompt = """
                You are expert in writing reply for positive reviews.
                You need to encourage the user to share their experience on social media.
                Review: {review}
                Answer:"""

positive_template = ChatPromptTemplate.from_template(positive_prompt)
positive_chain = positive_template | llm | StrOutputParser()
# Negative
negative_prompt = """
                You are expert in writing reply for negative reviews.
                You need first to apologize for the inconvenience caused to the user.
                You need to encourage the user to share their concern on following Email:'udemy@kgptalkie.com'.
                Review: {review}
                Answer:"""


negative_template = ChatPromptTemplate.from_template(negative_prompt)
negative_chain = negative_template | llm | StrOutputParser()
#
# define a rounter
def route(info):
    if 'positive' in info['sentiment'].lower():
        return positive_chain
    else:
        return negative_chain
#
# Runnable Lambda - similar to lambda function ... runnableLambda is a pass through function
full_chain = {"sentiment": myChain, 'review': lambda x:x['review']} | RunnableLambda(route)
#review = "Thank you so much for providing such a great plateform for learning. I am really happy with the service."
review = "I am not happy with the service. It is not good."
output = full_chain.invoke({'review': review })
print(output)
print(F"{'*' * 5}\n{'*' * 5}")
#
### Make Custom Chain Runnables with RunnablePassthrough and RunnableLambda
#This is useful for formatting or when you need functionality not provided by other LangChain components,
# and custom functions used as Runnables are called RunnableLambdas.
def char_counts(text):
    return len(text)

def word_counts(text):
    return len(text.split())

prompt = ChatPromptTemplate.from_template("Explain these inputs in 5 sentences: {input1} and {input2}")
myChain = prompt | llm | StrOutputParser() | {'char_counts': RunnableLambda(char_counts),
                                            'word_counts': RunnableLambda(word_counts),
                                            'outout': RunnablePassthrough()}
output = myChain.invoke({'input1': 'Earth is planet', 'input2': 'Sun is star'})
print(output)
