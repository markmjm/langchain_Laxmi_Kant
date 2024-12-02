import os
import streamlit as st

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate,
                                    ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
import tiktoken

load_dotenv()
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

base_url = "http://localhost:11434"
model = 'llama3.2:3b'
llm = ChatOllama(base_url=base_url, model=model)
###
# PDF LOADERS
### Read the list of PDFs in the dir
pdfs = []
for root, dirs, files in os.walk("../rag-dataset"):
    # print(root, dirs, files)
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))
docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    temp = loader.load()
    docs.extend(temp)
# print(f'number of pages (the sum) in all PDF files: {len(docs)}') #the len represents the sum of all pages in all documents.    Each element in docs is one page in the original PDF file

def format_docs(docs):
    return "\n\n".join([x.page_content for x in docs])


context = format_docs(docs)
#print(context)
#
# can we use LLAMA to process this context.  IS llama context window large enough for our context
# let check each doc page token size.
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
# for i in range(0, len(docs)):
#     print(f'Tokens in docs[{i}]: {len(encoding.encode((docs[i].page_content)))}')
# print(f'Tokens in context: {len(encoding.encode(context))}')
#
# context token = 60271 token.  llama can handles 128K
# #####
# LLM Chat with PDF
# #####
system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who answer user question based on the provided context. 
                                                    Do not answer in more than {words} words""")
prompt = """Answer user question based on the provided context ONLY! If the answer is not in the context, just say "I don't know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:"""
prompt = HumanMessagePromptTemplate.from_template(prompt)
messages = [system, prompt]
template = ChatPromptTemplate(messages)
qna_chain = template | llm | StrOutputParser()
# question = "How to gain muscle mass?"
question = "what is the radius of earth?"
response = qna_chain.invoke({'context': context, 'question': question, 'words':50})
print(response)