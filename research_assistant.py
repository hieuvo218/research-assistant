import os
import streamlit as st
import pickle 
import time
import langchain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv


# Interface construction
st.title("Research Assistant")


load_dotenv()

## Chat model
llm = ChatOpenAI(
    model = "gpt-3.5-turbo",   
    temperature = 0.5,
)

## Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

main_placeholder = st.empty()

## vector store
loader = PyPDFLoader("rag_research.pdf")
pages = []
for page in loader.lazy_load():
    pages.append(page)

# data = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# all_splits = text_splitter.split_documents(data)

# vectorstore = FAISS.from_documents(all_splits, embeddings)
vectorstore = FAISS.from_documents(pages, embeddings)
time.sleep(2)


query = main_placeholder.text_input("Question: ")

if query:
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    # {"answer": "", "sources": []}
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n") # Split the sources by newline
        for source in sources_list:
            st.write(source)

# Storing vectorstore in a file
# with open("vectorstore.pkl", "wb") as f:
#     pickle.dump(vectorstore, f)

# if os.path.exists("vectorstore.pkl"):
#     with open("vectorstore.pkl", "rb") as f:
#         vectorstore = pickle.load(f)

# chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

# query = "How many chefs involve in research?"
# langchain.debug = True
# chain({"question": query}, return_only_outputs=True)