from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import requests

from typing import List
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Load CSVs
parts_df = pd.read_csv("./Data/parts.csv")
systems_df = pd.read_csv("./Data/systems.csv")
scenarios_df = pd.read_csv("./Data/automotive_scenarios.csv")

# Combine them into one list of documents
def make_text(row):
    return " | ".join(str(x) for x in row if pd.notnull(x))

texts = parts_df.apply(make_text, axis=1).tolist() + \
        systems_df.apply(make_text, axis=1).tolist() + \
        scenarios_df.apply(make_text, axis=1).tolist()
        
class LocalServerEmbeddings(Embeddings):
    def __init__(self, base_url: str):
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(f"{self.base_url}/embeddings", json={"input": texts})
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(f"{self.base_url}/embeddings", json={"input": [text]})
        return response.json()["data"][0]["embedding"]

embedding = LocalServerEmbeddings(base_url="http://localhost:1234/v1")

from langchain.docstore.document import Document

documents = [Document(page_content=t) for t in texts]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(documents)

template = """Use the following context to answer the question.
If unsure, say "I don't know". Keep answers short. End with: "Thanks for asking!"
Context: {context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Chroma db
persist_directory = 'chroma_store'
    
chromadb_vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="llama-3.2-3b-instruct"
)

qa_chain_chroma = RetrievalQA.from_chain_type(
    llm,
    retriever=chromadb_vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

@tool
def rag_qa(query: str) -> str:
    """
    Answer a user’s question by retrieving from the automotive knowledge base.
    
    Args:
        query: A natural‑language question about automotive parts, systems, or scenarios.
    Returns:
        A concise answer based on retrieved context.
    """
    result = qa_chain_chroma.invoke({"query": query})
    return result["result"]


clarify_template = ChatPromptTemplate.from_template(
    """I want to make sure I understand the user's question or request.
    
    Here is what they said: {query}

    First, determine if this message is ambiguous or needs clarification.
    If it's clear and specific enough to provide a good response, respond with just: "NO_CLARIFY"

    If it's ambiguous or missing important details, formulate ONE specific clarifying question that would help you provide a better response.

    Your response should be EITHER:
    1. "NO_CLARIFY" (if no clarification is needed)
    OR
    2. A single, concise clarifying question (if clarification is needed)
    """
)

clarify_chain = clarify_template | llm | StrOutputParser()

@tool
def clarify_llm(query: str) -> str:
    """
    Use an LLM to decide if the last user message needs clarification.
    Returns a follow-up question if needed, otherwise returns an empty string.
    """
    result = clarify_chain.invoke({"query": query})
    if "NO_CLARIFY" in result:
        return ""
    else:
        return result

tools = [
    rag_qa,
    clarify_llm
]

memory = MemorySaver()

agent_executor = create_react_agent(
    llm, 
    tools, 
    checkpointer=memory)
    
# FastAPI setup
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    content: str
    conversation_id: str

@app.post("/chat")
async def chat(message: Message):
    config = {"configurable": {"thread_id": message.conversation_id}}
    response = agent_executor.invoke(
        {
            "messages": [HumanMessage(content=message.content)]
        },
        config
    )
    return {
        "responses": [response["messages"][-1].content]
    }
