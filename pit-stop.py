
#
# ATTEMPT AT USING LANGCHAIN INSTEAD OF LANGGRAPH
#

from typing import List
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

import os
import pandas as pd
import requests

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
    model="deepseek-r1-distill-qwen-7b"
)

qa_chain_chroma = RetrievalQA.from_chain_type(
    llm,
    retriever=chromadb_vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent

# 1. Build the Clarifier chain
clarify_prompt = PromptTemplate.from_template(
    "I don't have enough context to answer this yet.\n"
    "User asked: {query}\n"
    "Generate a single concise follow‑up question to clarify their intent."
)
clarifier_chain = LLMChain(llm=llm, prompt=clarify_prompt)

# 2. Wrap both as Tools
tools = [
    Tool(
        name="AnswerQuestion",
        func=lambda q: qa_chain_chroma.run(q),
        description="Use when you have enough context to answer the user's question."
    ),
    Tool(
        name="Clarify",
        func=lambda q: clarifier_chain.run(query=q),
        description="Use when you need to ask the user for more details."
    )
]

# 3. Create a Zero‑Shot Agent with a simple instruction
agent_prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=(
        "You are an assistant that first decides whether you can answer based on "
        "retrieved context. If you can, call the AnswerQuestion tool. Otherwise, "
        "call the Clarify tool."
    ),
    format_instructions="Use the following format:\n\n"
                        "Question: the user’s input\n"
                        "Thought: Do I have enough context? (Based on retrieval scores)\n"
                        "Action: the tool to call\n"
                        "Action Input: the text to pass\n\n"
                        "When giving the final answer, use the AnswerQuestion tool."
)

agent = ZeroShotAgent(llm_chain=LLMChain(llm=llm, prompt=agent_prompt),
                     tools=tools,
                     verbose=True)

# 4. Add memory so each turn accumulates
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 5. Run in a loop
print("Agent ready. Say something:")
while True:
    user_input = input(">> ")
    if user_input.lower().strip() in ("exit", "quit"):
        break
    response = executor.run(input=user_input)
    print(response)
