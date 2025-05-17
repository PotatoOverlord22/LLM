from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
import json
import logging
import time
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.docstore.document import Document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("automotive_assistant.log"), logging.StreamHandler()],
)
logger = logging.getLogger("automotive_assistant")

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_TOP_K = 5
PERSIST_DIRECTORY = "chroma_store"
MODEL_BASE_URL = "http://localhost:1234/v1"
MODEL_API_KEY = "lm-studio"
MODEL_NAME = "llama-3.2-3b-instruct"

# Custom system message with comprehensive guidelines
SYSTEM_MESSAGE = """
You are an expert automotive technician assistant specialized in vehicle systems, parts, repairs, and maintenance.

SCOPE OF EXPERTISE:
- Vehicle systems (electrical, mechanical, hydraulic, etc.)
- Parts identification and specifications
- Diagnostic procedures and troubleshooting
- Maintenance schedules and procedures
- Repair techniques and best practices
- Safety protocols for automotive work

RESPONSE GUIDELINES:
1. If the input is a greeting (e.g., "Hi", "Hello", "Hey"):
   - Respond politely and briefly
   - Invite the user to ask about their automotive issue
   - Do NOT invoke any tools for simple greetings

2. ALWAYS verify if the query is automotive-related using the verify_automotive_domain tool

3. For automotive queries:
   - Check if clarification is needed using the clarify_llm tool
   - If clear enough, retrieve information using the rag_qa tool
   - Prioritize safety in all recommendations
   - When suggesting repairs, indicate complexity level (DIY vs professional)
   - Include part numbers or specification ranges when relevant

4. For non-automotive queries:
   - Politely explain you're specialized in automotive topics
   - Suggest how to rephrase as an automotive question if possible
   - Do not attempt to answer non-automotive questions

5. RESPONSE FORMAT:
   - Begin with direct answer to question (1-2 sentences)
   - Provide supporting details/explanation
   - Include safety warnings when relevant
   - End with brief follow-up suggestion or advice

6. LIMITATIONS:
   - Make clear when information might vary by specific vehicle make/model/year
   - Acknowledge when a professional diagnosis is recommended
   - Don't guess at recall information or legal requirements
"""

# Enhanced prompt template for RAG
RETRIEVAL_PROMPT = """
You are an automotive expert assistant helping with technical information. Use the following context to answer the question.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain relevant information, say "I don't have enough information about this specific topic"
3. Keep answers concise and technical but easy to understand
4. When citing specific parts or procedures, mention the source (parts catalog, system manual, etc.)
5. Include part numbers or specifications when available in the context
6. Add a safety warning if the repair/maintenance procedure has safety implications

Answer:
"""


class LocalServerEmbeddings(Embeddings):
    """Custom embedding class for local LLM server"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        # Test connection
        try:
            self.embed_query("test connection")
            logger.info("Successfully connected to embedding server")
        except Exception as e:
            logger.error(f"Failed to connect to embedding server: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents"""
        try:
            # Split into manageable batches to avoid timeout/memory issues
            batch_size = 20
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = requests.post(
                    f"{self.base_url}/embeddings", json={"input": batch}, timeout=30
                )
                response.raise_for_status()
                embeddings = [item["embedding"] for item in response.json()["data"]]
                all_embeddings.extend(embeddings)

                # Sleep briefly to avoid overwhelming the server
                time.sleep(0.1)

            return all_embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a query string"""
        try:
            response = requests.post(
                f"{self.base_url}/embeddings", json={"input": [text]}, timeout=10
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise


def preprocess_automotive_data(data_dir="./Data"):
    """
    Process automotive data from CSV files into structured documents with metadata.

    Args:
        data_dir: Directory containing the CSV files

    Returns:
        List of Document objects ready for chunking and embedding
    """
    logger.info("Starting data preprocessing")

    # Load CSVs with error handling
    try:
        parts_df = pd.read_csv(f"{data_dir}/parts.csv")
        systems_df = pd.read_csv(f"{data_dir}/systems.csv")
        scenarios_df = pd.read_csv(f"{data_dir}/automotive_scenarios.csv")

        logger.info(
            f"Loaded {len(parts_df)} parts, {len(systems_df)} systems, {len(scenarios_df)} scenarios"
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Better text formatting for vector embedding with semantic structure
    def format_part(row):
        # Handle missing values gracefully
        part_type = row.get("part_type", "Unknown")
        name = row.get("name", "Unnamed Part")
        function = row.get("function", "No function information available")
        common_issues = row.get("common_issues", "No common issues documented")
        maintenance_info = row.get(
            "maintenance_info", "No maintenance information available"
        )

        return (
            f"Part Type: {part_type} | Name: {name} | "
            f"Function: {function} | Common Issues: {common_issues} | "
            f"Maintenance: {maintenance_info}"
        )

    def format_system(row):
        # Handle missing values gracefully
        system_type = row.get("system_type", "Unknown")
        name = row.get("name", "Unnamed System")
        description = row.get("description", "No description available")
        interconnected_parts = row.get(
            "interconnected_parts", "No interconnected parts listed"
        )

        return (
            f"System Type: {system_type} | Name: {name} | "
            f"Description: {description} | Interconnected Parts: {interconnected_parts}"
        )

    def format_scenario(row):
        # Handle missing values gracefully
        scenario_type = row.get("scenario_type", "Unknown Scenario")
        symptoms = row.get("symptoms", "No symptoms described")
        possible_causes = row.get("possible_causes", "No possible causes listed")
        contextual_advice = row.get(
            "contextual_advice", "No contextual advice available"
        )

        return (
            f"Scenario Type: {scenario_type} | Symptoms: {symptoms} | "
            f"Possible Causes: {possible_causes} | Contextual Advice: {contextual_advice}"
        )

    # Create formatted documents with rich metadata
    documents = []

    # Process parts with detailed metadata
    for i, row in parts_df.iterrows():
        # Extract key fields for metadata, defaulting to safe values
        part_type = str(row.get("part_type", "Unknown"))
        name = str(row.get("name", "Unnamed Part"))

        doc = Document(
            page_content=format_part(row),
            metadata={
                "source": "parts",
                "part_type": part_type,
                "name": name,
                "doc_id": f"part_{i}",
            },
        )
        documents.append(doc)

    # Process systems with detailed metadata
    for i, row in systems_df.iterrows():
        name = str(row.get("name", f"unknown_system_{i}"))
        system_type = str(row.get("system_type", "Unknown"))

        doc = Document(
            page_content=format_system(row),
            metadata={
                "source": "systems",
                "name": name,
                "system_type": system_type,
                "doc_id": f"system_{i}",
            },
        )
        documents.append(doc)

    # Process scenarios with detailed metadata
    for i, row in scenarios_df.iterrows():
        scenario_type = str(row.get("scenario_type", f"unknown_scenario_{i}"))

        doc = Document(
            page_content=format_scenario(row),
            metadata={
                "source": "scenarios",
                "scenario_type": scenario_type,
                "doc_id": f"scenario_{i}",
            },
        )
        documents.append(doc)
    logger.info(f"Created {len(documents)} preprocessed documents")
    return documents


def create_chunks(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Create semantic chunks from documents with appropriate size and overlap.

    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of chunked Document objects
    """
    logger.info(f"Creating chunks with size={chunk_size}, overlap={chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # More semantic separators to preserve context
        separators=["\n\n", "\n", " | ", ": ", " ", ""],
        # Keep metadata from original documents
        keep_separator=True,
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def create_vectorstore(chunks, embedding_model, persist_directory=PERSIST_DIRECTORY):
    """
    Create and persist a vector store from document chunks.

    Args:
        chunks: List of chunked Document objects
        embedding_model: Embedding model to use
        persist_directory: Directory to persist the vector store

    Returns:
        Chroma vector store
    """
    logger.info(f"Creating vector store in {persist_directory}")

    # Create directory if it doesn't exist
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory=persist_directory
    )

    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vectorstore


def create_rag_chain(vectorstore, llm):
    """
    Create a RAG chain for retrieving and answering questions.

    Args:
        vectorstore: Vector store to retrieve from
        llm: LLM to use for answering

    Returns:
        RetrievalQA chain
    """
    logger.info("Creating RAG chain")

    # Create retriever with configurable search parameters
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVER_TOP_K,  # Number of documents to retrieve
            # "fetch_k": RETRIEVER_TOP_K * 2,  # Fetch more and filter for diversity
        },
    )

    # Create prompt template
    qa_prompt = PromptTemplate.from_template(RETRIEVAL_PROMPT)

    # Create chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )

    logger.info("RAG chain created successfully")
    return qa_chain


@tool
def verify_automotive_domain(query: str) -> str:
    """
    Verify if a query is related to automotive topics.

    Args:
        query: The user's question or request.
    Returns:
        JSON string with {"is_automotive": boolean, "reason": string, "suggestion": string}
    """
    # Log the query
    logger.debug(f"Verifying domain for: {query}")

    # Enhanced domain verification prompt with specific examples
    domain_check_template = ChatPromptTemplate.from_template(
        """Determine if this query is related to automotive topics (vehicles, parts, repairs, maintenance):
        
        Query: "{query}"
        
        Automotive topics include:
        - Vehicle parts and systems (engine, transmission, brakes, etc.)
        - Vehicle diagnostics and repairs
        - Maintenance procedures and schedules
        - Vehicle specifications and performance
        - Driving and vehicle operation
        - Vehicle safety features and systems
        - Car buying, selling, and ownership
        - Automotive tools and equipment
        
        Return a JSON object with:
        1. "is_automotive": true if automotive-related, false if not
        2. "reason": brief explanation of your decision (max 15 words)
        3. "suggestion": if not automotive, suggest how to make it automotive-related; otherwise empty string
        
        Example:
        {"is_automotive": false, "reason": "Weather is not automotive-related", "suggestion": "To make this automotive-related, you could ask how weather affects vehicle performance"}
        """
    )

    try:
        domain_check_chain = domain_check_template | llm | StrOutputParser()
        result = domain_check_chain.invoke({"query": query})

        # Validate JSON structure
        try:
            parsed = json.loads(result)
            required_keys = ["is_automotive", "reason", "suggestion"]
            for key in required_keys:
                if key not in parsed:
                    logger.warning(
                        f"Missing key in domain verification response: {key}"
                    )
                    parsed[key] = "" if key != "is_automotive" else False
            result = json.dumps(parsed)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in domain verification: {result}")
            result = json.dumps(
                {
                    "is_automotive": False,
                    "reason": "Error processing domain verification",
                    "suggestion": "Please ask a clearly automotive-related question",
                }
            )

        # Log result summary
        logger.debug(f"Domain verification result: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Error in verify_automotive_domain: {e}")
        return json.dumps(
            {
                "is_automotive": False,
                "reason": "Error processing domain verification",
                "suggestion": "Please try again with a clear automotive question",
            }
        )


@tool
def clarify_llm(query: str) -> str:
    """
    Determine if a user query needs clarification for better automotive assistance.

    Args:
        query: The user's question or request about automotive topics.
    Returns:
        JSON string with format {"needs_clarification": boolean, "question": string, "reason": string}
    """
    # Log the query
    logger.debug(f"Checking clarification needs for: {query}")

    # Enhanced clarification prompt with specific automotive details
    clarify_template = ChatPromptTemplate.from_template(
        """Analyze this automotive query: "{query}"
        
        Common automotive details that might be missing:
        - Vehicle make, model, and year
        - Engine type/size (V6, 2.0L, diesel, etc.)
        - Transmission type (manual, automatic, CVT)
        - Specific symptoms or conditions (noises, vibrations, warning lights)
        - Maintenance history relevance
        - Driving conditions (city, highway, towing)
        - Mileage or vehicle age
        
        Determine if clarification is needed to provide a precise answer.
        
        Return a JSON object with:
        1. "needs_clarification": true if clarification needed, false if clear enough
        2. "question": if clarification needed, ONE specific question to ask; otherwise empty string
        3. "reason": brief reason why clarification is/isn't needed (max 20 words)
        
        Example output:
        {"needs_clarification": true, "question": "What year and model is your vehicle?", "reason": "Vehicle specifications needed for accurate diagnosis"}
        
        or
        
        {"needs_clarification": false, "question": "", "reason": "Query contains sufficient details for answering"}
        """
    )

    try:
        clarify_chain = clarify_template | llm | StrOutputParser()
        result = clarify_chain.invoke({"query": query})

        # Validate JSON structure
        try:
            parsed = json.loads(result)
            required_keys = ["needs_clarification", "question", "reason"]
            for key in required_keys:
                if key not in parsed:
                    logger.warning(f"Missing key in clarification response: {key}")
                    parsed[key] = "" if key != "needs_clarification" else False
            result = json.dumps(parsed)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in clarification: {result}")
            result = json.dumps(
                {
                    "needs_clarification": True,
                    "question": "Could you provide more details about your automotive question?",
                    "reason": "Error processing clarification check",
                }
            )

        # Log result summary
        logger.debug(f"Clarification check result: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Error in clarify_llm: {e}")
        return json.dumps(
            {
                "needs_clarification": True,
                "question": "Could you provide more details about your automotive question?",
                "reason": "Error processing clarification check",
            }
        )


@tool
def rag_qa(query: str) -> str:
    """
    Answer a user's question by retrieving from the automotive knowledge base.

    Args:
        query: A natural‑language question about automotive parts, systems, or scenarios.
    Returns:
        A concise answer based on retrieved context.
    """
    # Log the query
    logger.info(f"Retrieving information for: {query}")

    try:
        # Start timer for performance tracking
        start_time = time.time()

        # Get search results with source tracking
        result = qa_chain_chroma.invoke({"query": query})

        # Log performance and source information
        elapsed_time = time.time() - start_time
        sources = []
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "unknown")
            doc_id = doc.metadata.get("doc_id", "unknown")
            sources.append(f"{src}:{doc_id}")

        logger.info(f"RAG query completed in {elapsed_time:.2f}s")
        logger.info(f"Sources used: {sources}")

        # Check if we have a valid result
        answer = result.get("result", "")
        if not answer or answer.lower().startswith("i don't know"):
            logger.warning("RAG query returned insufficient information")
            return "I don't have enough specific information about this automotive topic in my knowledge base. Consider providing more details or consulting a professional mechanic for this particular issue."

        # Return the result
        return answer
    except Exception as e:
        logger.error(f"Error in rag_qa: {e}")
        return "I'm sorry, I encountered an error retrieving information from my automotive knowledge base. Please try rephrasing your question or asking about a different automotive topic."


# @tool
# def diagnose_automotive_problem(symptoms: str) -> str:
#     """
#     Analyze automotive symptoms and suggest possible causes and diagnostic steps.

#     Args:
#         symptoms: Description of vehicle symptoms or issues
#     Returns:
#         Structured diagnosis with possible causes and next steps
#     """
#     logger.info(f"Diagnosing automotive problem: {symptoms}")

#     try:
#         diagnose_template = ChatPromptTemplate.from_template(
#             """Given these automotive symptoms: "{symptoms}"

#             Provide a structured diagnosis with:
#             1. Most likely causes (in order of probability)
#             2. Simple diagnostic steps a user could perform
#             3. Warning signs that would indicate professional help needed

#             Format your response as a JSON object with "causes", "diagnostic_steps", and "warnings" keys.
#             """
#         )
#         diagnose_chain = diagnose_template | llm | StrOutputParser()
#         result = diagnose_chain.invoke({"symptoms": symptoms})


#         # Try to validate JSON structure
#         try:
#             parsed = json.loads(result)
#             return result
#         except json.JSONDecodeError:
#             logger.warning(f"Invalid JSON in diagnosis response: {result}")
#             # Return the text anyway, it might still be useful even if not proper JSON
#             return result
#     except Exception as e:
#         logger.error(f"Error in diagnose_automotive_problem: {e}")
#         return json.dumps(
#             {
#                 "causes": ["Unable to analyze symptoms due to processing error"],
#                 "diagnostic_steps": [
#                     "Please provide more specific symptoms or consult a mechanic"
#                 ],
#                 "warnings": [
#                     "This appears to be a complex issue requiring professional assessment"
#                 ],
#             }
#         )
@tool
def diagnose_automotive_problem(symptoms: str) -> str:
    """
    Analyze automotive symptoms by retrieving relevant knowledge base information
    and suggesting possible causes, diagnostic steps, and warnings.

    Args:
        symptoms (str): Description of vehicle symptoms or issues provided by the user.

    Returns:
        str: A JSON-formatted string containing:
            - "causes": A list of most likely causes ordered by probability.
            - "diagnostic_steps": Simple diagnostic steps the user can perform.
            - "warnings": Warning signs indicating when professional help is needed.

    This tool internally uses the knowledge retrieval tool (rag_qa) to enhance
    the diagnosis with relevant automotive information from the knowledge base,
    while abstracting the retrieval process from the user.
    """
    logger.info(f"Diagnosing automotive problem: {symptoms}")

    # Step 1: Get relevant info from knowledge base
    kb_context = rag_qa(symptoms)

    try:
        diagnose_template = ChatPromptTemplate.from_template(
            """
            Given these automotive symptoms: "{symptoms}"
            And the following relevant information retrieved from our database: "{kb_context}"
            
            Provide a structured diagnosis with:
            1. Most likely causes (in order of probability)
            2. Simple diagnostic steps a user could perform
            3. Warning signs that would indicate professional help needed
            
            Format your response as a JSON object with "causes", "diagnostic_steps", and "warnings" keys.
            """
        )
        diagnose_chain = diagnose_template | llm | StrOutputParser()
        result = diagnose_chain.invoke({"symptoms": symptoms, "kb_context": kb_context})

        # Validate JSON structure
        try:
            parsed = json.loads(result)
            return result
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in diagnosis response: {result}")
            return result
    except Exception as e:
        logger.error(f"Error in diagnose_automotive_problem: {e}")
        return json.dumps(
            {
                "causes": ["Unable to analyze symptoms due to processing error"],
                "diagnostic_steps": [
                    "Please provide more specific symptoms or consult a mechanic"
                ],
                "warnings": [
                    "This appears to be a complex issue requiring professional assessment"
                ],
            }
        )


def init_application():
    """Initialize the application components"""
    logger.info("Initializing automotive assistant application")

    # Initialize embedding model
    embedding = LocalServerEmbeddings(base_url=MODEL_BASE_URL)

    # Initialize LLM
    global llm
    llm = ChatOpenAI(
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
        model=MODEL_NAME,
        temperature=0.1,  # Lower temperature for more factual responses
    )

    # Check if vectorstore exists, otherwise create it
    persist_dir = Path(PERSIST_DIRECTORY)
    if persist_dir.exists() and any(persist_dir.iterdir()):
        logger.info(f"Loading existing vectorstore from {PERSIST_DIRECTORY}")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY, embedding_function=embedding
        )
    else:
        logger.info("Creating new vectorstore")
        # Process data
        documents = preprocess_automotive_data()
        # Chunk documents
        chunks = create_chunks(documents)
        # Create vectorstore
        vectorstore = create_vectorstore(chunks, embedding)

    # Create RAG chain
    global qa_chain_chroma
    qa_chain_chroma = create_rag_chain(vectorstore, llm)

    # Define tools for the agent
    tools = [
        verify_automotive_domain,
        clarify_llm,
        rag_qa,
        diagnose_automotive_problem,
    ]

    # Create memory for conversation context
    memory = MemorySaver()

    # Create agent
    global agent_executor
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    logger.info("Application initialization complete")
    return FastAPI()


# FastAPI setup
app = init_application()

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
    """
    Process a chat message and return the automotive assistant's response.

    Args:
        message: Message object with content and conversation_id

    Returns:
        Dictionary with responses list
    """
    try:
        logger.info(
            f"Received message: '{message.content[:50]}...' (convo: {message.conversation_id})"
        )

        # Configuration for the agent
        config = {"configurable": {"thread_id": message.conversation_id}}

        # Process message with the agent
        response = agent_executor.invoke(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": message.content},
                ]
            },
            config,
        )

        # Extract response content
        response_content = response["messages"][-1].content

        # Log response summary
        logger.info(f"Response generated: '{response_content[:50]}...'")

        return {"responses": [response_content]}

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {
            "responses": [
                "I apologize, but I encountered a technical issue while processing your automotive question. Please try again with a rephrased question about vehicles, parts, or repairs."
            ]
        }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the application is running properly"""
    return {"status": "healthy", "timestamp": time.time()}


# If this file is being run directly, start the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
