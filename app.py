import time
from fastapi import FastAPI, File, UploadFile, Response, Path, Body, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
import json
import shutil  # Import the shutil module

# from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from typing import List
import os
import dotenv
# from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma

from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from pydantic import BaseModel

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain.schema.output_parser import StrOutputParser

from typing import Dict

from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, WebBaseLoader
from datetime import datetime

from langchain.docstore.document import Document

from langchain.tools import DuckDuckGoSearchResults, YouTubeSearchTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import ast  # Import the ast module
import re

from langchain.tools.retriever import create_retriever_tool
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.agents import AgentActionMessageLog, AgentFinish

from langchain.agents import AgentType, initialize_agent, load_tools, Tool

from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, auth

from langchain_google_genai import ChatGoogleGenerativeAI


import asyncio  # Add this import statement

# Load environment variables from .env file
dotenv.load_dotenv()

# Check if FIREBASE_CREDENTIALS is loaded correctly
firebase_credentials_str = os.getenv("FIREBASE_CREDENTIALS")
if not firebase_credentials_str:
    raise ValueError("FIREBASE_CREDENTIALS environment variable not set or empty")

print(f"FIREBASE_CREDENTIALS: {firebase_credentials_str}")  # Add this line for logging

# firebase_credentials_str = firebase_credentials_str.replace("\\n", "\n")

print(f"FIREBASE_CREDENTIALS ATER STR: {firebase_credentials_str}")  # Add this line for logging


try:
    firebase_credentials = json.loads(firebase_credentials_str)
except json.JSONDecodeError as e:
    raise ValueError(f"Error decoding FIREBASE_CREDENTIALS: {e}")

# Initialize Firebase Admin SDK
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred)

dotenv.load_dotenv()

app = FastAPI()

# Enable CORS
origins = [
    "https://intelligent-study-buddy-674a4384987e.herokuapp.com",
    "http://127.0.0.1:8000",  # Add other origins as needed
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        print(f"Decoded token: {decoded_token}")  # Add this line for logging
        return decoded_token
    except Exception as e:
        print(f"Authentication error: {e}")  # Add this line for logging
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

class QuestionRequest(BaseModel):
    question: str
    chat_history: List[Dict]

class VideoRequest(BaseModel):
    query: str

class ExplanationRequest(BaseModel):
    question: str
    options: List[str]
    correctAnswer: str

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  

os.environ['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")

# Set up embeddings
embeddings = OpenAIEmbeddings()

# Set up the LLM
MODEL="gpt-4o"

llm = ChatOpenAI(model=MODEL, temperature=0.2)

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.environ['GEMINI_API_KEY'],
    max_tokens=800,
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
)


# Set up memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up text splitter
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=lambda text: len(text.split()),  # Use word count for chunking
    # Optional: Add a custom separator for semantic chunking
    # separators=["\n\n", "\n", ".", "?", "!", ";", ":", "(", ")"]
)


# Store for vectorstores
vectorstores = {}

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), user=Depends(get_current_user)):
    """
    Uploads a file (PDF, Word, or PowerPoint) and stores it in a vector store.

    Args:
        file: The uploaded file.

    Returns:
        A message indicating successful upload and storage.
    """
    user_id = user['uid']
    user_upload_dir = f"uploads/{user_id}"
    os.makedirs(user_upload_dir, exist_ok=True)
    file_path = f"{user_upload_dir}/{file.filename}"
    
    # Check if the file already exists
    if os.path.exists(file_path):
        return JSONResponse(content={"message": "File already exists"}, status_code=400)
    

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Extract text from the file
    try:
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
        elif file.filename.endswith((".pptx", ".ppt")):
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            return {"message": "Unsupported file type"}
        documents = loader.load()
        if not documents:
            return {"message": "No text extracted from the file"}
        print("Extracted documents:", documents)
    except Exception as e:
        return {"message": f"Error loading file: {e}"}
    
    try:
        # Split the text into chunks
        texts = text_splitter.split_documents(documents)
        if not texts:
            return {"message": "No text chunks created from the file"}

        print("Text chunks created:", texts)

        # Get upload datetime and file size
        upload_datetime = datetime.now().isoformat()
        file_size = os.path.getsize(file_path)

        # Check if the persist_directory exists
        persist_dir = f"chroma_db/{user_id}/{file.filename}"
        if os.path.exists(persist_dir):
            # Handle the existing directory
            # Option 1: Delete the existing directory
            shutil.rmtree(persist_dir)
        
        # Create a new vectorstore for this file
        file_vectorstore = Chroma.from_texts(
            [text.page_content for text in texts if text.page_content is not None],
            embeddings,
            persist_directory=persist_dir,
            metadatas=[{"page_number": text.metadata.get("page", None)+1} for text in texts if text.page_content is not None]  # Correct metadata format
        )

    
        # Store the vectorstore in the dictionary
        vectorstores[file.filename] = file_vectorstore

        # Add datetime and size as properties to the vectorstore
        metadata = {
            "upload_datetime": upload_datetime,
            "file_size": file_size
        }

        print("test")
        # Save the metadata to a JSON file
        metadata_path = os.path.join(persist_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Save the metadata to the Chroma database
        file_vectorstore.persist()
        print("test 2")

    except Exception as e:
        return {"vectoring": f"Error vectoring file: {e}"}

    # Fetch the updated list of files
    files = await fetch_files(user)

    return {"message": f"File '{file.filename}' uploaded and stored successfully.", "files": files}


@app.get("/files")
async def fetch_files(user: dict = Depends(get_current_user)):
    """
    Fetches the list of uploaded files for the user.

    Args:
        user: The authenticated user.

    Returns:
        A list of uploaded files.
    """
    user_id = user['uid']
    print("hi", user_id)
    user_upload_dir = f"uploads/{user_id}"
    if not os.path.exists(user_upload_dir):
        return []

    files = []
    for file_name in os.listdir(user_upload_dir):
        file_path = os.path.join(user_upload_dir, file_name)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            files.append({"file_name": file_name, "file_size": file_size})
    print(files)
    return files


@app.get("/search/{file_name}")
async def search_file(file_name: str, query: str):
    """
    Searches a specific file for a query.

    Args:
        file_name: The name of the file to search.
        query: The search query.

    Returns:
        A list of relevant text chunks.
    """
    if file_name in vectorstores:
        file_vectorstore = vectorstores[file_name]
        results = file_vectorstore.similarity_search(query, k=3)
        return {"results": [result.page_content for result in results]}
    else:
        return {"message": f"File '{file_name}' not found."}


@app.on_event("startup")
async def startup_event():
    """
    Loads vectorstores from persistent storage on startup.
    """
    for user_id in os.listdir("chroma_db"):
        user_dir = os.path.join("chroma_db", user_id)
        if os.path.isdir(user_dir):
            for filename in os.listdir(user_dir):
                if filename != ".DS_Store":
                    persist_directory = f"chroma_db/{user_id}/{filename}"
                    if os.path.exists(persist_directory):
                        if os.path.exists(persist_directory) and not os.path.isdir(persist_directory):
                            os.remove(persist_directory)  # Remove the file if it's not a directory
                        file_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                        vectorstores[filename] = file_vectorstore


@app.delete("/delete/{file_name}")
async def delete_file(file_name: str, user=Depends(get_current_user)):
    """
    Deletes a file and its associated vectorstore.

    Args:
        file_name: The name of the file to delete.

    Returns:
        A message indicating successful deletion.
    """
    user_id = user['uid']
    user_upload_dir = f"uploads/{user_id}"
    file_path = os.path.join(user_upload_dir, file_name)
    db_dir = f"chroma_db/{user_id}/{file_name}"

    if file_name in vectorstores:
        # Remove the vectorstore from the dictionary
        del vectorstores[file_name]

    # Delete the Chroma database directory
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    
    # Delete the uploaded file
    if os.path.exists(file_path):
        os.remove(file_path)

        return {"message": f"File '{file_name}' deleted successfully."}
    else:
        return {"message": f"File '{file_name}' not found."}


@app.get("/download/{file_name}")
async def download_file(file_name: str, user=Depends(get_current_user)):
    """
    Downloads a file by its filename.

    Args:
        file_name: The name of the file to download.

    Returns:
        The file content as a response.
    """
    user_id = user['uid']
    user_upload_dir = f"uploads/{user_id}"
    file_path = os.path.join(user_upload_dir, file_name)

    # Debugging statement to log the file path
    print(f"Attempting to download file from path: {file_path}")

    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            file_content = f.read()
        return Response(content=file_content, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={file_name}"})
    else:
        return {"message": f"File '{file_name}' not found or is a directory."}


# Define a generator function to stream responses
async def gemini_response_generator(filename, question, chat_history, user: str):
    
    user_id = user['uid']
    persist_directory = f"chroma_db/{user_id}/{filename}"
    
    # Debugging statements to log inputs
    print(f"Filename: {filename}, Question: {question}, Chat History: {chat_history}")
    
    # Check if the persist directory exists and is a directory
    if os.path.exists(persist_directory):
        if not os.path.isdir(persist_directory):
            print(f"Error: {persist_directory} is not a directory.")
            os.remove(persist_directory)  # Remove the file if it's not a directory
            os.makedirs(persist_directory, exist_ok=True)
    else:
        os.makedirs(persist_directory, exist_ok=True)
    
    # Load the vectorstore from ChromaDB
    try:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    except Exception as e:
        print(f"Error initializing Chroma vectorstore: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing Chroma vectorstore: {e}")
    
    # k is the number of chunks to retrieve
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Use the retriever to search for relevant documents
    relevant_docs = retriever.get_relevant_documents(question)

    # Check if relevant documents were found
    if not relevant_docs:
        print("No relevant documents found.")
        return JSONResponse(content={"message": "No relevant documents found."})

    # Output the relevant documents
    print("Relevant documents:")
    for doc in relevant_docs:
        print(f"Content: {doc.page_content}")

    # Convert chat_history to a list of HumanMessage and AIMessage objects
    chat_history_messages = []
    for item in chat_history:
        content = str(item["content"])  # Ensure content is a string
        if item["role"] == "user":
            chat_history_messages.append(HumanMessage(content=content, role=item["role"]))
        else:
            chat_history_messages.append(AIMessage(content=content, role=item["role"]))

    chat_history_messages.append(HumanMessage(content=question, role="user"))

    print("Chat history messages:")
    print(chat_history_messages)

    # Create a tool from the retriever
    retriever_tool = create_retriever_tool(retriever, "retrieve_from_notes", "Retrieve relevant information from the document.")

    search_tool = DuckDuckGoSearchResults(max_results=3)

    # Create agent
    tools = [
        Tool(
            name="retrieve_from_notes",
            func=retriever_tool,
            description="Retrieve relevant information from the document for given user question"
        ),
        Tool(
            name="search",
            func=search_tool,
            description="Search for relevant information."
        ),
    ]

    # Define the tools for the agent
    # tools = [retriever_tool, search_tool]

    # Prompt for creating Tool Calling Agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful teacher, very friendly. Keep your responses short. Use emojis in your replies. You have access to a tool called 'retrieve_from_notes' which can retrieve relevant information from the notes. 
                Always use retrieve_from_notes first.
                If students need more explanations on notes, you can used the search_tool to search the and present an answer. If question not related to notes, remind student to ask from notes.
                Strictly answer questions in the notes context.
                Empty responses strictly not allowed.
                """,
            ),
            ("human", "{input}"),
            MessagesPlaceholder("chat_history"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
  
    # Create the agent
    agent = create_tool_calling_agent(llm_gemini, tools, prompt)

    # Define the AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True, return_intermediate_steps=True)
   
    # Run the agent and capture the result
    result = agent_executor.invoke({"input": question + ". Refer to notes.", "chat_history": chat_history_messages})

    # Log the result
    print("Agent result:", result)

    # Check if the result is empty or not in expected format
    if not result or 'output' not in result:
        print("Gemini produced an empty response.")
        return JSONResponse(content={"message": "Gemini produced an empty response."})

    print("Check out the result", result)

    # Check if intermediate_steps list is empty
    if not result['intermediate_steps']:
        print("The agent did not use any tools.")
        result['context'] = []
    else:
        print("The agent used the following tools:")
        # Clean up the context
        cleaned_contexts = []

        for doc in relevant_docs:
            print(doc)
            cleaned_content = clean_up_document_content(doc.page_content)
            cleaned_contexts.append({"page_content": cleaned_content, "page_number": doc.metadata.get("page_number", None)})

        result['context'] = cleaned_contexts
    
    del result['intermediate_steps']
    del result['chat_history']

    chat_history = {
            "role": "user",
            "content": question,
            "user_id": user['uid'],
            "filename": filename
        }
        
    save_chat_history(chat_history)

    chat_history = {
            "role": "ai",
            "content": result['output'],
            "user_id": user['uid'],
            "filename": filename
        }
    
    save_chat_history(chat_history) 

    print(result)

    # Return the result
    return result


@app.post("/ask/{file_name}")
async def ask_question(file_name: str = Path(..., description="The name of the file to ask about"), request: QuestionRequest=None, user=Depends(get_current_user)):
    """
    Asks a question about a specific file, including chat history.

    Args:
        file_name: The name of the file to ask about.
        question: The question to ask.
        chat_history: A list of previous questions and answers.

    Returns:
        The answer to the question.
    """
    user_id = user['uid']
    chat_history = load_chat_history(user_id, file_name)

    question = request.question
    print("hey", file_name, question, chat_history)
    if file_name in vectorstores:
        # Directly call the generator function and collect the result
        result = await gemini_response_generator(file_name, question, chat_history, user)
        return JSONResponse(content=result, media_type='application/json; charset=utf-8')
    else:
        return {"message": f"File '{file_name}' not found."}


def parse_retriever_input(params: Dict):
    return params["input"][-1].content


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


async def conversation_q_and_a(filename, question, chat_history, user):
    """
    Performs conversational question answering using a given file and question,
    including chat history.

    Args:
        filename: The name of the file to retrieve.
        question: The question to ask.
        chat_history: A list of previous questions and answers.
        user: The authenticated user.

    Returns:
        The answer to the question.
    """
    user_id = user['uid']
    persist_directory = f"chroma_db/{user_id}/{filename}"
    
    # Debugging statements to log the persist directory
    print(f"Persist directory: {persist_directory}")
    
    # Check if the persist directory exists and is a directory
    if os.path.exists(persist_directory):
        if not os.path.isdir(persist_directory):
            print(f"Error: {persist_directory} is not a directory.")
            os.remove(persist_directory)  # Remove the file if it's not a directory
            os.makedirs(persist_directory, exist_ok=True)
    else:
        os.makedirs(persist_directory, exist_ok=True)
    
    # Load the vectorstore from ChromaDB
    try:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    except Exception as e:
        print(f"Error initializing Chroma vectorstore: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing Chroma vectorstore: {e}")
    
    # k is the number of chunks to retrieve
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Use the retriever to search for relevant documents
    relevant_docs = retriever.get_relevant_documents(question)

    # Output the relevant documents
    print("Relevant documents:")
    for doc in relevant_docs:
        print(f"Content: {doc.page_content}")

    # Convert chat_history to a list of HumanMessage and AIMessage objects
    chat_history_messages = []
    for item in chat_history:
        content = str(item["content"])  # Ensure content is a string
        if item["role"] == "user":
            chat_history_messages.append(HumanMessage(content=content, role=item["role"]))
        else:
            chat_history_messages.append(AIMessage(content=content, role=item["role"]))

    print("Chat history messages:")
    print(chat_history_messages)

    # Create a tool from the retriever
    retriever_tool = create_retriever_tool(retriever, "retrieve_from_notes", "Retrieve relevant information from the document.")

    search_tool = DuckDuckGoSearchResults(max_results=3)

    # Define the tools for the agent
    tools = [retriever_tool, search_tool]

    # Prompt for creating Tool Calling Agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful teacher, very friendly. Keep your responses short. Use emojis in your replies. You have access to a tool called 'retrieve_from_notes' which can retrieve relevant information from the notes. Use it where necessary to check if an answer exists. 
                If students need more explanations on notes, you can used the search_tool to search the and present an answer. If question not related to notes, remind student to ask from notes.
                Strictly answer questions in the notes context.
                """,
            ),
            ("human", "{input}"),
            MessagesPlaceholder("chat_history"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Construct the Tool Calling Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    # Run Agent
    result = agent_executor.invoke({"input": question, "chat_history": chat_history_messages,})

    # Check if intermediate_steps list is empty
    if not result['intermediate_steps']:
        print("The agent did not use any tools.")
        result['context'] = []
    else:
        print("The agent used the following tools:")
        # Clean up the context
        cleaned_contexts = []

        for doc in relevant_docs:
            print(doc)
            cleaned_content = clean_up_document_content(doc.page_content)
            cleaned_contexts.append({"page_content": cleaned_content, "page_number": doc.metadata.get("page_number", None)})

        result['context'] = cleaned_contexts
    
    
    
    del result['intermediate_steps']
    del result['chat_history']

    print(result)

    # Stream the answer
    for chunk in result:
        yield json.dumps(chunk)
        await asyncio.sleep(0.1)  # Simulate streaming delay


def clean_up_document_content(document_content):
    """
    Uses the LLM to clean up the context string, making it more readable.

    Args:
        document_content: The raw context string from the retriever.

    Returns:
        A cleaned-up version of the context string.
    """

    cleaned_content = document_content

    return cleaned_content  # Return both cleaned content and page number


def parsing_duckduckgo_search_results(search_results):
    # Splitting the string by '], [' to separate each search result
    # results_list = re.split(r'\], \[', search_results.strip('[]'))

    links = re.findall(r'link: (https?://\S+)', search_results)

    cleaned_links = [link.rstrip('],') for link in links]

    return cleaned_links

@app.post("/explain/{file_name}")
def web_search_and_explain(file_name: str = Path(..., description="The name of the file to ask about"), request: QuestionRequest = Body(...), user=Depends(get_current_user)) -> Dict:
    """
    Performs a web search based on the input text, retrieves relevant information,
    and provides an explanation along with referenced links.

    Args:
        request: The request body containing the question.

    Returns:
        A dictionary containing:
            - explanation: A human-readable explanation of the search results.
            - links: A list of URLs referenced in the explanation.
    """
    # 1. Use DuckDuckGo Search to get relevant URLs

    search_tool = DuckDuckGoSearchResults(max_results=3)
    search_results = search_tool.run(request.question)
    print(search_results)
    links = parsing_duckduckgo_search_results(search_results)  
    print(links)
    
    # Load the vectorstore from ChromaDB
    user_id = user['uid']
    vectorstore = Chroma(persist_directory=f"chroma_db/{user_id}/{file_name}", embedding_function=OpenAIEmbeddings())
    # k is the number of chunks to retrieve
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(retriever, "retrieve_from_notes", "Retrieve relevant information from the document.")

    tools = [search_tool, retriever_tool]

    # Prompt for creating Tool Calling Agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful teacher, very friendly. Keep your responses short. Use emojis in your replies.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Construct the Tool Calling Agent
    agent = create_tool_calling_agent(llm_gemini, tools, prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run Agent
    query = request.question
    explanation = agent_executor.invoke({"input": query})
    return JSONResponse(content={"explanation": explanation, "links": links}, media_type='application/json; charset=utf-8')


@app.post("/video_search")
async def video_search(query: VideoRequest, user=Depends(get_current_user)):
    """
    Performs a video search using LangChain's YoutubeSearch tool.

    Args:
        query: The search query.

    Returns:
        A list of video links.
    """
    youtube_search = YouTubeSearchTool()
    try:
        video_links = youtube_search.run(query.query)
        video_links_list = ast.literal_eval(video_links)  # Use ast.literal_eval for safe parsing
        return {"video_links": video_links_list}
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": "An unexpected error occurred."}, status_code=500)

class MCQRequest(BaseModel):
    filename: str
    num_questions: int
    difficulty: str


class MCQStructure(BaseModel):
    question: str = Field(description="the queston generated")
    options: List[str] = Field(description="list of mcq options with one correct answer and three incorrect options")
    correct_answer: str = Field(description="the correct option to the question")


class MCQList(BaseModel):
    questions: List[MCQStructure] = Field(description="list of mcq questions")


from fastapi.logger import logger

@app.post("/generate-mcqs")
def generate_mcqs_endpoint(request: MCQRequest, user=Depends(get_current_user)):
    try:
        mcqs = generate_mcqs(request.filename, request.num_questions, request.difficulty, user)
        return JSONResponse(content=mcqs, media_type='application/json; charset=utf-8')
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))


def generate_mcqs(filename, num_questions, difficulty, user):
    # Read the content of the file
    print(filename)
    try:
        user_id = user['uid']
        vectorstore = Chroma(persist_directory=f"chroma_db/{user_id}/{filename}", embedding_function=OpenAIEmbeddings())
        collections = vectorstore.get()
        content = collections['documents']
        print(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading ChromaDB file: {str(e)}")

    question_query = "generate questions"

    parser = JsonOutputParser(pydantic_object=MCQList)

    prompt = PromptTemplate(
        template="""Based on the following content, generate {num_questions} multiple-choice questions 
                 with {difficulty} difficulty level. Each question should have one correct answer and three incorrect options. 
                 Return the correct answer at end of each question. 
                 Content:{content}
                 {format_instructions}
                 """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions(), "content": content, "num_questions": num_questions, "difficulty": difficulty},
    )

    chain = prompt | llm_gemini | parser

    response = chain.invoke({"query": question_query})
    
    print(response)
    return response


@app.post("/explain")
async def explain(request: ExplanationRequest, user=Depends(get_current_user)):
    """
    Provides an explanation for the given question, options, and correct answer using the existing LLM.

    Args:
        request: The request body containing the question, options, and correct answer.

    Returns:
        A dictionary containing the explanation.
    """
    question = request.question
    options = request.options
    correct_answer = request.correctAnswer

    # Generate the explanation using the existing LLM
    explanation = generate_explanation_with_llm(question, options, correct_answer, user)

    return JSONResponse(content={'explanation': explanation}, media_type='application/json; charset=utf-8')

def generate_explanation_with_llm(question, options, correct_answer, user):
    # Construct the prompt for the LLM
    prompt = PromptTemplate.from_template(f"""
    Question: {question}
    Options: {', '.join(options)}
    Correct Answer: {correct_answer}

    State the {correct_answer} in bold.

    Then in the next paragraph provide a concise and short explanation for why the correct answer is '{correct_answer}'.

    Be friendly, use emojis in your response.
    """)

    chain = prompt | llm

    response = chain.invoke({"question": question, "options": options, "correct_answer": correct_answer})

    print(response)
    return response.content

def load_chat_history(user_id: str, file_name: str):
    user_chat_history_dir = f"chat_history/{user_id}"
    chat_history_file = f"{user_chat_history_dir}/{file_name}.json"
    
    # Load existing chat history if available
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as f:
            chat_history = json.load(f)
            return chat_history[-10:]  # Limit to the last 10 messages
    else:
        return []

@app.get("/chat_history/{file_name}")
async def load_chat_history_endpoint(file_name: str, user=Depends(get_current_user)):
    """
    Loads the chat history for a specific file.

    Args:
        file_name: The name of the file for which to load chat history.

    Returns:
        The chat history as a list of messages.
    """
    user_id = user['uid']
    chat_history = load_chat_history(user_id, file_name)
    return JSONResponse(content={"chat_history": chat_history}, media_type='application/json; charset=utf-8')

def save_chat_history(chat_history):
    user_id = chat_history['user_id']  # Assuming user_id is part of chat_history
    user_chat_history_dir = f"chat_history/{user_id}"
    os.makedirs(user_chat_history_dir, exist_ok=True)
    chat_history_file = f"{user_chat_history_dir}/{chat_history['filename']}.json"
    
    # Load existing chat history if it exists
    if os.path.exists(chat_history_file):
        with open(chat_history_file, "r") as f:
            existing_history = json.load(f)
    else:
        existing_history = []  # Initialize as empty if file does not exist

    existing_history.append(chat_history)  # Append new chat history

    with open(chat_history_file, "w") as f:
        json.dump(existing_history, f)  # Save the updated history

