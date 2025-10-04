---
layout: post
title: How I built a Voice AI
subtitle: sounds cool, eh? heres how I did it and you can too...
gh-repo: striderr1o1/AI-City-Info-BOT
# gh-badge: [star, fork, follow]
# tags: [test]
comments: true
mathjax: true
author: Mustafa
---

## The Idea
The background of this is, I was taking a Generative AI course, organized by Dicecamp. Our instructor had assigned us a project. And for that reason, I built this thing. It functions as a Voice AI tool with tool-calling and RAG in the background. For this project, i used mainly the following technologies:
- ElevenLabs
- ChromaDB
- FastAPI
- Streamlit
- Groqcloud (for LLM inference)
- Tavily Search Engine
- ...

## Intial Stages
In the early stages, I didnt know what would be the use-case for my project, i.e what problem would I be solving? Shall I make a customer support tool, or something else. And as I went on, I came up with the idea of a City Info Voice bot.

## How I built it:
I started off by creating an ingestion pipeline. For context, we need to ingest documents to store in the vector database to perform Retrieval-Augmented Generation (RAG). For this project, I used ChromaDB.

First you need to create a .env file and fill it with the API keys:
```
GROQ_API_KEY=""
OLLAMA_EMBEDDING_MODEL="mxbai-embed-large:latest"
CHAT_GROQ_MODEL="llama-3.3-70b-versatile"
ELEVEN_LAB_APIKEY = ""
TAVILY_API_KEY=""
```

### Ingestion.py:
importing the splitter to split text into chunks and creating a function over it:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter


def SplitText(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 30)
    splitted_text = splitter.split_text(text)
    return splitted_text
```
create a function for creating embeddings of the chunks. For this, you need to have Ollama embedding model installed on your system locally.
```python
from langchain_community.embeddings import 
from dotenv import load_dotenv
import os
load_dotenv()
def createEmbeddings(chunks):
    response = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBEDDING_MODEL"), base_url="http://localhost:11434").embed_documents(chunks)
    return response
```
and chromaDB for storing the embeddings, splitted text and IDs of the document.
**Note**: the collection says "IslamabadDocs" because I was storing a PDF file with information of Islamabad in it. Anyways, the code:
```python
def storeInChromaDB(embeddings, texts):
    try:
        client = chromadb.PersistentClient(path="ChromaDB")
        collection = client.get_or_create_collection(name="IslamabadDocs")
        collection.add(
        embeddings=embeddings,
        documents=texts,
        ids= [str(i) for i in range(len(texts))]
        )
        return "Successfully ingested"
    except Exception as e:
        return f"Exception: {e}"
```

### Tools For our Ai(tools.py):
import modules from other files:
```python
from ingestion import createEmbeddings
from tavily import TavilyClient
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
from groq import Groq
import chromadb

# load the environment variables
load_dotenv()
```

initialize the llm (i'm using groq):
```python
llm = ChatGroq(
    model=os.environ.get("CHAT_GROQ_MODEL"),
    api_key=os.environ.get("GROQ_API_KEY")
)
```
and now we program the retrieval tool:
```python
def retrieveContext(query):
    client = chromadb.PersistentClient(path="ChromaDB")
    embeddings = createEmbeddings([query])
    
    collection = client.get_or_create_collection(name="IslamabadDocs")
    results = collection.query(
        query_embeddings=embeddings[0],
        n_results=5
    )
    return results["documents"][0]
```
and the weather tool. For this, we use the tavily search engine:
```python
def GetWeatherFromWeb(City):
    tavily_client = TavilyClient(api_key=os.environ.get('TAVILY_API_KEY'))
    response = tavily_client.search(f"Search the current weather in city: {City}")
    answer = response['results']
    return answer
```
Now, to pass these tools or functions to the LLM, we need to design their schemas. I used this resource to accomplish this: [link](https://medium.com/@manojkotary/exploring-function-calling-capabilities-with-groq-a-step-by-step-guide-586ab7a165aa).
```python
tools = [
    {
         "type": "function",
        "function": {
            "name": "GetWeatherFromWeb",
            "description": "Getting weather details using web search",
            "parameters": {
                "type": "object",
                "properties": {
                    "City": {
                        "type": "string",
                        "description": "City name for fetching weather data",
                    }
                },
            
            },
        },
    },
        {
         "type": "function",
        "function": {
            "name": "retrieveContext",
            "description": "retrieve context from database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query for searching docs through database",
                    }
                },
            "required": ["query"]
            },
        },
    }
]
```
we initialize the groq client:
```python
client = Groq(
    api_key=os.environ.get('GROQ_API_KEY')
)
MODEL_NAME = "llama3-70b-8192"
```
and now we define the function in which we do all the tool/function calling with the help of an LLM. We pass the tools schema to the function. And then, to the LLM. We also instruct the LLM how to use the tools.
**Note**: Retrieval Tool is for RAG and weather tool is for searching weather on the tavily search engine.
```python
def CallLLM(query, tools=tools):
    response1 = client.chat.completions.create(
    model=MODEL_NAME,  # or llama3-8b, mixtral, etc.
    messages=[
        {"role": "system", "content": """Only call the weather tool if the user explicitly
                                         asks for weather information. If the word 'weather' is
                                         mentioned only then. Otherwise, call the retrieveContext tool.
                                         When calling retrieveContext tool, pass the whole user query 
                                         AS IT IS to the tool,"""},
        {"role": "user", "content": f"{query}"}
    ],
    tools=tools,
    tool_choice="auto"
    )
    answer1 = response1.choices[0].message.content
    
    tool_call = response1.choices[0].message.tool_calls
    print(answer1)
    print(tool_call)
    
    functionsAvailable = {
        "retrieveContext": retrieveContext,
        "GetWeatherFromWeb": GetWeatherFromWeb
    }
    
    for i in range(len(tool_call)):
        args = json.loads(tool_call[i].function.arguments)
        functionName = tool_call[i].function.name
        function_to_call = functionsAvailable[functionName]
        functionResp = function_to_call(**args)
        print(functionResp)
    
    
    response2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "When u get the results from previous agent,"
                                          "you analyze them and give precise answer according to user query"},
            {"role": "user", "content": f"Query: {query}...results: {functionResp}"}
        ]
    )
    answer2 = response2.choices[0].message.content
    return answer2
 
```

### Voice-Functions(voicefunctions.py):
