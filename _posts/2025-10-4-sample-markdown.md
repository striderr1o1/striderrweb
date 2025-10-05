---
layout: post
title: How I built a Voice AI
subtitle: Heres how I did it and you can too...
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

First we need to create a .env file and fill it with the API keys:
```
GROQ_API_KEY=""
OLLAMA_EMBEDDING_MODEL="mxbai-embed-large:latest"
CHAT_GROQ_MODEL="llama-3.3-70b-versatile"
ELEVEN_LAB_APIKEY = ""
TAVILY_API_KEY=""
```

Install these requirements:
```
streamlit
PyPDF2
langchain
langchain_groq
chromadb
ollama
langchain_community
sentence-transformers
elevenlabs
groq
numpy
pygame
sounddevice
fastapi
multipart
tavily-python
streamlit-audiorec
```
Its possible that i've missed something, in such a case, you can contact me on my linkedin or email me.

### Ingestion.py:
In this file, create the ingestion functions.
Read PDF using this function:
```python
import PyPDF2
def readPDF(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return text
```
import the splitter to split text into chunks and create a function over it:
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
### Create FastAPI Endpoint to Upload PDFs:
Make a seperate file for this. Either create a FastAPI endpoint, or upload documents through streamlit by creating a new navigation in the streamlit.py file(coming ahead). I'm going with FastAPI endpoint. Import the modules:
```python
from fastapi import FastAPI, UploadFile, File
from ingestion import readPDF, SplitText, createEmbeddings, storeInChromaDB
app = FastAPI()
```
And now the endpoint:
```python
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    text = readPDF(file.file)
    print(text)
    splittedText = SplitText(text)
    embeddings = []
    for text in splittedText:
        embedded = createEmbeddings(text)
        embeddings.append(embedded[0])

    
    status = storeInChromaDB(embeddings, splittedText)
    
    return {
        "Texts": splittedText,
        "Ingestion in Chroma Status": status 
    }
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
and now program the retrieval tool:
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
and the weather tool. For this, use the tavily search engine:
```python
def GetWeatherFromWeb(City):
    tavily_client = TavilyClient(api_key=os.environ.get('TAVILY_API_KEY'))
    response = tavily_client.search(f"Search the current weather in city: {City}")
    answer = response['results']
    return answer
```
Now, to pass these tools or functions to the LLM, you need to design their schemas. I used this resource to accomplish this: [link](https://medium.com/@manojkotary/exploring-function-calling-capabilities-with-groq-a-step-by-step-guide-586ab7a165aa)
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
and now we define the function in which we do all the tool/function calling with the help of an LLM. We pass the tools schema to the function. And then, to the LLM. We also instruct the LLM how to use the tools.We also set the tool_choice to auto.
**Note**: Retrieval Tool is for RAG from uploaded PDFs and weather tool is for searching weather on the tavily search engine.
```python
def CallLLM(query, tools=tools):
    response1 = client.chat.completions.create(
    model=MODEL_NAME,  # or llama3-8b, etc.
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
Much of the code here I took from AI tools, which I don't recommend mostly for some reasons, but anyways. We start off by making this function:
```python
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

def record_voice(filename="input.wav", duration=5, sample_rate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    # Convert float32 (-1..1) to int16 for WAV
    audio_int16 = np.int16(audio * 32767)
    wav.write(filename, sample_rate, audio_int16)
    print(f"Saved to {filename}")
```
This records your voice and make a .wav file of your audio.
Next, initialize the ElevenLabs clients:
```python
import os
from elevenlabs import ElevenLabs, play
from io import BytesIO
client = ElevenLabs(
    api_key=os.environ.get("ELEVEN_LAB_APIKEY"),
)
```
And now the elevenlabs function, that basically takes your .wav audio file, hard-coded model ID, and some other things to convert your speech to text. For this, I also referred to the official docs of elevenLabs: [link](https://elevenlabs.io/docs/cookbooks/speech-to-text/quickstart)
```python
def SpeechToText(audio_file, model_id="scribe_v1", language_code="eng",
                 tag_audio_events=True, diarize=True):
    with open(audio_file, "rb") as f:
        audio_data = BytesIO(f.read())

    transcription = client.speech_to_text.convert(
        file=audio_data,
        model_id=model_id,
        tag_audio_events=tag_audio_events,
        language_code=language_code,  # None for auto-detect
        diarize=diarize
    )
    
    return transcription.text
```
Now, we make another function that converts the AI response to speech. Also referred to ElevenLab docs for this: [link](https://elevenlabs.io/docs/quickstart)
```python
import streamlit as st
def TextToSpeech(text):
    audio = client.text_to_speech.convert(

    text=text,

    voice_id="nPczCjzI2devNBz1zQrb",

    model_id="eleven_multilingual_v2",

    output_format="mp3_44100_128",

    )
    st.header('AI Reply:')
    
    play(audio)
    st.write(text)
```
These were the tts-stt related modules. Some streamit related stuff "st.header('AI Reply:') and st.write(text)" is also used which displays the output to the screen. Now we move towards connecting stuff.

### Linking All the Modules(main.py/streamlit.py):
At this, you can create a simple streamlit frontend and have the voice input received via the frontend. So, you import the modules:
```python
import streamlit as st
from st_audiorec import st_audiorec
from voicefunctions import SpeechToText, TextToSpeech
from tools import CallLLM

```
Make a sidebar navigation and radio buttons to navigate:
```python
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Voice Chat"])
```
Then comes the implementation:
```python
if page == "Voice Chat":
    st.title("🎙️ Voice Chat with AI Bot")
    
    wav_audio_data = st_audiorec()
    
    if wav_audio_data is not None:
        # st.audio(wav_audio_data, format="audio/wav")
    
        with open("input.wav", "wb") as f:
            f.write(wav_audio_data)
    
        st.success("Saved recording to input.wav")
        text = SpeechToText("input.wav")
        st.header('Query:')
        st.write(text)
        reply = CallLLM(text)
        
        TextToSpeech(reply)
    else:
        st.info("Click the record button above to capture your voice.")
```

- st_audiorec() records your voice and it comes into wav_audio_data
- if wav_audio_data exists, it opens input.wav file and writes the audio data in it
- input.wav is sent to SpeechToText function which returns the text
- text is sent to CallLLM function which implements the actual Ai functionality. It returns the Ai reply
- reply is sent to TextToSpeech function which converts the AI's response to speech and plays the audio.

### Run the App:
In your terminal, type this command:
```
streamlit run streamlit.py
```
and for uploading your pdf document, run:
```bash
uvicorn <filename without including .py >:app --reload
```
Then you can run this url: http://127.0.0.1:8000/docs#/ (or just add docs infront of any ipaddress:port)

## Conclusion:
Enjoy your voice agent. Make sure you keep up with new posts. Here's the repo: [link](https://github.com/striderr1o1/AI-City-Info-BOT)