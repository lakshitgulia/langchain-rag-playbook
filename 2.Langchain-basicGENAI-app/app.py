import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM          ## Wrapper to connect to a local LLM running via Ollama
import streamlit as st                          ## Streamlit lets us build a simple web UI in pure Python
from langchain_core.prompts import ChatPromptTemplate     ## Used to build reusable chat prompt templates
from langchain_core.output_parsers import StrOutputParser ## Extracts plain text from the model's raw response

load_dotenv()  ## Loads variables from a .env file (like API keys) into the environment

## LANGSMITH TRACING
## LangSmith lets you log/monitor/debug every call your chain makes (optional but useful for dev)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"          ## Turns tracing ON
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")  ## Groups traces under this project name

## Prompt Template
## ChatPromptTemplate defines the structure of the conversation sent to the model.
## Chat models expect a LIST of role-tagged messages, not just one string.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ## ^ "system" role = sets the model's behavior/persona. Never shown to the user,
        ##   but shapes how it responds to every question.

        ("user", "Question:{question}")
        ## ^ "user" role = represents what the human is asking.
        ##   {question} is a placeholder — it gets filled in later with the real question,
        ##   at the moment we call chain.invoke({"question": input_text})
    ]
)

## Streamlit Framework
## This section builds the actual web page UI
st.title("Langchain Demo With Gemma Model")   ## Page title shown in the browser
input_text = st.text_input("What question you have in mind?")  ## A text box for user input
## input_text will hold whatever the user types, and updates automatically on every rerun

## Ollama Gemma Model
## Connects to a LOCAL model (no cloud API call) served by Ollama on your machine
llm = OllamaLLM(model="gemma:2b")   
## "gemma:2b" = Google's Gemma model, 2-billion-parameter version (small & lightweight)
## NOTE: this model has NO access to any external documents/database —
## it can only answer from what it learned during training (closed-book, no RAG here)

output_parser = StrOutputParser()   
## The raw model response is a structured message object.
## StrOutputParser() strips that down to just the plain text string.

## THE CHAIN — this is LCEL (LangChain Expression Language)
## The "|" pipes the output of one step directly into the input of the next step,
## like a Unix pipeline: prompt -> llm -> output_parser
chain = prompt | llm | output_parser
## Step 1: prompt   -> fills {question} and builds the formatted message list
## Step 2: llm      -> sends those messages to Gemma and gets a raw response back
## Step 3: parser   -> extracts just the clean text from that raw response

## Runs everything ONLY when the user has typed something into the text box
if input_text:
    st.write(chain.invoke({"question": input_text}))
    ## chain.invoke(...) runs the full pipeline start to finish in one call:
    ## fills the prompt -> sends to LLM -> parses output -> returns final string
    ## st.write(...) displays that string on the Streamlit page

## ------------------------------------------------------------------
## KEY THING TO REMEMBER:
## This app has NO vector database, NO embeddings, NO retriever.
## The model answers purely from its own training knowledge (closed-book).
## Compare this to a RAG pipeline (load -> chunk -> embed -> store in vector DB
## -> retrieve relevant chunks -> stuff into prompt -> then answer),
## which grounds answers in YOUR actual data instead of just model memory.
## ------------------------------------------------------------------