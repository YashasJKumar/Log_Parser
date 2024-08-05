# ------------------------------------- IMPORT STATEMENTS --------------------------------------------------------------

import os
import time
import datetime
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from parser import parse_log
from langchain_groq import ChatGroq

# ------------------------------------ Loading .env file ---------------------------------------------------------------

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def greet_user():
    current_hour = datetime.datetime.now().hour

    if 5 <= current_hour < 12:
        greeting = "Good Morning!"
    elif 12 <= current_hour < 16:
        greeting = "Good Afternoon!"
    elif 16 <= current_hour < 21:
        greeting = "Good Evening!"
    else:
        greeting = "Good Night!"
    return greeting


# Function to generate the dynamic header
def dynamic_header(header_text, place="app", delay=0.1):
    # Create a placeholder for the header
    if place == "sidebar":
        header_placeholder = st.sidebar.empty()
    else:
        header_placeholder = st.empty()

    # Display each letter of the header text one by one
    for i in range(len(header_text) + 1):
        time.sleep(delay)
        header_placeholder.markdown(f'<h1>{header_text[:i]}</h1>', unsafe_allow_html=True)

    # Ensure the full header text remains visible
    header_placeholder.markdown(f'<h1>{header_text}</h1>', unsafe_allow_html=True)


def validate_log_file(users_file):
    first_five_lines = []
    with open(users_file, 'r') as file:
        for _ in range(5):
            line = file.readline()
            if not line:
                break
            first_five_lines.append(line.strip())
    loaded_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    check_prompt = validation_template.invoke({"context": first_five_lines})
    response = loaded_llm.invoke(check_prompt)
    if response.content.lower() == "yes":
        return True
    return False


def file_parser(uploaded_file, file_path, display_messages, session):
    if uploaded_file is not None:
        with st.spinner(text="Parsing the Log File..."):
            type_of_log, msg = parse_log(file_path)

        if type_of_log is not None and msg == "success":
            display_messages.append(st.success("Detected " + type_of_log + " Logs."))
            time.sleep(1)
            display_messages.append(st.success("Parsing Completed Successfully!"))
        else:
            display_messages.append(st.warning("File didn't match predefined logs."))
            time.sleep(1)
            display_messages.append(st.warning("Not Parsing."))
            session.not_log = True


def create_vector_embeddings(session, file_path):
    if "vectors" not in session:

        # If no parsing is needed
        if "not_log" in session:
            session.loader = TextLoader(file_path=file_path)
        else:
            session.loader = CSVLoader(file_path="parsed_log_data.csv")

        session.log_file = session.loader.load()

        # Create a text splitter
        session.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=60,
                                                               length_function=len)

        # Splitting into smaller chunks.
        split_logs = session.text_splitter.split_documents(session.log_file)

        # Unlink the temporary file after use.
        os.unlink(file_path)

        try:
            # Converting into embeddings.
            session.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Initializing the FAISS DB & storing the embeddings in it
            session.vectors = FAISS.from_documents(documents=split_logs,
                                                   embedding=session.embeddings)
        except Exception:
            st.warning("Please Check your embedding model & Refresh the page.")


def load_llm(session):
    try:
        session.llm = ChatGroq(groq_api_key=groq_api_key, model_name=LLM_OPTIONS[session.selected_llm])
    except Exception:
        st.warning("Please Check your Groq API key.")


def create_chains(session):
    # Create a chain for LLM & Prompt Template to inject to LLM for inferencing
    document_chain = create_stuff_documents_chain(llm=session.llm, prompt=prompt_template)

    # Creating a retriever to fetch top 2 chunks related to User_Prompt by making similarity search.
    retriever = session.vectors.as_retriever(search_kwargs={'k': 2})

    # Create a retrieval chain which links the retriever & document chain
    session.retrieval_chain = create_retrieval_chain(retriever, document_chain)


def clear_cache(session):
    keys = list(session.keys())
    for key in keys:
        session.pop(key)
    try:
        os.remove("parsed_log_data.csv")
        print("File deleted successfully")
    except FileNotFoundError:
        print("File not found")


# ------------------------------------- DEFINE PROMPT TEMPLATE ---------------------------------------------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an advanced AI assistant integrated with a RAG (Retrieval-Augmented Generation) system,
                   "specialized in log analysis. Suggest next steps or further investigations when appropriate.
                   If you don't know the answer just say that you don't know.Don't try to make up an answer based on your
                   assumptions.
                   "Response Format:
                    Structure your responses clearly, using sections or bullet points for complex analyses.
                    Include the log entry which supports your answer
                    Include relevant log messages when explaining your findings.
                    Clearly distinguish between information from logs, retrieved knowledge, and your own analysis.
                    Provide the final answer to the question first in bold.
                    Clarification and Precision: If log formats or contents are unclear, ask for clarification.
                    """
         ),
        ("user", "The Log Data is as follows : {context}. User Question : {input}")
    ]
)


validation_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", """You are a Log File checking assistant. 
            You think how a log file will look like & then answer this.
            If a user uploads their generic txt files, don't answer them.
            If the file matches any log format, then consider it as a log file.
            Log files typically contain timestamps, log levels (e.g., INFO, ERROR, DEBUG, WARN, etc), and structured formats.
            You should reply on whether the given text file is a log file or not. You have to just reply with either Yes 
            or No. Don't say anything else."""
        ),
        (
            "user", "This is the content of the user file: {context}. Check if this is content of a "
                    "log file or not. Reply with either Yes or No."
        )
    ]
)

# These are the Meta LLM's currently available in Groq-Cloud
LLM_OPTIONS = {
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3.1 405B": "llama-3.1-405b-reasoning",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
}

style_header = '''
<style>
h1{
    font-size: 50px;
    background: -webkit-linear-gradient(left, #4a5ee5, #8a5ae5, #e55a5a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: left;
}
</style>
'''
