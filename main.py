# ------------------------------------- IMPORT STATEMENTS --------------------------------------------------------------

import time
import tempfile
import datetime
import streamlit as st
from helper_functions import (LLM_OPTIONS, greet_user, style_header, file_parser,
                              dynamic_header, create_vector_embeddings, clear_cache,
                              create_chains, load_llm)

# ------------------------------------- STREAMLIT UI -------------------------------------------------------------------

st.set_page_config(
    page_icon="💀",
    layout="wide",
    page_title="AI Powered Log Parser",
    initial_sidebar_state="auto"
)

st.sidebar.write("")
st.sidebar.write(datetime.datetime.now())
st.sidebar.write(":blue[Powered by ..]")
st.sidebar.image("./meta_21.png", use_column_width=True)

with st.sidebar:
    st.subheader(":red[Disclaimer:]")
    st.write("\nUpload your Log File & unleash the power of Llama 3 & 3.1 powered by Groq inferencing engine to "
             "answer your queries.\n")
    st.write("")
    st.session_state.selected_llm = st.selectbox(label= "Choose your Llama 3 variant here 👇",
                                                 label_visibility="visible",
                                                 options=LLM_OPTIONS)
    st.markdown(style_header, unsafe_allow_html=True)

# ------------------------------------- LOADING THE LOG FILE -----------------------------------------------------------

uploaded_file = st.sidebar.file_uploader("Upload your Log File here: ", type=["log", "txt"],
                                         accept_multiple_files=False)

if "greet" not in st.session_state:
    dynamic_header(header_text=greet_user(), place="sidebar")
    st.session_state.greet = True

load_llm(session=st.session_state)

if "header_rendered" not in st.session_state:
    dynamic_header(header_text="AI Powered Log Parser")

    # Ensure that this dynamic header doesn't display again,Push it to the session state.
    st.session_state.header_rendered = True
else:
    st.markdown('<h1>AI Powered Log Parser</h1>', unsafe_allow_html=True)

st.header('', divider='violet', anchor=False)


temp_file_path = None
if uploaded_file is not None:
    # Create a temporary file and write the uploaded file's content to it
    if "vectors" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name


if "vectors" not in st.session_state and uploaded_file is not None:
    display_messages = []

    file_parser(uploaded_file=uploaded_file, file_path=temp_file_path,
                display_messages=display_messages, session=st.session_state)

    with st.spinner("Creating Vector Embeddings..."):
        create_vector_embeddings(session=st.session_state, file_path=temp_file_path)

    if "vectors" in st.session_state:
        display_messages.append(st.success("Embedding are ready.."))
        create_chains(st.session_state)
        # Remove the success_msg
        time.sleep(1)

    for message in display_messages:
            message.empty()
            time.sleep(1)  # Clearing all temporary informative messages.

# User Prompting Part:
if uploaded_file is not None and "vectors" in st.session_state:
    if user_prompt := st.chat_input("Enter your query: ", key="prompt_for_llm"):
        message_container = st.container(height=500, border=False)
        message_container.subheader(":orange[User Prompt: ]")
        message_container.write(user_prompt)
        start_time = time.time()
        with st.spinner(text="Generating Response..."):
            chain = st.session_state.retrieval_chain.pick('answer')
        message_container.subheader(":blue[Response:]")
        message_container.write_stream(chain.stream({'input': user_prompt}))
        st.sidebar.subheader("\n\n\n:green[Response Time : ]" + " " +
                             str(round((time.time() - start_time), 2)) + " sec.")


# This block is to remove the chains, retrievers & embeddings from the session when user removes his log file.
if uploaded_file is None:
    st.warning("Please upload your Log file before asking questions.")
    if "vectors" in st.session_state:
        clear_cache(session=st.session_state)
