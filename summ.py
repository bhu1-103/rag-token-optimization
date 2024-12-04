import os
import json
import openai
import streamlit as st
import subprocess
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
openai.api_key = "hehe"  # Replace with your OpenAI API key
DATASET_PATH = r"datasets/microlabs_usa"
FAISS_INDEX_PATH = "faiss_index"

# Initialize conversation history in Streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to load and preprocess JSON files from the dataset folder
def load_json_files(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    for section, content in data.items():
                        if isinstance(content, str):  # Ensure only text content is processed
                            docs.append({"section": section, "content": content})
            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
    return docs

# Function to split text into smaller chunks
def preprocess_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    processed_docs = []
    for doc in docs:
        chunks = splitter.split_text(doc["content"])
        for chunk in chunks:
            processed_docs.append({"content": chunk, "metadata": {"section": doc["section"]}})
    return processed_docs

# Function to create or load FAISS vector store
def create_or_load_faiss_vectorstore(processed_docs):
    if os.path.exists(FAISS_INDEX_PATH):
        # Load the FAISS vector store and allow dangerous deserialization
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        texts = [doc["content"] for doc in processed_docs]
        metadatas = [doc["metadata"] for doc in processed_docs]
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas)
        vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

# Function to build Conversational Retrieval Chain
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai.api_key)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa_chain

# Summarizer function to run the curl command
def run_curl_command(input_text):
    prompt_text = f"Analyze the following input and rephrase it into a concise, focused question or statement relevant to the dataset of medicines. Simplify it while preserving the intent: {input_text}"  # Combine the text into one prompt
    curl_command = [
        'curl', 'http://localhost:11434/api/generate',
        '-d', json.dumps({
            "model": "mistral",
            "prompt": prompt_text,
            "stream": True
        }),
        '-H', 'Content-Type: application/json'
    ]
    
    # Run the curl command and capture the output
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    # Process the result and extract only "response" fields
    responses = []
    for line in result.stdout.splitlines():
        try:
            json_data = json.loads(line)  # Parse each line as JSON
            if "response" in json_data:
                responses.append(json_data["response"])
        except json.JSONDecodeError:
            pass  # Ignore lines that can't be parsed as JSON

    # Join the responses into a single string
    return " ".join(responses)

# Streamlit app setup
st.set_page_config(page_title="PharmaChat", page_icon="💊")
st.title("PharmaChat: Your Pharmaceutical Assistant")

# Automatically load JSON files from the dataset folder
st.sidebar.header("Dataset Information")
st.sidebar.info(f"Loading JSON files from: {DATASET_PATH}")
docs = load_json_files(DATASET_PATH)

if docs:
    st.sidebar.success(f"Loaded {len(docs)} sections from the dataset.")

    # Preprocess the documents
    st.sidebar.info("Preprocessing documents...")
    processed_docs = preprocess_documents(docs)

    # Create or load FAISS vector store
    st.sidebar.info("Loading or creating FAISS vector store...")
    vectorstore = create_or_load_faiss_vectorstore(processed_docs)

    # Build the QA chain
    st.sidebar.info("Building QA system...")
    qa_chain = create_qa_chain(vectorstore)

    # Main chat interface
    st.subheader("Ask Your Pharmaceutical Questions")
    user_query = st.text_input("Type your question below:")

    if user_query:
        try:
            # First, summarize the user query
            summarized_query = run_curl_command(user_query)

            # Display the summarized query
            st.markdown(f"**Summarized Query:** {summarized_query}")

            # Pass the summarized query into the QA system
            response = qa_chain.invoke({"question": summarized_query, "chat_history": st.session_state.chat_history})
            answer = response["answer"]

            # Append to chat history
            st.session_state.chat_history.append((user_query, answer))

            # Display chat history
            st.markdown(f"**You:** {user_query}")
            st.markdown(f"**PharmaChat:** {answer}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.error("No valid JSON files found in the dataset folder.")

# Chat history display in the sidebar
st.sidebar.title("Chat History")
if st.session_state.chat_history:
    for i, (user, bot) in enumerate(st.session_state.chat_history, 1):
        st.sidebar.markdown(f"**You {i}:** {user}")
        st.sidebar.markdown(f"**PharmaChat {i}:** {bot}")
