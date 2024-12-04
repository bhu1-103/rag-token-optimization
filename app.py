import os
import json
import logging
import numpy as np
import streamlit as st
from typing import List, Optional
from dataclasses import dataclass, field
import openai
from langgraph.graph import StateGraph, END

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# State Definition
@dataclass
class MedicalInfoState:
    messages: List[dict] = field(default_factory=list)
    query: Optional[str] = None

# Assistant Configuration
ASSISTANT_NAME = "MedInsight"
SYSTEM_INSTRUCTIONS = (
    "You are MedInsight, an advanced medical information assistant. "
    "Provide comprehensive, accurate, and empathetic information about medical topics. "
    "Your goal is to offer clear, well-researched insights while always recommending "
    "professional medical consultation for specific health concerns."
)

# Model Selection
MODEL = "gpt-4o-mini"

def load_medical_dataset():
    """Load medical information dataset."""
    try:
        with open('medical_knowledge_base.json', 'r', encoding='utf-8') as f:
            medical_data = json.load(f)
        logger.info("Medical dataset successfully loaded")
        return medical_data
    except FileNotFoundError:
        st.error("Medical knowledge base not found.")
        return []

def generate_text_embeddings(medical_data):
    """Create embeddings for medical information."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    texts = []
    for entry in medical_data:
        content = " ".join([
            entry.get('topic', ''),
            entry.get('description', ''),
            entry.get('details', '')
        ])
        texts.append(content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = []
    for text in texts:
        split_texts.extend(text_splitter.split_text(text))

    def get_embedding(text, model="text-embedding-ada-002"):
        response = openai.Embedding.create(input=[text], model=model)
        return response['data'][0]['embedding']

    embeddings = [get_embedding(text) for text in split_texts]
    return list(zip(split_texts, embeddings))

def semantic_search(query, embeddings_data):
    """Perform semantic search on medical dataset."""
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_query_embedding(text):
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        return response['data'][0]['embedding']

    query_embedding = get_query_embedding(query)
    similarities = [cosine_similarity(query_embedding, emb) for _, emb in embeddings_data]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
    
    return "\n\n".join([embeddings_data[i][0] for i in top_indices])

def external_knowledge_search(query):
    """Perform external web search."""
    import requests
    try:
        response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        results = response.json()
        return results.get('AbstractText', 'No external information found.')
    except Exception as e:
        logger.error(f"External search error: {e}")
        return "Unable to retrieve external information."

# LangGraph Workflow Nodes
def retrieval_node(state: MedicalInfoState):
    medical_data = load_medical_dataset()
    embeddings_data = generate_text_embeddings(medical_data)
    
    retrieved_text = semantic_search(state.query, embeddings_data)
    new_messages = state.messages + [{"role": "context", "content": retrieved_text}]
    
    return {"messages": new_messages, "query": state.query}

def external_search_node(state: MedicalInfoState):
    external_results = external_knowledge_search(state.query)
    new_messages = state.messages + [{"role": "external_info", "content": external_results}]
    
    return {"messages": new_messages, "query": state.query}

def response_generation_node(state: MedicalInfoState):
    context = next((msg['content'] for msg in state.messages if msg.get('role') == 'context'), '')
    external_info = next((msg['content'] for msg in state.messages if msg.get('role') == 'external_info'), '')

    full_context = f"Internal Context: {context}\n\nExternal Information: {external_info}"

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Context: {full_context}\n\nQuery: {state.query}"}
    ]

    response = openai.ChatCompletion.create(model=MODEL, messages=messages)
    
    new_messages = messages + [response.choices[0].message]
    return {"messages": new_messages, "query": state.query}

def completeness_check_node(state: MedicalInfoState):
    try:
        check_response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Assess the completeness of the last response."},
                {"role": "user", "content": f"Query: {state.query}\nResponse: {state.messages[-1]['content']}\n\nRespond with JSON: {{\"completed\": true/false}}"}
            ]
        )
        
        completion_flag = json.loads(check_response.choices[0].message.content)
        is_complete = completion_flag.get("completed", False)
        
        return {
            "messages": state.messages,
            "query": None if is_complete else state.query
        }
    except Exception as e:
        logger.error(f"Completeness check error: {e}")
        return {"messages": state.messages, "query": state.query}

def create_medical_workflow():
    workflow = StateGraph(MedicalInfoState)
    
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("external_search", external_search_node)
    workflow.add_node("response_generation", response_generation_node)
    workflow.add_node("completeness_check", completeness_check_node)
    
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "external_search")
    workflow.add_edge("external_search", "response_generation")
    workflow.add_edge("response_generation", "completeness_check")
    
    workflow.add_conditional_edges(
        "completeness_check",
        lambda state: "END" if state.query is None else "retrieval",
        {"END": END, "retrieval": "retrieval"}
    )
    
    return workflow.compile()

def main():
    st.set_page_config(page_title="MedInsight", page_icon="🩺")
    st.title("MedInsight: Medical Information Assistant")
    st.subheader("Get comprehensive medical insights")

    query = st.text_input("What medical information can I help you with today?")
    
    if st.button("Get Insights"):
        with st.spinner("Analyzing medical information..."):
            workflow = create_medical_workflow()
            
            initial_state = {"messages": [], "query": query}
            final_state = workflow.invoke(initial_state)
            
            if final_state['messages']:
                st.success("Medical Insights:")
                st.write(final_state['messages'][-1].get('content', 'No insights generated.'))
            else:
                st.warning("Could not generate medical insights.")

if __name__ == "__main__":
    main()
