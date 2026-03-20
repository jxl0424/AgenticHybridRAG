"""
Streamlit UI for RAG application.
"""
import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Clinical Document Ingestion", page_icon="🏥", layout="centered")

# Inject Custom CSS for Medical Aesthetic
st.markdown("""
<style>
    /* Professional Medical Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    
    /* Global Background and Text to force Light Mode */
    .stApp, .ea3mdgi1 {
        background-color: #f4f8fb !important; /* Extremely soft, premium light blue/grey */
        color: #1b263b !important; /* Slate navy bold text */
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0b5949 !important; /* Deep clinical teal */
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Override ALL Streamlit Inputs to enforce Light Box with Dark Text */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="input"] > input,
    .stTextInput input, 
    .stNumberInput input, 
    div[data-baseweb="base-input"],
    .stChatInputContainer input {
        background-color: #ffffff !important;
        color: #1b263b !important;
        border: 1px solid #dbe4eb !important;
        caret-color: #0d836b !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease;
    }
    
    /* Input Focus State for glassmorphism / pop out */
    div[data-baseweb="input"] > div:focus-within, 
    .stTextInput input:focus, 
    .stChatInputContainer input:focus {
        border-color: #0d836b !important;
        box-shadow: 0 4px 12px rgba(13, 131, 107, 0.15) !important;
    }
    
    .stChatInputContainer {
        padding-bottom: 2rem !important;
    }
    
    /* File Uploader Box */
    div[data-testid="stFileUploader"] > div {
        background-color: #ffffff !important;
        color: #1b263b !important;
        border: 2px dashed #b1c6d8 !important;
        border-radius: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    }
    div[data-testid="stFileUploader"] > div:hover {
        border-color: #0d836b !important;
        background-color: #f7fcfa !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(13, 131, 107, 0.08);
    }
    
    /* File Uploader Text Elements */
    div[data-testid="stFileUploader"] span, 
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] p {
        color: #4b6278 !important;
    }
    
    /* Chat Messages - Beautiful Card Styling */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e2eaf0;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(11, 89, 73, 0.03), 0 2px 4px rgba(11, 89, 73, 0.02);
        transition: box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
         box-shadow: 0 8px 16px rgba(11, 89, 73, 0.06), 0 4px 8px rgba(11, 89, 73, 0.04);
    }
    
    /* User Chat Message Distinction */
    .stChatMessage:nth-child(even) {
        background-color: #f0f7f5;
        border-color: #d1e8e2;
    }
    
    /* Primary Buttons */
    .stButton>button {
        background-color: #0d836b !important; /* Vibrant Medical Teal */
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px rgba(13, 131, 107, 0.25) !important;
    }
    .stButton>button:hover {
        background-color: #0b6955 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(13, 131, 107, 0.35) !important;
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(13, 131, 107, 0.2) !important;
    }
    
    /* Expanders & Status Containers */
    .streamlit-expanderHeader, div[data-testid="stStatusWidget"] {
        background-color: #edf2f7 !important; /* Soft gray blue */
        color: #1b263b !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: 1px solid #dbe4eb !important;
        transition: background-color 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: #e2e8f0 !important;
    }
    
    div[data-testid="stStatusWidget"] {
        border-radius: 14px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.03) !important;
    }
    
    /* Info Blocks (st.info, st.success, st.warning) */
    .stAlert {
        border-radius: 12px !important;
        background-color: #ffffff !important;
        color: #1b263b !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.03) !important;
        border: 1px solid #e2eaf0 !important;
    }
    div[data-testid="stAlert"]:has(> div > div > div > svg[color="#118770"]) {
       border-left: 4px solid #0d836b !important; /* Stronger Teal Accent */
    }
    
    /* All Markdown text */
    .stMarkdown p, .stMarkdown span, .stMarkdown li {
        color: #33485f !important;
        line-height: 1.7;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important; /* Pure white sidebar */
        border-right: 1px solid #e2eaf0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.02);
    }
    section[data-testid="stSidebar"] hr {
        border-color: #e2eaf0 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
        serializer=inngest.PydanticSerializer(),
    )


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("data/raw")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


# ---------------------------------------------------------
# Sidebar Configuration & Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Configuration")
    top_k = st.slider("Max Reference Contexts (Top-K)", min_value=1, max_value=20, value=5, step=1)
    
    st.divider()
    st.header("📄 Document Ingestion")
    uploaded = st.file_uploader("Upload Clinical PDF", type=["pdf"], accept_multiple_files=False)
    
    if uploaded is not None:
        with st.spinner("Indexing document..."):
            path = save_uploaded_pdf(uploaded)
            asyncio.run(send_rag_ingest_event(path))
        st.success(f"Queued for ingestion: {path.name}")

st.title("Medical Assistant AI")
st.caption("Agentic Graph RAG for Clinical Intelligence")

# ---------------------------------------------------------
# Chat Interface Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I am the Clinical AI Assistant. Please upload a document in the sidebar, or ask a question.", "logs": None, "sources": None}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "assistant" else "👤"):
        st.write(msg["content"])
        
        # Display agent logs if they exist for this message
        if msg.get("logs") and len(msg["logs"]) > 0:
            with st.expander("🧠 View Agent Reasoning", expanded=False):
                for step in msg["logs"]:
                    step_type = step.get("type")
                    content = step.get("content", "")
                    if step_type == "thought":
                        st.markdown(f"**🤔 Thought:** {content}")
                    elif step_type == "action":
                        st.info(f"**🛠️ Action:** `{content}`")
                    elif step_type == "observation":
                        st.success(f"**🔍 Observation:** {content}")
                        
        if msg.get("sources") and len(msg["sources"]) > 0:
            st.caption("Sources:")
            for s in set(msg["sources"]):
                st.write(f"- {s}")

# ---------------------------------------------------------
# Chat Input & Processing
# ---------------------------------------------------------
if prompt := st.chat_input("Enter clinical query..."):
    # Append user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt, "logs": None, "sources": None})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # Process assistant response
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        status_container = st.status("Agent reasoning...", expanded=True)
        with status_container:
            st.write("Querying pipeline...")
            response = requests.post(
                "http://localhost:8000/api/query",
                json={"question": prompt, "top_k": int(top_k), "use_hybrid": True},
                timeout=120,
            )
            response.raise_for_status()
            hybrid_output = response.json()
            status_container.update(label="Reasoning Complete", state="complete", expanded=False)
            
        # Display the final answer
        answer = hybrid_output.get("answer", "No synthesized data returned.")
        st.write(answer)
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "logs": hybrid_output.get("thought_process", []),
            "sources": hybrid_output.get("sources", [])
        })
        
        # Immediately show sources for the current message
        sources = hybrid_output.get("sources", [])
        if sources:
            st.caption("Sources:")
            for s in set(sources):
                st.write(f"- {s}")
