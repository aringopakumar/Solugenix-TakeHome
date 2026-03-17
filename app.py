"""Streamlit UI for the FAQ chatbot."""

import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components

from src.loader import load_documents
from src.chunker import chunk_documents
from src.vectorstore import build_vectorstore, load_vectorstore
from src.qa_chain import build_qa_chain

st.set_page_config(
    page_title="Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 780px;
    }

    /* Page title */
    h1 {
        font-size: 1.45rem;
        font-weight: 600;
        letter-spacing: -0.3px;
        margin-bottom: 0.1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.12);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Status badge */
    .status-ready {
        display: inline-block;
        background: rgba(34,197,94,0.12);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.04em;
    }
    .status-empty {
        display: inline-block;
        background: rgba(128,128,128,0.1);
        color: rgba(180,180,180,0.8);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.04em;
    }

    /* Indexed file list */
    .file-item {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 5px 8px;
        border-radius: 5px;
        background: rgba(128,128,128,0.07);
        margin: 3px 0;
        font-size: 0.8rem;
        opacity: 0.85;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .file-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #4ade80;
        flex-shrink: 0;
    }

    /* Sidebar buttons */
    .stButton > button {
        width: 100%;
        border: 1px solid rgba(128,128,128,0.25);
        background: transparent;
        border-radius: 6px;
        font-size: 0.82rem;
        padding: 0.38rem 0.9rem;
        transition: border-color 0.15s;
    }
    .stButton > button:hover {
        border-color: rgba(128,128,128,0.6);
        background: transparent;
    }
    .stButton > button:disabled { opacity: 0.3; cursor: not-allowed; }

    div[data-testid="stFileUploader"] { padding: 0; }
    div[data-testid="stFileUploader"] section {
        padding: 0.6rem;
        border-radius: 6px;
        border: 1px dashed rgba(128,128,128,0.3) !important;
    }

    /* Source references */
    .ref-card {
        border-left: 2px solid rgba(128,128,128,0.25);
        padding: 6px 0 6px 12px;
        margin: 5px 0;
    }
    .ref-meta {
        font-size: 0.73rem;
        font-weight: 600;
        opacity: 0.5;
        margin-bottom: 3px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .ref-text {
        font-size: 0.8rem;
        opacity: 0.65;
        line-height: 1.5;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

USER_AVATAR = (
    "data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 36 36'>"
    "<circle cx='18' cy='18' r='18' fill='%23374151'/>"
    "<text x='18' y='23' text-anchor='middle' font-size='10' "
    "fill='%239ca3af' font-family='system-ui,sans-serif' font-weight='600' letter-spacing='0.5'>YOU</text>"
    "</svg>"
)
AI_AVATAR = (
    "data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 36 36'>"
    "<circle cx='18' cy='18' r='18' fill='%231f2937'/>"
    "<text x='18' y='23' text-anchor='middle' font-size='11' "
    "fill='%236b7280' font-family='system-ui,sans-serif' font-weight='600' letter-spacing='0.5'>AI</text>"
    "</svg>"
)

with st.sidebar:
    indexed_files = st.session_state.get("indexed_files", [])
    status_html = (
        "<span class='status-ready'>Ready</span>"
        if indexed_files else
        "<span class='status-empty'>No index</span>"
    )
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:12px'>"
        f"<span style='font-weight:600;font-size:0.95rem'>Knowledge Base</span>"
        f"{status_html}</div>",
        unsafe_allow_html=True,
    )

    if indexed_files:
        for fname in indexed_files:
            st.markdown(
                f"<div class='file-item'><div class='file-dot'></div>{fname}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    build_clicked = st.button("Build index", disabled=not uploaded_files)

    if build_clicked and uploaded_files:
        with st.spinner("Indexing..."):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for f in uploaded_files:
                        dest = os.path.join(tmp_dir, f.name)
                        with open(dest, "wb") as out:
                            out.write(f.read())
                    docs = load_documents(tmp_dir)
                    chunks = chunk_documents(docs)
                    vectorstore = build_vectorstore(chunks)

                st.session_state["vectorstore"] = vectorstore
                st.session_state["indexed_files"] = [f.name for f in uploaded_files]
                st.session_state.pop("qa_chain", None)
                st.session_state["messages"] = []
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if indexed_files:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Clear knowledge base"):
            for key in ["vectorstore", "qa_chain", "messages", "indexed_files"]:
                st.session_state.pop(key, None)
            import shutil
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.caption("Upload HR policies, guides, or any company docs.")

is_ready = bool(st.session_state.get("indexed_files"))

if is_ready:
    if "vectorstore" not in st.session_state:
        try:
            st.session_state["vectorstore"] = load_vectorstore()
        except FileNotFoundError:
            pass
    if "vectorstore" in st.session_state and "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = build_qa_chain(st.session_state["vectorstore"])

qa_chain = st.session_state.get("qa_chain") if is_ready else None

st.title("Knowledge Assistant")
st.caption("Answers are based on your uploaded documents.")

def render_sources(source_docs):
    if not source_docs:
        return
    with st.expander(f"Referenced from {len(source_docs)} source{'s' if len(source_docs) != 1 else ''}", expanded=False):
        for doc in source_docs:
            name = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", None)
            location = f"page {page + 1}" if page is not None else "full document"
            excerpt = doc.page_content.strip().replace("\n", " ")
            if len(excerpt) > 240:
                excerpt = excerpt[:240] + "..."
            st.markdown(
                f"<div class='ref-card'>"
                f"<div class='ref-meta'>{name} &middot; {location}</div>"
                f"<div class='ref-text'>{excerpt}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not is_ready or not qa_chain:
    st.markdown(
        "<div style='text-align:center;margin-top:5rem;opacity:0.35;font-size:0.9rem;line-height:1.8'>"
        "Upload your documents in the sidebar<br>and click <strong>Build index</strong> to get started."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

for msg in st.session_state["messages"]:
    avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

def handle_query(user_input):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.spinner("Searching knowledge base..."):
            result = qa_chain.invoke({"query": user_input})

        answer = result["result"]
        source_docs = result["source_documents"]

        st.markdown(answer)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": source_docs,
    })
    st.rerun()

if qa_chain:
    if user_input := st.chat_input("Ask a question about your documents..."):
        handle_query(user_input)

components.html("""
<script>
    const main = window.parent.document.querySelector('section[data-testid="stMain"]');
    if (main) main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
</script>
""", height=0)
