import os
import tempfile
import streamlit as st

# Load API key from Streamlit Cloud secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(
    page_title="Enterprise AI Knowledge Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from document_processor import process_uploaded_file
from vector_store import add_documents_to_store, get_all_document_names, delete_document_from_store
from rag_chain import build_rag_chain, query_chain, format_sources


def init_state():
    for k, v in {"messages": [], "rag_chain": None, "upload_dir": tempfile.mkdtemp()}.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def get_chain():
    if st.session_state.rag_chain is None:
        st.session_state.rag_chain = build_rag_chain()
    return st.session_state.rag_chain


def reset_session():
    st.session_state.rag_chain = None
    st.session_state.messages = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 Document Manager")
    st.caption("Upload files to build your knowledge base")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md", "html"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("⚡ Index Documents", type="primary", use_container_width=True):
            success = 0
            with st.spinner("Processing..."):
                for f in uploaded_files:
                    try:
                        chunks, name = process_uploaded_file(f, st.session_state.upload_dir)
                        n = add_documents_to_store(chunks)
                        st.success(f"✅ {name} → {n} chunks")
                        success += 1
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
            if success:
                reset_session()
                st.info(f"Indexed {success} file(s). Chat reset.")

    st.divider()
    st.subheader("📚 Knowledge Base")
    docs = get_all_document_names()

    if not docs:
        st.info("No documents yet.")
    else:
        st.caption(f"{len(docs)} document(s)")
        for doc in docs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"📄 {doc}")
            if c2.button("🗑️", key=f"del_{doc}"):
                delete_document_from_store(doc)
                reset_session()
                st.rerun()

    st.divider()
    if st.button("🔄 Clear Conversation", use_container_width=True):
        reset_session()
        st.rerun()

    st.caption("LangChain · ChromaDB · Gemini · Streamlit")


# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🧠 Enterprise AI Knowledge Hub")
st.caption("Ask questions — answers are grounded in your uploaded documents.")

chat_col, info_col = st.columns([3, 1])

with info_col:
    st.metric("Docs indexed", len(get_all_document_names()))
    st.markdown("---")
    st.info("1. Upload docs\n2. Click **Index Documents**\n3. Ask anything\n4. Check **View Sources**")
    for fmt in ["📕 PDF", "📘 DOCX", "📄 TXT", "📝 Markdown", "🌐 HTML"]:
        st.markdown(fmt)

with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 View Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['document']}** · Page: `{s['page']}`")
                        st.caption(s["preview"])
                        st.divider()

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        if not get_all_document_names():
            st.warning("Upload and index documents first.")
        else:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        answer, source_docs = query_chain(get_chain(), user_input)
                        sources = format_sources(source_docs)
                        st.markdown(answer)
                        if sources:
                            with st.expander("📎 View Sources"):
                                for s in sources:
                                    st.markdown(f"**{s['document']}** · Page: `{s['page']}`")
                                    st.caption(s["preview"])
                                    st.divider()
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
