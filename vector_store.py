from typing import List
import time
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Use relative path so it works on Streamlit Cloud
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "enterprise_rag"
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )
def add_documents_to_store(chunks: List[Document]) -> int:
    vector_store = get_vector_store()
    batch_size = 20  # Processes 20 chunks at a time
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        time.sleep(1)  # 1-second pause to prevent rate-limiting
        
    return len(chunks)


def get_all_document_names() -> List[str]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(COLLECTION_NAME)
        result = col.get(include=["metadatas"])
        names = {m.get("document_name", "") for m in result["metadatas"]}
        return sorted(n for n in names if n)
    except Exception:
        return []


def delete_document_from_store(document_name: str) -> bool:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(COLLECTION_NAME)
        result = col.get(where={"document_name": document_name})
        if result["ids"]:
            col.delete(ids=result["ids"])
            return True
    except Exception:
        pass
    return False


def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(search_kwargs={"k": k})
