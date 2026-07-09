import os
from pathlib import Path
from typing import List, Tuple

from langchain_core import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    name = Path(file_path).name

    if ext == ".pdf":
        docs = PyPDFLoader(file_path).load()
    elif ext == ".docx":
        docs = Docx2txtLoader(file_path).load()
    elif ext in (".txt", ".md"):
        docs = TextLoader(file_path, encoding="utf-8").load()
    elif ext == ".html":
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        docs = [Document(page_content=text, metadata={"source": name})]
    else:
        raise ValueError(f"Unsupported format: {ext}")

    for doc in docs:
        doc.metadata["source"] = name
    return docs


def split_documents(documents: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_uploaded_file(uploaded_file, save_dir: str) -> Tuple[List[Document], str]:
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = load_document(file_path)
    chunks = split_documents(docs)
    for chunk in chunks:
        chunk.metadata["document_name"] = uploaded_file.name
    return chunks, uploaded_file.name
