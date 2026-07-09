from typing import List, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from vector_store import get_retriever

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are an expert Enterprise Knowledge Assistant.
Answer using ONLY the context below. If the answer is not in the context, say:
"I could not find relevant information in the uploaded documents."
Never invent facts. Be concise and professional.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:""",
)


def build_rag_chain() -> ConversationalRetrievalChain:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        convert_system_message_to_human=True,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_retriever(k=4),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
    )


def query_chain(chain, question: str) -> Tuple[str, List[Document]]:
    result = chain.invoke({"question": question})
    return result["answer"], result.get("source_documents", [])


def format_sources(source_docs: List[Document]) -> List[dict]:
    seen, formatted = set(), []
    for doc in source_docs:
        name = doc.metadata.get("document_name", "Unknown")
        page = doc.metadata.get("page", "N/A")
        key = f"{name}_{page}"
        if key not in seen:
            seen.add(key)
            preview = doc.page_content[:250].strip()
            if len(doc.page_content) > 250:
                preview += "..."
            formatted.append({"document": name, "page": page, "preview": preview})
    return formatted
