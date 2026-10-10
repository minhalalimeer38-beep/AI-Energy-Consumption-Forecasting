import os
from pathlib import Path

import streamlit as st
from config import PROJECT_DIR

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


APP_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = PROJECT_DIR / "Knowledge"

VECTOR_STORE_DIR = (
    PROJECT_DIR
    / "VectorStore"
    / "energy_knowledge_db_gemini"
)

COLLECTION_NAME = "energy_knowledge"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TOP_K = 4

@st.cache_resource

def get_embeddings():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Add GEMINI_API_KEY to .env locally or to Streamlit Cloud secrets."
        )
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )

def load_knowledge_documents():
    """
    Load knowledge files from the Knowledge folder.

    Supported:
    - TXT
    - MD
    - PDF
    """

    KNOWLEDGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = []

    text_files = list(
        KNOWLEDGE_DIR.glob("*.txt")
    )

    markdown_files = list(
        KNOWLEDGE_DIR.glob("*.md")
    )

    for file_path in (
        text_files + markdown_files
    ):

        text = file_path.read_text(
            encoding="utf-8"
        )

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name
                    }
                )
            )

    pdf_files = list(
        KNOWLEDGE_DIR.glob("*.pdf")
    )

    for file_path in pdf_files:
        reader = PdfReader(str(file_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path.name, "page": page_number},
                    )
                )

    if not documents:

        raise FileNotFoundError(
            "No knowledge documents found. "
            "Add TXT, MD, or PDF files inside the "
            "Knowledge folder."
        )

    return documents

def split_documents(documents):
    """
    Split documents into smaller chunks
    for semantic retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks

def build_vector_store():
    """
    Create the Chroma vector database
    from knowledge documents.
    """

    documents = load_knowledge_documents()

    chunks = split_documents(
        documents
    )

    embeddings = get_embeddings()

    VECTOR_STORE_DIR.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(
            VECTOR_STORE_DIR
        )
    )

    return vector_store

@st.cache_resource

def load_vector_store():
    """
    Load an existing Chroma vector database.
    """

    if not VECTOR_STORE_DIR.exists():

        return build_vector_store()

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(
            VECTOR_STORE_DIR
        )
    )

    return vector_store

def retrieve_documents(
    query,
    k=TOP_K
):
    """
    Retrieve the most relevant knowledge
    documents for a user question.
    """

    vector_store = load_vector_store()

    documents = (
        vector_store
        .similarity_search(
            query,
            k=k
        )
    )

    return documents

def retrieve_context(
    query,
    k=TOP_K
):
    """
    Convert retrieved documents into
    context for the AI model.
    """

    documents = retrieve_documents(
        query,
        k=k
    )

    if not documents:

        return ""

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"Source: {source}\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )
