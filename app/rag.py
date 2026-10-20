import hashlib
import os
import shutil
from pathlib import Path

import streamlit as st
from config import PROJECT_DIR

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


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

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@st.cache_resource
def get_embeddings():
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Add GEMINI_API_KEY to .env locally "
            "or to Streamlit Cloud secrets."
        )

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )


def load_knowledge_documents():
    """Load PDF, TXT, and Markdown files from Knowledge/."""

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    documents = []

    files = sorted(
        (
            path
            for path in KNOWLEDGE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )

    for file_path in files:
        if file_path.suffix.lower() in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path.name},
                    )
                )

        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_path.name,
                                "page": page_number,
                            },
                        )
                    )

    if not documents:
        raise FileNotFoundError(
            "No readable PDF, TXT, or MD documents found "
            "inside the Knowledge folder."
        )

    return documents


def knowledge_fingerprint():
    """Create a signature that changes when a Knowledge file changes."""

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()

    files = sorted(
        (
            path
            for path in KNOWLEDGE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )

    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(file_path.read_bytes())

    return digest.hexdigest()


def split_documents(documents):
    """Split documents into smaller searchable text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


def build_vector_store():
    """Build a Chroma database from the current Knowledge files."""

    documents = load_knowledge_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    VECTOR_STORE_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
    )


@st.cache_resource
def load_vector_store(fingerprint):
    """Load the matching database, or rebuild it if documents changed."""

    marker_path = (
        VECTOR_STORE_DIR.parent
        / "energy_knowledge_db_gemini.sha256"
    )

    if VECTOR_STORE_DIR.exists() and marker_path.exists():
        saved_fingerprint = marker_path.read_text(
            encoding="utf-8"
        ).strip()

        if saved_fingerprint == fingerprint:
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=str(VECTOR_STORE_DIR),
            )

    # Remove only this generated database, not the whole VectorStore folder.
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)

    vector_store = build_vector_store()

    marker_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    marker_path.write_text(
        fingerprint,
        encoding="utf-8",
    )

    return vector_store


def retrieve_documents(query, k=TOP_K):
    """Find the most relevant document chunks for a question."""

    fingerprint = knowledge_fingerprint()
    vector_store = load_vector_store(fingerprint)

    return vector_store.similarity_search(
        query,
        k=k,
    )


def retrieve_context(query, k=TOP_K):
    """Format retrieved chunks for the AI Assistant."""

    documents = retrieve_documents(query, k=k)

    if not documents:
        return ""

    context_parts = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page")

        source_label = source
        if page is not None:
            source_label += f", page {page}"

        context_parts.append(
            f"Source: {source_label}\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)
