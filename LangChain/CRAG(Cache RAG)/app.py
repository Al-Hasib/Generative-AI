import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import docx
from groq import Groq
import tempfile
import io

# Page configuration
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'knowledge_base_ready' not in st.session_state:
    st.session_state.knowledge_base_ready = False
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

# App title and description
st.title("📚 Document Q&A Assistant")
st.markdown("Upload your PDF documents and ask questions about their content!")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
    help="Get your API key from https://console.groq.com/"
)

# Document upload section
st.header("📄 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=['pdf'],
    accept_multiple_files=True,
    help="Upload one or more PDF documents to create your knowledge base"
)

@st.cache_data
def load_pdf_from_bytes(file_bytes):
    """Load PDF content from bytes"""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

@st.cache_resource
def load_embedder():
    """Load the sentence transformer model"""
    return SentenceTransformer("all-MiniLM-L6-v2")

def process_documents(files, embedder):
    """Process uploaded documents and create FAISS index"""
    documents = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Load documents
    for i, file in enumerate(files):
        status_text.text(f"Processing {file.name}...")
        file_bytes = file.read()
        text = load_pdf_from_bytes(file_bytes)
        if text.strip():
            documents.append(text)
        progress_bar.progress((i + 1) / (len(files) + 2))

    if not documents:
        st.error("No valid documents found!")
        return None, None

    # Split into chunks
    status_text.text("Splitting documents into chunks...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    progress_bar.progress((len(files) + 1) / (len(files) + 2))

    if not chunks:
        st.error("No chunks created from documents!")
        return None, None

    # Create embeddings and FAISS index
    status_text.text("Creating embeddings and search index...")
    try:
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        progress_bar.progress(1.0)
        status_text.text("✅ Knowledge base created successfully!")
        return index, chunks
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None, None

def retriever(query, embedder, index, chunks, k=3):
    """Retrieve relevant chunks for a query"""
    q_emb = embedder.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]

def generator(query, docs, groq_client):
    """Generate answer using Groq"""
    context = " ".join(docs)
    prompt = f"""
You are an AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer clearly and concisely:
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def cache_rag(query, embedder, index, chunks, groq_client, cache, threshold=0.85):
    """RAG with caching functionality"""
    q_emb = embedder.encode(query)

    # Check cache
    for cached_q, entry in cache.items():
        c_emb = entry["embedding"]
        sim = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
        if sim > threshold:
            return entry["answer"], True  # Cache hit

    # Cache miss - retrieve and generate
    docs = retriever(query, embedder, index, chunks)
    ans = generator(query, docs, groq_client)
    cache[query] = {"embedding": q_emb, "answer": ans}
    return ans, False  # Cache miss

# Process documents when uploaded
if uploaded_files and groq_api_key:
    if st.button("🔄 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            # Load embedder
            if st.session_state.embedder is None:
                st.session_state.embedder = load_embedder()

            # Process documents
            index, chunks = process_documents(uploaded_files, st.session_state.embedder)

            if index is not None and chunks:
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.knowledge_base_ready = True
                st.success(f"✅ Successfully processed {len(uploaded_files)} documents with {len(chunks)} chunks!")
            else:
                st.session_state.knowledge_base_ready = False

elif uploaded_files and not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key to process documents.")
elif not uploaded_files:
    st.info("📤 Please upload PDF documents to get started.")

# Q&A Section
if st.session_state.knowledge_base_ready and groq_api_key:
    st.header("❓ Ask Questions")

    # Initialize Groq client
    try:
        groq_client = Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        st.stop()

    # Question input
    query = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of the document?",
        help="Ask any question about the content of your uploaded documents"
    )

    if query and st.button("🔍 Get Answer", type="primary"):
        with st.spinner("Searching for answer..."):
            try:
                answer, is_cached = cache_rag(
                    query,
                    st.session_state.embedder,
                    st.session_state.index,
                    st.session_state.chunks,
                    groq_client,
                    st.session_state.cache
                )

                # Display answer
                st.subheader("💡 Answer")
                st.write(answer)

                # Show cache status
                if is_cached:
                    st.success("✅ Answer retrieved from cache")
                else:
                    st.info("🔍 New answer generated")

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

    # Display cache statistics
    if st.session_state.cache:
        st.sidebar.subheader("📊 Cache Statistics")
        st.sidebar.write(f"Cached queries: {len(st.session_state.cache)}")

        if st.sidebar.button("🗑️ Clear Cache"):
            st.session_state.cache = {}
            st.sidebar.success("Cache cleared!")

# Footer
st.markdown("---")
st.markdown(
    """
    **Instructions:**
    1. Enter your Groq API key in the sidebar
    2. Upload one or more PDF documents
    3. Click 'Process Documents' to build the knowledge base
    4. Ask questions about your documents

    The app uses caching to speed up similar queries!
    """
)