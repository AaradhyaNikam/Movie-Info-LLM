import streamlit as st
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Movie Bot (Groq Hybrid Edition)",
    page_icon="🎬",
    layout="centered"
)

# ==============================
# IMPORTS
# ==============================
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    # Local embeddings (CPU, unlimited)
    from langchain_huggingface import HuggingFaceEmbeddings

    # Groq LLM
    from langchain_groq import ChatGroq

except ImportError as e:
    st.error(
        "❌ Missing dependencies.\n\n"
        "Run:\n"
        "`pip install -r requirements.txt`\n\n"
        f"Error: {e}"
    )
    st.stop()

# ==============================
# CONFIG
# ==============================
PDF_FILES = [
    "Movies_A-F.pdf",
    "Movies_G-L.pdf",
    "Movies_M-R.pdf",
    "Movies_S-Z.pdf"
]

# ==============================
# SIDEBAR – GROQ API KEY
# ==============================
st.sidebar.header("🔑 Setup")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        st.warning("⚠️ Please enter your Groq API Key.")
        st.info("Get a free key here: https://console.groq.com/keys")
        st.stop()

os.environ["GROQ_API_KEY"] = api_key

# ==============================
# VECTOR STORE (LOCAL)
# ==============================
@st.cache_resource
def get_vectorstore():
    all_documents = []

    progress_text = "📚 Loading Movie Database..."
    progress_bar = st.progress(0, text=progress_text)

    total_files = len(PDF_FILES)

    for i, pdf in enumerate(PDF_FILES):
        if os.path.exists(pdf):
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            all_documents.extend(docs)

        progress_bar.progress((i + 1) / total_files, text=f"Processed: {pdf}")

    if not all_documents:
        st.error("❌ No PDF files found. Please check the directory.")
        st.stop()

    progress_bar.progress(0.85, text="🧠 Creating local embeddings (CPU)...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)

    progress_bar.empty()
    return vectorstore

# ==============================
# GROQ LLM
# ==============================
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",  # Best quality
        temperature=0
    )

# ==============================
# MAIN APP
# ==============================
st.title("🎬 Movie Expert (Groq Hybrid)")
st.caption("Powered by Local Embeddings + Groq LLaMA-3")

try:
    with st.spinner("⚙️ Initializing AI Engine..."):
        vectorstore = get_vectorstore()
        llm = get_llm()
except Exception as e:
    st.error(f"System initialization failed: {e}")
    st.stop()

# ==============================
# CHAT HISTORY
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 🎥 Ask me about any movie."
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==============================
# USER INPUT
# ==============================
if user_query := st.chat_input("Ex: Inception, Zulu, Futureworld"):
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔍 *Searching movie database...*")

        try:
            # 1. Retrieve context
            results = vectorstore.similarity_search(user_query, k=10)
            context = "\n\n".join(
                doc.page_content for doc in results
            )

            # 2. Prompt
            prompt = f"""
You are a movie expert.

QUESTION:
Find the movie titled "{user_query}" and provide a short summary.
If the exact movie is not found, say:
"I couldn't find that movie in the database."

CONTEXT:
{context}
"""

            # 3. LLM Response
            response = llm.invoke(prompt)
            answer = response.content

            placeholder.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            st.error(f"AI Error: {e}")
