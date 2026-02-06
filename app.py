import streamlit as st
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Movie Bot (Groq Persistent)",
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
    st.error(f"❌ Missing dependencies. Error: {e}")
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

# Folder where the database will be saved on disk
VECTORSTORE_PATH = "faiss_index_movies"

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
        st.stop()

os.environ["GROQ_API_KEY"] = api_key

# ==============================
# VECTOR STORE (CACHED & PERSISTENT)
# ==============================
@st.cache_resource
def get_vectorstore():
    # 1. Define the Embedding Model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. CHECK: Does the database already exist on disk?
    if os.path.exists(VECTORSTORE_PATH):
        st.info("📂 Database found on disk. Loading...")
        try:
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.success("✅ Loaded from disk!")
            return vectorstore
        except Exception as e:
            st.error(f"Error loading existing database: {e}")
            st.warning("Rebuilding database...")

    # 3. BUILD: If not found (or error), create it from scratch
    all_documents = []
    progress_text = "⚙️ Building Database (Runs once)..."
    progress_bar = st.progress(0, text=progress_text)
    
    total_files = len(PDF_FILES)
    files_found = 0

    for i, pdf in enumerate(PDF_FILES):
        if os.path.exists(pdf):
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            all_documents.extend(docs)
            files_found += 1
        progress_bar.progress((i + 1) / total_files, text=f"Processed: {pdf}")

    if files_found == 0:
        st.error("❌ No PDF files found. Please check your folder.")
        st.stop()

    progress_bar.progress(0.9, text="🧠 creating embeddings...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_documents)

    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 4. SAVE: Write to disk for next time
    vectorstore.save_local(VECTORSTORE_PATH)
    
    progress_bar.empty()
    st.success(f"✅ Database created and saved to '{VECTORSTORE_PATH}'!")
    
    return vectorstore

# ==============================
# GROQ LLM
# ==============================
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

# ==============================
# MAIN APP
# ==============================
st.title("🎬 Movie Expert")
st.caption("Powered by Persistent FAISS + Groq")

try:
    # This will now be instant on the 2nd run
    vectorstore = get_vectorstore()
    llm = get_llm()
except Exception as e:
    st.error(f"System initialization failed: {e}")
    st.stop()

# ==============================
# CHAT INTERFACE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! 🎥 Ask me about any movie."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_query := st.chat_input("Ex: Inception, Zulu, Futureworld"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔍 *Searching...*")

        try:
            results = vectorstore.similarity_search(user_query, k=10)
            context = "\n\n".join(doc.page_content for doc in results)

            prompt = f"""
            You are a movie expert.
            QUESTION: Find the movie titled "{user_query}" and provide a short summary.
            If not found, say "I couldn't find that movie."
            CONTEXT:
            {context}
            """
            
            response = llm.invoke(prompt)
            placeholder.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})

        except Exception as e:
            st.error(f"AI Error: {e}")
