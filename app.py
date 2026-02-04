import streamlit as st
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Movie Bot (Hybrid Edition)", page_icon="🎬")

# --- IMPORTS ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    # 1. Local Embeddings (Unlimited, runs on CPU)
    from langchain_huggingface import HuggingFaceEmbeddings
    # 2. Google Chat (Smart, Free Tier)
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"Missing Dependencies. Please run: pip install -r requirements.txt\nError: {e}")
    st.stop()

# --- CONFIGURATION ---
PDF_FILES = [
    "Movies_A-F.pdf", 
    "Movies_G-L.pdf", 
    "Movies_M-R.pdf", 
    "Movies_S-Z.pdf"
]

# --- SIDEBAR: API KEY ---
st.sidebar.header("🔑 Setup")
api_key = st.sidebar.text_input("Enter Google API Key", type="password")

if not api_key:
    # Check secrets for Cloud deployment
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        st.warning("⚠️ Please enter your Google API Key to continue.")
        st.info("Get a free key here: https://aistudio.google.com/app/apikey")
        st.stop()

# Set key
os.environ["GOOGLE_API_KEY"] = api_key

# --- CACHED FUNCTIONS ---
@st.cache_resource
def get_vectorstore():
    """Loads PDFs using LOCAL embeddings to avoid Google Rate Limits."""
    all_documents = []
    
    progress_text = "Loading Movie Database... Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    files_found = 0
    total_files = len(PDF_FILES)
    
    for i, pdf_file in enumerate(PDF_FILES):
        if os.path.exists(pdf_file):
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                all_documents.extend(docs)
                files_found += 1
            except Exception as e:
                st.error(f"Error loading {pdf_file}: {e}")
        
        my_bar.progress((i + 1) / total_files, text=f"Processed {pdf_file}")

    if not all_documents:
        st.error("❌ No documents found! Please ensure PDF files are in the folder.")
        st.stop()

    my_bar.progress(0.9, text="Creating Local Embeddings (CPU)...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_documents)

    # HYBRID FIX: Use Local Embeddings instead of Google's
    # This runs on your computer and has NO LIMITS.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    my_bar.empty()
    return vectorstore

@st.cache_resource
def get_llm():
    """Initializes the Google Gemini Brain for chatting."""
    # We only use Google for the Answer, not the Embeddings.
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- MAIN APP ---
st.title("🎬 Movie Expert (Hybrid)")
st.caption("Powered by Local Embeddings + Google Gemini")

# Initialize
try:
    with st.spinner("Initializing Hybrid AI Engine..."):
        vectorstore = get_vectorstore()
        llm = get_llm()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready. Ask me about any movie!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if movie_name := st.chat_input("Ex: Inception, Futureworld, Zulu"):
    st.session_state.messages.append({"role": "user", "content": movie_name})
    with st.chat_message("user"):
        st.write(movie_name)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔎 *Searching database...*")
        
        try:
            # 1. Search
            results = vectorstore.similarity_search(movie_name, k=10)
            context_text = "\n\n".join([doc.page_content for doc in results])
            
            # 2. Prompt
            prompt = f"""
            You are a movie expert.
            
            QUESTION:
            Find the movie titled '{movie_name}' and provide a summary. 
            If the exact movie is not found, say "I couldn't find that movie."
            
            CONTEXT:
            {context_text}
            """
            
            # 3. Answer
            response = llm.invoke(prompt)
            full_response = response.content
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"AI Error: {e}")