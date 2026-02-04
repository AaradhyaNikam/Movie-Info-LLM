import streamlit as st
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Movie Bot (Gemini Edition)", page_icon="🎬")

# --- IMPORTS ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    # NEW: Google Gemini Imports
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"Missing Dependencies. Error: {e}")
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
    # Check if key is in secrets (for Cloud deployment)
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        st.warning("⚠️ Please enter your Google API Key (starts with AIza...) to continue.")
        st.info("Get a free key here: https://aistudio.google.com/app/apikey")
        st.stop()

# Set the key for the session
os.environ["GOOGLE_API_KEY"] = api_key

# --- CACHED FUNCTIONS ---
@st.cache_resource
def get_vectorstore():
    """Loads PDFs and builds the vector database."""
    all_documents = []
    
    # Progress UI
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
        st.error("❌ No documents found! Please make sure the PDF files are in the folder.")
        st.stop()

    my_bar.progress(0.9, text="Creating AI Embeddings...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_documents)

    # NEW: Use Google's Embeddings (Free)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    my_bar.empty()
    return vectorstore

@st.cache_resource
def get_llm():
    """Initializes the Google Gemini Brain."""
    # Gemini 1.5 Flash is free, fast, and very smart
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- MAIN APP ---
st.title("🎬 Movie Expert AI (Gemini)")
st.caption("Powered by Google Gemini 1.5 Flash")

# Initialize
try:
    with st.spinner("Connecting to Google AI..."):
        vectorstore = get_vectorstore()
        llm = get_llm()
except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.stop()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready. Ask me about any movie!"}]

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
        message_placeholder.markdown("🔎 *Searching...*")
        
        try:
            # 1. Search (k=10 ensures we find deep matches)
            results = vectorstore.similarity_search(movie_name, k=10)
            context_text = "\n\n".join([doc.page_content for doc in results])
            
            # 2. Prompt
            prompt = f"""
            You are a helpful movie assistant.
            
            QUESTION:
            Find the movie titled '{movie_name}' and provide a summary. 
            If the exact movie is not found, say "I couldn't find that movie."
            
            CONTEXT FROM DATABASE:
            {context_text}
            """
            
            # 3. Answer
            response = llm.invoke(prompt)
            full_response = response.content
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"AI Error: {e}")