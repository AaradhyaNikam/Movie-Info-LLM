# 🎬 Movie Bot (Groq Hybrid Edition)

An **AI-powered Movie Expert App** built using **Streamlit + LangChain**, featuring a **Hybrid RAG Architecture**:

- 🔍 **Local Embeddings (FAISS + HuggingFace)** for unlimited, fast search  
- ⚡ **Groq LLaMA-3.1** for ultra-fast and reliable AI responses  

This design avoids API rate limits and model deprecation issues while remaining cloud-deployable.

---

## 🚀 Features

- 📄 Loads movie data from multiple PDF files (A–Z split)
- 🧠 Local embeddings (CPU-based, unlimited usage)
- ⚡ Groq-powered LLaMA-3.1 inference
- 💬 Chat-style conversational UI
- ♻️ Cached vector store for fast reloads
- ☁️ Works locally & on Streamlit Cloud

---

## 🏗️ Tech Stack

- **Frontend**: Streamlit  
- **LLM Provider**: Groq  
- **Model**: `llama-3.1-8b-instant`  
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`  
- **Vector DB**: FAISS  
- **Framework**: LangChain  

---

## 📁 Project Structure

```
movie-bot/
│
├── app.py
├── Movies_A-F.pdf
├── Movies_G-L.pdf
├── Movies_M-R.pdf
├── Movies_S-Z.pdf
├── requirements.txt
└── README.md
```

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/movie-bot.git
cd movie-bot
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Groq API Key Setup

### Option 1: Streamlit Sidebar (Local)
Paste your Groq API key directly in the sidebar.

### Option 2: Environment Variable
```bash
export GROQ_API_KEY="your_api_key"
```

### Option 3: Streamlit Cloud
Add this to **Secrets**:
```
GROQ_API_KEY = "your_api_key"
```

👉 Get a free key: https://console.groq.com/keys

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Movie PDFs are loaded and split into chunks  
2. Local embeddings are generated using HuggingFace  
3. FAISS performs similarity search  
4. Relevant context is sent to Groq LLaMA-3.1  
5. The model generates a clean movie summary  

---

## 🛡️ Why Hybrid Architecture?

| Component | Benefit |
|--------|--------|
| Local Embeddings | No API limits, free |
| FAISS | Fast similarity search |
| Groq LLaMA-3.1 | Ultra-low latency |
| Streamlit | Rapid deployment |

---

## 🧪 Example Queries

- Inception  
- Zulu  
- Futureworld  
- The Godfather  

---

## 📜 License

MIT License

---

## 🙌 Author

Built with ❤️ using LangChain, Groq, and Streamlit
