# 🎥 YouTube AI Agent: Video Summarization & Q&A Using RAG

This project is a **Streamlit-based AI Agent** that extracts transcripts
from YouTube videos and enables **context-aware question answering** and
**summarization** using a **Retrieval-Augmented Generation (RAG)**
pipeline.


------------------------------------------------------------------------

## 🚀 Features

-   📥 **YouTube Transcript Extraction**
    Automatically fetches captions (English/Hindi supported) from any
    YouTube video.

-   ✂️ **Transcript Chunking**
    Uses `RecursiveCharacterTextSplitter` for optimal chunking.

-   🧠 **Vector Embeddings with FAISS**
    Stores embeddings using FAISS for fast semantic search.

-   🔍 **Context-Aware Retrieval**
    Retrieves the most relevant transcript chunks for user queries.

-   🤖 **LLM-Powered Responses**
    Uses:

    -   **HuggingFace OpenRouter Model:** `moonshotai/Kimi-K2-Thinking`
    -   **OpenRouter Embeddings:** `text-embedding-3-small`

-   📝 **Video Summarization**

-   ❓ **Q&A Based on Transcript Only**

-   🧼 **Clear Chat History**

-   🔐 **Option to Use Custom API Keys**

------------------------------------------------------------------------

## 🏗️ Architecture Overview

### 1. **Transcript Fetching**

-   Extracts video ID from multiple YouTube URL formats.
-   Uses `youtube-transcript-api` to download captions.

### 2. **Document Splitting**

Chunks transcript into manageable pieces:

    chunk_size = 1000
    chunk_overlap = 200

### 3. **Embedding + Vectorstore**

-   Embeddings via OpenRouter API
-   Stores vectors in FAISS
-   Retriever performs top-8 semantic search

### 4. **RAG Pipeline**

A LangChain pipeline built from: - Retriever\
- Prompt Template\
- ChatOpenAI LLM\
- Output Parser

The agent answers **strictly from transcript context**.

------------------------------------------------------------------------

## 🧰 Tech Stack

  Component             Library
  --------------------- ---------------------------------
  UI                    Streamlit
  Transcript Fetching   youtube-transcript-api
  Text Splitting        langchain-text-splitters
  Embeddings            langchain-openai
  LLM                   ChatOpenAI (HuggingFace Router)
  Vector DB             FAISS
  RAG Framework         LangChain
  App Deployment        Streamlit

------------------------------------------------------------------------

## 📦 Installation

### 1. Clone the repository

``` bash
git clone https://github.com/MOHAMMED-ANAS-IQBAL/YouTube-AI-Agent-Video-Summarization-and-QnA-Using-RAG.git
cd youtube-ai-agent
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install requirements

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🔑 Environment Variables

Create a `.env` file:

    HF_TOKEN=your_huggingface_token
    OPENAI_API_KEY=your_openrouter_api_key

You may also enter keys manually inside the app UI.

------------------------------------------------------------------------

## ▶️ Run the App

``` bash
streamlit run youtube_summarizer_app.py
```

------------------------------------------------------------------------

## 📂 Project Structure

    📁 YouTube-RAG-Agent
    │── youtube_summarizer_app.py    # Main Streamlit application
    │── requirements.txt             # Python dependencies
    │── README.md                    # Project documentation
    │── .env (optional)              # API keys

------------------------------------------------------------------------

## 🖥️ How to Use

1.  Enter (optional) API keys in the sidebar
2.  Paste a YouTube URL
3.  Click **Process Video**
4.  Ask questions about the video transcript
5.  Use suggested prompts for quick insights

------------------------------------------------------------------------

## 🧪 Example URLs

    https://www.youtube.com/watch?v=Gfr50f6ZBvo
    https://youtu.be/Gfr50f6ZBvo
    Gfr50f6ZBvo

------------------------------------------------------------------------

## 📘 Dependencies

Installed via `requirements.txt`: - streamlit
- youtube-transcript-api
- langchain
- langchain-community
- langchain-openai
- faiss-cpu
- tiktoken
- openai

------------------------------------------------------------------------

## 🙌 Acknowledgements

This project uses: 
- **YouTube Transcript API** 
- **LangChain** 
- **FAISS** 
- **HuggingFace** 
- **OpenRouter**

------------------------------------------------------------------------

## 👨‍💻 Author

**MOHAMMED ANAS IQBAL**
