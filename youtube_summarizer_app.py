import streamlit as st
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="YouTube AI Agent: Video Summarization & Q&A Using RAG",
    page_icon="🎥",
    layout="wide"
)


st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html = True)


if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_id' not in st.session_state:
    st.session_state.video_id = None


def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    
    return None


@st.cache_resource
def initialize_models(hf_token, openai_key):
    llm = ChatOpenAI(
        base_url = "https://router.huggingface.co/v1",
        api_key = hf_token,
        model = "moonshotai/Kimi-K2-Thinking"
    )
    
    embeddings = OpenAIEmbeddings(
        base_url = "https://openrouter.ai/api/v1",
        api_key = openai_key,
        model = "text-embedding-3-small"
    )
    
    return llm, embeddings


def get_transcript_from_url(youtube_url):
    
    video_id = extract_video_id(youtube_url)
    
    if not video_id:
        raise ValueError("Invalid YouTube URL. Please provide a valid YouTube link.")
    
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id, languages=["en", "hi"]).to_raw_data()
        transcript = " ".join(chunk["text"] for chunk in transcript_data)
        return transcript, video_id
    except TranscriptsDisabled:
        raise Exception("No captions available for this video.")
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")


def process_video(youtube_url, llm, embeddings):

    transcript, video_id = get_transcript_from_url(youtube_url)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    docs = text_splitter.create_documents([transcript])
    
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs = {"k": 8})
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        
        Context: {context}"""),
        ("user", "{question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, video_id, len(docs), len(transcript)


st.title("🎥 YouTube AI Agent: Video Summarization and Q&A Using RAG")
st.markdown("Ask questions about any YouTube video with captions!")


with st.sidebar:
    
    st.header("⚙️ Configuration")
    
    use_custom_keys = st.checkbox("Enter your own API keys?", value=False)
    
    if use_custom_keys:
        hf_token = st.text_input(
            "HuggingFace Token",
            type="password",
            help="get it from https://huggingface.co/settings/tokens"
        )
        openai_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="get it from https://openrouter.ai/settings/keys"
        )
    
    else:
        hf_token = os.getenv("HF_TOKEN", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        st.info("Using default API keys")
    
    st.markdown("---")
    
    if st.session_state.video_processed:
        st.success("✅ Video processed!")
        st.info(f"**Video ID:** {st.session_state.video_id}")
        st.markdown("---")
        if st.button("🔄 Process New Video"):
            st.session_state.rag_chain = None
            st.session_state.video_processed = False
            st.session_state.chat_history = []
            st.session_state.video_id = None
            st.rerun()
    
    st.markdown("### 📖 How to use:")
    st.markdown("""
    1. Enter your API keys (optional)
    2. Paste a YouTube URL
    3. Click '🚀 Process Video'
    4. Ask questions about the video!
    """)


if not st.session_state.video_processed:

    st.markdown("### 🔗 Enter YouTube Video URL")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )
    
    with col2:
        process_button = st.button("🚀 Process Video", use_container_width = True)
    
    with st.expander("💡 Example URLs"):
        st.code("https://www.youtube.com/watch?v=Gfr50f6ZBvo")
        st.code("https://youtu.be/Gfr50f6ZBvo")
        st.code("Gfr50f6ZBvo")
    
    if process_button:
        if not hf_token or not openai_key:
            st.error("⚠️ Please enter both API keys in the sidebar!")
        elif not youtube_url:
            st.error("⚠️ Please enter a YouTube URL!")
        else:
            try:
                with st.spinner("🔄 Processing video... This may take a moment."):

                    llm, embeddings = initialize_models(hf_token, openai_key)
                    
                    rag_chain, video_id, num_chunks, transcript_length = process_video(youtube_url, llm, embeddings)
                    
                    st.session_state.rag_chain = rag_chain
                    st.session_state.video_processed = True
                    st.session_state.video_id = video_id
                    
                    st.success(f"""
                    ✅ **Video processed successfully!**
                    """)
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

else:
    st.markdown("### 💬 Ask Questions About the Video")
    
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🙋 You:</strong><br>
                {question}
            </div>
            """, unsafe_allow_html = True)
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong><br>
                {answer}
            </div>
            """, unsafe_allow_html = True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Your question",
            placeholder = "What is this video about?",
            key = "question_input",
            label_visibility = "collapsed"
        )
    
    with col2:
        ask_button = st.button("📤 Ask", use_container_width = True)
    
    st.markdown("**💡 Suggested questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Summarize the video"):
            question = "Can you summarize the main points of this video?"
    
    with col2:
        if st.button("🔑 Key takeaways"):
            question = "What are the key takeaways from this video?"
    
    with col3:
        if st.button("📊 Main topics"):
            question = "What are the main topics discussed in this video?"
    
    if ask_button or (question and question not in [q for q, _ in st.session_state.chat_history]):
        if question:
            try:
                with st.spinner("🤔 Thinking..."):
                    answer = st.session_state.rag_chain.invoke(question)
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question!")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()