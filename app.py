#!/usr/bin/env python3
"""
Chatbot Interface - Farhat Abbas University Sétif 1
Uses Streamlit + RAG + Groq LLM
"""
import streamlit as st
import sys
import time

# Import your existing modules
from LLM import StreamSession, GroqClient, PromptBuilder, detect_language
import rag

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="University Chatbot - Sétif 1",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 5px solid #4CAF50;
    }
    .source-link {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .score-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────────
if "session" not in st.session_state:
    st.session_state.session = StreamSession()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

# ── Initialize RAG once ─────────────────────────────────────
@st.cache_resource
def init_rag():
    """Initialize RAG retriever once and cache it."""
    try:
        _ = rag._get_retriever()
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG: {e}")
        return False

if not st.session_state.rag_initialized:
    with st.spinner("🔄 Initializing RAG system..."):
        st.session_state.rag_initialized = init_rag()

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/university.png", width=80)
    st.title("🎓 University Chatbot")
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.info(
        "Farhat Abbas University Sétif 1\n\n"
        "Ask questions about:\n"
        "- 📚 Courses & Programs\n"
        "- 👨‍🏫 Teachers & Staff\n"
        "- 📅 Schedules & Exams\n"
        "- 🏛️ Faculties & Departments\n"
        "- 📄 Administrative Info"
    )
    
    st.markdown("---")
    
    # Language selector
    lang = st.selectbox(
        "🌐 Interface Language",
        ["Auto-detect", "English", "Français", "العربية"],
        index=0
    )
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session.reset()
        st.rerun()
    
    st.markdown("---")
    st.caption("v1.0 - Powered by RAG + Groq LLM")

# ── Main Header ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 University Chatbot</h1>
    <p>Farhat Abbas University Sétif 1 — Your Academic Assistant</p>
</div>
""", unsafe_allow_html=True)

# ── Example Questions ───────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Master CS specializations", use_container_width=True):
            st.session_state.initial_query = "What are master computer science specializations?"
            st.rerun()
        if st.button("👨‍🏫 CS teachers", use_container_width=True):
            st.session_state.initial_query = "Who are the computer science teachers?"
            st.rerun()
    
    with col2:
        if st.button("📅 Exam schedule L2", use_container_width=True):
            st.session_state.initial_query = "Emploi des examens L2 informatique"
            st.rerun()
        if st.button("🏛️ Faculty structure", use_container_width=True):
            st.session_state.initial_query = "Structure de la faculté des sciences"
            st.rerun()
    
    with col3:
        if st.button("تخصصات الماستر إعلام آلي", use_container_width=True):
            st.session_state.initial_query = "ما هي تخصصات الماستر في الإعلام الآلي"
            st.rerun()
        if st.button("أساتذة قسم الرياضيات", use_container_width=True):
            st.session_state.initial_query = "من هم أساتذة قسم الرياضيات"
            st.rerun()

# ── Chat Input ──────────────────────────────────────────────
if "initial_query" in st.session_state:
    prompt = st.session_state.initial_query
    del st.session_state.initial_query
else:
    prompt = st.chat_input("Ask a question about the university...")

# ── Display Chat History ────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- [{src['title']}]({src['url']}) `[{src['score']}]`")

# ── Process Query ───────────────────────────────────────────
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        
        try:
            # Stream response
            for chunk in st.session_state.session.chat(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Extract sources from last generation
            if hasattr(st.session_state.session._generator, '_last_sources'):
                raw_sources = st.session_state.session._generator._last_sources
                sources = [
                    {
                        "title": s.title,
                        "url": s.url or s.pdf_url,
                        "score": round(s.chunk_score, 2)
                    }
                    for s in raw_sources[:5] if s.url or s.pdf_url
                ]
            
            # Show sources
            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        score_color = "green" if src["score"] > 0.5 else "orange"
                        st.markdown(
                            f"- [{src['title']}]({src['url']}) "
                            f"<span style='color:{score_color}'>[{src['score']}]</span>",
                            unsafe_allow_html=True
                        )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            full_response = "Sorry, an error occurred. Please try again."
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
    
    # Reset session for next question
    if st.session_state.session.finished:
        st.session_state.session.reset()

# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ This chatbot uses official university data. Verify critical information on the official website.")
