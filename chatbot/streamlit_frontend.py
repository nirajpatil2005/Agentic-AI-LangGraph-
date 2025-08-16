import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# Page configuration with modern theme
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main page styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 10px;
        padding: 15px 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    
    /* Chat containers */
    .user-message {
        background-color: #4b6cb7;
        color: white;
        padding: 15px 20px;
        border-radius: 18px 18px 0 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background-color: white;
        color: #333;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 0;
        margin: 10px 0;
        max-width: 80%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Code blocks and structured content */
    pre {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 15px;
        border-radius: 10px;
        overflow-x: auto;
        margin: 10px 0;
    }
    
    /* Chat input styling */
    .stChatInput {
        background-color: white;
        border-radius: 25px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #4b6cb7 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
    
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = f"thread_{uuid.uuid4().hex}"

# Header with gradient
st.markdown(
    """
    <div class="header">
        <h1 style="margin:0; padding:0;">🤖 AI Assistant</h1>
        <p style="margin:0; opacity:0.8;">Ask anything and get structured, detailed responses</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar for controls
with st.sidebar:
    st.subheader("Conversation Settings")
    
    # Model selection
    model_type = st.selectbox(
        "Response Style",
        ("Concise", "Detailed", "Technical", "Creative"),
        index=1
    )
    
    # Clear button
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.message_history = []
        st.session_state.thread_id = f"thread_{uuid.uuid4().hex}"
        st.rerun()
    
    st.divider()
    st.caption("💡 Tip: Ask complex questions to see structured responses")
    st.caption("✨ Responses include formatted content with code blocks when needed")

# Display chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.message_history:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">'
                f'<strong>You:</strong><br>{message["content"]}'
                f'</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-message">'
                f'<strong>Assistant:</strong><br>{message["content"]}'
                f'</div>', 
                unsafe_allow_html=True
            )

# Chat input and processing
user_input = st.chat_input("Type your message here...", key="chat_input")

if user_input:
    # Add user message to history
    st.session_state.message_history.append({"role": "user", "content": user_input})
    
    # Create config with dynamic thread ID
    CONFIG = {'configurable': {'thread_id': st.session_state.thread_id}}
    
    # Get AI response
    with st.spinner("Generating response..."):
        response = chatbot.invoke(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG
        )
        ai_content = response['messages'][-1].content
    
    # Format response with markdown support
    formatted_content = ai_content
    
    # Add assistant response to history
    st.session_state.message_history.append({
        "role": "assistant",
        "content": formatted_content
    })
    
    # Rerun to update the display
    st.rerun()