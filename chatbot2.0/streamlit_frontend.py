# streamlit_frontend.py
"""
Enhanced Streamlit chat UI with improved markdown and code block handling
"""

import streamlit as st
import html
import os
import json
import time
import uuid
from datetime import datetime
from typing import List
from langchain_core.messages import HumanMessage
import re
from langgraph_backend import chat_stream, chat_sync, STRUCTURED_PREFIX

# Page config
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Styles (dark theme + modern gradients) ------------------
st.markdown(
    """
<style>
/* Main app background */
body, .stApp {
    background: #0b1220 !important;
    color: #e6eef8;
}

/* Header card with gradient */
.header {
    background: linear-gradient(90deg, #0f1724 0%, #121827 100%);
    color: white;
    border-radius: 10px;
    padding: 15px 25px;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(2,6,23,0.6);
}

/* Chat bubbles with improved gradients */
.user-message {
    background: linear-gradient(90deg, #0b63d1 0%, #0b4fa8 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 18px 18px 0 18px;
    margin: 10px 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 8px 20px rgba(11,99,209,0.12);
    word-wrap: break-word;
    font-family: "Source Code Pro", monospace;
}

.assistant-message {
    background: linear-gradient(180deg, #0f1724 0%, #0b1220 100%);
    border: 1px solid rgba(255,255,255,0.03);
    color: #e6eef8;
    padding: 15px 20px;
    border-radius: 18px 18px 18px 0;
    margin: 10px 0;
    max-width: 80%;
    box-shadow: 0 4px 16px rgba(2,6,23,0.4);
    word-wrap: break-word;
    font-family: "Source Code Pro", monospace;
}

/* Code block styling */
pre {
    background: #0b1220 !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02) !important;
    font-family: "Source Code Pro", monospace !important;
    font-size: 13px !important;
    white-space: pre-wrap !important;
    overflow-x: auto !important;
    margin: 8px 0 !important;
}

/* Small metadata text */
.meta { 
    font-size: 0.77em; 
    color: #9fb0d6; 
    margin-bottom: 8px; 
}

/* Markdown-style formatting */
.md-bold { font-weight: bold; }
.md-italic { font-style: italic; }
.md-list { padding-left: 20px; margin-top: 4px; }
.md-list-item { margin-bottom: 4px; }

/* Small utilities */
.title-muted { color:#869bb8; font-size:0.85em; }
.details-summary { 
    cursor: pointer; 
    color: #a8c0ff; 
    text-decoration: underline; 
}

/* Button styling with gradient */
.stButton>button {
    background: linear-gradient(90deg, #0b63d1 0%, #0b4fa8 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 25px;
    font-weight: 600;
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

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(135deg, #0f1724 0%, #121827 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Chat input styling */
.stChatInput {
    background-color: rgba(15, 23, 36, 0.8);
    border-radius: 25px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}
</style>
""",
    unsafe_allow_html=True,
)

def render_markdown(content: str) -> str:
    """Enhanced markdown renderer with proper code block handling."""
    # Handle code blocks first
    content = re.sub(
        r'```(.*?)```', 
        r'<pre>\1</pre>', 
        content, 
        flags=re.DOTALL
    )
    
    # Handle inline code
    content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
    
    # Handle bold (**text**)
    content = re.sub(r'\*\*(.*?)\*\*', r'<span class="md-bold">\1</span>', content)
    # Handle italics (*text*)
    content = re.sub(r'\*(.*?)\*', r'<span class="md-italic">\1</span>', content)
    # Handle lists (lines starting with - or *)
    lines = content.split('\n')
    in_list = False
    result = []
    for line in lines:
        if line.startswith('- ') or line.startswith('* '):
            if not in_list:
                result.append('<div class="md-list">')
                in_list = True
            result.append(f'<div class="md-list-item">{line[2:]}</div>')
        else:
            if in_list:
                result.append('</div>')
                in_list = False
            result.append(line)
    if in_list:
        result.append('</div>')
    return '<br>'.join(result)

# ------------------ Session state ------------------
if "message_history" not in st.session_state:
    st.session_state.message_history = []  # each item: {"role","content","ts", "meta": optional}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "last_stream_error" not in st.session_state:
    st.session_state.last_stream_error = None

CHAT_FILE = "chat_history.json"

def load_chats():
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_chats():
    try:
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save chats: {e}")

def new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.message_history = []
    st.session_state.active_chat = st.session_state.thread_id

def load_chat(chat_id):
    data = st.session_state.chat_history.get(chat_id)
    if data:
        st.session_state.thread_id = chat_id
        st.session_state.message_history = data["messages"].copy()
        st.session_state.active_chat = chat_id

# load saved chats
if not st.session_state.chat_history:
    st.session_state.chat_history = load_chats() or {}

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("## 🤖 Conversations")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat()

    st.markdown("### Recent")
    chats_sorted = sorted(st.session_state.chat_history.items(), 
                         key=lambda kv: kv[1].get("timestamp", 0), 
                         reverse=True)
    for chat_id, chat_data in chats_sorted:
        title = chat_data.get("title") or "Untitled"
        if st.button(f"{title[:28]}{'...' if len(title)>28 else ''}", 
                    key=f"c_{chat_id}"):
            load_chat(chat_id)

    st.divider()
    st.markdown("### Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.6, 0.05)
    st.caption("Streaming will be used where the backend LLM supports it.")
    st.divider()
    if st.button("🧹 Clear All History", use_container_width=True):
        st.session_state.chat_history = {}
        save_chats()
        st.rerun()

# ------------------ Header ------------------
st.markdown(
    f"""
<div class="header">
  <h2 style="margin:0">AI Assistant</h2>
  <div style="margin-top:6px; opacity:0.9">Streaming chat with markdown support</div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------ Main layout ------------------
left_col, right_col = st.columns((3,1))

with left_col:
    # Render message history
    for message in st.session_state.message_history:
        role = message.get("role")
        content = message.get("content", "")
        ts = message.get("ts", "")
        meta = message.get("meta")
        
        safe_content_html = render_markdown(html.escape(content).replace("\n", "<br>"))
        
        if role == "user":
            st.markdown(
                f'<div class="user-message"><div class="meta"><strong>You</strong> <span class="title-muted">{ts}</span></div>{safe_content_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Assistant message
            st.markdown(
                f'<div class="assistant-message"><div class="meta"><strong>Assistant</strong> <span class="title-muted">{ts}</span></div>{safe_content_html}',
                unsafe_allow_html=True,
            )

            if meta:
                try:
                    meta_preview = json.dumps({k: v for k, v in meta.items() if k != "content"}, 
                                            indent=2, ensure_ascii=False)
                except Exception:
                    meta_preview = str(meta)
                st.markdown(
                    f'<details style="margin-top:8px"><summary class="details-summary">Metadata (click to open)</summary><pre style="margin-top:8px">{html.escape(meta_preview)}</pre></details></div>',
                    unsafe_allow_html=True,
                )

    # Input box
    user_input = st.chat_input("Type a message (press Enter to send)")

    if user_input:
        now_ts = datetime.now().strftime("%b %d %H:%M")
        st.session_state.message_history.append({"role":"user","content":user_input,"ts":now_ts})
        assistant_placeholder = st.empty()
        assistant_placeholder.markdown(
            f'<div class="assistant-message"><div class="meta"><strong>Assistant</strong> <span class="title-muted">...</span></div>Thinking…</div>', 
            unsafe_allow_html=True
        )

        # Prepare backend messages
        messages_to_send = [HumanMessage(content=user_input)]

        # Build invoke kwargs
        invoke_kwargs = {"temperature": float(temperature)}

        # Streaming loop
        full_text = ""
        stream_error = None
        assistant_meta = None
        try:
            for chunk in chat_stream(messages_to_send, **invoke_kwargs):
                if not chunk:
                    continue

                # Handle structured payload
                if isinstance(chunk, str) and chunk.startswith(STRUCTURED_PREFIX):
                    raw = chunk[len(STRUCTURED_PREFIX):]
                    try:
                        payload = json.loads(raw)
                        content = payload.get("content", "")
                        if content:
                            full_text = content
                        assistant_meta = {k: v for k, v in payload.items() if k != "content"}
                        
                        # Render the structured content
                        safe_content = render_markdown(html.escape(content).replace("\n","<br>"))
                        meta_preview = json.dumps(assistant_meta, indent=2, ensure_ascii=False)
                        
                        assistant_placeholder.markdown(
                            f'<div class="assistant-message"><div class="meta"><strong>Assistant</strong> <span class="title-muted">{datetime.now().strftime("%b %d %H:%M")}</span></div>{safe_content}<details style="margin-top:8px"><summary class="details-summary">Metadata (click)</summary><pre style="margin-top:8px">{html.escape(meta_preview)}</pre></details></div>',
                            unsafe_allow_html=True,
                        )
                        continue
                    except Exception:
                        assistant_placeholder.markdown(
                            f'<div class="assistant-message"><div class="meta"><strong>Assistant</strong> <span class="title-muted">{datetime.now().strftime("%b %d %H:%M")}</span></div><pre>{html.escape(raw)}</pre></div>',
                            unsafe_allow_html=True,
                        )
                        continue

                # Handle stream error marker
                if isinstance(chunk, str) and chunk.startswith("[__STREAM_ERROR__]"):
                    stream_error = chunk.replace("[__STREAM_ERROR__]","")
                    break

                # Normal text chunk
                try:
                    piece = str(chunk)
                except Exception:
                    piece = repr(chunk)
                full_text += piece
                safe_partial = render_markdown(html.escape(full_text).replace("\n","<br>"))

                assistant_placeholder.markdown(
                    f'<div class="assistant-message"><div class="meta"><strong>Assistant</strong> <span class="title-muted">{datetime.now().strftime("%b %d %H:%M")}</span></div>{safe_partial}</div>',
                    unsafe_allow_html=True,
                )

                # Smoothing for streaming effect
                time.sleep(0.01)

        except Exception as e:
            stream_error = str(e)

        # Finalize
        if stream_error:
            st.session_state.last_stream_error = stream_error
            assistant_placeholder.markdown(
                f'<div class="assistant-message"><div class="meta"><strong>Assistant</strong> <span class="title-muted">error</span></div>Could not stream response: {html.escape(stream_error)}</div>', 
                unsafe_allow_html=True
            )

            # Fallback synchronous attempt
            try:
                fallback_raw = chat_sync(messages_to_send, **invoke_kwargs)
                final_meta = None
                final_text = fallback_raw
                if isinstance(fallback_raw, str) and fallback_raw.startswith(STRUCTURED_PREFIX):
                    try:
                        payload = json.loads(fallback_raw[len(STRUCTURED_PREFIX):])
                        final_text = payload.get("content", "")
                        final_meta = {k: v for k, v in payload.items() if k != "content"}
                    except Exception:
                        final_text = fallback_raw
                if not final_text:
                    final_text = "(no text returned)"
                    
                st.session_state.message_history.append({
                    "role": "assistant",
                    "content": final_text,
                    "ts": datetime.now().strftime("%b %d %H:%M"), 
                    "meta": final_meta
                })
                
                safe_final_text = render_markdown(html.escape(final_text).replace("\n","<br>"))
                assistant_placeholder.markdown(
                    f'<div class="assistant-message"><div class="meta"><strong>Assistant</strong> <span class="title-muted">{datetime.now().strftime("%b %d %H:%M")}</span></div>{safe_final_text}</div>', 
                    unsafe_allow_html=True
                )
                
                # Save conversation
                chat_title = user_input[:60] + ("..." if len(user_input)>60 else "")
                st.session_state.chat_history[st.session_state.thread_id] = {
                    "title": chat_title, 
                    "messages": st.session_state.message_history, 
                    "timestamp": datetime.now().timestamp()
                }
                save_chats()
            except Exception as e:
                assistant_placeholder.markdown(
                    f'<div class="assistant-message"><div class="meta"><strong>Assistant</strong> <span class="title-muted">failed</span></div>Final attempt failed: {html.escape(str(e))}</div>', 
                    unsafe_allow_html=True
                )

        else:
            # Streaming succeeded
            saved = {
                "role": "assistant",
                "content": full_text,
                "ts": datetime.now().strftime("%b %d %H:%M")
            }
            if assistant_meta:
                saved["meta"] = assistant_meta
            st.session_state.message_history.append(saved)
            
            chat_title = user_input[:60] + ("..." if len(user_input)>60 else "")
            st.session_state.chat_history[st.session_state.thread_id] = {
                "title": chat_title, 
                "messages": st.session_state.message_history, 
                "timestamp": datetime.now().timestamp()
            }
            save_chats()

        # Refresh to show final saved message
        st.rerun()

with right_col:
    st.markdown("### Conversation Tools")
    if st.session_state.message_history:
        if st.button("Download chat (.txt)"):
            lines: List[str] = []
            for m in st.session_state.message_history:
                who = "You" if m["role"] == "user" else "Assistant"
                ts = m.get("ts","")
                lines.append(f"[{ts}] {who}:")
                lines.append(m["content"])
                lines.append("")
            payload = "\n".join(lines)
            st.download_button(
                "Click to download", 
                payload, 
                file_name=f"chat_{st.session_state.thread_id}.txt", 
                mime="text/plain"
            )
            
        if st.button("Copy transcript to clipboard"):
            transcript = "\n\n".join([
                f'{("You" if m["role"]=="user" else "Assistant")}: {m["content"]}' 
                for m in st.session_state.message_history
            ])
            st.text_area("Transcript (select all and copy)", value=transcript, height=220)

    st.divider()
    st.markdown("### Quick Actions")
    if st.button("Regenerate last reply"):
        # Remove last assistant message and re-run
        last_user = None
        for m in reversed(st.session_state.message_history):
            if m["role"] == "user":
                last_user = m
                break
        if last_user:
            if st.session_state.message_history and st.session_state.message_history[-1]["role"] == "assistant":
                st.session_state.message_history.pop()
            st.rerun()
        else:
            st.warning("No user message found to regenerate.")

    if st.session_state.last_stream_error:
        st.error(f"Last stream error: {st.session_state.last_stream_error}")

    st.divider()
    st.markdown("### Session")
    st.write(f"Thread id: `{st.session_state.thread_id}`")
    if st.button("Clear current chat"):
        st.session_state.message_history = []
        st.rerun()