# langgraph_backend.py
"""
langgraph_backend.py

Robust backend wrapper for streaming & sync calls with improved markdown support.
"""

from typing import Any, List, Iterable, Mapping
from collections.abc import Iterable as IterableABC
from dotenv import load_dotenv
import os
import re
import json
import traceback
import logging

# Attempt to import langchain message classes / ChatGroq (best-effort)
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
except Exception:
    # Define light-weight stand-ins so type hints won't crash if package missing
    class BaseMessage: ...
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
    class AIMessage:
        def __init__(self, content: str):
            self.content = content

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("test_groq") or os.getenv("GROQ") or os.getenv("GROQ_API")

if ChatGroq is None:
    logger.warning("langchain_groq not available; llm invocation will fail until ChatGroq is installed or replaced.")
if not GROQ_API_KEY:
    logger.warning("GROQ API key not found in environment variables. Set GROQ_API_KEY or test_groq for production.")

# Instantiate LLM wrapper (if available)
if ChatGroq is not None:
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="openai/gpt-oss-20b", temperature=0.7)
else:
    llm = None

# Structured payload prefix for structured SDK-style objects
STRUCTURED_PREFIX = "[__STRUCTURED__]"

# ----------------- Helper utilities -----------------

def _to_primitive(obj: Any):
    """Recursively convert object to JSON-serializable primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, IterableABC) and not isinstance(obj, (str, bytes, bytearray, dict)):
        return [_to_primitive(x) for x in obj]
    try:
        if hasattr(obj, "__dict__"):
            return _to_primitive(vars(obj))
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return repr(obj)

def _looks_like_sdk_response(obj: Any) -> bool:
    """Heuristic to detect SDK-like LLM responses that include metadata."""
    if obj is None:
        return False
    if isinstance(obj, Mapping):
        keys = set(obj.keys())
        return bool(keys & {"content", "additional_kwargs", "response_metadata"})
    for attr in ("content", "additional_kwargs", "response_metadata"):
        if hasattr(obj, attr):
            return True
    return False

def _is_code_block(text: str) -> bool:
    """Check if text appears to be a code block."""
    code_keywords = ["def ", "class ", "import ", "from ", "if ", "for ", "while ", "try ", "except ", "return "]
    return any(keyword in text for keyword in code_keywords) or ("\n" in text and "```" in text)

def _safe_extract_text(item: Any) -> str:
    """
    Extract text from an LLM response piece with improved code block handling.
    If an SDK-like object is detected, return a JSON string prefixed with STRUCTURED_PREFIX.
    """
    if item is None:
        return ""

    if isinstance(item, str):
        # Check for existing markdown code blocks
        if "```" in item:
            return item
        # Auto-format code blocks if detected
        if _is_code_block(item):
            return f"```\n{item}\n```"
        return item

    # mapping case
    if isinstance(item, Mapping):
        if _looks_like_sdk_response(item):
            payload = {}
            for k in ("content", "additional_kwargs", "response_metadata", "type", "id", "name"):
                if k in item:
                    payload[k] = _to_primitive(item[k])
            for k, v in item.items():
                if k not in payload:
                    payload[k] = _to_primitive(v)
            try:
                return STRUCTURED_PREFIX + json.dumps(payload, ensure_ascii=False)
            except Exception:
                return STRUCTURED_PREFIX + json.dumps(_to_primitive(payload))
        
        # Convert simple dicts to markdown
        try:
            return json.dumps(_to_primitive(item), indent=2, ensure_ascii=False)
        except Exception:
            return str(item)

    # object-like case
    if _looks_like_sdk_response(item):
        payload = {}
        for attr in ("content", "additional_kwargs", "response_metadata", "type", "id", "name"):
            if hasattr(item, attr):
                try:
                    payload[attr] = _to_primitive(getattr(item, attr))
                except Exception:
                    payload[attr] = str(getattr(item, attr))
        try:
            if hasattr(item, "__dict__"):
                for k, v in vars(item).items():
                    if k not in payload:
                        payload[k] = _to_primitive(v)
        except Exception:
            pass
        try:
            return STRUCTURED_PREFIX + json.dumps(payload, ensure_ascii=False)
        except Exception:
            return STRUCTURED_PREFIX + json.dumps(_to_primitive(payload))

    # try content/text/message attribute names
    for attr in ("content", "text", "message"):
        if hasattr(item, attr):
            val = getattr(item, attr)
            if isinstance(val, str):
                if _is_code_block(val):
                    return f"```\n{val}\n```"
                return val
            if val is not None:
                return str(val)

    try:
        text = str(item)
        if _is_code_block(text):
            return f"```\n{text}\n```"
        return text
    except Exception:
        return repr(item)

def _invoke_with_kw_retry(invoke_fn, messages, stream_flag: bool, invoke_kwargs: dict):
    """
    Attempt to call invoke_fn with stream flag and kwargs. If TypeError indicates
    an unexpected kw, remove it and retry.
    """
    invoke_kwargs = dict(invoke_kwargs or {})
    try:
        if stream_flag:
            return invoke_fn(messages, stream=True, **invoke_kwargs)
        else:
            return invoke_fn(messages, **invoke_kwargs)
    except TypeError as e:
        msg = str(e)
        m = re.search(r"unexpected keyword argument '([^']+)'", msg) or re.search(r"got an unexpected keyword argument '([^']+)'", msg)
        if m:
            bad_kw = m.group(1)
            if bad_kw in invoke_kwargs:
                logger.warning("Removing unsupported invoke kw '%s' and retrying", bad_kw)
                del invoke_kwargs[bad_kw]
                return _invoke_with_kw_retry(invoke_fn, messages, stream_flag, invoke_kwargs)
        raise
    except Exception:
        raise

def _iter_from_stream_response(stream_resp):
    """Normalize a streaming response (various forms) into an iterator."""
    if hasattr(stream_resp, "__iter__") and not isinstance(stream_resp, (str, bytes, dict)):
        return iter(stream_resp)
    if hasattr(stream_resp, "stream") and callable(getattr(stream_resp, "stream")):
        return iter(stream_resp.stream())
    return iter([stream_resp])

# ----------------- Public API: chat_stream & chat_sync -----------------

def chat_stream(messages: List[BaseMessage], **invoke_kwargs) -> Iterable[str]:
    """
    Generator yielding incremental string chunks with proper markdown formatting.
    Tries streaming entrypoints first and gracefully falls back to synchronous.
    """
    try:
        if llm is None:
            yield "[__STREAM_ERROR__]LLM not initialized (ChatGroq missing)."
            return

        # 1) try invoke with stream=True
        try:
            stream_resp = _invoke_with_kw_retry(llm.invoke, messages, stream_flag=True, invoke_kwargs=invoke_kwargs)
            it = _iter_from_stream_response(stream_resp)
            for piece in it:
                yield _safe_extract_text(piece)
            return
        except TypeError:
            pass
        except Exception as e:
            logger.debug("invoke(stream=True) attempt failed: %s", e)

        # 2) try candidate streaming methods
        for candidate in ("stream_invoke", "invoke_stream", "stream", "streaming_invoke"):
            if hasattr(llm, candidate):
                method = getattr(llm, candidate)
                try:
                    stream_resp = method(messages, **invoke_kwargs)
                    it = _iter_from_stream_response(stream_resp)
                    for piece in it:
                        yield _safe_extract_text(piece)
                    return
                except TypeError:
                    try:
                        stream_resp = method(messages)
                        it = _iter_from_stream_response(stream_resp)
                        for piece in it:
                            yield _safe_extract_text(piece)
                        return
                    except Exception:
                        continue
                except Exception as e:
                    logger.debug("candidate %s failed: %s", candidate, e)
                    continue

        # 3) fallback synchronous
        resp = _invoke_with_kw_retry(llm.invoke, messages, stream_flag=False, invoke_kwargs=invoke_kwargs)
        yield _safe_extract_text(resp)
        return

    except Exception as e:
        traceback.print_exc()
        yield f"[__STREAM_ERROR__]{str(e)}"

def chat_sync(messages: List[BaseMessage], **invoke_kwargs) -> str:
    """Synchronous safe call returning the final text (or structured JSON)."""
    if llm is None:
        raise RuntimeError("LLM not initialized (ChatGroq missing).")
    resp = _invoke_with_kw_retry(llm.invoke, messages, stream_flag=False, invoke_kwargs=invoke_kwargs)
    return _safe_extract_text(resp) or ""

# Export names
__all__ = ["llm", "STRUCTURED_PREFIX", "chat_stream", "chat_sync"]