

import os
import hashlib
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
ZEP_API_KEY: Optional[str] = os.getenv("ZEP_API_KEY")
USER_ID: str = os.getenv("USER_ID", "default_user")
SESSION_ID: str = os.getenv("SESSION_ID", "")  # if provided, used as thread id
SHORT_HISTORY_MAX: int = int(os.getenv("SHORT_HISTORY_MAX", "6"))

# SDK imports (deferred)
try:
    from zep_cloud.client import Zep as ZepClient
    from zep_cloud.types import Message as ZepMessage
except Exception:
    ZepClient = None  # type: ignore
    ZepMessage = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Globals
_openai_client = None
_zep_client = None
__bg_tasks = set()

# ---------- utilities ----------
def stable_thread_id(user_id: str = USER_ID, session_id: str = SESSION_ID) -> str:
    if session_id:
        return session_id
    suffix = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:8]
    return f"{user_id}-{suffix}"

def trim_short_history(history: List[Dict[str, str]], max_len: int = SHORT_HISTORY_MAX) -> List[Dict[str, str]]:
    return history if len(history) <= max_len else history[-max_len:]

def build_system_prompt() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the user-provided Zep memory context and short-term "
            "chat history to answer precisely. If the memory contains facts, prefer those as ground truth."
        ),
    }

async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

# ---------- init clients ----------
def init_openai_client():
    global _openai_client
    if OpenAI is None:
        raise RuntimeError("Missing 'openai' package. Install: pip install openai")
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
    return _openai_client

def init_zep_client():
    global _zep_client
    if ZepClient is None:
        print("[warn] zep-cloud SDK not installed. Install: pip install zep-cloud")
        _zep_client = None
        return None
    if not ZEP_API_KEY:
        print("[warn] ZEP_API_KEY not set. Zep features disabled.")
        _zep_client = None
        return None
    _zep_client = ZepClient(api_key=ZEP_API_KEY)
    return _zep_client

# ---------- zep-sync wrappers (callable in executor) ----------
def _zep_user_add_sync(client: Any, user_id: str, first_name: Optional[str]=None, last_name: Optional[str]=None, email: Optional[str]=None):
    if client is None: return None
    return client.user.add(user_id=user_id, first_name=first_name, last_name=last_name, email=email)

def _zep_thread_create_sync(client: Any, thread_id: str, user_id: str):
    if client is None: return None
    return client.thread.create(thread_id=thread_id, user_id=user_id)

def _zep_thread_add_messages_sync(client: Any, thread_id: str, messages: List[Any]):
    if client is None: return None
    return client.thread.add_messages(thread_id=thread_id, messages=messages)

def _zep_get_user_context_sync(client: Any, thread_id: str, template_id: Optional[str]=None):
    if client is None: return ""
    if template_id:
        mem = client.thread.get_user_context(thread_id=thread_id, template_id=template_id)
    else:
        mem = client.thread.get_user_context(thread_id=thread_id)
    return getattr(mem, "context", "") or ""

def _zep_graph_search_sync(client: Any, query: Optional[str]=None, limit: int = 100):
    """
    Run client.graph.search() if available. Some Zep versions expose different signatures.
    Return raw JSON-like Python object.
    """
    if client is None:
        return {}
    # best-effort call - may need adjustment for specific sdk version
    try:
        if query is None:
            return client.graph.search(limit=limit)
        return client.graph.search(query=query, limit=limit)
    except TypeError:
        # fallback
        return client.graph.search(limit=limit)

# ---------- OpenAI call (sync) ----------
def _call_openai_chat_sync(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.2) -> str:
    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized")
    _model = model or OPENAI_MODEL
    resp = _openai_client.chat.completions.create(model=_model, messages=messages, temperature=temperature, max_tokens=800)
    try:
        choice = resp.choices[0]
    except Exception:
        try:
            choice = resp.get("choices", [None])[0]
        except Exception:
            choice = None
    if not choice:
        return str(resp)
    try:
        msg = getattr(choice, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content
            try:
                return content.get("text")
            except Exception:
                pass
    except Exception:
        pass
    try:
        return choice.get("message", {}).get("content", "")
    except Exception:
        return str(resp)

# ---------- background scheduling ----------
def schedule_zep_add_messages(zep_client, thread_id: str, messages: List[Any]):
    async def _bg():
        try:
            await run_in_thread(_zep_thread_add_messages_sync, zep_client, thread_id, messages)
        except Exception as e:
            print(f"[warn] background write to zep failed: {e}")

    try:
        t = asyncio.create_task(_bg())
        __bg_tasks.add(t)
        def _on_done(tt): __bg_tasks.discard(tt)
        t.add_done_callback(_on_done)
    except RuntimeError:
        try:
            _zep_thread_add_messages_sync(zep_client, thread_id, messages)
        except Exception as e:
            print(f"[warn] background (sync) write to zep failed: {e}")

# ---------- public helpers ----------
async def ensure_user_and_thread(zep_client, thread_id: str, user_id: str):
    """
    Ensure user exists and thread exists (create if not).
    Runs SDK calls in executor.
    """
    if zep_client is None:
        return
    try:
        await run_in_thread(_zep_user_add_sync, zep_client, user_id, None, None, None)
    except Exception:
        # ignore if already exists
        pass
    try:
        await run_in_thread(_zep_thread_create_sync, zep_client, thread_id, user_id)
    except Exception:
        # ignore if already exists
        pass

async def get_zep_context(zep_client, thread_id: str, template_id: Optional[str] = None) -> str:
    """
    Return context block string (may be empty on error).
    """
    if zep_client is None:
        return ""
    try:
        return await run_in_thread(_zep_get_user_context_sync, zep_client, thread_id, template_id)
    except Exception as e:
        print(f"[warn] failed fetching zep context: {e}")
        return ""

async def graph_search(zep_client, query: Optional[str] = None, limit:int=200) -> Any:
    if zep_client is None:
        return {}
    try:
        return await run_in_thread(_zep_graph_search_sync, zep_client, query, limit)
    except Exception as e:
        print(f"[warn] zep graph.search failed: {e}")
        return {}

async def call_openai_chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    return await run_in_thread(_call_openai_chat_sync, messages, model or OPENAI_MODEL)

async def add_user_message(zep_client, thread_id: str, user_id: str, text: str):
    """
    Synchronously add the user message to the thread (so context includes it).
    """
    if zep_client is None or ZepMessage is None:
        return
    try:
        msg = ZepMessage(name=user_id, role="user", content=text)
        await run_in_thread(_zep_thread_add_messages_sync, zep_client, thread_id, [msg])
    except Exception as e:
        print(f"[warn] failed to add user message to zep: {e}")

async def add_assistant_message_background(zep_client, thread_id: str, assistant_text: str):
    if zep_client is None or ZepMessage is None:
        return
    try:
        msg = ZepMessage(name="AI Assistant", role="assistant", content=assistant_text)
        schedule_zep_add_messages(zep_client, thread_id, [msg])
    except Exception as e:
        print(f"[warn] scheduling assistant message failed: {e}")

async def handle_user_turn(zep_client, thread_id: str, user_id: str, user_text: str, short_history: List[Dict[str,str]]) -> str:
    """
    Full turn: add user -> fetch context -> build messages -> call LLM -> persist assistant (bg) -> return assistant_text
    """
    # add user message to zep so get_user_context sees it
    await add_user_message(zep_client, thread_id, user_id, user_text)

    # fetch zep context block
    context_block = await get_zep_context(zep_client, thread_id)

    # build messages
    messages: List[Dict[str,str]] = []
    messages.append(build_system_prompt())
    if context_block:
        # put context into system message (compat with all OpenAI chat endpoints)
        messages.append({"role":"system", "content": f"[ZEP MEMORY]\n{context_block}"})
    # short history (list of {"role","content"})
    if short_history:
        messages.extend(short_history)
    # current user
    messages.append({"role":"user", "content": user_text})

    # call openai
    try:
        assistant_text = await call_openai_chat(messages, OPENAI_MODEL)
    except Exception as e:
        print(f"[error] OpenAI call failed: {e}")
        assistant_text = "Sorry — I couldn't get a response from the model."

    # schedule assistant persistence in background
    await add_assistant_message_background(zep_client, thread_id, assistant_text)

    return assistant_text

# ---------- simple chunker ----------
def chunk_text_simple(text: str, max_words: int = 200, overlap: int = 30) -> List[str]:
    """
    Very simple chunker: split by words into overlapping chunks.
    Returns list of chunk strings.
    """
    words = text.strip().split()
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + max_words, n)
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == n:
            break
        i = max(end - overlap, end)  # move forward with overlap
    return chunks

# helper to chunk a pair of messages and create ZepMessage objects (useful for app)
def create_chunk_messages_for_thread(zep_client, user_id: str, thread_id: str, user_text: str, assistant_text: str, max_words:int=200) -> List[Any]:
    """
    Returns a list of ZepMessage instances (user then assistant) for the provided text chunks.
    If ZepMessage is not available, returns a list of simple dict payloads.
    """
    chunks = []
    combined = f"User: {user_text}\n\nAssistant: {assistant_text}"
    text_chunks = chunk_text_simple(combined, max_words=max_words, overlap=40)
    msgs = []
    for t in text_chunks:
        if ZepMessage is None:
            msgs.append({"name": user_id, "role":"system", "content": t})
        else:
            # store as 'system' messages to avoid role enforcement issues in some SDKs
            msgs.append(ZepMessage(name=user_id, role="system", content=t))
    return msgs