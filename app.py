# app.py
"""
Streamlit app for Zep Knowledge-Graph Chat
Run:
    streamlit run app.py
Place this file next to agent.py and your .env
"""

import streamlit as st
import asyncio
from dotenv import load_dotenv
from typing import List, Dict
import agent  # local module we created

load_dotenv()

st.set_page_config(page_title="Zep Memory Chat", layout="wide")

# ---------- UI state init ----------
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "short_history" not in st.session_state:
    st.session_state.short_history = []  # list of {"role","content"}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = agent.stable_thread_id()
if "graph_json" not in st.session_state:
    st.session_state.graph_json = None
if "zep_client" not in st.session_state:
    st.session_state.zep_client = None
if "openai_client" not in st.session_state:
    st.session_state.openai_client = None

# ---------- async wrapper ----------
def run_async(coro):
    """
    Run a coroutine safely in Streamlit (even if no current event loop exists)
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

# ---------- initialization ----------
st.title("Zep Knowledge-Graph Chat (Streamlit)")
with st.expander("Configuration", expanded=True):
    st.write("Using env values for OpenAI and Zep. Make sure `.env` has OPENAI_API_KEY and ZEP_API_KEY set.")
    st.write(f"USER_ID: **{agent.USER_ID}**")
    st.write(f"Thread id: **{st.session_state.thread_id}**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize / Re-init clients"):
            # OpenAI client
            try:
                st.session_state.openai_client = agent.init_openai_client()
            except Exception as e:
                st.error(f"OpenAI init failed: {e}")
                st.session_state.openai_client = None
            # Zep client
            try:
                st.session_state.zep_client = agent.init_zep_client()
            except Exception as e:
                st.error(f"Zep init failed: {e}")
                st.session_state.zep_client = None
            # ensure user/thread
            if st.session_state.zep_client:
                try:
                    run_async(agent.ensure_user_and_thread(st.session_state.zep_client, st.session_state.thread_id, agent.USER_ID))
                    st.success("Zep user & thread ensured.")
                except Exception as e:
                    st.warning(f"Could not ensure thread/user: {e}")
            st.session_state.initialized = True
    with col2:
        st.write("Zep client available." if st.session_state.zep_client else "Zep client NOT available.")

# ---------- Graph refresh ----------
st.markdown("---")
st.subheader("Graph / Knowledge view")
col_graph, col_actions = st.columns([3,1])
with col_actions:
    if st.button("Refresh Graph?"):
        if st.session_state.zep_client is None:
            st.warning("Zep client not initialized.")
        else:
            try:
                data = run_async(agent.graph_search(st.session_state.zep_client, query=""))
                st.session_state.graph_json = data
                st.success("Graph refreshed.")
            except Exception as e:
                st.error(f"Graph refresh failed: {e}")

with col_graph:
    if st.session_state.graph_json:
        st.write("Graph JSON (trimmed):")
        st.json(st.session_state.graph_json)
    else:
        st.info("No graph loaded. Click 'Refresh Graph?'")

# ---------- Chat UI ----------
st.markdown("---")
st.subheader("Chat")
chat_col, right_col = st.columns([3,1])
with chat_col:
    # show conversation
    if st.session_state.short_history:
        for m in st.session_state.short_history:
            if m["role"] == "user":
                st.chat_message("user").write(m["content"])
            else:
                st.chat_message("assistant").write(m["content"])

    message = st.text_input("You:", key="input_text", placeholder="Type a message and press Enter")
    if st.button("Send") or (message and st.session_state.get("last_sent") != message):
        user_text = message.strip()
        if not user_text:
            st.warning("Please enter a message.")
        else:
            # append locally
            st.session_state.short_history.append({"role":"user","content":user_text})
            st.session_state.last_sent = user_text

            # ensure zep client
            if st.session_state.zep_client is None:
                st.warning("Zep client not initialized; attempting init...")
                try:
                    st.session_state.zep_client = agent.init_zep_client()
                    run_async(agent.ensure_user_and_thread(st.session_state.zep_client, st.session_state.thread_id, agent.USER_ID))
                except Exception as e:
                    st.error(f"Failed to init Zep client: {e}")
                    st.session_state.zep_client = None

            # call turn handler
            assistant_text = run_async(agent.handle_user_turn(
                st.session_state.zep_client,
                st.session_state.thread_id,
                agent.USER_ID,
                user_text,
                st.session_state.short_history
            ))

            # show assistant
            st.session_state.short_history.append({"role":"assistant","content":assistant_text})
            st.session_state.short_history = agent.trim_short_history(st.session_state.short_history, agent.SHORT_HISTORY_MAX)

            # no experimental_rerun; Streamlit will re-render automatically

with right_col:
    st.markdown("**Actions**")
    if st.button("Export convo to markdown"):
        md = []
        for m in st.session_state.short_history:
            who = "You" if m["role"]=="user" else "Assistant"
            md.append(f"**{who}**: {m['content']}\n")
        md_text = "\n\n".join(md)
        st.download_button("Download MD", md_text, file_name="conversation.md", mime="text/markdown")

# ---------- memory persistence ----------
st.markdown("---")
if st.session_state.short_history:
    if st.button("Force chunk-last-turn -> save to Zep"):
        try:
            hist = st.session_state.short_history
            last_user = next((item["content"] for item in reversed(hist) if item["role"]=="user"), None)
            last_assistant = next((item["content"] for item in reversed(hist) if item["role"]=="assistant"), "")

            if not last_user:
                st.warning("No user message found in history.")
            else:
                msgs = agent.create_chunk_messages_for_thread(
                    st.session_state.zep_client,
                    agent.USER_ID,
                    st.session_state.thread_id,
                    last_user,
                    last_assistant or "",
                    max_words=250
                )
                # schedule background save
                agent.schedule_zep_add_messages(st.session_state.zep_client, st.session_state.thread_id, msgs)
                st.success("Chunk messages scheduled for Zep.")
        except Exception as e:
            st.error(f"Chunk & save failed: {e}")
