import streamlit as st
import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# ============================================================
st.set_page_config(page_title="🔍 Arxiv Agent", page_icon="📄")


# ============================================================


@st.cache_resource
def setup_agent():
    model = ChatOllama(model="llama3.2:latest")
    client = MultiServerMCPClient(
        {
            "arxiv_mcp": {
                "url": "http://localhost:3000/sse",
                "transport": "sse",
            }
        }
    )
    return model, client


# ============================================================


async def run_agent(query: str):
    model, client = setup_agent()
    async with client:
        agent = create_react_agent(model=model, tools=client.get_tools())
        return await agent.ainvoke({"messages": query})


# ============================================================


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm Arxiv Agent. Ask me everything your want.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_async(run_agent(prompt))

            last_message = response["messages"][-1]

            content = (
                last_message.content
                if hasattr(last_message, "content")
                else "No content found."
            )

            st.session_state.messages.append({"role": "assistant", "content": content})
            st.write(content)


# ============================================================
