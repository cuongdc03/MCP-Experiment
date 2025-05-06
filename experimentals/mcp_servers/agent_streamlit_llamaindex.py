import streamlit as st
import asyncio
import nest_asyncio
import dotenv

from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.llms.ollama import Ollama

# ================== Setup =====================
st.set_page_config(page_title="🔍 Arxiv Agent", page_icon="📄")
dotenv.load_dotenv()
nest_asyncio.apply()

llm = Ollama(model="llama3.2:latest", request_timeout=60.0)

SYSTEM_PROMPT = """\
You are an AI assistant for Tool Calling.

Before you help a user, you need to work with tools to interact with MCP tools
"""


@st.cache_resource
def setup():
    mcp_client = BasicMCPClient("http://127.0.0.1:3000/sse")
    mcp_tool = McpToolSpec(client=mcp_client)
    tools = asyncio.run(mcp_tool.to_tool_list_async())

    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with Our Database software.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    ctx = Context(agent)
    return agent, ctx


async def handle_stream(message_content: str, agent: FunctionAgent, ctx: Context):
    handler = agent.run(message_content, ctx=ctx)
    tool_name = None
    tool_output = None
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            tool_name = event.tool_name
            tool_output = event.tool_output
    return tool_name, tool_output


# ================== Chat UI ===================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Arxiv Agent. Ask me anything."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        msg_placeholder = st.empty()

        async def run_and_stream():
            agent, ctx = setup()
            tool_name, tool_output = await handle_stream(prompt, agent, ctx)
            full_response = (
                f"🔧 Using tool: `{tool_name}`\n\n✅ Final response: {tool_output}"
            )
            msg_placeholder.markdown(full_response)
            return full_response

        # Chạy async function an toàn với nest_asyncio + Streamlit
        full_response = asyncio.run(run_and_stream())
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
