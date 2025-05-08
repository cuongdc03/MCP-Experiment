import streamlit as st
import asyncio
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import aiohttp
from typing import Dict, Any

# ============================================================
st.set_page_config(page_title="🔍 Arxiv Agent", page_icon="📄")

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the latest research papers and courses with links.
You has 2 MCP servers to get the information from:
- Arxiv MCP server: for research papers
- Course MCP server: for courses

You can use the following tools to get the information while using the following prompt you have to extract keywords from the user's question:
- Arxiv MCP server: for research papers
- Course MCP server: for courses

"""

# ============================================================


async def check_server_status(url: str, timeout: int = 5) -> bool:
    """Check if a server is running and accessible."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                return response.status == 200
    except:
        return False


async def wait_for_servers(server_configs: Dict[str, Any], max_retries: int = 3, retry_delay: int = 2) -> bool:
    """Wait for servers to become available."""
    for _ in range(max_retries):
        all_servers_ready = True
        for server_name, config in server_configs.items():
            if not await check_server_status(config["url"]):
                all_servers_ready = False
                break
        if all_servers_ready:
            return True
        await asyncio.sleep(retry_delay)
    return False


@st.cache_resource
def setup_agent():
    model = ChatOllama(model="llama3.2:latest")
    
    # Define server configurations
    server_configs = {
        "arxiv_mcp": {
            "url": "http://localhost:3000/sse",
            "transport": "sse",
        },
        "course_mcp": {
            "url": "http://localhost:3001/sse",
            "transport": "sse",
        },
    }
    
    return model, server_configs


# ============================================================


async def run_agent(query: str):
    model, server_configs = setup_agent()
    
    # Check server availability
    if not await wait_for_servers(server_configs):
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Unable to connect to required services after multiple attempts. Please ensure the MCP servers are running:\n"
                             "- http://localhost:3000 (arxiv_mcp)\n"
                             "- http://localhost:3001 (course_mcp)\n\n"
                             "You can start them using:\n"
                             "python -m mcp.server.arxiv_mcp --port 3000\n"
                             "python -m mcp.server.course_mcp --port 3001"
                }
            ]
        }
    
    try:
        # Create client with retry mechanism
        for attempt in range(3):
            try:
                client = MultiServerMCPClient(server_configs)
                async with client:
                    agent = create_react_agent(model=model, tools=client.get_tools(),prompt=SYSTEM_PROMPT)
                    formatted_query = [{"role": "user", "content": query}]
                    response = await agent.ainvoke({"messages": formatted_query})
                    return response
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(2)  # Wait before retry
                
    except Exception as e:
        error_msg = str(e)
        print("Error in run_agent:", error_msg)
        
        if "Connection refused" in error_msg or "Cannot connect to host" in error_msg:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Connection to services failed. Please check if the servers are running and try again."
                    }
                ]
            }
        elif "TaskGroup" in error_msg:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Service connection timed out. Please try again in a few moments."
                    }
                ]
            }
        else:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"An error occurred: {error_msg}\nPlease try again or check the server status."
                    }
                ]
            }


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
            "content": "Hi! I'm Arxiv Agent. Ask me everything you want.",
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
            try:
                response = run_async(run_agent(prompt))
                
                last_message = response["messages"][-1]
                content = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else last_message
                )
                # Search all messages for tool_calls
                tool_names = []
                for msg in response["messages"]:
                    tc = None
                    if hasattr(msg, "tool_calls"):
                        tc = msg.tool_calls
                    elif isinstance(msg, dict) and "tool_calls" in msg:
                        tc = msg["tool_calls"]
                    if tc:
                        if isinstance(tc, list):
                            tool_names.extend([t.get("name") or t.get("tool_name") or str(t) for t in tc])
                        else:
                            tool_names.append(str(tc))
                tool_info = f"\n\n**Tool(s) used:** {', '.join(tool_names)}" if tool_names else ""
                st.session_state.messages.append(
                    {"role": "assistant", "content": content + tool_info}
                )
                st.write(content + tool_info)
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                print("Error in main loop:", error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                st.write(error_msg)


# ============================================================
