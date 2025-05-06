import asyncio
import dotenv
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.llms.ollama import Ollama

dotenv.load_dotenv()

# Load LLM
llm = Ollama(model="llama3.2:latest", request_timeout=120)

SYSTEM_PROMPT = """\
You are an AI assistant for Tool Calling.

Before you help a user, you need to work with tools to interact with Our Database
"""


async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with Our Database software.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


async def handle_user_message(
    message_content: str,
    agent: FunctionAgent,
    agent_context: Context,
    verbose: bool = False,
):
    handler = agent.run(message_content, ctx=agent_context)
    async for event in handler.stream_events():
        if verbose and isinstance(event, ToolCall):
            print(
                f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
        elif verbose and isinstance(event, ToolCallResult):
            print(f"Tool {event.tool_name} returned {event.tool_output}")

    response = await handler
    return str(response)


async def main():
    mcp_client = BasicMCPClient("http://127.0.0.1:3000/sse")
    mcp_tool = McpToolSpec(client=mcp_client)

    # get the agent
    agent = await get_agent(mcp_tool)

    # create the agent context
    agent_context = Context(agent)

    while True:
        user_input = input("Enter your message: ")
        if user_input.lower() == "exit":
            break
        print("User: ", user_input)
        response = await handle_user_message(
            user_input, agent, agent_context, verbose=True
        )
        print("Agent: ", response)


if __name__ == "__main__":
    asyncio.run(main())
