import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

model = ChatOllama(model="llama3.2:latest")


async def main():
    async with MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:3000/sse",  # 👈 Đảm bảo server đang chạy
                "transport": "sse",
            }
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())

        while True:
            user_input = input("Enter your message: ")
            if user_input.lower() == "exit":
                break
            response = await agent.ainvoke({"messages": user_input})
            print("Agent:", response)


if __name__ == "__main__":
    asyncio.run(main())
