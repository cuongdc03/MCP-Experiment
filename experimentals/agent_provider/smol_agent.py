from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

model = InferenceClientModel()
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

agent.run(
    "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
)
