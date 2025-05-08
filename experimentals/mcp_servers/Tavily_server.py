from tools import tavily_tool
from fastmcp import FastMCP

# ============================================================
mcp = FastMCP("Course_MCP", port=3001)
tavily = tavily_tool.TavilyTools()

# ============================================================


@mcp.tool()
def search_course(query: str) -> str:
    """
    Search for papers on arXiv using the given query.
    """
    result = tavily.search_course(query)
    if not result:
        return "No results found."
    else:
        return result


# ============================================================


if __name__ == "__main__":
    mcp.run(transport="sse")
