from fastmcp import FastMCP
import arxiv

# ============================================================
mcp = FastMCP("arXiv", port=3000)
client = arxiv.Client()


# ============================================================


@mcp.tool()
def search_arxiv(query: str) -> str:
    """
    Search for papers on arXiv using the given query.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, max_results=5, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = list(client.results(search))
    if not results:
        return "No results found."
    else:
        return "\n".join([f"{result.title} ({result.entry_id})" for result in results])


# ============================================================


@mcp.tool()
def download_arxiv_paper(paper_id: str) -> str:
    """
    Download the paper with the given arXiv ID.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        paper = next(client.results(search))
        paper.download_pdf(dirpath="papers")
        return f"Paper downloaded: {paper.title} ({paper.entry_id})"
    except Exception as e:
        return f"Error downloading paper: {e}"


# ============================================================


if __name__ == "__main__":
    mcp.run(transport="sse")
