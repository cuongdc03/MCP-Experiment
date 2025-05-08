from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


class TavilyTools:
    def __init__(self):
        self.client = TavilyClient("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")

    def search_course(self, query: str) -> str:
        modified_query = "Udemey course " + query
        response = self.client.search(query=modified_query)
        results = response.get("results", [])
        if not results:
            return "No results found."

        return "\n".join(
            [
                f"{result.get('title', 'No Title')} ({result.get('url', 'No URL')})"
                for result in results
            ]
        )
