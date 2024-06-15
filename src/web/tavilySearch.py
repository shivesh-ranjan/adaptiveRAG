from langchain_community.tools.tavily_search import TavilySearchResults

def GetWebSearchTool(k:int = 3):
    return TavilySearchResults(max_results=k)
