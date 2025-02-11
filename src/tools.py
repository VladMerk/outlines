from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


@tool
async def search_engine(query: str):
    """Search engine to the internet"""
    search = DuckDuckGoSearchResults(num_results=4)
    results = await search.arun(query)

    return results


wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=3
        )  # type: ignore
    ).run,
    description="Поиск по Wikipedia",
    # verbose=True,
)


if __name__ == "__main__":
    import asyncio
    from pprint import pprint

    async def main():
        results = await search_engine.ainvoke("словарь в python")
        pprint(results)

    asyncio.run(main())
