from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


@tool
async def search_engine(query: str):
    """Search engine to the internet"""
    search = DuckDuckGoSearchResults(num_results=4)
    results = await search.arun(query)

    return results


if __name__ == "__main__":
    import asyncio
    from pprint import pprint

    async def main():
        results = await search_engine.ainvoke("словарь в python")
        pprint(results)

    asyncio.run(main())
