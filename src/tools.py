import warnings

from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.rate_limiters import InMemoryRateLimiter

warnings.catch_warnings()
warnings.simplefilter("ignore")

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1, check_every_n_seconds=0.1, max_bucket_size=1
)


@tool
async def search_engine(query: str):
    """Search engine to the internet"""
    search = DuckDuckGoSearchResults(
        num_results=2,
    )
    await rate_limiter.aacquire()

    return await search.arun(query)


wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2)  # type: ignore
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
