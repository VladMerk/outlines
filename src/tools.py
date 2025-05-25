import warnings

import httpx
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from markdownify import markdownify

from llms import llm

warnings.catch_warnings()
warnings.simplefilter("ignore")

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1, check_every_n_seconds=0.1, max_bucket_size=1
)


async def scrape_pages(title: str, url: str) -> str:

    # Create an async HTTP client
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:

        # Fetch each URL and convert to markdown
        try:
            # Fetch the content
            response = await client.get(url)
            response.raise_for_status()

            # Convert HTML to markdown if successful
            if response.status_code == 200:
                # Handle different content types
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    # Convert HTML to markdown
                    markdown_content = markdownify(response.text)
                    result = markdown_content
                else:
                    # For non-HTML content, just mention the content type
                    result = f"Content type: {content_type} (not converted to markdown)"
            else:
                result = f"Error: Received status code {response.status_code}"

        except Exception as e:
            # Handle any exceptions during fetch
            return f"Error fetching URL: {str(e)}"

        # Create formatted output
        formatted_output = "Search results: \n\n"
        formatted_output += f"\n\n--- SOURCE: {title} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"FULL CONTENT:\n {result}"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


@tool
async def search_engine(query: str):
    """Search engine to the internet"""
    search = DuckDuckGoSearchResults(num_results=2, output_format="list")
    await rate_limiter.aacquire()

    results: list[dict] = await search.arun(query)

    prompt = ChatPromptTemplate.from_template(
        """
        Вы редактор текстов.
        Вам необходимо выделить наиболее важные, ключевые моменты из предоставленного текста:
        ### Заголовок: {title}
        ### Content
        {text}

        И вернуть получившийся текст
        """
    )

    chain = prompt | llm | StrOutputParser()

    output_text = ""

    for result in results:
        title = result["title"]
        parsed_text = await scrape_pages(title=title, url=result["link"])
        output_text += await chain.ainvoke({"title": title, "text": parsed_text})
        output_text += "\n\n"

    return output_text.strip()


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

    async def main():
        results = await search_engine.ainvoke("Жизнь города в Средневековой Германии")
        print(results)

    asyncio.run(main())
