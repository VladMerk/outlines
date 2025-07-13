import os
import warnings

import httpx
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from markdownify import markdownify
from tavily import TavilyClient

from llms import llm

warnings.catch_warnings()
warnings.simplefilter("ignore")

load_dotenv()

# Инициализация Tavily клиента
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
async def search_engine(query: str) -> str:
    """Enhanced search engine using Tavily API"""
    try:
        # Tavily поиск с настройками для технического контента
        response = tavily_client.search(
            query=query,
            max_results=3,
            search_depth="advanced",  # Более глубокий поиск
            include_answer=True,  # Включить краткий ответ
            include_raw_content=True,  # Включить полный контент
        )

        output_text = ""

        # Если есть краткий ответ
        if response.get("answer"):
            output_text += f"**Краткий ответ:** {response['answer']}\n\n"

        # Обработка результатов поиска
        results = response.get("results", [])

        for i, result in enumerate(results, 1):
            title = result.get("title", "Без названия")
            url = result.get("url", "")
            content = result.get("content", "")

            output_text += f"**Источник {i}: {title}**\n"
            output_text += f"URL: {url}\n"
            output_text += f"Содержание: {content}\n"
            output_text += "-" * 80 + "\n\n"

        return output_text.strip()

    except Exception as e:
        return f"Ошибка поиска: {str(e)}"


# Альтернативный поиск специально для кода
@tool
async def code_search_engine(query: str) -> str:
    """Specialized search for code examples and technical documentation"""
    try:
        # Добавляем ключевые слова для поиска кода
        code_query = f"{query} code example documentation tutorial"

        response = tavily_client.search(
            query=code_query,
            max_results=2,
            search_depth="advanced",
            include_domains=["github.com", "docs.rs", "doc.rust-lang.org", "stackoverflow.com"],
            include_raw_content=True,
        )

        output_text = "**Примеры кода и документация:**\n\n"

        results = response.get("results", [])

        for i, result in enumerate(results, 1):
            title = result.get("title", "Без названия")
            url = result.get("url", "")
            content = result.get("content", "")

            output_text += f"**Источник {i}: {title}**\n"
            output_text += f"URL: {url}\n"
            output_text += f"Содержание: {content}\n"
            output_text += "-" * 80 + "\n\n"

        return output_text.strip()

    except Exception as e:
        return f"Ошибка поиска кода: {str(e)}"


rate_limiter = InMemoryRateLimiter(requests_per_second=0.1, check_every_n_seconds=0.1, max_bucket_size=1)


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
async def ddg_search(query: str):
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

        И вернуть обощенный результат в одном-двух предложениях.
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

# Список всех инструментов
tools = [search_engine, code_search_engine, wikipedia_tool]


if __name__ == "__main__":
    import asyncio

    async def main():
        # print("\n=== Тест DuckDuckGo Search ===")
        # results_ddg = await ddg_search.ainvoke("Жизнь города в Средневековой Германии")
        # print(results_ddg)

        print("=== Тест обычного поиска ===")
        result1 = await search_engine.ainvoke("Builder pattern in Rust")
        print(result1[:500] + "..." if len(result1) > 500 else result1)

        print("\n=== Тест поиска кода ===")
        result2 = await code_search_engine.ainvoke("Rust Builder pattern implementation")
        print(result2[:500] + "..." if len(result2) > 500 else result2)

        print("\n=== Тест Wikipedia ===")
        result3 = wikipedia_tool.run("Builder pattern")
        print(result3[:300] + "..." if len(result3) > 300 else result3)

    asyncio.run(main())
