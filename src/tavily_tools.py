import os
import warnings
# from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.tools import Tool, tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient

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


# Оставляем Wikipedia как есть
wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2)).run,
    description="Поиск по Wikipedia для общих знаний",
)

# Список всех инструментов
tools = [search_engine, code_search_engine, wikipedia_tool]

# Тестирование
if __name__ == "__main__":
    import asyncio

    async def main():
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

