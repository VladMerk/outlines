from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain as as_runnable
from langgraph.prebuilt import create_react_agent

from llms import llm
from models import Outline
from states import ArticleState
from tools import search_engine

sections_prompt = ChatPromptTemplate.from_template(
    """Вам необходимо написать раздел большой статьи посвященной теме:

    {topic}.

    Заголовок раздела статьи:
    {title}

    Краткое описание секции:
    {description}

    Дайте развернутый и содержательный ответ, соответствующий заголовку секции статьи.
    Используйте поиск в интернете для получения актуальной информации для формирования ответа.
    При необходимости используйте формат Markdown, для форматирования текста."""
)

agent = create_react_agent(model=llm, tools=[search_engine], debug=True)

section_chain = sections_prompt | agent


@as_runnable
async def generate_sections(state: ArticleState):
    topic = state["topic"]
    outline: Outline = state["outline"]

    results = []
    for section in outline.sections:
        try:
            result = await section_chain.ainvoke(
                {
                    "topic": topic,
                    "title": section.section_title,
                    "description": section.content,
                },
            )
            results.append(result["messages"][-1].content)
        except Exception:
            print("==========DEBUG===========: Error in the Search Enging!")

    return {**state, "sections": results}


if __name__ == "__main__":
    import asyncio
    from pprint import pprint

    async def main():
        topic = "Использование словарей в Python"
        section_title = "Создание и доступ к элементам словаря"
        description = (
            "- Создание пустого словаря или с данными <br>- Доступ к элементу по ключу "
        )

        results = await section_chain.ainvoke(
            {"topic": topic, "title": section_title, "description": description}
        )

        pprint(results["messages"][-1].content)

    asyncio.run(main())
