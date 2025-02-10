from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain as as_runnable

from llms import llm
from models import Outline
from states import ArticleState

sections_prompt = ChatPromptTemplate.from_template(
    """Вам необходимо написать раздел большой статьи посвященной теме:

    {topic}.

    Заголовок для которой необходимо написать раздел статьи:
    {title}

    Краткое содержание секции:
    {description}

    Дайте развернутый и содержательный ответ, соответствующий заголовку секции статьи.
    При необходимости используйте формат Markdown, для форматирования текста."""
)

section_chain = sections_prompt | llm | StrOutputParser()


@as_runnable
async def generate_sections(state: ArticleState):
    topic = state["topic"]
    outline: Outline = state["outline"]

    results = []
    for section in outline.sections:
        result = await section_chain.ainvoke(
            {
                "topic": topic,
                "title": section.section_title,
                "description": section.content,
            }
        )
        results.append(result)

    return {**state, "sections": results}


if __name__ == "__main__":
    import asyncio

    async def main():
        topic = "Использование словарей в Python"
        section_title = "Создание и доступ к элементам словаря"
        description = "- Создание пустого словаря или с данными <br>- Доступ к элементу по ключу "

        results = section_chain.ainvoke(
            {"topic": topic, "title": section_title, "description": description}
        )

        print(results)

    asyncio.run(main())
