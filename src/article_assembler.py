from langchain.prompts import ChatPromptTemplate

from llms import think_llm
from states import ArticleState


async def assemble_article(state: ArticleState):
    topic = state["topic"]
    sections: list[str] = state["sections"]

    # Создаем промпт для генерации целостной статьи
    compile_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Вы - опытный технический редактор. Ваша задача - создать целостную, "
            "хорошо структурированную статью на основе предоставленных секций."
            "Ваша задача - склеить подсекции в единый связный текст."
            "Устраните повторяющиеся определения и фразы, обеспечьте плавные переходы между темами и сохраните "
            "логическую структуру. Используйте Markdown для форматирования текста, там где нужно."
            "Схемы можно чертить посредством языка Mermaid."
        ),
        (
            "user",
            """
            Тема статьи: {topic}

            Готовые секции для статьи:
            {sections}

            Верните только готовый текст.
            """
        ),
    ])

    # Объединяем все секции в один текст для контекста
    sections_text = "\n".join(sections)

    think_llm.temperature = 0

    # Генерируем целостную статью
    result = await think_llm.ainvoke(compile_prompt.format(
        topic=topic,
        sections=sections_text
    ))

    # Форматируем финальную статью с заголовком
    article = f"# {topic}\n" + "\n".join(str(section) for section in sections)

    return {**state, "article": article}
