from langchain.prompts import ChatPromptTemplate

from llms import llm
from nodes.planning.strategy import determine_article_type


async def practical_planning_phase(state):
    """Планирование без академической воды, с фокусом на практику"""

    # Сначала определяем стратегию
    state_with_strategy = await determine_article_type(state)
    article_strategy = state_with_strategy["article_strategy"]

    planning_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            Вы - практик-методист, создающий планы для действенных статей.

            ПРИНЦИПЫ:
            - Никаких академических введений/заключений
            - Прямо к сути: определение → пример → практика
            - Минимум теории, максимум практики
            - Конкретные примеры с кодом/действиями
            - Красная нить через всю статью

            ЗАПРЕЩЕНО:
            - "В этом разделе мы изучим..."
            - "Заключение: мы рассмотрели..."
            - "Введение в тему..."
            - Абстрактные рассуждения без примеров
            """,
            ),
            (
                "user",
                """
            **Стратегия статьи:** {article_strategy}

            **Секция:** {title}
            **Описание:** {description}
            **Исследовательские данные:** {research_data}

            Создайте практический план БЕЗ воды:

            **Структура секции:**
            1. **Прямое определение** (1 абзац, без "введений")
            2. **Базовый пример** (конкретный код/случай)
            3. **Развитие темы** (усложнение, вариации)
            4. **Практические моменты** (что важно знать)
            5. **Подводные камни** (частые ошибки)

            **Требования к содержанию:**
            - Конкретные примеры кода (если применимо)
            - Реальные сценарии использования
            - Практические советы
            - НЕТ абстрактных рассуждений

            **Если это часть проекта:**
            - Как эта секция связана с общим проектом
            - Какую часть функциональности реализуем

            Будьте конкретны и практичны!
            """,
            ),
        ]
    )

    # topic = state["topic"]
    research_results = state["research_results"]
    plans = []

    for research in research_results:
        result = await llm.ainvoke(
            planning_prompt.format(
                article_strategy=article_strategy,
                title=research["section_title"],
                description=research["description"],
                research_data=research.get("research_data", "Нет данных"),
            )
        )

        plans.append({"section_title": research["section_title"], "plan": result.content})

    return {**state_with_strategy, "plans": plans}
