import tiktoken
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent

from llms import llm
from models import Section
from tools import tools


async def research_phase(state):
    """Оптимизированная фаза исследования - меньше запросов, больше эффективности"""

    topic = state["topic"]
    sections = [Section.model_validate(section) for section in state["sections"]]
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    print(f"🔍 Начинаем оптимизированное исследование для {len(sections)} секций...")

    # ЭТАП 1: Глобальный анализ - что вообще нужно искать
    global_analysis = await _analyze_research_needs(topic, sections)

    # ЭТАП 2: Пакетный поиск - делаем 5-8 запросов вместо 30+
    batch_results = await _conduct_batch_search(topic, global_analysis, encoding)

    # ЭТАП 3: Распределение результатов по секциям
    section_results = await _distribute_results(sections, batch_results, encoding)

    print(f"✅ Исследование завершено. Использовано запросов: {batch_results['search_count']}")

    return {**state, "research_results": section_results}


async def _analyze_research_needs(topic: str, sections: list[Section]) -> dict:
    """Анализ всех секций сразу для определения общих потребностей"""

    analysis_prompt = ChatPromptTemplate.from_template("""
Проанализируйте ВСЕ секции статьи и определите общие потребности в исследовании:

**Тема:** {topic}
**Секции:** {sections_info}

Определите:
1. **Общие ключевые термины** для поиска (3-5 терминов)
2. **Типы информации** (теоретическая/практическая/историческая)
3. **Нужен ли код** (да/нет + какой именно)
4. **Специфические запросы** (2-3 уникальных запроса)

Цель: минимизировать количество поисков, покрыв максимум потребностей.

Формат ответа:
Ключевые термины: [список]
Типы информации: [список]
Нужен код: [да/нет + детали]
Специфические запросы: [список]
""")

    # Формируем сводку всех секций
    sections_info = "\n".join([f"- {section.section_title}: {section.content}" for section in sections])

    result = await llm.ainvoke(analysis_prompt.format(topic=topic, sections_info=sections_info))

    return {"analysis": result.content, "sections_count": len(sections)}


async def _conduct_batch_search(topic: str, global_analysis: dict, encoding) -> dict:
    """Пакетный поиск - делаем мало запросов, получаем много информации"""

    # Создаем умного агента для пакетного поиска
    batch_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            Вы - эксперт-исследователь. Ваша задача - эффективно собрать информацию
            для ВСЕЙ статьи, делая минимум поисков.

            Стратегия:
            1. Делайте ШИРОКИЕ поисковые запросы, покрывающие несколько тем
            2. Используйте разные инструменты стратегически
            3. Максимум 6-8 поисков на всю статью
            4. Фокус на качество, а не количество

            Инструменты:
            - wikipedia_tool: для общих концепций
            - search_engine: для широкого поиска
            - code_search_engine: только если нужен код

            ВАЖНО: Делайте широкие запросы, покрывающие несколько аспектов темы сразу!
            """,
            ),
            (
                "user",
                """
            Тема статьи: {topic}
            Анализ потребностей: {analysis}

            Соберите всю необходимую информацию за минимум поисков.
            Делайте широкие запросы, покрывающие несколько аспектов темы.
            """,
            ),
        ]
    )

    # Создаем агента БЕЗ max_iterations (его нет в create_react_agent)
    batch_agent = create_react_agent(model=llm, tools=tools)

    batch_chain = batch_prompt | batch_agent

    # Ограничиваем через prompt и контроль результата
    result = await batch_chain.ainvoke({"topic": topic, "analysis": global_analysis["analysis"]})

    # Извлекаем все результаты поиска
    tool_messages: list[ToolMessage] = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]

    combined_results = "\n\n".join(str(msg.content) for msg in tool_messages)

    # Контроль токенов - УМЕНЬШАЕМ лимиты
    max_tokens = 8000
    if len(encoding.encode(combined_results)) > max_tokens:
        tokens = encoding.encode(combined_results)
        truncated_tokens = tokens[:max_tokens]
        combined_results = encoding.decode(truncated_tokens)
        combined_results += "\n\n[ПАКЕТНЫЕ ДАННЫЕ ОБРЕЗАНЫ ДЛЯ ОПТИМИЗАЦИИ]"

    return {
        "combined_research": combined_results,
        "search_count": len(tool_messages),  # Количество использованных результатов
    }


async def _distribute_results(sections: list[Section], batch_results: dict, encoding) -> list:
    """Распределение пакетных результатов по секциям"""

    distribution_prompt = ChatPromptTemplate.from_template("""
Распределите найденную информацию по конкретной секции:

**Секция:** {section_title}
**Описание:** {section_description}
**Общие исследовательские данные:** {research_data}

Выберите и структурируйте ТОЛЬКО ту информацию, которая относится к этой секции:

• Ключевые понятия: [для этой секции]
• Примеры: [конкретные примеры для этой секции]
• Практические советы: [для этой секции]
• Подводные камни: [для этой секции]

Если информации недостаточно, используйте базовые знания и напишите основные моменты по секции.
""")

    section_results = []
    research_data = batch_results["combined_research"]

    for section in sections:
        result = await llm.ainvoke(
            distribution_prompt.format(
                section_title=section.section_title, section_description=section.content, research_data=research_data
            )
        )

        # Контроль размера для каждой секции
        distributed_content = result.content
        max_section_tokens = 1000  # Меньше лимит для распределенных данных

        if len(encoding.encode(distributed_content)) > max_section_tokens:
            tokens = encoding.encode(distributed_content)
            truncated_tokens = tokens[:max_section_tokens]
            distributed_content = encoding.decode(truncated_tokens)
            distributed_content += "\n\n[ДАННЫЕ СЕКЦИИ СЖАТЫ]"

        section_results.append(
            {
                "section_title": section.section_title,
                "description": section.content,
                "research_data": distributed_content,
            }
        )

    return section_results
