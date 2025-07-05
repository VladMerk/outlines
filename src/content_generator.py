import tiktoken
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

from llms import llm
from models import Section, SubSection
from states import ContentGenerationState
from tools import tools

# ========== ОПТИМИЗИРОВАННЫЙ RESEARCH PHASE ==========


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


async def vector_store_node(state: ContentGenerationState):
    # Инициализация векторного хранилища с локальной моделью эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings)

    # Индексация собранных данных
    for research in state["research_results"]:
        if research["research_data"]:  # Проверка на пустые данные
            vectorstore.add_texts(
                texts=[research["research_data"]],
                metadatas=[{"section": research["section_title"], "topic": state["topic"]}],
            )

    # Поиск релевантной информации для каждой секции
    enhanced_results = []
    for research in state["research_results"]:
        query = f"{state['topic']} {research['section_title']}"
        similar_docs = vectorstore.similarity_search(query, k=min(3, vectorstore._collection.count()))

        # Объединение найденной информации с исходными данными
        enhanced_data = research["research_data"]
        if similar_docs:
            additional_content = "\n\n".join([doc.page_content for doc in similar_docs])
            enhanced_data += f"\n\n### Связанная информация:\n\n{additional_content}"

        enhanced_results.append(
            {
                "section_title": research["section_title"],
                "description": research["description"],
                "research_data": enhanced_data,
            }
        )

    return {**state, "research_results": enhanced_results}


async def determine_article_type(state):
    """Определение типа статьи и общей стратегии"""

    type_prompt = ChatPromptTemplate.from_template("""
Проанализируйте запрос и определите тип статьи:

**Тема:** {topic}
**Пожелания:** {wishes}

**Типы статей:**
1. **ОБЪЯСНИТЕЛЬНАЯ** - разбор концепций, паттернов, теорий
   (например: "Что такое Builder pattern", "Как работает async/await")

2. **ПРОЕКТНАЯ** - пошаговое создание конкретного проекта
   (например: "Создаем HTTP клиент", "Пишем телеграм бота")

3. **ОБУЧАЮЩАЯ** - правила, методики, пошаговое изучение
   (например: "Правила немецкой грамматики", "Алгоритмы сортировки")

4. **АНАЛИТИЧЕСКАЯ** - сравнения, обзоры, анализ вариантов
   (например: "FastAPI vs Django", "Rust vs C++ для системного программирования")

**Определите тип и обоснуйте выбор одним предложением.**

Если ПРОЕКТНАЯ - предложите конкретный мини-проект для демонстрации концепций.

Формат ответа:
Тип: [ТИП]
Обоснование: [почему этот тип]
Проект (если нужен): [описание мини-проекта]
""")

    result = await llm.ainvoke(type_prompt.format(topic=state["topic"], wishes=state.get("wishes", "")))

    return {**state, "article_strategy": result.content}


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
            - НИкаких академических введений/заключений
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


async def role_selector_phase(state: ContentGenerationState):
    topic = state["topic"]
    wishes = state["wishes"]

    prompt = ChatPromptTemplate.from_template(
        """
        На основе темы и пожеланий пользователя (если они есть) определите, кто должен быть автором текста.

        Формат ответа:
        Должность

        Пример:
            Тема: Алгоритм быстрой сортировки
            Пожелания: подробные объснения как работает алогоритм и примеры кода на Python.

            Ответ: Python-программист, преподаватель университета.
            ---

            Тема: Развитие сюрреализма в цифровом искусстве
            Пожелания: рассказать об истории и влиянии

            Ответ: Историк современного искусства.
            ---

            Тема: История США конца XVIII века
            Пожелания: развитие сельского хозяйства в США в этот период

            Ответ: Историк, преподаватель истории США

        Теперь, пожалуйста, сформулируйте должность для следующей темы:
        - Тема: {topic}
        - Пожелания: {wishes}

        нужно вернуть только должность, без дополнений и посторонних слов.
        """
    )

    result = await llm.ainvoke(prompt.format(topic=topic, wishes=wishes))

    return {**state, "writer_role": result.content}


async def writing_phase(state: ContentGenerationState):
    writing_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Вы - {role}.
                Ваша задача - написать раздел статьи, посвященную теме и подразделу предоставленную пользователем.
                Редактором был составлен план для написания статьи, поэтому четко следуйте инструкциям:
                - Четко следуйте этому плану, учитывайте описание для подтем. Они согласованы с пользователем.
                - Объясните понятно, последовательно, с примерами и пояснениями.
                - Учитывайте предыдущий контекст, если он добавлен, в нем предыдущие написанные подраздели,
                  чтобы избежать повторов и сделать плавные переходы между темами.
                - Используйте подготовленные исследовательские данные.
                - Используйте Markdown для форматирования текста
                - Испльзуйте Mermaid для Markdown для построения схем
                - Не нужно добавлять "Введение" и "Заключение" к подсекции - нужны только ответы на описываемые темы
                для секций статьи.

                ВАЖНО ПРО СТРУКТУРУ ПРОЕКТА:
                - Если это проектная статья, ОБЯЗАТЕЛЬНО указывайте названия файлов и структуру проекта
                - Показывайте, в какой файл помещать код: src/main.rs, app/models.py, etc.
                - Создавайте четкую структуру папок и файлов
                - Объясняйте, как организовать код в проекте

                - Обязательно нужно добавить секцию с рекомендациями для чтения/просмотра с различными полезными рессурсами,
                которые могут помочь расширить знания только по указанному разделу статьи. Учитывайте рекомендации из предыдущего
                контекста, чтобы избежать повторов. Если к разделу нет хороших рекомендаций какого то типа,
                тогда оставить рекомендации пустыми.
                    - книги: автор, название, дополнительная информация о книги для упрощения поиска:
                    isbn номер, ссылка в интернет-магазине и т.д. Но если вы не уверены в существовании такой книги,
                    тогда не пишите ничего. Оставьте поле пустым.
                    - сслылки на рессурсы в интернете
                    - документация
                    - качественные запросы в поисковые системы по теме
                    - и другое, что посчитаете нужным.

                Цель: сделать сложную тему понятной и практичной.
                """,
            ),
            (
                "user",
                """
                    Тема статьи: {topic}
                    Подтема: {title}
                    Описание: {description}
                    Предыдущий контекст: {context}
                    План раздела: {plan}
                    Исследовательские данные: {research_data}

                    Напишите полный текст для этого раздела статьи.
                """,
            ),
        ]
    )

    topic = state["topic"]
    plans = state["plans"]
    research_results = state["research_results"]
    role = state["writer_role"]
    final_sections: list[str] = []
    llm.temperature = 0.3

    writing_llm = writing_prompt | llm.with_structured_output(SubSection)

    for i, plan in enumerate(plans):
        research_data = research_results[i]["research_data"]
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

        # УЛУЧШЕННЫЙ КОНТРОЛЬ ТОКЕНОВ
        # 1. Обрезаем research_data если он слишком большой
        # max_research_tokens = 2000
        # if len(encoding.encode(research_data)) > max_research_tokens:
        #     tokens = encoding.encode(research_data)
        #     truncated_tokens = tokens[:max_research_tokens]
        #     research_data = encoding.decode(truncated_tokens)
        #     research_data += "\n\n[ИССЛЕДОВАТЕЛЬСКИЕ ДАННЫЕ ОБРЕЗАНЫ]"

        # 2. Контроль контекста (твой проверенный костыль)
        if i == 0:
            context = ""
        else:
            N = min(3, len(final_sections))
            context = "\n".join([str(item) for item in final_sections[-N:]])
            while len(encoding.encode(context)) > 3000 and N > 1:
                N -= 1
                context = "\n".join([str(item) for item in final_sections[-N:]])

        # 3. Проверяем общий размер промпта
        plan_content = plan["plan"]
        if len(encoding.encode(plan_content)) > 1000:
            tokens = encoding.encode(plan_content)
            truncated_tokens = tokens[:1000]
            plan_content = encoding.decode(truncated_tokens)
            plan_content += "\n\n[ПЛАН ОБРЕЗАН]"

        result = SubSection.model_validate(
            await writing_llm.ainvoke(
                {
                    "topic": topic,
                    "title": plan["section_title"],
                    "description": research_results[i]["description"],
                    "context": context,
                    "plan": plan_content,  # Используем обрезанный план
                    "role": role,
                    "research_data": research_data,  # Используем обрезанные данные
                }
            )
        )

        final_sections.append(str(result))

    return {**state, "sections": final_sections}


graph_builder = StateGraph(ContentGenerationState)

tool_node = ToolNode(tools=tools)

graph_builder.add_node("tools", tool_node)
graph_builder.add_node("research_phase", research_phase)
graph_builder.add_node("practical_planning_phase", practical_planning_phase)  # Используем практическое планирование
graph_builder.add_node("role_selector_phase", role_selector_phase)
graph_builder.add_node("writing_phase", writing_phase)
graph_builder.add_node("vector_store_node", vector_store_node)

graph_builder.add_edge(START, "research_phase")
graph_builder.add_edge("research_phase", "vector_store_node")
graph_builder.add_edge("vector_store_node", "practical_planning_phase")
graph_builder.add_edge("practical_planning_phase", "role_selector_phase")
graph_builder.add_edge("role_selector_phase", "writing_phase")
graph_builder.add_edge("writing_phase", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    import asyncio

    async def test_optimized():
        from models import Section

        sections = [
            Section(section_title="Основы ownership", content="Понятие владения в Rust"),
            Section(section_title="Borrowing", content="Заимствования и ссылки"),
            Section(section_title="Lifetimes", content="Время жизни переменных"),
        ]

        state = {
            "topic": "Система владения в Rust",
            "wishes": "Практические примеры с кодом",
            "sections": sections,
            "messages": [],
        }

        print("=== ТЕСТ ОПТИМИЗИРОВАННОГО ИССЛЕДОВАНИЯ ===")
        result = await research_phase(state)

        print(f"Секций обработано: {len(result['research_results'])}")
        for research in result["research_results"]:
            print(f"\nСекция: {research['section_title']}")
            print(f"Данные: {research['research_data'][:200]}...")

    asyncio.run(test_optimized())
