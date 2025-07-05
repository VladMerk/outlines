import tiktoken
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition

from llms import llm
from models import Section, SubSection
from states import ContentGenerationState
from tavily_tools import search_engine, code_search_engine, wikipedia_tool


async def _plan_research(topic: str, section: Section, encoding) -> str:
    """Этап планирования - что именно нужно искать"""

    planning_prompt = ChatPromptTemplate.from_template("""
Определите, что именно нужно исследовать для секции:

**Тема:** {topic}
**Секция:** {title}
**Описание:** {description}

Кратко ответьте (максимум 100 слов):
1. Тип информации: теоретическая/практическая/техническая
2. Нужны ли примеры кода: да/нет, какие именно
3. Ключевые термины для поиска: [список]
4. Источники: Wikipedia/поиск/код

Пример ответа:
"Техническая информация. Нужны примеры кода: Builder struct, методы build(). Термины: Rust builder pattern, ownership, type state. Источники: код + поиск."
""")

    result = await llm.ainvoke(planning_prompt.format(topic=topic, title=section.section_title, description=section.content))

    return result.content


async def _conduct_targeted_search(topic: str, section: Section, plan: str, encoding) -> str:
    """Этап поиска с ограничением токенов"""

    search_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            Вы - исследователь. Найдите КОНКРЕТНУЮ информацию согласно плану.
            
            Инструменты:
            - wikipedia_tool: базовые концепции
            - search_engine: актуальная информация  
            - code_search_engine: примеры кода, документация
            
            ВАЖНО: Делайте не более 2-3 поисковых запросов!
            Ищите только самое важное согласно плану.
            """,
            ),
            (
                "user",
                """
            План исследования: {plan}
            Секция: {title}
            
            Найдите ключевую информацию согласно плану.
            """,
            ),
        ]
    )

    # Ограничиваем количество итераций для контроля бюджета и токенов
    search_agent = create_react_agent(
        model=llm,
        tools=[wikipedia_tool, search_engine, code_search_engine],
        # max_iterations=2,  # Максимум 2 итерации
    )

    search_chain = search_prompt | search_agent

    result = await search_chain.ainvoke({"plan": plan, "title": section.section_title})

    # Извлекаем и объединяем результаты поиска
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    combined_results = "\n\n".join([msg.content for msg in tool_messages])

    # КОНТРОЛЬ ТОКЕНОВ: обрезаем если слишком длинно
    max_tokens = 3000  # Лимит для исследовательских данных
    if len(encoding.encode(combined_results)) > max_tokens:
        # Обрезаем по токенам, а не по символам
        tokens = encoding.encode(combined_results)
        truncated_tokens = tokens[:max_tokens]
        combined_results = encoding.decode(truncated_tokens)
        combined_results += "\n\n[ДАННЫЕ ОБРЕЗАНЫ ДЛЯ ОПТИМИЗАЦИИ]"

    return combined_results


async def _synthesize_information(section: Section, search_results: str, encoding) -> str:
    """Этап синтеза - структурирование и сжатие информации"""

    if not search_results.strip():
        return f"Исследование для секции '{section.section_title}' не дало результатов."

    synthesis_prompt = ChatPromptTemplate.from_template("""
Кратко структурируйте найденную информацию (максимум 500 слов):

**Секция:** {title}
**Найденная информация:** {search_results}

Выделите ТОЛЬКО самое важное:
• Ключевые понятия: [определения]
• Примеры: [конкретные примеры кода или случаи]
• Практические советы: [рекомендации]
• Подводные камни: [частые ошибки]

Фокус на практической ценности для написания статьи!
""")

    synthesis_result = await llm.ainvoke(synthesis_prompt.format(title=section.section_title, search_results=search_results))

    synthesized = synthesis_result.content

    # Дополнительная проверка размера
    max_final_tokens = 1500  # Финальный лимит для каждой секции
    if len(encoding.encode(synthesized)) > max_final_tokens:
        tokens = encoding.encode(synthesized)
        truncated_tokens = tokens[:max_final_tokens]
        synthesized = encoding.decode(truncated_tokens)
        synthesized += "\n\n[ИНФОРМАЦИЯ СЖАТА]"

    return synthesized


async def research_phase(state):
    """Улучшенная фаза исследования с контролем токенов"""

    topic = state["topic"]
    sections = [Section.model_validate(section) for section in state["sections"]]
    research_results = []

    # Инициализация токенайзера для контроля размера
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    for section in sections:
        print(f"Исследуем секцию: {section.section_title}")

        # ЭТАП 1: Планирование исследования
        planning_result = await _plan_research(topic, section, encoding)

        # ЭТАП 2: Целенаправленный поиск
        search_results = await _conduct_targeted_search(topic, section, planning_result, encoding)

        # ЭТАП 3: Синтез и сжатие информации
        final_research = await _synthesize_information(section, search_results, encoding)

        research_results.append(
            {
                "section_title": section.section_title,
                "description": section.content,
                "research_data": final_research,
            }
        )

    return {**state, "research_results": research_results}


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


async def planning_phase(state: ContentGenerationState):
    planning_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Вы - технический редактор, структурирующий информацию для статьи. "
                "На основе собранных исследовательских данных создайте детальный план "
                "для написания подраздела. Выделите ключевые моменты, определите "
                "логическую последовательность изложения. Учитывайте приложенное описание "
                "для подтемы. Нужно только ответить на поставленный вопрос, поэтому составьте план,"
                "который будет содержать только ответ на поставленный вопрос. "
                "Не нужно добавлять введение и заключение, здесь нужен только ответ"
                "на тему подтемы.",
            ),
            (
                "user",
                """
            Тема статьи: {topic}
            Подтема: {title}
            Описание подтемы: {description}
            Собранные данные: {research_data}

            Создайте структурированный план для написания этой подтемы.
            """,
            ),
        ]
    )

    topic = state["topic"]
    research_results = state["research_results"]
    plans = []

    for research in research_results:
        result = await llm.ainvoke(
            planning_prompt.format(
                topic=topic,
                # main_plan=sections_text,
                title=research["section_title"],
                description=research["description"],
                research_data=research["research_data"],
            ),
        )

        plans.append({"section_title": research["section_title"], "plan": result.content})

    return {
        **state,
        "plans": plans,
    }


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

        if i == 0:
            context = ""
        else:
            N = min(5, len(final_sections))
            context = "\n".join([str(item) for item in final_sections[-N:]])
            while len(encoding.encode(context)) > 2000 and N > 1:
                N -= 1
                context = "\n".join([str(item) for item in final_sections[-N:]])

        # context = "\n".join([str(item) for item in final_sections[-1:]]) if i > 0 else ""

        result = SubSection.model_validate(
            await writing_llm.ainvoke(
                {
                    "topic": topic,
                    "title": plan["section_title"],
                    "description": research_results[i]["description"],
                    "context": context,
                    "plan": plan["plan"],
                    "role": role,
                    "research_data": research_data,
                }
            )
        )

        final_sections.append(str(result))

    return {**state, "sections": final_sections}


graph_builder = StateGraph(ContentGenerationState)

tool_node = ToolNode(tools=[wikipedia_tool, search_engine, code_search_engine])

graph_builder.add_node("tools", tool_node)
graph_builder.add_node("research_phase", research_phase)
graph_builder.add_node("planning_phase", planning_phase)
graph_builder.add_node("role_selector_phase", role_selector_phase)
graph_builder.add_node("writing_phase", writing_phase)
graph_builder.add_node("vector_store_node", vector_store_node)

# graph_builder.add_edge("tools", "research_phase")
graph_builder.add_edge(START, "research_phase")
graph_builder.add_edge("research_phase", "vector_store_node")
graph_builder.add_edge("vector_store_node", "planning_phase")
graph_builder.add_edge("planning_phase", "role_selector_phase")
graph_builder.add_edge("role_selector_phase", "writing_phase")
graph_builder.add_edge("writing_phase", END)

# graph_builder.add_conditional_edges("research_phase", tools_condition)
# graph_builder.add_edge("tools", "research_phase")

graph = graph_builder.compile()


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        sections = [
            Section.model_validate(
                {
                    "section_title": "История Германии в Средневековье",
                    "content": "Развитие городов в Средние века",
                },
            ),
            Section.model_validate(
                {
                    "section_title": "Реформация в Германии",
                    "content": "Влияние Мартина Лютера на предпосылки к Реформации в Германии.",
                }
            ),
        ]
        state = {
            "topic": "Исстория Германии.",
            "wishes": "Развитие городов и городской жизни в истории Германии",
            "sections": sections,
            "messages": [],
        }

        result = await graph.ainvoke(state)

        for section in result["sections"]:
            print(section)
            print()

    asyncio.run(main())
