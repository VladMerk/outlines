from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition

from llms import llm
from models import Section
from states import ContentGenerationState
from tools import search_engine, wikipedia_tool


async def research_phase(state: ContentGenerationState):
    research_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Вы - исследователь, собирающий информацию по заданной теме. "
                "Используйте доступные инструменты для поиска информации. "
                "Соберите максимально полные данные, включая определения, ключевые концепции, "
                "примеры, сравнения и актуальные сведения. "
                "Структурируйте найденную информацию в виде списка фактов и ключевых моментов.",
            ),
            (
                "user",
                """
            Тема статьи: {topic}
            Подтема для исследования: {title}
            Описание подтемы: {description}

            Соберите всю необходимую информацию по этой подтеме.
            """,
            ),
        ]
    )

    research_agent = create_react_agent(
        model=llm, tools=[wikipedia_tool, search_engine]
    )
    research_chain = research_prompt | research_agent

    topic = state["topic"]
    sections = [Section.model_validate(section) for section in state["sections"]]
    research_results = []
    results: list[AIMessage] = []

    for section in sections:
        result = await research_chain.ainvoke(
            {
                "topic": topic,
                "title": section.section_title,
                "description": section.content,
            }
        )

        # Извлекаем только ответы модели
        tool_messages = [
            msg for msg in result["messages"] if isinstance(msg, ToolMessage)
        ]
        research_results.append(
            {
                "section_title": section.section_title,
                "description": section.content,
                "research_data": tool_messages[-1].content if tool_messages else "",
            }
        )
        results.extend(
            message for message in result["messages"] if isinstance(message, AIMessage)
        )

    return {**state, "research_results": research_results, "messages": results}


async def vector_store_node(state: ContentGenerationState):
    # Инициализация векторного хранилища с локальной моделью эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings)

    # Индексация собранных данных
    for research in state["research_results"]:
        if research["research_data"]:  # Проверка на пустые данные
            vectorstore.add_texts(
                texts=[research["research_data"]],
                metadatas=[
                    {"section": research["section_title"], "topic": state["topic"]}
                ],
            )

    # Поиск релевантной информации для каждой секции
    enhanced_results = []
    for research in state["research_results"]:
        query = f"{state['topic']} {research['section_title']}"
        similar_docs = vectorstore.similarity_search(query, k=3)

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

        plans.append(
            {"section_title": research["section_title"], "plan": result.content}
        )

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
                - Учитывайте предыдущий контекст, если он добавлен, чтобы избежать повторов
                    и сделать плавные переходы между темами.
                - Используйте подготовленные исследовательские данные.
                - Используйте Markdown для форматирования текста
                - Испльзуйте Mermaid для Markdown для построения схем
                - Не нужно добавлять "Введение" и "Заключение" к подсекции - нужны только ответы на описываемые темы
                для секций статьи.
                - Обязательно нужно добавить секцию с рекомендациями для чтения/просмотру с различными полезными рессурсами,
                которые могут помочь расширить знания по указанной теме секции:
                    - книги
                    - сслылки на рессурсы в интернете
                    - документация
                    - качественные запросы в поисковые системы по теме
                    - и т.д.

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

    for i, plan in enumerate(plans):
        research_data = research_results[i]["research_data"]

        result = await llm.ainvoke(
            writing_prompt.format(
                topic=topic,
                title=plan["section_title"],
                description=research_results[i]["description"],
                context=final_sections[i - 1] if i > 0 else "",
                plan=plan["plan"],
                role=role,
                research_data=research_data,
            ),
        )

        final_sections.append(result.content)  # type: ignore

    return {**state, "sections": final_sections}


graph_builder = StateGraph(ContentGenerationState)

tool_node = ToolNode(tools=[wikipedia_tool, search_engine])

graph_builder.add_node("tools", tool_node)
graph_builder.add_node("research_phase", research_phase)
graph_builder.add_node("planning_phase", planning_phase)
graph_builder.add_node("role_selector_phase", role_selector_phase)
graph_builder.add_node("writing_phase", writing_phase)
graph_builder.add_node("vector_store_node", vector_store_node)

graph_builder.add_edge("tools", "research_phase")
graph_builder.add_edge(START, "research_phase")
graph_builder.add_edge("research_phase", "vector_store_node")
graph_builder.add_edge("vector_store_node", "planning_phase")
graph_builder.add_edge("planning_phase", "role_selector_phase")
graph_builder.add_edge("role_selector_phase", "writing_phase")
graph_builder.add_edge("writing_phase", END)

graph_builder.add_conditional_edges("research_phase", tools_condition)
graph_builder.add_edge("tools", "research_phase")

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
