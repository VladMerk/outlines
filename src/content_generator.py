from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from typing_extensions import Annotated, TypedDict

from llms import llm
from models import Section, SectionsList
from tools import search_engine, wikipedia_tool


class ContentGenerationState(TypedDict):
    topic: str
    sections: SectionsList
    messages: Annotated[list, add_messages]
    research_results: list[dict[str, str]]
    plans: list[dict[str, str]]


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
            {"section_title": research["section_title"], "research_data": enhanced_data}
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
                "логическую последовательность изложения и укажите, какие примеры "
                "или иллюстрации следует включить.",
            ),
            (
                "user",
                """
            Тема статьи: {topic}
            Подтема: {title}
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
                title=research["section_title"],
                research_data=research["research_data"],
            )
        )

        plans.append(
            {"section_title": research["section_title"], "plan": result.content}
        )

    return {
        **state,
        "plans": plans,
    }


async def writing_phase(state: ContentGenerationState):
    writing_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Вы - опытный технический писатель. Ваша задача - написать детальный, "
                "информативный и хорошо структурированный раздел статьи на основе "
                "предоставленного плана и исследовательских данных. "
                "Используйте Markdown для форматирования. Включите примеры, "
                "сравнения и технические детали, где это уместно. "
                "Текст должен быть понятным, но глубоким по содержанию.",
            ),
            (
                "user",
                """
                    Тема статьи: {topic}
                    Подтема: {title}
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
    final_sections = []

    for i, plan in enumerate(plans):
        research_data = research_results[i]["research_data"]

        result = await llm.ainvoke(
            writing_prompt.format(
                topic=topic,
                title=plan["section_title"],
                plan=plan["plan"],
                research_data=research_data,
            )
        )

        final_sections.append(result.content)

    return {**state, "sections": final_sections}


graph_builder = StateGraph(ContentGenerationState)

tool_node = ToolNode(tools=[wikipedia_tool, search_engine])

graph_builder.add_node("tools", tool_node)
graph_builder.add_node("research_phase", research_phase)
graph_builder.add_node("planning_phase", planning_phase)
graph_builder.add_node("writing_phase", writing_phase)
graph_builder.add_node("vector_store_node", vector_store_node)

graph_builder.add_edge("tools", "research_phase")
graph_builder.add_edge(START, "research_phase")
graph_builder.add_edge("research_phase", "vector_store_node")
graph_builder.add_edge("vector_store_node", "planning_phase")
graph_builder.add_edge("planning_phase", "writing_phase")
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
                    "section_title": "История создания",
                    "content": "Исторические справки по созданию языка.",
                }
            ),
            Section.model_validate(
                {
                    "section_title": "Текущие изменения в языке",
                    "content": "Развитие языка на сегодняшний день. ",
                }
            ),
        ]
        state = {
            "topic": "Python",
            "sections": sections,
            "messages": [],
        }

        result = await graph.ainvoke(state)

        for section in result["sections"]:
            print(section)
            print()

    asyncio.run(main())
