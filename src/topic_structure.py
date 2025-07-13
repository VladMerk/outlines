import asyncio
import os
import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import chain as as_runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from llms import think_llm
from models import SectionsList
from states import OutlineState


async def thinking_phase(state: OutlineState):
    """Этап размышления о структуре статьи"""

    topic = state["topic"]
    wishes = (
        "\n".join([str(item.content) for item in state["wishes"]])
        if isinstance(state["wishes"], list) and "wishes" in state
        else state.get("wishes", "no additional wishes")
    )

    thinking_prompt = ChatPromptTemplate.from_template("""
Проанализируйте тему статьи пошагово:

Тема: {topic}
Пожелания: {wishes}

Размышления:
1. Какой тип статьи? (техническая/теоретическая/практическая)
2. Какой уровень сложности? (начальный/средний/продвинутый)
3. Какие ключевые концепции нужно объяснить?
4. Какая логическая последовательность? (от чего к чему)
5. Какие практические примеры понадобятся?
6. Есть ли сравнительные аспекты с другими подходами?
7. Какие "подводные камни" нужно осветить?

Напишите краткий план подхода к структурированию (3-5 предложений):
""")

    thinking_result = await think_llm.ainvoke(thinking_prompt.format(topic=topic, wishes=wishes))

    return {**state, "thinking_result": thinking_result.content}


async def generate_outline_improved(state: OutlineState):
    """Создание структуры на основе размышлений"""

    topic = state["topic"]
    wishes = (
        "\n".join([str(item.content) for item in state["wishes"]])
        if isinstance(state["wishes"], list) and "wishes" in state
        else state.get("wishes", "no additional wishes")
    )
    prev_sections = "\n".join([str(section) for section in state["sections"]]) if "sections" in state else "no sections"
    thinking_result = state.get("thinking_result", "")

    structure_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            Вы — экспертный технический редактор.
            
            На основе проведенного анализа создайте структуру статьи.
            
            Требования:
            - От простого к сложному
            - Каждая секция = один конкретный аспект темы
            - Детальное описание содержания (2-3 предложения)
            - Без "введения" и "заключения"
            - Технически точная терминология
            - Практические примеры в каждой секции
            """,
            ),
            (
                "user",
                """
            **Анализ темы:**
            {thinking_result}
            
            **Тема статьи:** {topic}
            **Пожелания:** {wishes}
            **Предыдущие секции:** {prev_sections}
            
            Создайте структуру статьи, следуя анализу.
            """,
            ),
        ]
    )

    generate_outline_chain = structure_prompt | think_llm.with_structured_output(SectionsList)

    sections = await generate_outline_chain.ainvoke(
        {"thinking_result": thinking_result, "topic": topic, "wishes": wishes, "prev_sections": prev_sections}
    )

    return {**state, "sections": sections}


async def display_sections(state: OutlineState):
    """Отображение секций пользователю"""

    sections = SectionsList.model_validate(state["sections"]).sections

    os.system("clear")
    print("\n=== АНАЛИЗ ТЕМЫ ===")
    print(state.get("thinking_result", ""))

    print("\n=== СТРУКТУРА СТАТЬИ ===")
    for i, section in enumerate(sections, start=1):
        print(f"\n[{i}] {section.section_title}")
        print(f"    {section.content}")

    return state


async def process_user_feedback(state: OutlineState):
    """Обработка обратной связи пользователя"""

    user_feedback: str = interrupt(
        {
            "wishes": state["wishes"],
            "messages": "\n>>> Скорректируйте структуру или напишите 'done': ",
        }
    )

    if user_feedback.lower() == "done":
        return Command(update={"wishes": state["wishes"]}, goto=END)

    new_wishes = state["wishes"] + [user_feedback] if user_feedback not in state["wishes"] else state["wishes"]

    return Command(
        update={"wishes": new_wishes},
        goto="thinking_phase",  # Начинаем с размышлений заново
    )


async def finalize_outline(state: OutlineState):
    """Финализация структуры"""

    print("\n=== ФИНАЛЬНАЯ СТРУКТУРА ===")
    for i, section in enumerate(state["sections"].sections, start=1):
        print(f"\n[{i}] {section.section_title}")
        print(f"    {section.content}")

    return Command(goto=END)


def get_improved_graph():
    """Создание улучшенного графа"""

    graph_builder = StateGraph(OutlineState)

    # Добавляем этап размышления
    graph_builder.add_node("thinking_phase", thinking_phase)
    graph_builder.add_node("generate_outline_improved", generate_outline_improved)
    graph_builder.add_node("display_sections", display_sections)
    graph_builder.add_node("process_user_feedback", process_user_feedback)
    graph_builder.add_node("finalize_outline", finalize_outline)

    # Новый граф: thinking -> generate -> display -> feedback
    graph_builder.add_edge(START, "thinking_phase")
    graph_builder.add_edge("thinking_phase", "generate_outline_improved")
    graph_builder.add_edge("generate_outline_improved", "display_sections")
    graph_builder.add_edge("display_sections", "process_user_feedback")

    graph_builder.set_finish_point("finalize_outline")

    checkpointer = MemorySaver()
    return graph_builder.compile(checkpointer=checkpointer)


@as_runnable
async def sections_generator(state: OutlineState):
    """Улучшенный генератор секций"""

    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})
    graph = get_improved_graph()

    async for chunk in graph.astream(
        {"topic": state["topic"], "wishes": state["wishes"]},
        config,
        stream_mode="updates",
    ):
        if "__interrupt__" in chunk:
            while True:
                user_feedback = await asyncio.get_event_loop().run_in_executor(None, input, ">>> Дополните свои пожелания: ")
                await graph.ainvoke(Command(resume=user_feedback), config)

                if user_feedback.lower() == "done":
                    break

    return graph.get_state(config).values["sections"]


# Тестирование
if __name__ == "__main__":

    async def main():
        topic = "Реализация паттерна Builder в Rust"
        wishes = "Хочу понять как правильно реализовать Builder pattern в Rust, особенно интересует работа с типами и lifetime параметрами. Также хотелось бы увидеть сравнение с тем, как это делается в других языках вроде Python"

        result = await sections_generator.ainvoke(input={"topic": topic, "wishes": wishes})  # type: ignore

        print("\n=== РЕЗУЛЬТАТ ===")
        print(result)

    asyncio.run(main())
