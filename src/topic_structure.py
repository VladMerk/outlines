import asyncio
import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import chain as as_runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict

from llms import think_llm
from models import SectionsList


class OutlineState(TypedDict):
    topic: str
    wishes: Annotated[list[str], add_messages]
    sections: SectionsList


async def generate_outline(state: OutlineState):

    topic = state["topic"]
    wishes = (
        "\n".join([str(item) for item in state["wishes"]])
        if isinstance(state["wishes"], list) and "wishes" in state
        else "no additional wishes"
    )
    prev_sections = (
        "\n".join([str(section) for section in state["sections"]])
        if "sections" in state
        else "no sections"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Вы - экспертный технический редактор. Ваша задача — **создать список тем и подтем** для статьи.

                Этап 1: определене типа запроса
                По заданной пользователем названию статьи и его пожеланиям (если они есть), определите его тип по
                следующим критериям:
                 - Исследовательская статья: широкая тема, требующего всестороннего освещения, введения в предметную
                 область, объяснения базовых концепций
                 - Фактический: конкретный вопрос, требующий прямого ответа, технической информации или пошаговых инструкций.

                Этап 2: создание структуры
                В зависимости от определенного типа запроса:

                Если статья **исследовательская**:
                - Пользователь скорее всего ничего о ней не знает, поэтому ему нужны те знания, которые введут его в курс дела
                - Составте список из 5-8 логически связанных подтем.
                - Начните с базовых концепций, постепенно переходя к более сложным.
                - Фокусируйтесь на основных аспектах темы.
                - Не нужно переходить к узким темам. Важнее дать общее понимание. Узкие темы пользователь раскроет другим запросом

                Если вопрос **фактический**:
                - Составте список из 3-5 конкретных подтем, напрямую отвечающих на вопрос.
                - Избегайте общих введений, фокусируйтесь на сути вопроса.
                - Если вопрос технический, включите подтему с практическими примерами.
                - Не нужно добавлять введение и заключение - пользователя интересует только ответ на вопрос.

                Этап 3: рекомендации для дальнейшего изучения
                Всегда добавляйте финальную подтему "Рекомендации для дальнейшего изучения", включающую:
                - 5-10 связыных тем или вопросов для углубления и раскрытия знаний.
                - конкретные формулировки запросов, которые пользователь может использовать.
                - Ссылки на авторитетные рессурсы (документация, книги, ресурсы интернета)

                Общие требования:
                🔹 Удалите подтемы, которые пользователь считает ненужными.
                🔹 Добавьте те подтемы, предложеныные в пожеланиях пользователя.
                🔹 НЕ дублируйте подтемы, если они уже есть.
                🔹 Структурируйте их так, чтобы они плавно раскрывали тему.
                🔹 Напишите название подтемы и очень подробное описание того, что будет в этой подтеме.
                Это должны быть четкие указания для редактора-исполнителя, который будет писать эту часть статьи.
                🔹 Для каждой подтемы предоставьте детальное описание содержания и рекомендации по написанию подтемы
                (не менее 2-3 предложений) - по этому описанию и рекомендациям будет написана собственно статья другим редактором.
                🔹 Используйте технически точную терминологию.


                **Тема статьи:** {topic}
                **Прошлые подтемы:** {sections}
                **Пожелания пользователя:** {wishes}
                """,
            ),
            ("user", "Обновите список подтем согласно указанным пожеланиям."),
        ]
    )

    generate_outline_chain = prompt | think_llm.with_structured_output(SectionsList)

    sections = await generate_outline_chain.ainvoke(
        {"topic": topic, "sections": prev_sections, "wishes": wishes}
    )

    return {"sections": sections, "wishes": state["wishes"]}


async def display_sections(state: OutlineState):

    sections = SectionsList.model_validate(state["sections"]).sections

    print("\nТекущий список подтем:")
    for i, section in enumerate(sections, start=1):
        print(f"[{i}] {section.section_title.capitalize()}:\n\t{section.content}")

    return state


async def process_user_feedback(state: OutlineState):

    user_feedback: str = interrupt(
        {
            "wishes": state["wishes"],
            "messages": "Скорректируйте полученные подтемы или напишите 'done': ",
        }
    )

    if user_feedback.lower() == "done":
        return Command(update={"wishes": state["wishes"]}, goto=END)

    new_wishes = (
        state["wishes"] + [user_feedback]
        if user_feedback not in state["wishes"]
        else state["wishes"]
    )

    return Command(
        update={"wishes": new_wishes},
        goto="generate_outline",
    )


async def finalize_outline(state: OutlineState):
    print("\nFinal node and finished values:")
    for i, section in enumerate(state["sections"].sections, start=1):
        print(f"[{i}] {section.section_title}\n\t{section.content}")

    return Command(goto=END)


def get_graph():
    graph_builder = StateGraph(OutlineState)

    graph_builder.add_node("generate_outline", generate_outline)
    graph_builder.add_node("display_sections", display_sections)
    graph_builder.add_node("process_user_feedback", process_user_feedback)
    graph_builder.add_node("finalize_outline", finalize_outline)

    graph_builder.add_edge(START, "generate_outline")
    graph_builder.add_edge("generate_outline", "display_sections")
    graph_builder.add_edge("display_sections", "process_user_feedback")

    graph_builder.set_finish_point("finalize_outline")

    checkpointer = MemorySaver()

    return graph_builder.compile(checkpointer=checkpointer)


@as_runnable
async def content_generator(state: OutlineState):
    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})

    graph = get_graph()

    async for chunk in graph.astream(
        {"topic": state["topic"], "wishes": state["wishes"]}, config
    ):
        for node_id, _ in chunk.items():
            if node_id == "__interrupt__":
                while True:
                    user_feedback = await asyncio.get_event_loop().run_in_executor(
                        None, input, ">>> Дополните свои пожелания: "
                    )
                    await graph.ainvoke(Command(resume=user_feedback), config)

                    if user_feedback.lower() == "done":
                        break

    return graph.get_state(config).values["sections"]


if __name__ == "__main__":
    import os

    async def main():
        config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})
        topic = await asyncio.get_event_loop().run_in_executor(
            None, input, "> Тема статьи: "
        )
        wishes = await asyncio.get_event_loop().run_in_executor(
            None, input, "> Пожелания: "
        )

        result = await content_generator.ainvoke(
            input={"topic": topic, "wishes": wishes}, config=config  # type: ignore
        )

        os.system("clear")
        print(result)

    asyncio.run(main())
