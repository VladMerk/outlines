import asyncio
import uuid
from pprint import pprint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import chain as as_runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict

from llms import think_llm
from models import SectionsList


class SectionsState(TypedDict):
    topic: str
    wishes: Annotated[list[str], add_messages]
    sections: SectionsList


async def first_node(state: SectionsState):
    print("\n=============First Node===============")

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

                🔹 Составте список из 5-10 подтем, таких чтобы они раскрывали поставленный вопрос и ничего лишнего.
                🔹 Удалите те подтемы, которые пользователь считает ненужными.
                🔹 Добавьте те подтемы, которые он предложил в своих пожеланиях.
                🔹 НЕ дублируйте подтемы, если они уже есть.
                🔹 Структурируйте их так, чтобы они плавно раскрывали тему.

                **Тема статьи:** {topic}
                **Прошлые подтемы:** {sections}
                **Пожелания пользователя:** {wishes}
                """,
            ),
            ("user", "Обновите список подтем согласно указанным пожеланиям."),
        ]
    )

    first_node_chain = prompt | think_llm.with_structured_output(SectionsList)

    sections = await first_node_chain.ainvoke(
        {"topic": topic, "sections": prev_sections, "wishes": wishes}
    )

    return {"sections": sections, "wishes": state["wishes"]}


async def display_sections(state: SectionsState):
    print("\n========Display Sections===========")

    sections = SectionsList.model_validate(state["sections"]).sections

    print("\nТекущий список подтем:")
    for i, section in enumerate(sections, start=1):
        print(f"[{i}] {section.section.capitalize()}:\n\t{section.description}")

    return state


async def human_node(state: SectionsState):
    print("\n===========Human Node=================")

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
        goto="first_node",
    )


async def end_node(state: SectionsState):
    print("\nFinal node and finished values:")
    for i, section in enumerate(state["sections"].sections, start=1):
        print(f"[{i}] {section.section}\n\t{section.description}")

    return Command(goto=END)


def get_graph():
    graph_builder = StateGraph(SectionsState)

    graph_builder.add_node("first_node", first_node)
    graph_builder.add_node("display_sections", display_sections)
    graph_builder.add_node("human_node", human_node)
    graph_builder.add_node("end_node", end_node)

    graph_builder.add_edge(START, "first_node")
    graph_builder.add_edge("first_node", "display_sections")
    graph_builder.add_edge("display_sections", "human_node")

    graph_builder.set_finish_point("end_node")

    checkpointer = MemorySaver()

    return graph_builder.compile(checkpointer=checkpointer)


@as_runnable
async def sections_subgraph(state: SectionsState):
    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})

    graph = get_graph()

    async for chunk in graph.astream(
        {"topic": state["topic"], "wishes": state["wishes"]}, config
    ):
        for node_id, _ in chunk.items():
            if node_id == "__interrupt__":
                while True:
                    user_feedback = await asyncio.get_event_loop().run_in_executor(
                        None, input, "> Дополните свои пожелания: "
                    )
                    await graph.ainvoke(Command(resume=user_feedback), config)

                    if user_feedback.lower() == "done":
                        break

    return graph.get_state(config).values


if __name__ == "__main__":

    async def main():
        config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})
        topic = await asyncio.get_event_loop().run_in_executor(
            None, input, "> Тема статьи: "
        )
        wishes = await asyncio.get_event_loop().run_in_executor(
            None, input, "> Пожелания: "
        )

        result = await sections_subgraph.ainvoke(
            input={"topic": topic, "wishes": wishes}, config=config  # type: ignore
        )

        pprint(result, indent=2, width=200)

    asyncio.run(main())
