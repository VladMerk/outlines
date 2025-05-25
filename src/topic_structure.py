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


async def generate_outline(state: OutlineState):

    topic = state["topic"]
    wishes = (
        "\n".join([str(item.content) for item in state["wishes"]])  # type: ignore
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
                Вы — экспертный автор и редактор.
                Ваша задача последовательно обдумать вопрос, предоставленный пользователем и ответить
                в виде списка тем, которые помогут пользователю изучать материал.

                Проанализируйте тему статьи и сформулируйте, какие крупные разделы необходимо включить в статью,
                чтобы она была полной, логичной и полезной. Думайте шаг за шагом:

                - Какого вида статья? (техническая, историческая или др.)
                - Какие ключевые аспекты охватывает эта тема?
                - Какие термины и понятия необходимо объяснить?
                - Что важно описать сначала, чтобы создать базу для остального?
                - Какие практические/технические моменты нужно раскрыть?
                - Какие частые ошибки или недосказанности встречаются по этой теме?
                - Нужно ли дополнить объяснение примерами, сравнениями, диаграммами?
                - Какую логическую структуру должны иметь будущие разделы?
                - Современное состояние вопроса: что сейчас считается стандартом, что недавно изменилось,
                какие есть новые подходы, рекомендации и прочее.
                - Есть ли разногласия старой трактовки и современного состояния в теме?

                Общие требования:
                - Удалите подтемы, которые пользователь считает ненужными.
                - Добавьте те подтемы, предложеныные в пожеланиях пользователя.
                - НЕ дублируйте подтемы, если они уже есть.
                - Структурируйте их так, чтобы они плавно раскрывали тему - от просто к сложному и от начального к продвинутому.
                - Напишите название подтемы и очень подробное описание того, что будет в этой подтеме.
                Это должны быть четкие указания для редактора-исполнителя, который будет писать эту часть статьи.
                - Для каждой подтемы предоставьте детальное описание содержания и рекомендации по написанию подтемы
                (не менее 2-3 предложений) - по этому описанию и рекомендациям будет написана собственно статья другим редактором.
                - Используйте технически точную терминологию.
                - Техническая статья должна иметь более "узкий" формат - не нужно "введния" и "заключения",
                  нужно более точно и полно раскрыть тему.
                """,
            ),
            (
                "user",
                """
                    **Тема статьи:** {topic}
                    **Прошлые подтемы:**
                    {sections}
                    **Пожелания пользователя:**
                    {wishes}
                """,
            ),
        ]
    )

    generate_outline_chain = prompt | think_llm.with_structured_output(SectionsList)

    sections = await generate_outline_chain.ainvoke(
        {"topic": topic, "sections": prev_sections, "wishes": wishes}
    )

    return {"sections": sections, "wishes": state["wishes"]}


async def display_sections(state: OutlineState):

    sections = SectionsList.model_validate(state["sections"]).sections

    os.system("clear")
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
async def sections_generator(state: OutlineState):
    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})

    graph = get_graph()

    async for chunk in graph.astream(
        {"topic": state["topic"], "wishes": state["wishes"]},
        config,
        stream_mode="updates",
    ):
        if "__interrupt__" in chunk:
            while True:
                user_feedback = await asyncio.get_event_loop().run_in_executor(
                    None, input, ">>> Дополните свои пожелания: "
                )
                await graph.ainvoke(Command(resume=user_feedback), config)

                if user_feedback.lower() == "done":
                    break

    return graph.get_state(config).values["sections"]


if __name__ == "__main__":

    async def main():
        config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})
        # topic = await asyncio.get_event_loop().run_in_executor(
        #     None, input, "> Тема статьи: "
        # )
        # wishes = await asyncio.get_event_loop().run_in_executor(
        #     None, input, "> Пожелания: "
        # )
        # topic = "Как работает FastAPI: архитектура, практические примеры и продвинутые приёмы"
        # wishes = "Интересует асинхронность, работа с БД и подготовка к деплою."
        topic = "Гармонозаместительная терапия (ГЗТ) тестостероном"
        wishes = (
            "Хотел бы разобраться чем полезна подобная терапия на пациентов старше 40 лет, "
            "занимающихся любительским спортом и ведущих здоровый образ жизни. Какие показания для начала проведения ГЗТ"
        )

        result = await sections_generator.ainvoke(
            input={"topic": topic, "wishes": wishes}, config=config  # type: ignore
        )

        os.system("clear")
        print(result)

    asyncio.run(main())
