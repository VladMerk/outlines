import asyncio
import uuid
from pprint import pprint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import chain as as_runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from llms import llm
from models import SectionsList
from states import SectionsState

sections_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы экспертный редактор для написания статей на различные темы."
            " Пользователь предоставит вопрос или тему для написания статьи и свои пожелания."
            "Вам необходимо составить список 5-10 ключевых подтем из которых может состоять статья и которые важны для этой темы."
            "Пользователь выберет из этого списка нужные ему подтемы."
            " Будьте всеобъемлющим и конкретным. Учитывайте пожелания пользователя.",
        ),
        ("user", "Тема: \n{topic}\n\nПожелания: {wishes}"),
    ]
)

add_sections_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы технический редактор статей."
            " Пользователь хочет получить статью по вопросу: {topic}."
            " Уже добавлены несколько подтем для статьи:"
            "\n\n{sections}"
            "\n\nВам необходимо предложить **ещё 5-10 новых подтем**, которые логично дополняют существующие, "
            "но **не дублируют их**. Они должны углублять понимание темы и охватывать важные аспекты, "
            "которые пока не были затронуты."
            # "\n\n⚠ **Формат ответа:**\n"
            # "- Название подтемы\n"
            # "- Краткое описание (1-2 предложения, зачем эта подтема нужна)"
            "Для форматирования текста используйте формат Markdown.",
        )
    ]
)

llm.temperature = 0.5
generate_question_chain = sections_prompt | llm.with_structured_output(SectionsList)


async def generate_questions(state: SectionsState):
    print("\n\n+++++++++++++++++DEBUG(generate_questions)+++++++++++++++++++++++")
    return {
        **state,
        "sections": SectionsList.model_validate(
            await generate_question_chain.ainvoke(
                {"topic": state["topic"], "wishes": state["wishes"]}
            )
        ).sections,
    }


add_sections_chain = add_sections_prompt | llm.with_structured_output(SectionsList)


async def select_sections(state: SectionsState):
    print("\n\n++++++++++++++++++++++++DEBUG(select_sections)+++++++++++\n")
    sections = state["sections"]
    if sections:
        for i, section in enumerate(sections):
            print(f"[{i+1}] {section.section}:\n{section.description} ")  # type: ignore
    answer = interrupt(
        "Выберите подтемы, которые вы хотели бы удалить: ",
    )

    return {**state, "human_answer": answer}


async def get_human_feedback(state: SectionsState):
    print("\n\n+++++++++++++++++DEBUG(get_human_feedback)+++++++++++++++++++++++")
    answer = state["human_answer"]
    human_sections = list(map(int, [a.strip() for a in answer.split(",")]))
    new_sections = [
        t for i, t in enumerate(state["sections"]) if i + 1 not in human_sections  # type: ignore
    ]

    return {**state, "new_sections": new_sections}


async def get_user_end(state: SectionsState):
    print("\n\n+++++++++++++++++DEBUG(get_user_end)+++++++++++++++++++++++")
    answer = interrupt("Достаточно или нет? yes/no: ")
    return answer == "yes"


async def add_sections(state: SectionsState):
    print("\n\n+++++++++++++DEBUG(add_sections)++++++++++++++\n")
    sections = SectionsList.model_validate(
        await add_sections_chain.ainvoke(
            {
                "topic": state["topic"],
                "sections": "\n\n".join(
                    [str(section) for section in state["new_sections"]]  # type: ignore
                ),
            }
        )
    ).sections
    return {**state, "sections": list(state["new_sections"]) + sections}  # type: ignore


def get_sections_graph() -> CompiledStateGraph:

    graph_builder = StateGraph(SectionsState)

    graph_builder.add_node("generate_sections", generate_questions)
    graph_builder.add_node("select_sections", select_sections)
    graph_builder.add_node("human_feedback", get_human_feedback)
    graph_builder.add_node("add_sections", add_sections)

    graph_builder.add_edge(START, "generate_sections")
    graph_builder.add_edge("generate_sections", "select_sections")
    graph_builder.add_edge("select_sections", "human_feedback")
    graph_builder.add_conditional_edges(
        "human_feedback", get_user_end, {True: END, False: "add_sections"}
    )
    graph_builder.add_edge("add_sections", "select_sections")

    checkpoint = MemorySaver()

    return graph_builder.compile(checkpointer=checkpoint)


@as_runnable
async def sections_subgraph(state: SectionsState):
    config = RunnableConfig(configurable={"thread_id": uuid.uuid4()})
    # topic = input("Topic: ")
    # wishes = input("Ваши пожелания к написанию статьи: ")
    state = {"topic": state["topic"], "wishes": state["wishes"]}

    graph = get_sections_graph()

    while True:
        async for chunk in graph.astream(state, config, stream_mode="updates"):
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                user_input = input(f"{chunk['__interrupt__'][0].value}")
                state = Command(resume=user_input)
                break
            else:
                if not isinstance(chunk, Command):
                    state = chunk
        else:
            break

    return {**state, "sections": "\n".join([str(section) for section in state["human_feedback"]["sections"]])}  # type: ignore


if __name__ == "__main__":

    async def main():
        state = await sections_subgraph()
        pprint(state, indent=2, width=200)

    asyncio.run(main())
