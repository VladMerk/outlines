import asyncio
import uuid
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from llms import llm
from tools import wikipedia_tool


class Section(BaseModel):
    section: str = Field(description="Подтема основной темы статьи.")
    description: str = Field(description="Краткое содержание подтемы.")

    def __str__(self):
        return f"{self.section}\n{self.description}"


class SectionsList(BaseModel):
    sections: list[Section] = Field(
        description="Список подтем для описания основной темы."
    )


class SectionsState(TypedDict):
    topic: str
    wishes: Optional[str]
    sections: Optional[SectionsList]
    new_sections: Optional[SectionsList]
    human_answer: str


sections_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы экспертный редактор для написания статей на различные темы."
            " Пользователь предоставит вопрос или тему для написания статьи. "
            "Вам необходимо составить список 5-10 ключевых подтем из которых может состоять статья и которые важны для этой темы."
            "Пользователь выберет из этого списка нужные ему подтемы."
            " Будьте всеобъемлющим и конкретным.",
        ),
        ("user", "Тема: {topic}"),
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
            await generate_question_chain.ainvoke({"topic": state["topic"]})
        ).sections,
    }


add_sections_chain = add_sections_prompt | llm.with_structured_output(SectionsList)


async def select_sections(state: SectionsState):
    print("\n\n++++++++++++++++++++++++DEBUG(select_sections)+++++++++++\n")
    sections = state["sections"]
    if sections:
        for i, section in enumerate(sections):
            print(f"[{i+1}] {section.section}:\n{section.description}\n")
    answer = interrupt(
        "Выберите подтемы, которые вы хотели бы удалить: ",
    )

    return {**state, "human_answer": answer}


async def get_human_feedback(state: SectionsState):
    print("\n\n+++++++++++++++++DEBUG(get_human_feedback)+++++++++++++++++++++++")
    answer = state["human_answer"]
    human_sections = list(map(int, [a.strip() for a in answer.split(",")]))
    new_sections = [
        t for i, t in enumerate(state["sections"]) if i + 1 not in human_sections
    ]

    return {**state, "new_sections": new_sections}


async def get_user_end(state: SectionsState):
    print("\n\n+++++++++++++++++DEBUG(get_user_end)+++++++++++++++++++++++")
    answer = interrupt("Достаточно или нет? yes/no: ")
    return answer == "yes"


async def add_sections(state: SectionsState):
    print("\n\n+++++++++++++DEBUG(add_sections)++++++++++++++\n")
    # print("\n+++++++++++++++DEBUG:", add_sections_prompt.format(**{'topic': state["topic"], 'sections': "\n\n".join([str(section) for section in state["new_sections"]])}))
    sections = SectionsList.model_validate(
        await add_sections_chain.ainvoke(
            {
                "topic": state["topic"],
                "sections": "\n\n".join(
                    [str(section) for section in state["new_sections"]]
                ),
            }
        )
    ).sections
    return {**state, "sections": list(state["new_sections"]) + sections}


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


if __name__ == "__main__":
    from pprint import pprint

    async def main():
        config = RunnableConfig(configurable={"thread_id": "1"})
        topic = input("Topic: ")
        state = {"topic": topic}

        graph = get_sections_graph()

        while True:
            async for chunk in graph.astream(state, config, stream_mode="updates"):
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    user_input = input(f"{chunk['__interrupt__'][0].value}")
                    state = Command(resume=user_input)
                    break
                else:
                    # pass
                    # pprint(chunk)
                    if isinstance(chunk, dict) and "human_feedback" in chunk:
                        pprint(chunk["human_feedback"])
            else:
                break

        pprint(state, indent=2, width=300)

    asyncio.run(main())
