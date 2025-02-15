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

llm.temperature = 0.5
generate_question_chain = sections_prompt | llm.with_structured_output(SectionsList)


async def generate_questions(state: SectionsState):
    return {
        "sections": SectionsList.model_validate(
            await generate_question_chain.ainvoke({"topic": state["topic"]})
        ).sections
    }


async def select_sections(state: SectionsState):
    sections = state["sections"]
    if sections:
        for i, section in enumerate(sections):
            section = Section.model_validate(section)
            print(f"[{i+1}] {section.section}:\n{section.description}\n")
    answer = interrupt(
        "Выберите интересующие подтемы Вашего вопроса и напишите их через запятую в одну строку: ",
    )

    return {**state, "human_answer": answer}


async def get_human_feedback(state: SectionsState):
    answer = state["human_answer"]
    # print(answer)
    human_sections = list(map(int, [a.strip() for a in answer.split(",")]))
    new_sections = [
        t for i, t in enumerate(state["sections"]) if i + 1 in human_sections
    ]
    return {"human_answer": new_sections}


def get_sections_graph() -> CompiledStateGraph:

    graph_builder = StateGraph(SectionsState)

    graph_builder.add_node("generate_sections", generate_questions)
    graph_builder.add_node("select_sections", select_sections)
    graph_builder.add_node("human_feedback", get_human_feedback)

    graph_builder.add_edge(START, "generate_sections")
    graph_builder.add_edge("generate_sections", "select_sections")
    graph_builder.add_edge("select_sections", "human_feedback")
    graph_builder.add_edge("human_feedback", END)

    checkpoint = MemorySaver()

    return graph_builder.compile(checkpointer=checkpoint)


if __name__ == "__main__":
    from pprint import pprint

    async def main():
        config = RunnableConfig(configurable={"thread_id": "1"})
        topic = input("Topic: ")

        graph = get_sections_graph()

        result = await graph.ainvoke({"topic": topic}, config, stream_mode="updates")
        pprint(result[-1])

        if isinstance(result[-1], dict) and "__interrupt__" in result[-1]:
            user_input = input(result[-1]["__interrupt__"][0].value)

            result = await graph.ainvoke(Command(resume=user_input), config)

        print(result["human_answer"])

    asyncio.run(main())
