from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
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


content_generator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы являетесь техническим писателем и исследователем. "
            "Вам необходимо написать содержательную часть раздела статьи. "
            "При этом необходимо использовать доступные инструменты для поиска дополнительной информации. "
            "Обязательно запрашивайте дополнительные данные из Википедии или интернета, "
            "если вы не уверены в ответе или тема требует уточнения. "
            "При необходимости укажите ссылки на источники. "
            "Ответ должен быть детальным, но понятным для опытного читателя. "
            "Используйте Markdown для форматирования.",
        ),
        (
            "user",
            """
            Заголовок статьи: {topic}

            Подзаголовок, для которого нужно написать содержктельную часть: {title}

            Описание подраздела:\n{description}
            """,
        ),
    ]
)


agent = create_react_agent(model=llm, tools=[wikipedia_tool, search_engine])


async def generate_section_content(state: ContentGenerationState):
    section_chain = content_generator_prompt | agent

    topic = state["topic"]
    sections = [Section.model_validate(section) for section in state["sections"]]

    results: list[AIMessage] = []
    for section in sections:
        result = await section_chain.ainvoke(
            {
                "topic": topic,
                "title": section.section_title,
                "description": section.content,
            }
        )
        results.extend(
            message for message in result["messages"] if isinstance(message, AIMessage)
        )
    return {"sections": [message.content for message in results], "messages": results}


tool_node = ToolNode(tools=[wikipedia_tool, search_engine])

graph_builder = StateGraph(ContentGenerationState)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("generate_section_content", generate_section_content)

graph_builder.add_conditional_edges(
    "generate_section_content",
    tools_condition,
)

graph_builder.add_edge("tools", "generate_section_content")
graph_builder.add_edge(START, "generate_section_content")
graph_builder.add_edge("generate_section_content", END)

graph = graph_builder.compile()


async def outline_generator(state: ContentGenerationState):

    return await graph.ainvoke(state)


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await outline_generator(
            {
                "topic": "Python",
                "sections": SectionsList.model_validate(
                    [
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
                ),
                "messages": [],
            }
        )

        print(result)

    asyncio.run(main())
