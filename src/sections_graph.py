from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from typing_extensions import Annotated, TypedDict

from llms import llm
from models import Section
from tools import search_engine, wikipedia_tool


class SectionState(TypedDict):
    topic: str
    sections: list[Section]
    messages: Annotated[list, add_messages]


# sections_prompt = ChatPromptTemplate.from_template(
#     """Вам необходимо написать раздел большой статьи посвященной теме:

#     {topic}.

#     Заголовок раздела статьи:
#     {title}

#     Краткое описание секции:
#     {description}

#     Для улучшения качества и формирования ответа вам доступны инструменты:
#     - search_engine - для поиска в интернете
#     - wikipedia - для поиска по Wikipedia

#     Дайте развернутый и содержательный ответ, соответствующий заголовку секции статьи.
#     К каждому разделу прикладывайте ссылки на используемый или полезный материал по теме раздела.
#     При необходимости используйте формат Markdown, для форматирования текста.
#     Ссылки форматируйте в виде сносок типа '[1]'."""
# )

sections_prompt = ChatPromptTemplate.from_template(
    """Вам необходимо написать **содержательную часть** раздела статьи на тему:

    {topic}.

    🔹 **Заголовок раздела**: {title}
    🔹 **Описание секции**: {description}

    📝 **Требования**:
    - **Не пишите введение и заключение** – только основную информацию.
    - **Будьте краткими и по делу**, не добавляйте "воду".
    - **Используйте Markdown** (заголовки, списки, примеры кода, таблицы).
    - **Если уместно, давайте примеры кода**.
    - **Ссылайтесь на источники** (например, '[1]').

    🎯 **Финальный текст должен содержать только полезную информацию по теме.**
    """
)


agent = create_react_agent(model=llm, tools=[wikipedia_tool, search_engine])


async def chatbot(state: SectionState):
    section_chain = sections_prompt | agent

    topic = state["topic"]
    sections = [Section.model_validate(section) for section in state["sections"]]

    results = []
    for section in sections:
        result = await section_chain.ainvoke(
            {
                "topic": topic,
                "title": section.section_title,
                "description": section.content,
            }
        )
        for message in result["messages"]:
            if isinstance(message, AIMessage):
                results.append(message)

    return {"sections": [message.content for message in results], "messages": results}


tool_node = ToolNode(tools=[wikipedia_tool, search_engine])

graph_builder = StateGraph(SectionState)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


async def sections_graph(state: SectionState):

    return await graph.ainvoke(state)


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await sections_graph(
            {
                "topic": "Python",
                "sections": [
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
                ],
                "messages": [],
            }
        )

        print(result)

    asyncio.run(main())
