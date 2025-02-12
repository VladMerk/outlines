import asyncio
import os

from langgraph.graph import END, START, StateGraph

from generate_article import get_article
from outline_generator import get_outline
from sections_graph import graph as sections_graph
from states import ArticleState

graph_builder = StateGraph(ArticleState)
graph_builder.add_node("outlines", get_outline)
graph_builder.add_node("sections_graph", sections_graph)

graph_builder.add_node("get_article", get_article)

graph_builder.add_edge(START, "outlines")
graph_builder.add_edge("outlines", "sections_graph")
graph_builder.add_edge("sections_graph", "get_article")

graph_builder.add_edge("get_article", END)

graph = graph_builder.compile()


async def main():
    os.makedirs("outputs", exist_ok=True)
    topic = input("Тема для статьи: ")

    result = await graph.ainvoke({"topic": topic})

    with open(f"outputs/{topic}.md", "w") as file:
        file.write(result["article"])

    print(result["article"])


asyncio.run(main())
