import asyncio
import os

from langgraph.graph import END, START, StateGraph

from generate_article import get_article
from generate_questions import sections_subgraph
from sections_graph import graph as sections_graph
from states import ArticleState

graph_builder = StateGraph(ArticleState)
graph_builder.add_node("sections_subgraph", sections_subgraph)
graph_builder.add_node("sections_graph", sections_graph)
graph_builder.add_node("get_article", get_article)

graph_builder.add_edge(START, "sections_subgraph")
graph_builder.add_edge("sections_subgraph", "sections_graph")
graph_builder.add_edge("sections_graph", "get_article")

graph_builder.add_edge("get_article", END)

graph = graph_builder.compile()


async def main():
    os.makedirs("outputs", exist_ok=True)
    topic = await asyncio.get_event_loop().run_in_executor(None, input, "Тема для статьи: ")
    wishes = await asyncio.get_event_loop().run_in_executor(None, input, "Пожелания к статье: ")

    result = await graph.ainvoke({"topic": topic, "wishes": wishes})

    with open(f"outputs/{topic}.md", "w") as file:
        file.write(result["article"])

    print(result["article"])


asyncio.run(main())
