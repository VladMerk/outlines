import asyncio
import os

from langgraph.graph import END, START, StateGraph

from article_assembler import assemble_article
from content_generator import graph as outline_generator
from states import ArticleState
from topic_structure import content_generator

graph_builder = StateGraph(ArticleState)
graph_builder.add_node("sections_subgraph", content_generator)
graph_builder.add_node("outline_generator", outline_generator)
graph_builder.add_node("assemble_article", assemble_article)

graph_builder.add_edge(START, "sections_subgraph")
graph_builder.add_edge("sections_subgraph", "outline_generator")
graph_builder.add_edge("outline_generator", "assemble_article")

graph_builder.add_edge("assemble_article", END)

graph = graph_builder.compile()


async def main():
    os.makedirs("outputs", exist_ok=True)
    topic = await asyncio.get_event_loop().run_in_executor(
        None, input, ">>> Тема для статьи: "
    )
    wishes = await asyncio.get_event_loop().run_in_executor(
        None, input, ">>> Пожелания к статье: "
    )

    result = await graph.ainvoke({"topic": topic, "wishes": wishes})

    with open(f"outputs/{topic}.md", "w") as file:
        file.write(result["article"])

    os.system("clear")
    print(result["article"])


asyncio.run(main())
