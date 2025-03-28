from states import ArticleState


async def get_article(state: ArticleState):
    topic = state["topic"]
    sections: list[str] = state["sections"]

    article = f"\n# {topic}\n" + "\n".join(list(sections))

    return {
        **state,
        "article": article
    }
