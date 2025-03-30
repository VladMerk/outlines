from states import ArticleState


async def assemble_article(state: ArticleState):
    topic = state["topic"]
    sections: list[str] = state["sections"]

    # Форматируем финальную статью с заголовком
    article = f"# {topic}\n" + "\n".join(str(section) for section in sections)

    return {**state, "article": article}
