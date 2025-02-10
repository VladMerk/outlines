from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain as as_runnable

from llms import llm
from models import Outline
from states import ArticleState

outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы технический писатель и специалист в области информационных технологий и языков программирования."
            "Напишите подробный план статьи для полного ответа на тему предоставленную пользователем."
            " Будьте всеобъемлющим и конкретным. Для форматирования, там где нужно, используйте формат Markdown.",
        ),
        ("user", "{topic}"),
    ]
)


@as_runnable
async def get_outline(state: ArticleState):
    outline_chain = outline_prompt | llm.with_structured_output(Outline)
    outline = await outline_chain.ainvoke(state["topic"])  # type: ignore

    return {**state, "outline": outline}
