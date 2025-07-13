from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from typing_extensions import Annotated, TypedDict

from models import SectionsList


class ArticleState(TypedDict):
    topic: str
    wishes: str
    sections: list[str]
    article: str


class OutlineState(TypedDict):
    topic: str
    wishes: Annotated[list[AnyMessage], add_messages]
    sections: SectionsList
    thinking_result: str


class ContentGenerationState(TypedDict):
    topic: str
    wishes: str
    sections: SectionsList
    messages: Annotated[list, add_messages]
    research_results: list[dict[str, str]]
    plans: list[dict[str, str]]
    writer_role: str
    article_strategy: str
