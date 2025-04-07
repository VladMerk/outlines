from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict

from models import SectionsList


class ArticleState(TypedDict):
    topic: str
    wishes: str
    sections: list[str]
    article: str


class OutlineState(TypedDict):
    topic: str
    wishes: Annotated[list[str], add_messages]
    sections: SectionsList


class ContentGenerationState(TypedDict):
    topic: str
    wishes: str
    sections: SectionsList
    messages: Annotated[list, add_messages]
    research_results: list[dict[str, str]]
    plans: list[dict[str, str]]
    writer_role: str
