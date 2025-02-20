from typing import Optional

from typing_extensions import TypedDict

from models import Outline, SectionsList


class ArticleState(TypedDict):
    topic: str
    wishes: str
    outline: Outline
    sections: list[str]
    article: str


class SectionsState(TypedDict):
    topic: str
    wishes: Optional[str]
    sections: Optional[SectionsList]
    new_sections: Optional[SectionsList]
    human_answer: Optional[str]
