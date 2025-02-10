from typing_extensions import TypedDict

from models import Outline


class ArticleState(TypedDict):
    topic: str
    outline: Outline
    sections: list[str]
    article: str
