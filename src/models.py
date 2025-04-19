from pydantic import BaseModel, Field


class Section(BaseModel):
    """Класс секции предварительного списка подсекций статьи"""

    section_title: str = Field(description="Заголовок секции статьи")
    content: str = Field(
        description="Подробное описание того, что должна содержать данная секция статьи."
    )

    def __str__(self) -> str:
        return f"\n## {self.section_title}\n\n{self.content}\n"


class SectionsList(BaseModel):
    """Класс предварительного списка подсекций статьи"""

    sections: list[Section] = Field(
        description="Список подтем для описания основной темы."
    )

    def __str__(self):
        return "\n".join([str(section) for section in self.sections])


class SubSection(BaseModel):
    """Класс текста подсекции статьи"""

    subsection_title: str = Field(description="Заголовок подсекции статьи")
    subsection_content: str = Field(description="Текст подсекции статьи")
    remmendations: str = Field(description="Рекомендации к текущей подсекции статьи")

    def __str__(self):
        return f"## {self.subsection_title}\n\n{self.subsection_content}\n\n\n###Рекомендации:\n{self.remmendations}"
