from pydantic import BaseModel, Field


class Section(BaseModel):
    section_title: str = Field(description="Заголовок секции статьи")
    content: str = Field(
        description="Описание того, что должна содержать данная секция статьи."
    )

    def __str__(self) -> str:
        return f"\n## {self.section_title}\n\n{self.content}\n"


class Outline(BaseModel):
    title: str = Field(description="Заголовок статьи")
    sections: list[Section] = Field(
        default_factory=list, description="Заголовки и содержания всех секций статьи"
    )

    def __str__(self) -> str:
        return f"# {self.title}" + "\n".join(str(section) for section in self.sections)


class SubSection(BaseModel):
    section: str = Field(description="Подтема основной темы статьи.")
    description: str = Field(description="Краткое содержание подтемы.")

    def __str__(self):
        return f"{self.section}\n{self.description}"


class SectionsList(BaseModel):
    sections: list[SubSection] = Field(
        description="Список подтем для описания основной темы."
    )
