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


class RecomendatedBook(BaseModel):
    author: str = Field(description="Автор книги")
    name: str = Field(description="Название книги")
    isbn: str = Field(description="Номер ISBN")

    def __str__(self):
        return f"{self.author} - {self.name}({self.isbn})"


class RecommendationBlock(BaseModel):
    books: list[RecomendatedBook] = Field(
        default_factory=list, description="Рекомендуемые книги."
    )
    links: list[str] = Field(
        default_factory=list, description="Ссылки на рессурсы или статьи в интернете."
    )
    documentation: list[str] = Field(
        default_factory=list,
        description=(
            "Ссылки на страницы документации. "
            "[ВАЖНО!] Нужны только тех случаев, где действительно может быть какая то документация! "
            "К примеру - программирование. "
            "Во всех остальных случаях оставлять пустым."
        ),
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Запросы к поисковым системам, которые могут помочь в расширенном объяснении вопроса.",
    )
    other: list[str] = Field(
        default_factory=list,
        description="Какие то другие рессурсы, которые не относятся к уже описанным, но так же могут быть полезными.",
    )

    def __str__(self) -> str:

        recomendations: str = ""
        if self.books:
            recomendations += (
                "#### Книги:\n\t"
                + "\n\t".join(str(book) for book in self.books)
                + "\n\n"
            )
        if self.links:
            recomendations += "#### Ссылки:\n- " + "\n- ".join(self.links) + "\n\n"
        if self.documentation:
            recomendations += (
                "#### Документация:\n\t" + "\n\t".join(self.documentation) + "\n\n"
            )
        if self.search_queries:
            recomendations += (
                "#### Поисковые запросы:\n- "
                + "\n- ".join(self.search_queries)
                + "\n\n"
            )
        if self.other:
            recomendations += "#### Другое:\n\t" + "\n\t".join(self.other)

        return recomendations

    def is_empty(self):
        return all(
            len(item) <= 0
            for item in [
                self.books,
                self.links,
                self.documentation,
                self.search_queries,
                self.other,
            ]
        )


class SubSection(BaseModel):
    """Класс текста подсекции статьи"""

    subsection_title: str = Field(description="Заголовок подсекции статьи")
    subsection_content: str = Field(description="Текст подсекции статьи")
    recomendations: RecommendationBlock = Field(
        description="Рекомендации к текущей подсекции статьи"
    )

    def __str__(self):
        if not self.recomendations.is_empty():
            return f"## {self.subsection_title}\n\n{self.subsection_content}\n\n### Рекомендации:\n{self.recomendations}"
        else:
            return f"## {self.subsection_title}\n\n{self.subsection_content}"
